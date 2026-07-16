"""Confined argv execution for one command-capable Agent Skill."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import jsonschema

from client_backend.core.config import client_settings
from client_backend.core.paths import PathSecurityError, is_under_root, validate_workspace_path
from client_backend.services.local_skills_registry import LocalSkillsRegistry, get_skills_registry
from client_backend.services.skill_runtime.audit import SkillAuditWriter, new_audit_id
from client_backend.services.skill_runtime.environment import SkillEnvironmentManager
from client_backend.services.skill_runtime.manager import (
    COMMAND_INPUT_SCHEMA,
    RUN_SKILL_COMMAND,
    SkillRuntimeManager,
)
from client_backend.services.skill_runtime.secrets import SkillSecretStore, redact_secret_values
from shared.skills.commands import is_supported_bundle_command
from shared.skills.errors import (
    CAPABILITY_NOT_FOUND,
    COMMAND_NOT_FOUND,
    EXECUTION_TIMEOUT,
    INVALID_ARGUMENTS,
    OUTPUT_TOO_LARGE,
    PERMISSION_REQUIRED,
    RUNTIME_ERROR,
    SKILL_NOT_READY,
    SKILL_RUNTIME_STALE,
    SkillRuntimeError,
    error_payload,
)

DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 300
MAX_OUTPUT_BYTES = 1_000_000

_ENV_PASSTHROUGH_NAMES = frozenset(
    name.upper()
    for name in {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "TERM",
        "TMPDIR",
        "PYTHONIOENCODING",
        "PYTHONUNBUFFERED",
        "PYTHONDONTWRITEBYTECODE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "PATHEXT",
        "COMSPEC",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "OS",
    }
)


class _OutputLimitExceeded(Exception):
    """Internal signal used to terminate a child while reading its output."""


class _JobBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("per_process_user_time_limit", ctypes.c_int64),
        ("per_job_user_time_limit", ctypes.c_int64),
        ("limit_flags", ctypes.c_ulong),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", ctypes.c_ulong),
        ("affinity", ctypes.c_size_t),
        ("priority_class", ctypes.c_ulong),
        ("scheduling_class", ctypes.c_ulong),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("read_operation_count", ctypes.c_uint64),
        ("write_operation_count", ctypes.c_uint64),
        ("other_operation_count", ctypes.c_uint64),
        ("read_transfer_count", ctypes.c_uint64),
        ("write_transfer_count", ctypes.c_uint64),
        ("other_transfer_count", ctypes.c_uint64),
    ]


class _JobExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("basic_limit_information", _JobBasicLimitInformation),
        ("io_info", _IoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    ]


class _WindowsKillJob:
    """A kill-on-close Windows Job Object containing one command tree."""

    _KILL_ON_JOB_CLOSE = 0x2000
    _EXTENDED_LIMIT_INFORMATION = 9
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100

    def __init__(self, kernel32, handle) -> None:
        self._kernel32 = kernel32
        self._handle = handle

    @classmethod
    def attach(cls, pid: int) -> _WindowsKillJob | None:
        if os.name != "nt":
            return None

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle_type = ctypes.c_void_p
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel32.CreateJobObjectW.restype = handle_type
        kernel32.SetInformationJobObject.argtypes = [
            handle_type,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        kernel32.SetInformationJobObject.restype = ctypes.c_int
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = handle_type
        kernel32.AssignProcessToJobObject.argtypes = [handle_type, handle_type]
        kernel32.AssignProcessToJobObject.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [handle_type]
        kernel32.CloseHandle.restype = ctypes.c_int

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        process_handle = None
        try:
            limits = _JobExtendedLimitInformation()
            limits.basic_limit_information.limit_flags = cls._KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                job,
                cls._EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(limits),
                ctypes.sizeof(limits),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            process_handle = kernel32.OpenProcess(
                cls._PROCESS_TERMINATE | cls._PROCESS_SET_QUOTA,
                False,
                pid,
            )
            if not process_handle:
                raise ctypes.WinError(ctypes.get_last_error())
            if not kernel32.AssignProcessToJobObject(job, process_handle):
                raise ctypes.WinError(ctypes.get_last_error())
            return cls(kernel32, job)
        except OSError:
            kernel32.CloseHandle(job)
            raise
        finally:
            if process_handle:
                kernel32.CloseHandle(process_handle)

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


class SkillExecutionEngine:
    """Execute the reserved command capability without a shell or global lookup."""

    def __init__(
        self,
        registry: LocalSkillsRegistry | None = None,
        environment_manager: SkillEnvironmentManager | None = None,
        secret_store: SkillSecretStore | None = None,
        manager: SkillRuntimeManager | None = None,
        audit_writer: SkillAuditWriter | None = None,
    ) -> None:
        self._registry = registry if registry is not None else get_skills_registry()
        self._environment = environment_manager or SkillEnvironmentManager()
        self._secrets = secret_store or SkillSecretStore()
        self._manager = manager or SkillRuntimeManager(self._environment)
        self._audit = audit_writer or SkillAuditWriter()

    async def execute(
        self,
        qualified_tool_id: str,
        arguments: dict | None = None,
        context: dict | None = None,
    ) -> dict:
        start = time.monotonic()
        audit_id = new_audit_id()
        working_arguments = dict(arguments or {})
        call_context = dict(context or {})
        skill_name = ""
        secret_values: set[str] = set()

        try:
            skill_name, capability = self._parse_qualified_id(qualified_tool_id)
            if capability != RUN_SKILL_COMMAND:
                raise SkillRuntimeError(
                    CAPABILITY_NOT_FOUND,
                    f"skill '{skill_name}' exposes only '{RUN_SKILL_COMMAND}'",
                )
            skill = self._registry.get_skill(skill_name)
            if skill is None or not skill.enabled:
                raise SkillRuntimeError(
                    CAPABILITY_NOT_FOUND, f"skill '{skill_name}' is not available"
                )

            readiness = self._manager.evaluate_readiness(skill)
            if readiness.status != "ready":
                raise SkillRuntimeError(
                    SKILL_NOT_READY,
                    f"skill '{skill_name}' is not ready",
                    repair=readiness.to_summary(),
                )

            self._validate_arguments(working_arguments)
            working_arguments.setdefault("cwd", "workspace")

            if call_context.get("mutation_approved") is not True:
                raise SkillRuntimeError(
                    PERMISSION_REQUIRED,
                    f"running a command from skill '{skill_name}' requires approval",
                    repair={
                        "type": "approve_skill_command",
                        "skill": skill_name,
                        "source_hash": skill.source_hash,
                    },
                )

            try:
                live_source_hash = await asyncio.to_thread(
                    LocalSkillsRegistry._compute_source_hash,
                    skill.bundle_root,
                )
            except (OSError, ValueError) as exc:
                raise SkillRuntimeError(
                    SKILL_RUNTIME_STALE,
                    f"skill '{skill_name}' changed or became unsafe after publication",
                    repair={"type": "reload_skill", "skill": skill_name},
                ) from exc
            if live_source_hash != skill.source_hash:
                raise SkillRuntimeError(
                    SKILL_RUNTIME_STALE,
                    f"skill '{skill_name}' changed after its command tool was published",
                    repair={"type": "reload_skill", "skill": skill_name},
                )

            skill_secrets = self._skill_secrets(skill_name)
            secret_values = {value for value in skill_secrets.values() if value}
            argv = list(working_arguments["argv"])
            command, runtime_root = self._resolve_owned_command(skill, argv[0])
            cmd = [*command, *argv[1:]]
            cwd = self._resolve_cwd(skill.bundle_root, working_arguments["cwd"], call_context)
            env = self._build_environment(
                bundle_root=skill.bundle_root,
                runtime_root=runtime_root,
                secrets=skill_secrets,
            )
            timeout = self._clamp_timeout(call_context.get("timeout_seconds"))
            stdout, stderr, returncode = await self._run_and_collect(
                cmd, cwd, env, timeout, secret_values
            )
            if returncode != 0:
                raise SkillRuntimeError(
                    RUNTIME_ERROR,
                    f"skill command exited with code {returncode}: {stderr[-4000:]}",
                )
            try:
                result = json.loads(stdout)
            except json.JSONDecodeError:
                result = stdout

            self._write_audit(
                audit_id,
                skill_name,
                qualified_tool_id,
                working_arguments,
                secret_values,
                "ok",
                start,
                call_context,
            )
            return self._success_envelope(skill_name, result, stdout, stderr, start, audit_id)
        except SkillRuntimeError as exc:
            self._write_audit(
                audit_id,
                skill_name,
                qualified_tool_id,
                working_arguments,
                secret_values,
                "error",
                start,
                call_context,
                error_code=exc.code,
            )
            return self._failure_envelope(
                skill_name,
                error_payload(exc.code, exc.message, exc.repair),
                start,
                audit_id,
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            self._write_audit(
                audit_id,
                skill_name,
                qualified_tool_id,
                working_arguments,
                secret_values,
                "error",
                start,
                call_context,
                error_code=RUNTIME_ERROR,
            )
            return self._failure_envelope(
                skill_name,
                error_payload(RUNTIME_ERROR, f"unexpected runtime error: {exc.__class__.__name__}"),
                start,
                audit_id,
            )

    @staticmethod
    def _parse_qualified_id(value: str) -> tuple[str, str]:
        parts = value.split("::")
        if len(parts) != 3 or parts[0] != "skill" or not parts[1] or not parts[2]:
            raise SkillRuntimeError(
                CAPABILITY_NOT_FOUND,
                "malformed skill tool id; expected skill::<name>::run_skill_command",
            )
        return parts[1], parts[2]

    @staticmethod
    def _validate_arguments(arguments: dict) -> None:
        try:
            jsonschema.validate(arguments, COMMAND_INPUT_SCHEMA)
        except jsonschema.ValidationError as exc:
            path = ".".join(str(part) for part in exc.absolute_path) or "<root>"
            raise SkillRuntimeError(
                INVALID_ARGUMENTS,
                f"argument validation failed at '{path}' (constraint: {exc.validator})",
            ) from exc

    def _resolve_owned_command(self, skill, requested: str) -> tuple[list[str], Path | None]:
        if not requested or requested in {".", ".."}:
            raise SkillRuntimeError(COMMAND_NOT_FOUND, "skill command is empty or invalid")
        normalized = requested.replace("\\", "/")
        if "/" in normalized:
            if not normalized.startswith("scripts/") or not normalized.endswith(".py"):
                raise SkillRuntimeError(
                    COMMAND_NOT_FOUND,
                    f"command is not owned by skill '{skill.name}': {requested}",
                )
            script = (skill.bundle_root / normalized).resolve()
            scripts_root = (skill.bundle_root / "scripts").resolve()
            if (
                not is_under_root(script, scripts_root)
                or not script.is_file()
                or script.is_symlink()
            ):
                raise SkillRuntimeError(COMMAND_NOT_FOUND, f"script not found: {requested}")
            python = self._environment.python_executable(skill) or Path(sys.executable)
            return [str(python), str(script)], self._runtime_root(skill)

        runtime_inspection = self._environment.inspect(skill)
        runtime_dir = self._environment.command_directory(skill)
        search_dirs = [skill.bundle_root / "bin"]
        runtime_commands = {str(command) for command in runtime_inspection.get("commands", [])}
        if runtime_dir is not None and requested in runtime_commands:
            search_dirs.append(runtime_dir)
        for directory in search_dirs:
            for candidate_name in self._candidate_names(requested):
                candidate = (directory / candidate_name).resolve()
                if not is_under_root(candidate, directory) or not is_supported_bundle_command(
                    candidate
                ):
                    continue
                runtime_root = self._runtime_root(skill)
                if candidate.suffix.lower() == ".py":
                    python = self._environment.python_executable(skill) or Path(sys.executable)
                    return [str(python), str(candidate)], runtime_root
                return [str(candidate)], runtime_root
        raise SkillRuntimeError(
            COMMAND_NOT_FOUND,
            f"command is not owned by skill '{skill.name}': {requested}",
            repair={"type": "inspect_skill_commands", "skill": skill.name},
        )

    @staticmethod
    def _candidate_names(requested: str) -> list[str]:
        if Path(requested).suffix:
            return [requested]
        return [requested, f"{requested}.py", f"{requested}.exe"]

    def _runtime_root(self, skill) -> Path | None:
        command_dir = self._environment.command_directory(skill)
        if command_dir is None:
            return None
        # <runtime>/venv/Scripts or <runtime>/venv/bin
        return command_dir.parent.parent

    @staticmethod
    def _resolve_cwd(bundle_root: Path, choice: str, context: dict) -> Path:
        if choice == "skill":
            return bundle_root
        configured_roots = [
            str(root).strip() for root in client_settings.workspace_roots if str(root).strip()
        ]
        workspace = Path(
            context.get("workspace_root")
            or (configured_roots[0] if configured_roots else Path.cwd())
        )
        try:
            return validate_workspace_path(workspace)
        except PathSecurityError as exc:
            raise SkillRuntimeError(INVALID_ARGUMENTS, str(exc)) from exc

    @staticmethod
    def _build_environment(
        *,
        bundle_root: Path,
        runtime_root: Path | None,
        secrets: dict[str, str],
    ) -> dict[str, str]:
        env = {
            name: value
            for name, value in os.environ.items()
            if name.upper() in _ENV_PASSTHROUGH_NAMES
        }
        path_parts = [str(bundle_root / "bin")]
        if runtime_root is not None:
            runtime_bin = runtime_root / "venv" / ("Scripts" if os.name == "nt" else "bin")
            path_parts.append(str(runtime_bin))
        env["PATH"] = os.pathsep.join(path_parts)
        env["SKILL_ROOT"] = str(bundle_root)
        env["SKILL_RUNTIME_ROOT"] = str(runtime_root) if runtime_root is not None else ""
        env.update(secrets)
        return env

    def _skill_secrets(self, skill_name: str) -> dict[str, str]:
        getter = getattr(self._secrets, "get_for_skill", None)
        if getter is None:
            return {}
        return {
            str(name): str(value) for name, value in getter(skill_name).items() if value is not None
        }

    @staticmethod
    def _clamp_timeout(requested: object) -> float:
        if not isinstance(requested, (int, float)) or isinstance(requested, bool):
            return float(DEFAULT_TIMEOUT_SECONDS)
        if not math.isfinite(requested) or requested <= 0:
            return float(DEFAULT_TIMEOUT_SECONDS)
        return float(min(requested, MAX_TIMEOUT_SECONDS))

    @staticmethod
    async def _run_and_collect(
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        timeout: float,
        secret_values: set[str],
    ) -> tuple[str, str, int]:
        try:
            process, windows_job = await SkillExecutionEngine._spawn_contained_process(
                cmd, cwd, env
            )
        except FileNotFoundError as exc:
            raise SkillRuntimeError(
                COMMAND_NOT_FOUND,
                "skill command executable disappeared",
            ) from exc
        except OSError as exc:
            raise SkillRuntimeError(
                RUNTIME_ERROR,
                "unable to establish the command process boundary",
            ) from exc

        async def read_limited(stream: asyncio.StreamReader) -> bytes:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = await stream.read(64 * 1024)
                if not chunk:
                    return b"".join(chunks)
                total += len(chunk)
                if total > MAX_OUTPUT_BYTES:
                    raise _OutputLimitExceeded
                chunks.append(chunk)

        stdout_task = asyncio.create_task(read_limited(process.stdout))
        stderr_task = asyncio.create_task(read_limited(process.stderr))
        wait_task = asyncio.create_task(process.wait())
        tasks = (stdout_task, stderr_task, wait_task)
        try:
            stdout_bytes, stderr_bytes, returncode = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=timeout,
            )
        except TimeoutError as exc:
            await SkillExecutionEngine._terminate_process_tree(process, windows_job)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise SkillRuntimeError(
                EXECUTION_TIMEOUT,
                f"skill command exceeded {timeout}s timeout",
            ) from exc
        except _OutputLimitExceeded as exc:
            await SkillExecutionEngine._terminate_process_tree(process, windows_job)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise SkillRuntimeError(
                OUTPUT_TOO_LARGE,
                f"skill command output exceeded {MAX_OUTPUT_BYTES} bytes",
            ) from exc
        except BaseException:
            await SkillExecutionEngine._terminate_process_tree(process, windows_job)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            if windows_job is not None:
                windows_job.close()

        stdout = redact_secret_values(
            stdout_bytes.decode("utf-8", errors="replace"), secret_values
        ).replace("\r\n", "\n")
        stderr = redact_secret_values(
            stderr_bytes.decode("utf-8", errors="replace"), secret_values
        ).replace("\r\n", "\n")
        return stdout, stderr, returncode

    @staticmethod
    async def _spawn_contained_process(
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
    ) -> tuple[asyncio.subprocess.Process, _WindowsKillJob | None]:
        windows = os.name == "nt"
        spawn_cmd = cmd
        stdin = asyncio.subprocess.DEVNULL
        if windows:
            launcher = Path(__file__).with_name("windows_job_launcher.py")
            spawn_cmd = [sys.executable, "-I", str(launcher)]
            stdin = asyncio.subprocess.PIPE

        process = await asyncio.create_subprocess_exec(
            *spawn_cmd,
            cwd=str(cwd),
            env=env,
            stdin=stdin,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=(
                subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if windows else 0
            ),
            start_new_session=not windows,
        )
        if not windows:
            return process, None

        windows_job: _WindowsKillJob | None = None
        try:
            windows_job = _WindowsKillJob.attach(process.pid)
            if windows_job is None or process.stdin is None:
                raise OSError("Windows process containment is unavailable")
            request = (
                json.dumps(
                    cmd,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            process.stdin.write(request)
            await process.stdin.drain()
            process.stdin.close()
            return process, windows_job
        except BaseException:
            if windows_job is not None:
                windows_job.close()
            if process.stdin is not None:
                process.stdin.close()
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            with contextlib.suppress(TimeoutError, ProcessLookupError):
                await asyncio.wait_for(process.wait(), timeout=5)
            raise

    @staticmethod
    async def _terminate_process_tree(
        process: asyncio.subprocess.Process,
        windows_job: _WindowsKillJob | None,
    ) -> None:
        if os.name == "nt":
            if windows_job is not None:
                windows_job.close()
            elif process.returncode is None:
                system_root = Path(os.environ.get("SYSTEMROOT") or r"C:\Windows")
                taskkill = system_root / "System32" / "taskkill.exe"
                if taskkill.is_file():
                    with contextlib.suppress(OSError):
                        killer = await asyncio.create_subprocess_exec(
                            str(taskkill),
                            "/PID",
                            str(process.pid),
                            "/T",
                            "/F",
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW,
                        )
                        await asyncio.wait_for(killer.wait(), timeout=5)
        else:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        with contextlib.suppress(TimeoutError, ProcessLookupError):
            await asyncio.wait_for(process.wait(), timeout=5)

    def _write_audit(
        self,
        audit_id: str,
        skill_name: str,
        qualified_id: str,
        arguments: dict,
        secret_values: set[str],
        status: str,
        start: float,
        context: dict,
        error_code: str | None = None,
    ) -> None:
        self._audit.write(
            audit_id=audit_id,
            skill=skill_name,
            capability=RUN_SKILL_COMMAND,
            qualified_id=qualified_id,
            arguments=arguments,
            secret_values=secret_values,
            status=status,
            duration_ms=self._elapsed_ms(start),
            error_code=error_code,
            device_id=context.get("device_id"),
            session_id=context.get("session_id"),
        )

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int((time.monotonic() - start) * 1000)

    def _success_envelope(
        self,
        skill_name: str,
        result: object,
        stdout: str,
        stderr: str,
        start: float,
        audit_id: str,
    ) -> dict:
        return {
            "ok": True,
            "skill": skill_name,
            "capability": RUN_SKILL_COMMAND,
            "result": result,
            "stdout": stdout,
            "stderr": stderr,
            "duration_ms": self._elapsed_ms(start),
            "audit_id": audit_id,
        }

    def _failure_envelope(
        self,
        skill_name: str,
        payload: dict,
        start: float,
        audit_id: str,
    ) -> dict:
        return {
            "ok": False,
            "skill": skill_name,
            "capability": RUN_SKILL_COMMAND,
            "error": payload,
            "duration_ms": self._elapsed_ms(start),
            "audit_id": audit_id,
        }
