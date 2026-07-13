"""Generic execution engine for skill capabilities.

Ties together the manifest, readiness, permission, and secret modules to
actually invoke one capability of one skill and return a normalized result
or error envelope. This is the only place in the skill runtime that spawns a
child process, so every safety property the plan requires (no shell, argv
rendered as a list never a string, permission check before secret
resolution/argv rendering/spawning, timeouts, output limits, secret
redaction) is enforced here.

Platform note: the sidecar forces a Selector event loop on Windows (so
psycopg keeps working), and ``asyncio.create_subprocess_exec`` raises
``NotImplementedError`` on a Selector loop on Windows. This module therefore
never uses ``asyncio`` subprocess APIs -- it runs ``subprocess.run`` (which
never invokes a shell when given an argv list) inside ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import jsonschema

from client_backend.core.logging import get_logger
from client_backend.core.paths import is_under_root
from client_backend.services.local_skills_registry import LocalSkillsRegistry, get_skills_registry
from client_backend.services.skill_runtime.audit import SkillAuditWriter, new_audit_id
from client_backend.services.skill_runtime.manager import SkillRuntimeManager
from client_backend.services.skill_runtime.permissions import (
    SkillPermissionEvaluator,
    SkillPermissionPolicy,
)
from client_backend.services.skill_runtime.secrets import SkillSecretStore
from shared.skills.errors import (
    CAPABILITY_NOT_FOUND,
    COMMAND_NOT_FOUND,
    EXECUTION_TIMEOUT,
    INVALID_ARGUMENTS,
    MISSING_SECRET,
    NON_JSON_OUTPUT,
    OUTPUT_TOO_LARGE,
    RUNTIME_ERROR,
    SKILL_NOT_READY,
    UNSUPPORTED_RUNTIME,
    SkillRuntimeError,
    error_payload,
)
from shared.skills.manifest import SkillCapabilitySpec, SkillManifest, SkillRuntimeSpec

logger = get_logger(__name__)

# A capability call may run for a while (network round trips, local file
# scans); 30s is a generous first-slice default. Callers can override per
# call via ``context={"timeout_seconds": ...}``.
DEFAULT_TIMEOUT_SECONDS = 30

# 1 MB. Generous enough for a normal JSON/text response, small enough that a
# runaway or malicious capability cannot exhaust sidecar memory buffering
# output that is fully captured before being returned.
MAX_OUTPUT_BYTES = 1_000_000

# Hard ceiling on a per-call timeout override so an untrusted caller (once
# Task 8 wires `context`) cannot request an effectively unbounded run.
MAX_TIMEOUT_SECONDS = 300

# Allow-list of environment variable NAMES passed through to a skill child
# process. The child gets ONLY these infrastructure vars plus the secrets this
# capability explicitly declares (injected separately) -- never the sidecar's
# full os.environ, which would hand every skill every OTHER skill's (and the
# app's own) secrets. Names are matched case-insensitively (Windows env vars
# are case-insensitive). Skill-specific configuration must arrive via declared
# secrets or arguments, not arbitrary inherited env.
_ENV_PASSTHROUGH_NAMES = frozenset(
    name.upper()
    for name in {
        # POSIX / cross-platform runtime + locale + TLS trust store.
        "PATH",
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
        # Windows: required for the interpreter to even start / spawn.
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

# Matches `{identifier}` placeholders inside one argv template element.
# Capability argument keys are schema property names (identifiers), so this
# intentionally does not attempt to handle arbitrary/nested expressions.
_ARGV_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

_REDACTION_PLACEHOLDER = "<redacted>"


def _redact(text: str, secret_values: set[str]) -> str:
    """Replace every occurrence of a resolved secret value with a placeholder.

    Values are replaced LONGEST-FIRST. If one secret value is a substring of
    another (e.g. ``"abc"`` and ``"abcdef"``), replacing the shorter one first
    would fragment the longer one and leak its tail; replacing the longest
    first makes redaction complete and independent of set iteration order
    (which is otherwise randomized by PYTHONHASHSEED). Empty values are skipped
    so an unset secret can never become a no-op ``str.replace("", ...)`` that
    would corrupt the output.
    """
    redacted = text
    for value in sorted(secret_values, key=len, reverse=True):
        if not value:
            continue
        redacted = redacted.replace(value, _REDACTION_PLACEHOLDER)
    return redacted


def _build_scoped_env(extra_env: dict[str, str], env_secrets: dict[str, str]) -> dict[str, str]:
    """Build a skill child's environment from an allow-list, never full os.environ.

    Passing the sidecar's whole ``os.environ`` would hand every skill every
    OTHER skill's (and the app's own) secrets, defeating the per-capability
    secret scoping this engine enforces. Instead the child gets only the
    infrastructure vars in ``_ENV_PASSTHROUGH_NAMES``, plus runtime-specific
    ``extra_env`` (e.g. PYTHONPATH) and this capability's own resolved
    ``env_secrets``.
    """
    env = {
        name: value
        for name, value in os.environ.items()
        if name.upper() in _ENV_PASSTHROUGH_NAMES
    }
    env.update(extra_env)
    env.update(env_secrets)
    return env


class SkillExecutionEngine:
    """Executes one skill capability by qualified id and returns a normalized envelope.

    Default permission policy (first slice):
    ``SkillPermissionPolicy(granted=frozenset({"*"}), allow_mutation=False)``.
    This trusts the resource permissions (network/filesystem/process/domain
    tokens) a user-installed skill declares, but unconditionally blocks
    ``mutation: true`` capabilities -- those always come back as
    ``PERMISSION_REQUIRED`` here, which is the hand-off point to Task 9's
    human-in-the-loop approval flow. The reserved ``shell`` runtime is
    hard-denied by :class:`SkillPermissionEvaluator` regardless of policy.
    This is deliberately more permissive on resource grants than
    ``SkillPermissionPolicy.deny_all()`` and MUST NOT default to
    ``allow_all()`` (which would also allow mutation outright). A stricter,
    per-resource-scoped policy can be injected via ``permission_policy`` once
    a device-level permission UI exists.
    """

    def __init__(
        self,
        registry: LocalSkillsRegistry | None = None,
        secret_store: SkillSecretStore | None = None,
        permission_policy: SkillPermissionPolicy | None = None,
        manager: SkillRuntimeManager | None = None,
        audit_writer: SkillAuditWriter | None = None,
    ) -> None:
        self._registry = registry if registry is not None else get_skills_registry()
        self._secret_store = secret_store if secret_store is not None else SkillSecretStore()
        self._audit = audit_writer if audit_writer is not None else SkillAuditWriter()
        # Default the readiness manager onto the SAME secret store the engine
        # uses, so readiness and execution can never disagree about which
        # secrets exist (they only diverge if a caller deliberately passes a
        # mismatched manager). This matters once a non-os.environ store (Task
        # 10) is injected.
        self._manager = (
            manager if manager is not None else SkillRuntimeManager(secret_store=self._secret_store)
        )
        self._policy = (
            permission_policy
            if permission_policy is not None
            else SkillPermissionPolicy(granted=frozenset({"*"}), allow_mutation=False)
        )

    async def execute(
        self,
        qualified_tool_id: str,
        arguments: dict | None = None,
        context: dict | None = None,
    ) -> dict:
        """Run one capability and return a normalized success/failure envelope.

        Never raises for expected failures (parse errors, missing
        capability, not-ready skill, invalid arguments, permission denial,
        missing secret, command-not-found, timeout, output-too-large,
        non-JSON output, non-zero exit). Any truly unexpected exception is
        also caught and converted into a ``RUNTIME_ERROR`` envelope rather
        than propagating, so a skill-runtime bug can never crash the caller.
        """
        start = time.monotonic()
        audit_id = new_audit_id()
        working_arguments: dict = dict(arguments) if arguments else {}
        call_context: dict = context or {}
        skill_name = ""
        capability_name = ""
        secret_values: set[str] = set()

        try:
            skill_name, capability_name = self._parse_qualified_tool_id(qualified_tool_id)

            skill = self._registry.get_skill(skill_name)
            if skill is None or not skill.enabled:
                raise SkillRuntimeError(
                    CAPABILITY_NOT_FOUND, f"skill '{skill_name}' is not available"
                )

            if skill.manifest is None:
                raise SkillRuntimeError(
                    SKILL_NOT_READY,
                    f"skill '{skill_name}' has no executable manifest (instruction-only)",
                )

            manifest = skill.manifest
            capability = self._find_capability(manifest, capability_name)
            if capability is None:
                raise SkillRuntimeError(
                    CAPABILITY_NOT_FOUND,
                    f"capability '{capability_name}' not found on skill '{skill_name}'",
                )

            readiness = self._manager.evaluate_readiness(manifest, skill.manifest_error)
            if readiness.status != "ready":
                raise SkillRuntimeError(
                    SKILL_NOT_READY,
                    f"skill '{skill_name}' is not ready (status={readiness.status})",
                    repair=readiness.to_summary(),
                )

            self._validate_arguments(capability, working_arguments)

            decision = SkillPermissionEvaluator(self._policy).evaluate(
                capability=capability, runtime=manifest.runtime
            )
            if not decision.allowed:
                logger.info(
                    "skill capability %s::%s blocked by permission policy (%s)",
                    skill_name,
                    capability_name,
                    decision.code,
                )
                # A permission denial (blocked mutation, reserved shell, ungranted
                # resource) is a security-relevant outcome and must be audited too,
                # even though it returns rather than raising. secret_values is still
                # empty here (this is before secret resolution).
                self._audit.write(
                    audit_id=audit_id,
                    skill=skill_name,
                    capability=capability_name,
                    qualified_id=qualified_tool_id,
                    arguments=working_arguments,
                    secret_values=secret_values,
                    status="error",
                    duration_ms=self._elapsed_ms(start),
                    error_code=decision.code,
                    device_id=call_context.get("device_id"),
                    session_id=call_context.get("session_id"),
                )
                return self._failure_envelope(
                    skill_name,
                    capability_name,
                    decision.to_error_payload(working_arguments),
                    start,
                    audit_id,
                )

            secret_values, env_secrets = self._resolve_secrets(manifest, capability)

            rendered_argv = self._render_argv(capability.execution.argv, working_arguments)

            skill_dir = skill.path.parent
            cmd, extra_env = self._build_command(manifest.runtime, rendered_argv, skill_dir)

            env = _build_scoped_env(extra_env, env_secrets)

            timeout = self._clamp_timeout(call_context.get("timeout_seconds"))

            stdout_text, stderr_text, returncode = await self._run_and_collect(
                cmd, skill_dir, env, timeout, secret_values
            )

            if returncode != 0:
                stderr_tail = stderr_text[-4000:]
                raise SkillRuntimeError(
                    RUNTIME_ERROR,
                    f"capability exited with code {returncode}: {stderr_tail}",
                )

            result: object
            if capability.execution.json_output:
                try:
                    result = json.loads(stdout_text)
                except json.JSONDecodeError as exc:
                    raise SkillRuntimeError(
                        NON_JSON_OUTPUT,
                        "capability declared JSON output but stdout was not valid JSON: "
                        f"{stdout_text[-2000:]}",
                    ) from exc
            else:
                result = stdout_text

            self._audit.write(
                audit_id=audit_id,
                skill=skill_name,
                capability=capability_name,
                qualified_id=qualified_tool_id,
                arguments=working_arguments,
                secret_values=secret_values,
                status="ok",
                duration_ms=self._elapsed_ms(start),
                device_id=call_context.get("device_id"),
                session_id=call_context.get("session_id"),
            )
            return self._success_envelope(
                skill_name, capability_name, result, stdout_text, stderr_text, start, audit_id
            )

        except SkillRuntimeError as exc:
            logger.warning(
                "skill capability %s failed: %s: %s", qualified_tool_id, exc.code, exc.message
            )
            self._audit.write(
                audit_id=audit_id,
                skill=skill_name,
                capability=capability_name,
                qualified_id=qualified_tool_id,
                arguments=working_arguments,
                secret_values=secret_values,
                status="error",
                duration_ms=self._elapsed_ms(start),
                error_code=exc.code,
                device_id=call_context.get("device_id"),
                session_id=call_context.get("session_id"),
            )
            return self._failure_envelope(
                skill_name,
                capability_name,
                error_payload(exc.code, exc.message, exc.repair),
                start,
                audit_id,
            )
        except Exception as exc:  # pragma: no cover - defensive: never let this propagate
            logger.exception("unexpected error executing skill capability %s", qualified_tool_id)
            self._audit.write(
                audit_id=audit_id,
                skill=skill_name,
                capability=capability_name,
                qualified_id=qualified_tool_id,
                arguments=working_arguments,
                secret_values=secret_values,
                status="error",
                duration_ms=self._elapsed_ms(start),
                error_code=RUNTIME_ERROR,
                device_id=call_context.get("device_id"),
                session_id=call_context.get("session_id"),
            )
            return self._failure_envelope(
                skill_name,
                capability_name,
                error_payload(RUNTIME_ERROR, f"unexpected error: {exc}"),
                start,
                audit_id,
            )

    # -- envelope helpers ---------------------------------------------------

    @staticmethod
    def _clamp_timeout(requested: object) -> float:
        """Clamp a caller-supplied timeout into (0, MAX_TIMEOUT_SECONDS].

        Falls back to the default for a missing/non-numeric value, and never
        lets an untrusted caller request an effectively unbounded run.
        """
        if not isinstance(requested, (int, float)) or isinstance(requested, bool):
            return float(DEFAULT_TIMEOUT_SECONDS)
        # Reject NaN/inf: min(nan, MAX) returns nan, which subprocess.run then
        # rejects with a confusing ValueError instead of running bounded.
        if not math.isfinite(requested) or requested <= 0:
            return float(DEFAULT_TIMEOUT_SECONDS)
        return float(min(requested, MAX_TIMEOUT_SECONDS))

    def _elapsed_ms(self, start: float) -> int:
        return int((time.monotonic() - start) * 1000)

    def _success_envelope(
        self,
        skill_name: str,
        capability_name: str,
        result: object,
        stdout_text: str,
        stderr_text: str,
        start: float,
        audit_id: str | None = None,
    ) -> dict:
        return {
            "ok": True,
            "skill": skill_name,
            "capability": capability_name,
            "result": result,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "duration_ms": self._elapsed_ms(start),
            "audit_id": audit_id,
        }

    def _failure_envelope(
        self,
        skill_name: str,
        capability_name: str,
        payload: dict,
        start: float,
        audit_id: str | None = None,
    ) -> dict:
        return {
            "ok": False,
            "skill": skill_name,
            "capability": capability_name,
            "error": payload,
            "duration_ms": self._elapsed_ms(start),
            "audit_id": audit_id,
        }

    # -- parsing / lookup ----------------------------------------------------

    @staticmethod
    def _parse_qualified_tool_id(qualified_tool_id: str) -> tuple[str, str]:
        parts = qualified_tool_id.split("::")
        if len(parts) != 3 or parts[0] != "skill":
            raise SkillRuntimeError(
                CAPABILITY_NOT_FOUND,
                f"malformed qualified tool id: {qualified_tool_id!r}; expected "
                "'skill::<skill-name>::<capability-name>'",
            )
        _, skill_name, capability_name = parts
        return skill_name, capability_name

    @staticmethod
    def _find_capability(
        manifest: SkillManifest, capability_name: str
    ) -> SkillCapabilitySpec | None:
        for capability in manifest.capabilities:
            if capability.name == capability_name:
                return capability
        return None

    # -- argument validation ---------------------------------------------------

    @staticmethod
    def _validate_arguments(capability: SkillCapabilitySpec, arguments: dict) -> None:
        """Validate ``arguments`` then fill in schema defaults, in place.

        Validation runs against the raw arguments (so a missing required
        field is still reported even if some *other* field has a default).
        Defaults are only applied afterwards, so argv placeholders for
        optional arguments still resolve.
        """
        schema = capability.input_schema
        try:
            jsonschema.validate(instance=arguments, schema=schema)
        except jsonschema.ValidationError as exc:
            path = ".".join(str(part) for part in exc.absolute_path) or "<root>"
            # Deliberately built from the schema path/keyword only -- never
            # the offending instance value, so a caller cannot make an error
            # message echo back sensitive argument content.
            raise SkillRuntimeError(
                INVALID_ARGUMENTS,
                f"argument validation failed at '{path}' (constraint: {exc.validator})",
            ) from exc
        except jsonschema.SchemaError as exc:
            raise SkillRuntimeError(
                RUNTIME_ERROR, f"capability input_schema is invalid: {exc}"
            ) from exc

        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        for key, property_schema in properties.items():
            if key in arguments:
                continue
            if isinstance(property_schema, dict) and "default" in property_schema:
                arguments[key] = property_schema["default"]

    # -- secrets ---------------------------------------------------------------

    def _resolve_secrets(
        self, manifest: SkillManifest, capability: SkillCapabilitySpec
    ) -> tuple[set[str], dict[str, str]]:
        """Resolve every secret the capability declares.

        Returns ``(secret_values, env_secrets)``: ``secret_values`` is the
        set of resolved values to redact from output/logs; ``env_secrets``
        maps secret name -> value for injection into the child environment.
        A secret is required unless it has a matching, explicitly
        ``required=False`` entry in ``manifest.secrets`` -- a capability
        secret with no manifest-level entry defaults to required.
        """
        required_by_name = {spec.name: spec.required for spec in manifest.secrets}
        secret_values: set[str] = set()
        env_secrets: dict[str, str] = {}

        for name in capability.secrets:
            value = self._secret_store.get(name)
            required = required_by_name.get(name, True)
            if not value:
                if required:
                    raise SkillRuntimeError(
                        MISSING_SECRET,
                        f"required secret '{name}' is not configured",
                        repair={"type": "configure_secret", "secret": name},
                    )
                continue
            env_secrets[name] = value
            secret_values.add(value)

        return secret_values, env_secrets

    # -- argv rendering ----------------------------------------------------------

    def _render_argv(self, argv_templates: list[str], arguments: dict) -> list[str]:
        """Render each argv template independently into a discrete argv element.

        Never concatenates elements together -- each templated string in
        ``argv_templates`` becomes exactly one element of the returned list,
        so a value containing spaces or shell metacharacters is passed to
        the child process as a single argument, never re-split by a shell.
        """
        return [self._render_argv_element(template, arguments) for template in argv_templates]

    @staticmethod
    def _render_argv_element(template: str, arguments: dict) -> str:
        def _substitute(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in arguments:
                raise SkillRuntimeError(
                    INVALID_ARGUMENTS,
                    f"argv placeholder '{{{key}}}' has no matching argument",
                )
            return str(arguments[key])

        return _ARGV_PLACEHOLDER_RE.sub(_substitute, template)

    # -- command construction --------------------------------------------------

    @staticmethod
    def _build_command(
        runtime: SkillRuntimeSpec, argv: list[str], skill_dir: Path
    ) -> tuple[list[str], dict[str, str]]:
        """Build the argv list (never a shell string) and any extra env vars."""
        extra_env: dict[str, str] = {}

        if runtime.type == "binary":
            resolved = shutil.which(runtime.command) if runtime.command else None
            if resolved is None:
                raise SkillRuntimeError(
                    COMMAND_NOT_FOUND, f"command not found on PATH: {runtime.command}"
                )
            return [resolved, *argv], extra_env

        if runtime.type == "python_script":
            if not runtime.script:
                raise SkillRuntimeError(
                    RUNTIME_ERROR, "python_script runtime requires a 'script' path"
                )
            script_path = (skill_dir / runtime.script).resolve()
            if not is_under_root(script_path, skill_dir):
                raise SkillRuntimeError(
                    RUNTIME_ERROR,
                    f"script path escapes skill directory: {runtime.script}",
                )
            if not script_path.is_file():
                raise SkillRuntimeError(COMMAND_NOT_FOUND, f"script not found: {script_path}")
            return [sys.executable, str(script_path), *argv], extra_env

        if runtime.type == "python_module":
            if not runtime.module:
                raise SkillRuntimeError(
                    RUNTIME_ERROR, "python_module runtime requires a 'module'"
                )
            existing_pythonpath = os.environ.get("PYTHONPATH", "")
            extra_env["PYTHONPATH"] = (
                f"{skill_dir}{os.pathsep}{existing_pythonpath}"
                if existing_pythonpath
                else str(skill_dir)
            )
            return [sys.executable, "-m", runtime.module, *argv], extra_env

        raise SkillRuntimeError(UNSUPPORTED_RUNTIME, f"unsupported runtime type: {runtime.type}")

    # -- subprocess execution -------------------------------------------------

    async def _run_and_collect(
        self,
        cmd: list[str],
        skill_dir: Path,
        env: dict[str, str],
        timeout: float,
        secret_values: set[str],
    ) -> tuple[str, str, int]:
        """Run ``cmd`` in a worker thread and return redacted (stdout, stderr, returncode).

        Uses ``subprocess.run`` (never ``shell=True``, always an argv list)
        wrapped in ``asyncio.to_thread`` -- ``asyncio.create_subprocess_exec``
        raises ``NotImplementedError`` on the sidecar's Selector event loop
        on Windows.
        """

        def _run() -> subprocess.CompletedProcess[bytes]:
            # Argv list, never shell=True -- no shell string interpolation.
            return subprocess.run(
                cmd, cwd=str(skill_dir), env=env, capture_output=True, timeout=timeout
            )

        try:
            proc = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired as exc:
            raise SkillRuntimeError(
                EXECUTION_TIMEOUT, f"capability execution exceeded {timeout}s timeout"
            ) from exc
        except FileNotFoundError as exc:
            raise SkillRuntimeError(COMMAND_NOT_FOUND, f"command not found: {exc}") from exc

        stdout_bytes = proc.stdout or b""
        stderr_bytes = proc.stderr or b""
        if len(stdout_bytes) > MAX_OUTPUT_BYTES or len(stderr_bytes) > MAX_OUTPUT_BYTES:
            raise SkillRuntimeError(
                OUTPUT_TOO_LARGE, f"capability output exceeded {MAX_OUTPUT_BYTES} bytes"
            )

        stdout_text = _redact(stdout_bytes.decode("utf-8", errors="replace"), secret_values)
        stderr_text = _redact(stderr_bytes.decode("utf-8", errors="replace"), secret_values)
        return stdout_text, stderr_text, proc.returncode
