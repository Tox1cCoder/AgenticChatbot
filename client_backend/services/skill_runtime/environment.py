"""Profile-local runtime preparation for installable Python skill bundles."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
import venv
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

from client_backend.core.paths import (
    get_skill_runtimes_root,
    is_under_root,
    sanitize_filename,
)
from client_backend.services.local_skills_registry import SkillMetadata
from client_backend.services.upstream_auth import get_upstream_auth_service
from shared.skills.commands import is_link_like
from shared.skills.errors import (
    SKILL_SETUP_FAILED,
    SKILL_SETUP_REQUIRED,
    UNSAFE_BUNDLE_PATH,
    SkillRuntimeError,
)

RUNTIME_FORMAT_VERSION = 1
SETUP_TIMEOUT_SECONDS = 300
MAX_SETUP_LOG_BYTES = 64_000


def _default_venv_builder(venv_root: Path) -> None:
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_root)


def _default_runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, **kwargs)


class SkillEnvironmentManager:
    """Prepare, inspect, and remove isolated Python environments for skills."""

    def __init__(
        self,
        *,
        runtime_base: Path | None = None,
        venv_builder: Callable[[Path], None] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[bytes]] | None = None,
    ) -> None:
        self._runtime_base = runtime_base.resolve() if runtime_base is not None else None
        self._venv_builder = venv_builder or _default_venv_builder
        self._runner = runner or _default_runner

    def preview(self, skill: SkillMetadata) -> dict:
        pyproject = skill.bundle_root / "pyproject.toml"
        if not pyproject.is_file():
            return {
                "python_project": False,
                "dependencies": [],
                "build_requirements": [],
                "declared_commands": [],
                "dependency_lock": None,
                "confirmation_required": False,
            }
        if is_link_like(pyproject) or not is_under_root(
            pyproject.resolve(), skill.bundle_root.resolve()
        ):
            raise SkillRuntimeError(
                UNSAFE_BUNDLE_PATH,
                "Python project metadata escapes the skill bundle",
            )

        payload = self._read_pyproject(pyproject)
        project = payload.get("project") if isinstance(payload.get("project"), dict) else {}
        build_system = (
            payload.get("build-system") if isinstance(payload.get("build-system"), dict) else {}
        )
        scripts = project.get("scripts") if isinstance(project.get("scripts"), dict) else {}
        dependencies = project.get("dependencies")
        build_requirements = build_system.get("requires")
        return {
            "python_project": True,
            "dependencies": sorted(str(item) for item in dependencies if isinstance(item, str))
            if isinstance(dependencies, list)
            else [],
            "build_requirements": sorted(
                str(item) for item in build_requirements if isinstance(item, str)
            )
            if isinstance(build_requirements, list)
            else [],
            "declared_commands": sorted(str(name) for name in scripts),
            "dependency_lock": (
                "requirements.lock" if (skill.bundle_root / "requirements.lock").is_file() else None
            ),
            # Python builds execute project-controlled backend code, even when
            # the project has no remote dependencies.
            "confirmation_required": True,
        }

    def prepare(
        self,
        skill: SkillMetadata,
        *,
        approve_setup: bool,
        force: bool = False,
    ) -> dict:
        preview = self.preview(skill)
        if not preview["python_project"]:
            return {"status": "not_applicable", "commands": []}
        if preview["confirmation_required"] and not approve_setup:
            raise SkillRuntimeError(
                SKILL_SETUP_REQUIRED,
                f"Python setup for skill '{skill.name}' requires explicit approval",
                repair={
                    "type": "approve_skill_setup",
                    "skill": skill.name,
                    "source_hash": skill.source_hash,
                    "preview": preview,
                },
            )

        target = self._runtime_root(skill)
        if target.exists() and not force:
            inspection = self.inspect(skill)
            if inspection.get("status") == "ready":
                return inspection

        skill_runtime_root = target.parent
        skill_runtime_root.mkdir(parents=True, exist_ok=True)
        stage = skill_runtime_root / f"{target.name}.stage-{uuid.uuid4().hex}"
        backup = skill_runtime_root / f"{target.name}.backup-{uuid.uuid4().hex}"
        if not is_under_root(stage, self._base()):
            raise SkillRuntimeError(UNSAFE_BUNDLE_PATH, "runtime stage escapes profile root")

        try:
            stage.mkdir(parents=True)
            venv_root = stage / "venv"
            self._venv_builder(venv_root)
            python_path = self._venv_python(venv_root)
            pip_install = [
                str(python_path),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
            ]
            lock_path = skill.bundle_root / "requirements.lock"
            commands = []
            if lock_path.is_file():
                if not is_under_root(lock_path.resolve(), skill.bundle_root.resolve()):
                    raise SkillRuntimeError(
                        UNSAFE_BUNDLE_PATH,
                        "dependency lock escapes the skill bundle",
                    )
                commands.append([*pip_install, "-r", str(lock_path)])
                commands.append(
                    [
                        *pip_install,
                        "--no-build-isolation",
                        "--no-deps",
                        str(skill.bundle_root),
                    ]
                )
            else:
                commands.append([*pip_install, str(skill.bundle_root)])

            log_bytes = b""
            for command in commands:
                completed = self._runner(
                    command,
                    cwd=str(skill.bundle_root),
                    capture_output=True,
                    timeout=SETUP_TIMEOUT_SECONDS,
                    env=self._setup_environment(),
                )
                stdout = completed.stdout or b""
                stderr = completed.stderr or b""
                log_bytes = (log_bytes + stdout + b"\n" + stderr + b"\n")[-MAX_SETUP_LOG_BYTES:]
                (stage / "setup.log").write_bytes(log_bytes)
                if completed.returncode != 0:
                    raise SkillRuntimeError(
                        SKILL_SETUP_FAILED,
                        f"Python setup for skill '{skill.name}' failed with exit code "
                        f"{completed.returncode}",
                        repair={"type": "inspect_setup_failure", "skill": skill.name},
                    )

            commands = self._discover_runtime_commands(
                venv_root,
                preview["declared_commands"],
            )
            if not commands:
                raise SkillRuntimeError(
                    SKILL_SETUP_FAILED,
                    f"Python setup for skill '{skill.name}' produced no console commands",
                    repair={"type": "check_project_scripts", "skill": skill.name},
                )

            metadata = {
                "format_version": RUNTIME_FORMAT_VERSION,
                "skill": skill.name,
                "source_hash": skill.source_hash,
                "sidecar_python": str(Path(sys.executable).resolve()),
                "platform": self._platform_id(),
                "commands": commands,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (stage / "runtime.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            if target.exists():
                os.replace(target, backup)
            try:
                os.replace(stage, target)
            except Exception:
                if backup.exists() and not target.exists():
                    os.replace(backup, target)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        except SkillRuntimeError:
            if stage.exists():
                shutil.rmtree(stage)
            raise
        except Exception as exc:
            if stage.exists():
                shutil.rmtree(stage)
            raise SkillRuntimeError(
                SKILL_SETUP_FAILED,
                f"Python setup for skill '{skill.name}' failed: {exc.__class__.__name__}",
                repair={"type": "inspect_setup_failure", "skill": skill.name},
            ) from exc

        return self.inspect(skill)

    def inspect(self, skill: SkillMetadata) -> dict:
        try:
            target = self._runtime_root(skill)
        except SkillRuntimeError as exc:
            return {
                "status": "setup_required",
                "commands": [],
                "detail": exc.message,
            }
        if not target.is_dir():
            existing = self.inspect_existing_for_skill(skill)
            if existing.get("status") in {"ready", "stale"}:
                return {**existing, "status": "stale"}
            return {"status": "setup_required", "commands": []}
        return self._inspect_runtime_root(target, expected_skill=skill)

    def inspect_existing_for_skill(self, skill: SkillMetadata | str) -> dict:
        skill_name = skill.name if isinstance(skill, SkillMetadata) else skill
        skill_root = self._skill_runtime_root(skill_name)
        if not skill_root.is_dir():
            return {"status": "setup_required", "commands": []}
        candidates = sorted(path for path in skill_root.iterdir() if path.is_dir())
        for candidate in reversed(candidates):
            expected = skill if isinstance(skill, SkillMetadata) else None
            result = self._inspect_runtime_root(candidate, expected_skill=expected)
            if result.get("status") in {"ready", "stale"}:
                return result
        return {"status": "setup_required", "commands": []}

    def command_directory(self, skill: SkillMetadata) -> Path | None:
        inspection = self.inspect(skill)
        if inspection.get("status") != "ready":
            return None
        return self._venv_scripts(Path(inspection["runtime_root"]) / "venv")

    def python_executable(self, skill: SkillMetadata) -> Path | None:
        inspection = self.inspect(skill)
        if inspection.get("status") != "ready":
            return None
        return self._venv_python(Path(inspection["runtime_root"]) / "venv")

    def remove_skill(self, skill_name: str) -> None:
        target = self._skill_runtime_root(skill_name)
        base = self._base()
        if target.exists() and is_under_root(target, base):
            shutil.rmtree(target)

    def _inspect_runtime_root(
        self,
        target: Path,
        expected_skill: SkillMetadata | None = None,
    ) -> dict:
        metadata_path = target / "runtime.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"status": "failed", "commands": []}
        stale = (
            metadata.get("format_version") != RUNTIME_FORMAT_VERSION
            or metadata.get("sidecar_python") != str(Path(sys.executable).resolve())
            or metadata.get("platform") != self._platform_id()
            or (
                expected_skill is not None
                and metadata.get("source_hash") != expected_skill.source_hash
            )
        )
        commands = sorted(str(item) for item in metadata.get("commands", []))
        venv_root = target / "venv"
        missing_commands = [
            command
            for command in commands
            if self._runtime_command_path(venv_root, command) is None
        ]
        if not self._venv_python(venv_root).is_file() or missing_commands:
            return {
                "status": "failed",
                "commands": [],
                "runtime_id": target.name[:12],
                "runtime_root": str(target),
                "detail": "prepared runtime files are incomplete",
            }
        return {
            "status": "stale" if stale else "ready",
            "commands": commands,
            "runtime_id": target.name[:12],
            "runtime_root": str(target),
        }

    def _runtime_root(self, skill: SkillMetadata) -> Path:
        return self._skill_runtime_root(skill.name) / skill.source_hash

    def _skill_runtime_root(self, skill_name: str) -> Path:
        target = self._base() / sanitize_filename(skill_name)
        if not is_under_root(target, self._base()):
            raise SkillRuntimeError(UNSAFE_BUNDLE_PATH, "skill runtime path escapes profile root")
        return target

    def _base(self) -> Path:
        if self._runtime_base is not None:
            self._runtime_base.mkdir(parents=True, exist_ok=True)
            return self._runtime_base
        user_id = get_upstream_auth_service().get_current_user_id()
        if not user_id:
            raise SkillRuntimeError(SKILL_SETUP_FAILED, "no active user profile")
        base = get_skill_runtimes_root(user_id).resolve()
        base.mkdir(parents=True, exist_ok=True)
        return base

    @staticmethod
    def _read_pyproject(path: Path) -> dict:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _venv_scripts(venv_root: Path) -> Path:
        return venv_root / ("Scripts" if os.name == "nt" else "bin")

    @classmethod
    def _venv_python(cls, venv_root: Path) -> Path:
        name = "python.exe" if os.name == "nt" else "python"
        return cls._venv_scripts(venv_root) / name

    @classmethod
    def _discover_runtime_commands(
        cls,
        venv_root: Path,
        declared_commands: list[str],
    ) -> list[str]:
        return sorted(
            command
            for command in dict.fromkeys(declared_commands)
            if cls._runtime_command_path(venv_root, command) is not None
        )

    @classmethod
    def _runtime_command_path(cls, venv_root: Path, command: str) -> Path | None:
        scripts = cls._venv_scripts(venv_root)
        if not scripts.is_dir() or Path(command).name != command:
            return None
        for candidate_name in (command, f"{command}.py", f"{command}.exe"):
            candidate = scripts / candidate_name
            if candidate.is_file() and not candidate.is_symlink():
                return candidate
        return None

    @staticmethod
    def _platform_id() -> str:
        return f"{platform.system()}-{platform.machine()}"

    @staticmethod
    def _setup_environment() -> dict[str, str]:
        allowed = {
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "HOME",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
        }
        return {name: value for name, value in os.environ.items() if name.upper() in allowed}
