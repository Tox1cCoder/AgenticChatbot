"""Atomic installation of complete, standard one-skill bundles."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from client_backend.core.logging import get_logger
from client_backend.core.paths import get_installed_skills_root, is_under_root, sanitize_filename
from client_backend.services.local_skills_registry import (
    LocalSkillsRegistry,
    SkillMetadata,
    get_skills_registry,
)
from client_backend.services.runtime_bridge import get_runtime_bridge
from client_backend.services.skill_runtime.environment import SkillEnvironmentManager
from client_backend.services.skill_runtime.secrets import SkillSecretStore
from shared.skills.errors import (
    SKILL_INSTALL_CONFLICT,
    SKILL_INSTALL_INVALID,
    SKILL_SETUP_REQUIRED,
    UNSAFE_BUNDLE_PATH,
    SkillRuntimeError,
)
from shared.skills.front_matter import is_valid_skill_name, parse_skill_front_matter
from shared.skills.hashing import UnsafeSkillBundleError, compute_skill_bundle_hash

logger = get_logger(__name__)

_INSTALL_METADATA_FILENAME = "install.json"


def _compute_source_hash(source: Path) -> str:
    """Hash a bundle deterministically while rejecting every symlink."""
    try:
        return compute_skill_bundle_hash(source)
    except UnsafeSkillBundleError as exc:
        raise SkillRuntimeError(UNSAFE_BUNDLE_PATH, str(exc)) from exc


def _read_install_metadata(bundle_dir: Path) -> dict | None:
    path = bundle_dir / _INSTALL_METADATA_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Failed to read install metadata for %s: %s", bundle_dir, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _find_installed_bundle(install_root: Path, name: str) -> Path | None:
    if not install_root.is_dir():
        return None
    for entry in sorted(install_root.iterdir()):
        if not entry.is_dir() or ".stage-" in entry.name or ".backup-" in entry.name:
            continue
        metadata = _read_install_metadata(entry)
        if metadata is not None and metadata.get("bundle_name") == name:
            return entry
    return None


class SkillBundleInstaller:
    """Validate, copy, prepare, and remove device-local skill bundles."""

    def __init__(
        self,
        registry: LocalSkillsRegistry | None = None,
        environment_manager: SkillEnvironmentManager | None = None,
        secret_store: SkillSecretStore | None = None,
    ) -> None:
        self._registry = registry if registry is not None else get_skills_registry()
        self._environment = environment_manager or SkillEnvironmentManager()
        self._secrets = secret_store or SkillSecretStore()

    async def preview(self, source: str | Path) -> dict:
        skill, shape = await self._discover_source(source)
        setup = await asyncio.to_thread(self._environment.preview, skill)
        return {
            "name": skill.name,
            "source_hash": skill.source_hash,
            "bundle_shape": shape,
            "executable_assets": dict(skill.executable_assets),
            "setup": setup,
        }

    async def install(
        self,
        source: str | Path,
        *,
        expected_source_hash: str | None = None,
        approve_setup: bool = False,
    ) -> dict:
        source_skill, shape = await self._discover_source(source)
        setup_preview = await asyncio.to_thread(self._environment.preview, source_skill)
        preview = {
            "name": source_skill.name,
            "source_hash": source_skill.source_hash,
            "bundle_shape": shape,
            "executable_assets": dict(source_skill.executable_assets),
            "setup": setup_preview,
        }
        setup_requires_confirmation = bool(preview["setup"].get("confirmation_required"))
        if setup_requires_confirmation and not expected_source_hash:
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID,
                "Python setup requires the source hash returned by install preview",
            )
        if expected_source_hash and expected_source_hash != source_skill.source_hash:
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID,
                "bundle changed after preview; request a new preview before installing",
            )
        if setup_requires_confirmation and not approve_setup:
            raise SkillRuntimeError(
                SKILL_SETUP_REQUIRED,
                f"setup for skill '{source_skill.name}' requires explicit approval",
                repair={"type": "approve_skill_setup", "preview": preview},
            )

        await self._registry.initialize()
        existing = self._registry.get_skill(source_skill.name)
        if existing is not None and existing.enabled:
            raise SkillRuntimeError(
                SKILL_INSTALL_CONFLICT,
                f"a skill named '{source_skill.name}' is already active; "
                "disable or uninstall it first",
            )

        install_root = self._resolve_install_root()
        target = install_root / (
            f"{sanitize_filename(source_skill.name)}-{source_skill.source_hash[:12]}"
        )
        stage = install_root / f"{target.name}.stage-{uuid.uuid4().hex}"
        backup = install_root / f"{target.name}.backup-{uuid.uuid4().hex}"
        for path in (target, stage, backup):
            if not is_under_root(path, install_root):
                raise SkillRuntimeError(
                    UNSAFE_BUNDLE_PATH, "install path escapes the profile skill root"
                )

        stale_bundle = _find_installed_bundle(install_root, source_skill.name)
        try:
            await asyncio.to_thread(
                shutil.copytree,
                source_skill.bundle_root,
                stage,
                symlinks=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".venv"),
            )
            copied_hash = await asyncio.to_thread(_compute_source_hash, stage)
            if copied_hash != source_skill.source_hash:
                raise SkillRuntimeError(
                    SKILL_INSTALL_INVALID,
                    "bundle changed while it was being copied; request a new preview",
                )
            relative_skill_path = source_skill.path.relative_to(source_skill.bundle_root)
            install_payload = {
                "bundle_name": source_skill.name,
                "source_hash": source_skill.source_hash,
                "source": "profile",
                "source_path": str(source_skill.bundle_root),
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "enabled": True,
                "installed": True,
            }
            await asyncio.to_thread(
                (stage / _INSTALL_METADATA_FILENAME).write_text,
                json.dumps(install_payload, indent=2),
                encoding="utf-8",
            )
            installed_skill = SkillMetadata(
                name=source_skill.name,
                path=stage / relative_skill_path,
                bundle_root=stage,
                source_hash=source_skill.source_hash,
                executable_assets=self._registry._discover_executable_assets(stage),
                description=source_skill.description,
                content=source_skill.content,
                category=source_skill.category,
                tags=list(source_skill.tags),
                install_metadata=install_payload,
            )

            if installed_skill.executable_assets["python_project"]:
                runtime = await asyncio.to_thread(
                    self._environment.prepare,
                    installed_skill,
                    approve_setup=approve_setup,
                )
                runtime_status = str(runtime.get("status") or "not_ready")
            elif (
                installed_skill.executable_assets["bin"]
                or installed_skill.executable_assets["scripts"]
            ):
                runtime_status = "ready"
            else:
                runtime_status = "instruction_only"

            if target.exists():
                await asyncio.to_thread(os.replace, target, backup)
            try:
                await asyncio.to_thread(os.replace, stage, target)
            except Exception:
                if backup.exists() and not target.exists():
                    await asyncio.to_thread(os.replace, backup, target)
                raise
            if backup.exists():
                await asyncio.to_thread(shutil.rmtree, backup)
            if (
                stale_bundle is not None
                and stale_bundle != target
                and stale_bundle.exists()
                and is_under_root(stale_bundle, install_root)
            ):
                await asyncio.to_thread(shutil.rmtree, stale_bundle)
        except Exception:
            if stage.exists():
                await asyncio.to_thread(shutil.rmtree, stage)
            raise

        await self._registry.refresh()
        await self._refresh_runtime_bridge_catalogs_if_connected()
        return {
            "name": source_skill.name,
            "install_id": target.name,
            "source_hash": source_skill.source_hash,
            "runtime_status": runtime_status,
        }

    async def setup(
        self,
        name: str,
        *,
        expected_source_hash: str | None,
        approve_setup: bool,
    ) -> dict:
        await self._registry.initialize()
        skill = self._registry.get_skill(name)
        if skill is None:
            raise SkillRuntimeError(SKILL_INSTALL_INVALID, f"skill '{name}' was not found")
        if not expected_source_hash:
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID,
                "skill setup requires the current expected source hash",
            )
        try:
            live_source_hash = await asyncio.to_thread(
                _compute_source_hash,
                skill.bundle_root,
            )
        except SkillRuntimeError:
            raise
        if expected_source_hash != skill.source_hash or live_source_hash != skill.source_hash:
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID,
                "skill changed after preview; request a new preview before setup",
            )
        result = await asyncio.to_thread(
            self._environment.prepare,
            skill,
            approve_setup=approve_setup,
            force=True,
        )
        await self._registry.refresh()
        await self._refresh_runtime_bridge_catalogs_if_connected()
        return {"name": name, "source_hash": skill.source_hash, **result}

    async def uninstall(self, name: str) -> dict:
        install_root = self._resolve_install_root()
        bundle = _find_installed_bundle(install_root, name)
        if bundle is None:
            raise SkillRuntimeError(SKILL_INSTALL_INVALID, f"no installed skill named '{name}'")
        if not is_under_root(bundle, install_root):
            raise SkillRuntimeError(
                UNSAFE_BUNDLE_PATH, "installed bundle path escapes the profile root"
            )
        await asyncio.to_thread(shutil.rmtree, bundle)
        await asyncio.to_thread(self._environment.remove_skill, name)
        await asyncio.to_thread(self._secrets.remove_skill, name)
        await self._registry.refresh()
        await self._refresh_runtime_bridge_catalogs_if_connected()
        return {"name": name, "removed": True}

    def list_installed(self) -> list[dict]:
        user_id = self._registry._resolve_current_user_id()
        if not user_id:
            return []
        root = get_installed_skills_root(user_id)
        if not root.is_dir():
            return []
        return [
            metadata
            for entry in sorted(root.iterdir())
            if entry.is_dir()
            if (metadata := _read_install_metadata(entry)) is not None
        ]

    async def _discover_source(self, source: str | Path) -> tuple[SkillMetadata, str]:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_dir():
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID,
                f"source '{source}' does not exist or is not a directory",
            )
        source_hash = await asyncio.to_thread(_compute_source_hash, source_path)
        skill_files = sorted(source_path.rglob("SKILL.md"))
        if len(skill_files) != 1:
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID,
                f"bundle must contain exactly one SKILL.md file; found {len(skill_files)}",
            )
        skill_path = skill_files[0]
        raw = await asyncio.to_thread(skill_path.read_text, encoding="utf-8")
        parsed = parse_skill_front_matter(raw)
        if parsed is not None:
            if not parsed.name or not parsed.name.strip():
                raise SkillRuntimeError(
                    SKILL_INSTALL_INVALID, "malformed front matter: missing name"
                )
            name = parsed.name
            description = parsed.description
            content = parsed.body
            category = parsed.category
            tags = list(parsed.tags)
        else:
            name = sanitize_filename(skill_path.parent.name)
            description = f"Skill: {name}"
            content = raw
            category = None
            tags = []
        if not is_valid_skill_name(name):
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID,
                "skill name must use 1-64 lowercase letters, digits, or single hyphens",
            )
        shape = "direct" if skill_path.parent == source_path else "nested"
        return (
            SkillMetadata(
                name=name,
                path=skill_path,
                bundle_root=source_path,
                source_hash=source_hash,
                executable_assets=self._registry._discover_executable_assets(source_path),
                description=description,
                content=content,
                category=category,
                tags=tags,
            ),
            shape,
        )

    def _resolve_install_root(self) -> Path:
        user_id = self._registry._resolve_current_user_id()
        if not user_id:
            raise SkillRuntimeError(SKILL_INSTALL_INVALID, "no active user profile")
        root = get_installed_skills_root(user_id)
        root.mkdir(parents=True, exist_ok=True)
        return root

    async def _refresh_runtime_bridge_catalogs_if_connected(self) -> None:
        bridge = get_runtime_bridge()
        if bridge.is_connected() and bridge.get_registered_device_id():
            await bridge.refresh_catalogs()
