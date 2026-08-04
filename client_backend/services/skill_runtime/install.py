"""Atomic installation of complete, standard one-skill bundles."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from client_backend.core.logging import get_logger
from client_backend.core.paths import (
    get_installed_skills_root,
    get_skill_operations_root,
    is_under_root,
    sanitize_filename,
)
from client_backend.services.local_skills_registry import (
    LocalSkillsRegistry,
    SkillMetadata,
    get_skills_registry,
)
from client_backend.services.skill_runtime.collection import DiscoveredSkill
from client_backend.services.skill_runtime.environment import SkillEnvironmentManager
from client_backend.services.skill_runtime.locks import profile_lock
from client_backend.services.skill_runtime.secrets import SkillSecretStore
from client_backend.services.skill_runtime.state import atomic_write_json, read_json_object
from shared.skills.errors import (
    SKILL_CONFIGURED_ROOT_CONFLICT,
    SKILL_INSTALL_CONFLICT,
    SKILL_INSTALL_INVALID,
    SKILL_SETUP_REQUIRED,
    SKILL_SOURCE_CHANGED,
    UNSAFE_BUNDLE_PATH,
    SkillRuntimeError,
)
from shared.skills.front_matter import is_valid_skill_name, parse_skill_front_matter
from shared.skills.hashing import UnsafeSkillBundleError, compute_skill_bundle_hash

logger = get_logger(__name__)

_INSTALL_METADATA_FILENAME = "install.json"


class SkillInstallObserver(Protocol):
    """Cooperative lifecycle hooks for a long-running installation.

    Implemented by the async operation service so a client can watch progress and
    cancel. The installer calls these at points where stopping is safe; it never
    depends on an observer existing, and path-based callers pass none.
    """

    async def phase(self, name: str) -> None:
        """Report entry into a named phase."""
        ...

    async def before_commit(self) -> None:
        """Called immediately before the atomic promotion.

        The last chance to abort. An implementation either raises to cancel or
        durably records that the commit started, because no later point can be
        cleanly undone.
        """
        ...


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

    async def preview(self, source: str | Path | DiscoveredSkill) -> dict:
        skill, shape = await self._load_source(source)
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
        source: str | Path | DiscoveredSkill,
        *,
        expected_source_hash: str | None = None,
        approve_setup: bool = False,
        replace_source_hash: str | None = None,
        source_kind: Literal["path", "upload"] = "path",
        observer: SkillInstallObserver | None = None,
    ) -> dict:
        """Install or replace one skill bundle atomically.

        Args:
            source: Directory holding exactly one discoverable ``SKILL.md``.
            expected_source_hash: Hash the caller previewed; binds this request
                to content the user actually saw.
            approve_setup: Explicit authorization to run project build code.
            replace_source_hash: Required to overwrite an existing skill, and must
                equal that skill's current hash. Its absence means "new install",
                so a name collision is a conflict rather than a silent overwrite.
            source_kind: Provenance recorded in install metadata. ``"upload"``
                omits the source path, which points into upload staging.
            observer: Optional cooperative lifecycle hook used by the async
                operation service for phase reporting and cancellation.

        Returns:
            ``name``, ``install_id``, ``source_hash``, ``runtime_status``, and
            ``action`` (``"installed"`` or ``"updated"``).
        """
        await self._notify(observer, "validating")
        source_skill, shape = await self._load_source(source)
        preview = await self._build_preview(source_skill, shape)
        self._require_preview_agreement(
            source_skill,
            preview,
            expected_source_hash=expected_source_hash,
            approve_setup=approve_setup,
        )

        user_id = self._resolve_user_id()
        await self._notify(observer, "waitingForLock")
        async with profile_lock(user_id, f"skill:{source_skill.name}"):
            return await self._install_locked(
                source,
                user_id=user_id,
                expected_source_hash=expected_source_hash,
                approve_setup=approve_setup,
                replace_source_hash=replace_source_hash,
                source_kind=source_kind,
                observer=observer,
            )

    async def _build_preview(self, source_skill: SkillMetadata, shape: str) -> dict:
        setup_preview = await asyncio.to_thread(self._environment.preview, source_skill)
        return {
            "name": source_skill.name,
            "source_hash": source_skill.source_hash,
            "bundle_shape": shape,
            "executable_assets": dict(source_skill.executable_assets),
            "setup": setup_preview,
        }

    @staticmethod
    def _require_preview_agreement(
        source_skill: SkillMetadata,
        preview: dict,
        *,
        expected_source_hash: str | None,
        approve_setup: bool,
    ) -> None:
        """Refuse to install content the caller has not seen and approved."""
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

    async def _install_locked(
        self,
        source: str | Path | DiscoveredSkill,
        *,
        user_id: str,
        expected_source_hash: str | None,
        approve_setup: bool,
        replace_source_hash: str | None,
        source_kind: Literal["path", "upload"],
        observer: SkillInstallObserver | None,
    ) -> dict:
        """Perform the mutation while holding this skill's lock.

        The source is rediscovered and rehashed here rather than reusing the
        pre-lock result: between the caller's check and the lock being granted, the
        bundle on disk can change, and a copy of *different* content than the user
        approved is exactly what the hash binding exists to prevent.
        """
        source_skill, shape = await self._load_source(source)
        if expected_source_hash and expected_source_hash != source_skill.source_hash:
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID,
                "bundle changed after preview; request a new preview before installing",
            )

        await self._registry.initialize()
        existing = self._registry.get_skill(source_skill.name)
        action = self._resolve_replacement_action(
            source_skill.name,
            existing,
            user_id=user_id,
            replace_source_hash=replace_source_hash,
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
            await self._notify(observer, "copying")
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
                "source": "upload" if source_kind == "upload" else "profile",
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "enabled": True,
                "installed": True,
            }
            if source_kind == "path":
                # An upload's source path points into staging, which is deleted
                # after installation and must never be persisted or served.
                install_payload["source_path"] = str(source_skill.bundle_root)
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
                await self._notify(observer, "preparingRuntime")
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

            # Last point at which nothing user-visible has changed. An operation
            # observer either aborts here or durably records that the commit
            # began, because everything below is a single atomic promotion that
            # cannot be half-undone.
            await self._notify(observer, "committing")
            if observer is not None:
                await observer.before_commit()

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

        # Local visibility only. Runtime-catalog synchronization is the catalog
        # service's job: a bridge failure must not turn a committed install into a
        # failed one, and this method has already committed.
        await self._registry.refresh()
        return {
            "name": source_skill.name,
            "install_id": target.name,
            "source_hash": source_skill.source_hash,
            "runtime_status": runtime_status,
            "action": action,
        }

    def _resolve_replacement_action(
        self,
        name: str,
        existing: SkillMetadata | None,
        *,
        user_id: str,
        replace_source_hash: str | None,
    ) -> Literal["installed", "updated"]:
        """Decide whether this request may overwrite an existing skill.

        Ownership is decided by *location*. The registry copies
        ``install_metadata`` out of any ``install.json`` it finds under any
        configured root, so trusting that file's ``installed`` flag would let a
        hand-copied bundle claim to be profile-installed. Only a bundle under the
        profile's installed root is ours to replace.
        """
        if existing is None:
            if replace_source_hash is not None:
                raise SkillRuntimeError(
                    SKILL_SOURCE_CHANGED,
                    "no installed skill matches the requested replacement; reload the catalog",
                )
            return "installed"

        if not is_under_root(existing.bundle_root, get_installed_skills_root(user_id)):
            raise SkillRuntimeError(
                SKILL_CONFIGURED_ROOT_CONFLICT,
                f"skill '{name}' comes from a configured skills root and cannot be replaced",
            )
        if replace_source_hash is None:
            raise SkillRuntimeError(
                SKILL_INSTALL_CONFLICT,
                f"a skill named '{name}' already exists; confirm an explicit update",
            )
        if replace_source_hash != existing.source_hash:
            raise SkillRuntimeError(
                SKILL_SOURCE_CHANGED,
                "installed skill changed after preview; reload and confirm its current source hash",
            )
        return "updated"

    @staticmethod
    async def _notify(observer: SkillInstallObserver | None, phase: str) -> None:
        """Report a lifecycle phase, if anyone is listening."""
        if observer is not None:
            await observer.phase(phase)

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
        return {"name": name, "source_hash": skill.source_hash, **result}

    async def uninstall(self, name: str) -> dict:
        """Remove one installed bundle, its runtime, and its secrets.

        Removal spans three stores that cannot be updated atomically together, so
        progress is journaled in a cleanup receipt before anything is deleted. A
        crash or a failing step leaves ``cleanup_status: "pending"`` and a receipt
        that a later call resumes -- otherwise a half-removed skill would keep a
        prepared runtime and stored secrets with no bundle to bind them to.

        A skill with neither a bundle nor a receipt raises
        ``SKILL_INSTALL_INVALID``, which the route maps to 404 as before. Only a
        resumed receipt reports ``removed: False``.
        """
        user_id = self._resolve_user_id()
        async with profile_lock(user_id, f"skill:{name}"):
            install_root = self._resolve_install_root()
            bundle = _find_installed_bundle(install_root, name)
            receipt_path = self._cleanup_receipt_path(user_id, name)
            receipt = read_json_object(receipt_path) if receipt_path.is_file() else None

            if bundle is None and receipt is None:
                raise SkillRuntimeError(SKILL_INSTALL_INVALID, f"no installed skill named '{name}'")
            if bundle is not None and not is_under_root(bundle, install_root):
                raise SkillRuntimeError(
                    UNSAFE_BUNDLE_PATH, "installed bundle path escapes the profile root"
                )

            removed_now = bundle is not None
            progress = {
                "version": 1,
                "skill": name,
                "bundle_removed": bool(receipt.get("bundle_removed")) if receipt else False,
                "runtime_removed": bool(receipt.get("runtime_removed")) if receipt else False,
                "secrets_removed": bool(receipt.get("secrets_removed")) if receipt else False,
            }
            atomic_write_json(receipt_path, progress)

            failures: list[str] = []
            if bundle is not None:
                try:
                    await asyncio.to_thread(shutil.rmtree, bundle)
                    progress["bundle_removed"] = True
                    atomic_write_json(receipt_path, progress)
                except OSError as exc:
                    failures.append(f"bundle: {exc}")
            else:
                progress["bundle_removed"] = True

            for step, remover in (
                ("runtime_removed", self._environment.remove_skill),
                ("secrets_removed", self._secrets.remove_skill),
            ):
                if progress[step]:
                    continue
                try:
                    await asyncio.to_thread(remover, name)
                    progress[step] = True
                    atomic_write_json(receipt_path, progress)
                except Exception as exc:  # noqa: BLE001 - recorded, then retried later
                    logger.warning("skill cleanup step %s failed for %s: %s", step, name, exc)
                    failures.append(f"{step}: {exc}")

            complete = all(
                progress[key] for key in ("bundle_removed", "runtime_removed", "secrets_removed")
            )
            if complete:
                with contextlib.suppress(OSError):
                    receipt_path.unlink()

            await self._registry.refresh()
            return {
                "name": name,
                "removed": removed_now,
                "cleanup_status": "complete" if complete else "pending",
            }

    @staticmethod
    def _cleanup_receipt_path(user_id: str, name: str) -> Path:
        """Resolve one skill's cleanup receipt under the profile operations root.

        The filename is a digest so an arbitrary route-supplied name never becomes
        a path component.
        """
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:32]
        return get_skill_operations_root(user_id) / f"cleanup-{digest}.json"

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

    async def _load_source(
        self,
        source: str | Path | DiscoveredSkill,
    ) -> tuple[SkillMetadata, str]:
        if isinstance(source, DiscoveredSkill):
            return await self._load_discovered_skill(source)
        return await self._discover_source(source)

    async def _discover_source(self, source: str | Path) -> tuple[SkillMetadata, str]:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_dir():
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID,
                f"source '{source}' does not exist or is not a directory",
            )
        skill_files = sorted(source_path.rglob("SKILL.md"))
        if len(skill_files) != 1:
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID,
                f"bundle must contain exactly one SKILL.md file; found {len(skill_files)}",
            )
        return await self._load_discovered_skill(
            DiscoveredSkill(bundle_root=source_path, skill_file=skill_files[0])
        )

    async def _load_discovered_skill(
        self,
        discovered: DiscoveredSkill,
    ) -> tuple[SkillMetadata, str]:
        source_path = discovered.bundle_root.expanduser().resolve()
        skill_path = discovered.skill_file.expanduser().resolve()
        if not source_path.is_dir() or not skill_path.is_file():
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID,
                "the previewed skill bundle or SKILL.md no longer exists",
            )
        if skill_path.name != "SKILL.md" or not is_under_root(skill_path, source_path):
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID,
                "the previewed SKILL.md is outside its bundle root",
            )
        source_hash = await asyncio.to_thread(_compute_source_hash, source_path)
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

    def _resolve_user_id(self) -> str:
        """Return the profile that owns this mutation, or fail loudly.

        Every lock scope, receipt, and install root is per-user, so there is no
        sensible shared fallback when no session is active.
        """
        user_id = self._registry._resolve_current_user_id()
        if not user_id:
            raise SkillRuntimeError(SKILL_INSTALL_INVALID, "no active user profile")
        return str(user_id)

    def _resolve_install_root(self) -> Path:
        root = get_installed_skills_root(self._resolve_user_id())
        root.mkdir(parents=True, exist_ok=True)
        return root
