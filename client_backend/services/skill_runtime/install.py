"""Atomic installation of complete, standard one-skill bundles."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

import filelock

from client_backend.core.logging import get_logger
from client_backend.core.paths import (
    get_skill_operations_root,
    is_promotable_bundle,
    is_under_root,
    make_relative_to_root,
    resolve_skills_root,
    sanitize_filename,
)
from client_backend.services.local_skills_registry import (
    LocalSkillsRegistry,
    SkillMetadata,
    get_skills_registry,
    publishes_single_skill,
)
from client_backend.services.skill_runtime.collection import DiscoveredSkill
from client_backend.services.skill_runtime.environment import (
    PREPARATION_LEASE_FILENAME,
    SETUP_TIMEOUT_SECONDS,
    SkillEnvironmentManager,
)
from client_backend.services.skill_runtime.locks import (
    SKILLS_MUTATION_SCOPE,
    profile_lock,
)
from client_backend.services.skill_runtime.secrets import SkillSecretStore
from client_backend.services.skill_runtime.state import atomic_write_json, read_json_object
from shared.skills.errors import (
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
_BACKGROUND_PREPARATION_CLEANUPS: set[asyncio.Task] = set()


async def _settled_to_thread(function, /, *args, **kwargs):
    """Finish an in-flight thread operation before propagating cancellation."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def _cleanup_cancelled_preparation(
    preparation: asyncio.Task,
    *,
    stage: Path,
    stage_lease: filelock.FileLock,
) -> None:
    """Clean stages only after a cancelled worker stops touching them."""
    with contextlib.suppress(Exception):
        await preparation
    try:
        await asyncio.to_thread(stage_lease.release)
        (stage / PREPARATION_LEASE_FILENAME).unlink(missing_ok=True)
        if stage.exists():
            await asyncio.to_thread(shutil.rmtree, stage)
    except Exception:  # noqa: BLE001 - startup cleanup can retry orphaned stages
        logger.warning("cancelled skill preparation cleanup failed", exc_info=True)


def _track_background_cleanup(task: asyncio.Task) -> None:
    _BACKGROUND_PREPARATION_CLEANUPS.add(task)
    task.add_done_callback(_BACKGROUND_PREPARATION_CLEANUPS.discard)


def _prepare_environment_with_runtime_lock(
    environment,
    skill,
    *,
    approve_setup: bool,
    force: bool = False,
):
    lock_path = environment.preparation_lock_path(skill)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_lease = filelock.FileLock(str(lock_path), thread_local=False)
    try:
        runtime_lease.acquire(timeout=SETUP_TIMEOUT_SECONDS)
    except filelock.Timeout as exc:
        raise SkillRuntimeError(
            SKILL_SETUP_REQUIRED,
            f"runtime preparation for skill '{skill.name}' is already in progress",
        ) from exc
    try:
        return environment.prepare(skill, approve_setup=approve_setup, force=force)
    finally:
        runtime_lease.release()


def _remove_environment_with_runtime_lock(environment, skill_name: str) -> None:
    """Serialize runtime deletion with current and detached preparation workers."""
    lock_path = environment.preparation_lock_path_for_name(skill_name)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_lease = filelock.FileLock(str(lock_path), thread_local=False)
    try:
        runtime_lease.acquire(timeout=SETUP_TIMEOUT_SECONDS)
    except filelock.Timeout as exc:
        raise SkillRuntimeError(
            SKILL_SETUP_REQUIRED,
            f"runtime preparation for skill '{skill_name}' is already in progress",
        ) from exc
    try:
        environment.remove_skill(skill_name)
    finally:
        runtime_lease.release()


@dataclass(frozen=True)
class SkillInstallSpec:
    discovered: DiscoveredSkill
    expected_source_hash: str | None
    approve_setup: bool
    replace_source_hash: str | None
    source_kind: Literal["path", "upload"]


@dataclass
class PreparedSkillInstall:
    name: str
    source_hash: str
    action: Literal["installed", "updated"]
    stage: Path
    target: Path
    previous: Path | None
    backup: Path
    runtime_status: str
    previous_source_hash: str | None


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
        from client_backend.services.skill_runtime.transactions import (
            SkillInstallTransaction,
        )

        await self._notify(observer, "validating")
        discovered = await self._as_discovered(source)
        user_id = self._resolve_user_id()
        await self._notify(observer, "waitingForLock")
        transaction = SkillInstallTransaction(self, user_id)
        transaction_id = uuid.uuid4().hex
        try:
            results = await transaction.execute(
                [
                    SkillInstallSpec(
                        discovered=discovered,
                        expected_source_hash=expected_source_hash,
                        approve_setup=approve_setup,
                        replace_source_hash=replace_source_hash,
                        source_kind=source_kind,
                    )
                ],
                transaction_id=transaction_id,
                observer=observer,
            )
        except BaseException:
            with contextlib.suppress(RuntimeError):
                transaction.finalize()
            raise
        transaction.finalize()
        return results[0]

    async def install_many(
        self,
        specs: list[SkillInstallSpec],
        *,
        transaction_id: str,
        observer: SkillInstallObserver | None = None,
    ) -> list[dict]:
        """Install a complete collection and retain its journal for its receipt."""
        from client_backend.services.skill_runtime.transactions import (
            SkillInstallTransaction,
        )

        await self._notify(observer, "waitingForLock")
        transaction = SkillInstallTransaction(self, self._resolve_user_id())
        return await transaction.execute(
            specs,
            transaction_id=transaction_id,
            observer=observer,
        )

    def finalize_transaction(self, transaction_id: str) -> None:
        from client_backend.services.skill_runtime.transactions import (
            SkillInstallTransaction,
        )

        transaction = SkillInstallTransaction(self, self._resolve_user_id())
        transaction._transaction_id = transaction_id
        transaction.finalize()

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

    async def prepare_install(
        self,
        spec: SkillInstallSpec,
        *,
        observer: SkillInstallObserver | None = None,
        user_id: str | None = None,
    ) -> PreparedSkillInstall:
        """Validate and stage one install without changing installed bundles."""
        source_skill, shape = await self._load_discovered_skill(spec.discovered)
        preview = await self._build_preview(source_skill, shape)
        self._require_preview_agreement(
            source_skill,
            preview,
            expected_source_hash=spec.expected_source_hash,
            approve_setup=spec.approve_setup,
        )
        owner = user_id or self._resolve_user_id()
        await self._registry.initialize()
        existing = self._registry.get_skill(source_skill.name)
        action = self._resolve_replacement_action(
            source_skill.name,
            existing,
            user_id=owner,
            replace_source_hash=spec.replace_source_hash,
        )

        install_root = self._resolve_install_root()
        target = install_root / (
            f"{sanitize_filename(source_skill.name)}-{source_skill.source_hash[:12]}"
        )
        stage = install_root / f"{target.name}.stage-{uuid.uuid4().hex}"
        previous = self._resolve_previous_bundle(install_root, source_skill.name, existing)
        backup_basis = previous.name if previous is not None else target.name
        backup = install_root / f"{backup_basis}.backup-{uuid.uuid4().hex}"
        for path in (target, stage, backup):
            if not is_under_root(path, install_root):
                raise SkillRuntimeError(
                    UNSAFE_BUNDLE_PATH, "install path escapes the skills root"
                )

        runtime_cleanup_deferred = False
        try:
            await self._notify(observer, "copying")
            await _settled_to_thread(
                shutil.copytree,
                source_skill.bundle_root,
                stage,
                symlinks=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".venv"),
            )
            copied_hash = await _settled_to_thread(_compute_source_hash, stage)
            if copied_hash != source_skill.source_hash:
                raise SkillRuntimeError(
                    SKILL_INSTALL_INVALID,
                    "bundle changed while it was being copied; request a new preview",
                )
            relative_skill_path = source_skill.path.relative_to(source_skill.bundle_root)
            install_payload = {
                "bundle_name": source_skill.name,
                "source_hash": source_skill.source_hash,
                "source": "upload" if spec.source_kind == "upload" else "profile",
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "enabled": True,
                "installed": True,
            }
            if spec.source_kind == "path":
                install_payload["source_path"] = str(source_skill.bundle_root)
            await _settled_to_thread(
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
                stage_lease = filelock.FileLock(
                    str(stage / PREPARATION_LEASE_FILENAME),
                    thread_local=False,
                )
                try:
                    await _settled_to_thread(
                        stage_lease.acquire,
                        timeout=SETUP_TIMEOUT_SECONDS,
                    )
                except BaseException:
                    await asyncio.to_thread(stage_lease.release)
                    (stage / PREPARATION_LEASE_FILENAME).unlink(missing_ok=True)
                    raise
                preparation = asyncio.create_task(
                    asyncio.to_thread(
                        _prepare_environment_with_runtime_lock,
                        self._environment,
                        installed_skill,
                        approve_setup=spec.approve_setup,
                    )
                )
                try:
                    runtime = await asyncio.shield(preparation)
                except asyncio.CancelledError:
                    runtime_cleanup_deferred = True
                    cleanup = asyncio.create_task(
                        _cleanup_cancelled_preparation(
                            preparation,
                            stage=stage,
                            stage_lease=stage_lease,
                        )
                    )
                    _track_background_cleanup(cleanup)
                    raise
                finally:
                    if not runtime_cleanup_deferred:
                        await _settled_to_thread(stage_lease.release)
                        (stage / PREPARATION_LEASE_FILENAME).unlink(missing_ok=True)
                runtime_status = str(runtime.get("status") or "not_ready")
            elif (
                installed_skill.executable_assets["bin"]
                or installed_skill.executable_assets["scripts"]
            ):
                runtime_status = "ready"
            else:
                runtime_status = "instruction_only"
        except BaseException:
            if not runtime_cleanup_deferred and stage.exists():
                await _settled_to_thread(shutil.rmtree, stage)
            if not runtime_cleanup_deferred and (
                existing is None or existing.source_hash != source_skill.source_hash
            ):
                with contextlib.suppress(Exception):
                    await _settled_to_thread(
                        self._environment.remove_runtime,
                        source_skill.name,
                        source_skill.source_hash,
                    )
            raise

        return PreparedSkillInstall(
            name=source_skill.name,
            source_hash=source_skill.source_hash,
            action=action,
            stage=stage,
            target=target,
            previous=previous,
            backup=backup,
            runtime_status=runtime_status,
            previous_source_hash=existing.source_hash if existing is not None else None,
        )

    @staticmethod
    def _resolve_previous_bundle(
        install_root: Path,
        name: str,
        existing: SkillMetadata | None,
    ) -> Path | None:
        """Return the directory this install replaces, if any.

        The catalog decides, not ``install.json``: a bundle the operator wrote by
        hand carries no install metadata, so scanning for metadata alone would
        miss it and promote a second directory publishing the same skill name --
        two bundles, one name, and whichever the scanner reaches first wins.
        The metadata scan remains as a fallback for a bundle the registry has not
        picked up yet.
        """
        if existing is not None and is_promotable_bundle(existing.bundle_root, install_root):
            return existing.bundle_root
        return _find_installed_bundle(install_root, name)

    def _resolve_replacement_action(
        self,
        name: str,
        existing: SkillMetadata | None,
        *,
        user_id: str,
        replace_source_hash: str | None,
    ) -> Literal["installed", "updated"]:
        """Decide whether this request may overwrite an existing skill.

        There is one skill root and the sidecar owns it, so a colliding skill is
        replaceable -- but only deliberately, and only when it is a direct child
        of that root, which is the only shape promotion and rollback can move.
        """
        if existing is None:
            if replace_source_hash is not None:
                raise SkillRuntimeError(
                    SKILL_SOURCE_CHANGED,
                    "no installed skill matches the requested replacement; reload the catalog",
                )
            return "installed"

        skills_root = resolve_skills_root(user_id)
        if not is_under_root(existing.bundle_root, skills_root):
            # The registry scans the root the installer writes to, so a skill
            # outside it means those two disagree. Refuse rather than write
            # outside the directory we own.
            raise SkillRuntimeError(
                UNSAFE_BUNDLE_PATH,
                f"skill '{name}' resolves outside the skills root",
            )
        if not is_promotable_bundle(existing.bundle_root, skills_root):
            raise SkillRuntimeError(
                SKILL_INSTALL_CONFLICT,
                f"skill '{name}' is published by the skills root itself rather than by a "
                "folder inside it; move it into its own folder before installing",
            )
        if not publishes_single_skill(existing.bundle_root):
            raise SkillRuntimeError(
                SKILL_INSTALL_CONFLICT,
                f"skill '{name}' shares the folder "
                f"'{make_relative_to_root(existing.bundle_root, skills_root)}' with other "
                "skills, and replacing it would remove them too; give it its own folder "
                "inside the skills root first",
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
        user_id = self._resolve_user_id()
        async with profile_lock(user_id, SKILLS_MUTATION_SCOPE):
            await self._registry.refresh()
            skill = self._registry.get_skill(name)
            if skill is None:
                raise SkillRuntimeError(SKILL_INSTALL_INVALID, f"skill '{name}' was not found")
            if not expected_source_hash:
                raise SkillRuntimeError(
                    SKILL_INSTALL_INVALID,
                    "skill setup requires the current expected source hash",
                )
            live_source_hash = await asyncio.to_thread(
                _compute_source_hash,
                skill.bundle_root,
            )
            if expected_source_hash != skill.source_hash or live_source_hash != skill.source_hash:
                raise SkillRuntimeError(
                    SKILL_INSTALL_INVALID,
                    "skill changed after preview; request a new preview before setup",
                )
            result = await _settled_to_thread(
                _prepare_environment_with_runtime_lock,
                self._environment,
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
        async with profile_lock(user_id, SKILLS_MUTATION_SCOPE):
            install_root = self._resolve_install_root()
            bundle = _find_installed_bundle(install_root, name)
            receipt_path = self._cleanup_receipt_path(user_id, name)
            receipt = read_json_object(receipt_path) if receipt_path.is_file() else None

            if bundle is None and receipt is None:
                raise SkillRuntimeError(SKILL_INSTALL_INVALID, f"no installed skill named '{name}'")
            if bundle is not None and not is_under_root(bundle, install_root):
                raise SkillRuntimeError(
                    UNSAFE_BUNDLE_PATH, "installed bundle path escapes the skills root"
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
                (
                    "runtime_removed",
                    lambda skill_name: _remove_environment_with_runtime_lock(
                        self._environment, skill_name
                    ),
                ),
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
        root = resolve_skills_root(user_id)
        if not root.is_dir():
            return []
        installed: list[dict] = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            metadata = _read_install_metadata(entry)
            if metadata is None:
                continue
            try:
                live_hash = _compute_source_hash(entry)
            except (OSError, ValueError, SkillRuntimeError):
                continue
            installed.append({**metadata, "source_hash": live_hash})
        return installed

    async def _load_source(
        self,
        source: str | Path | DiscoveredSkill,
    ) -> tuple[SkillMetadata, str]:
        return await self._load_discovered_skill(await self._as_discovered(source))

    async def _as_discovered(
        self,
        source: str | Path | DiscoveredSkill,
    ) -> DiscoveredSkill:
        if isinstance(source, DiscoveredSkill):
            return source
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
        return DiscoveredSkill(bundle_root=source_path, skill_file=skill_files[0])

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
        root = resolve_skills_root(self._resolve_user_id())
        root.mkdir(parents=True, exist_ok=True)
        return root
