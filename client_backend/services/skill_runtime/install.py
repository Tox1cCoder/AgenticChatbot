"""Generic skill bundle installation.

Installs a local *directory* bundle (``SKILL.md`` plus an optional
``skill.json``) into the current user's profile skill root, so a user can add
a new provider-neutral skill without editing ``CLIENT_SKILLS_ROOTS`` or
restarting the sidecar with new environment variables.

Archive (zip) install is intentionally out of scope here -- the plan marks it
optional, and every path-traversal concern an archive would introduce is
already handled generically for the directory case via
:func:`client_backend.core.paths.is_under_root` on the resolved install
target. This module never installs a dependency, resolves a secret, or
executes anything; it only validates, copies, and records metadata.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from client_backend.core.logging import get_logger
from client_backend.core.paths import get_installed_skills_root, is_under_root, sanitize_filename
from client_backend.services.local_skills_registry import LocalSkillsRegistry, get_skills_registry
from client_backend.services.runtime_bridge import get_runtime_bridge
from shared.skills.errors import (
    SKILL_INSTALL_CONFLICT,
    SKILL_INSTALL_INVALID,
    SKILL_MANIFEST_INVALID,
    UNSAFE_BUNDLE_PATH,
    SkillRuntimeError,
)
from shared.skills.front_matter import parse_skill_front_matter
from shared.skills.manifest import load_manifest

logger = get_logger(__name__)

# Never hashed / never copied verbatim: build artifacts, VCS metadata, and our
# own install marker (which does not exist yet at hash time, but would appear
# on a re-install-over-existing-copy if we didn't exclude it defensively).
_IGNORED_DIR_NAMES = frozenset({"__pycache__", ".git"})
_INSTALL_METADATA_FILENAME = "install.json"


def _compute_source_hash(source: Path) -> str:
    """Hash a bundle directory deterministically: sorted relative path + bytes.

    Sorting by POSIX-style relative path makes the digest stable across
    platforms and directory-walk orders.

    Walks WITHOUT following symlinks (``os.walk(followlinks=False)``) and
    rejects any symlinked file or directory with ``UNSAFE_BUNDLE_PATH``. Skill
    bundles have no legitimate need for symlinks, and allowing them would let a
    bundle (a) reference files outside itself — copying arbitrary readable
    content into the installed skill, which is later served to the model — or
    (b) form a cycle that makes the walk unbounded. Rejecting here, before any
    copy, aborts the whole install for an unsafe bundle.
    """
    hasher = hashlib.sha256()
    entries: list[tuple[str, Path]] = []

    for root, dir_names, file_names in os.walk(source, followlinks=False):
        root_path = Path(root)
        # Prune ignored dirs so we neither descend into nor scrutinize them.
        dir_names[:] = [name for name in dir_names if name not in _IGNORED_DIR_NAMES]
        for dir_name in dir_names:
            if (root_path / dir_name).is_symlink():
                raise SkillRuntimeError(
                    UNSAFE_BUNDLE_PATH,
                    f"bundle contains a symlinked directory ('{dir_name}'); "
                    "symlinks are not allowed in skill bundles",
                )
        for file_name in file_names:
            file_path = root_path / file_name
            if file_path.is_symlink():
                raise SkillRuntimeError(
                    UNSAFE_BUNDLE_PATH,
                    f"bundle contains a symlinked file ('{file_name}'); "
                    "symlinks are not allowed in skill bundles",
                )
            relative_posix = file_path.relative_to(source).as_posix()
            if relative_posix == _INSTALL_METADATA_FILENAME:
                continue
            entries.append((relative_posix, file_path))

    entries.sort(key=lambda entry: entry[0])
    for relative_posix, path in entries:
        hasher.update(relative_posix.encode("utf-8"))
        hasher.update(path.read_bytes())

    return hasher.hexdigest()


def _find_installed_bundle(install_root: Path, name: str) -> Path | None:
    """Find the installed bundle directory whose install.json bundle_name matches."""
    if not install_root.exists():
        return None

    for entry in sorted(install_root.iterdir()):
        if not entry.is_dir():
            continue
        install_metadata = _read_install_metadata(entry)
        if install_metadata is not None and install_metadata.get("bundle_name") == name:
            return entry

    return None


def _read_install_metadata(bundle_dir: Path) -> dict | None:
    """Read and parse ``install.json`` from a bundle dir; tolerate corruption."""
    install_json_path = bundle_dir / _INSTALL_METADATA_FILENAME
    if not install_json_path.exists():
        return None

    try:
        payload = json.loads(install_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read install metadata for %s: %s", bundle_dir, exc)
        return None

    return payload if isinstance(payload, dict) else None


class SkillBundleInstaller:
    """Validates and installs local skill bundle directories into the profile root."""

    def __init__(self, registry: LocalSkillsRegistry | None = None) -> None:
        self._registry = registry if registry is not None else get_skills_registry()

    async def install(self, source: str | Path) -> dict:
        """Install a directory bundle. Returns a JSON-safe result summary."""
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists() or not source_path.is_dir():
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID,
                f"source '{source}' does not exist or is not a directory",
            )

        skill_md_path = source_path / "SKILL.md"
        if not skill_md_path.exists():
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID, "bundle is missing a SKILL.md file"
            )

        raw_skill_md = await asyncio.to_thread(skill_md_path.read_text, encoding="utf-8")
        name = self._resolve_bundle_name(raw_skill_md, source_path)
        manifest_status = await self._validate_manifest(source_path)
        source_hash = await asyncio.to_thread(_compute_source_hash, source_path)
        short_hash = source_hash[:12]

        await self._registry.initialize()
        existing = self._registry.get_skill(name)
        if existing is not None and existing.enabled:
            raise SkillRuntimeError(
                SKILL_INSTALL_CONFLICT,
                f"a skill named '{name}' is already active; disable or uninstall it first",
            )

        install_root = self._resolve_install_root()
        safe_name = sanitize_filename(name)
        target = install_root / f"{safe_name}-{short_hash}"

        if not is_under_root(target, install_root):
            raise SkillRuntimeError(
                UNSAFE_BUNDLE_PATH, "install target escapes the profile skill root"
            )

        # Any same-named bundle that reaches here is DISABLED (an enabled one
        # already raised SKILL_INSTALL_CONFLICT). Replace it: remove the stale
        # install directory so the reinstall is discoverable and scan dedup is
        # never left arbitrating between two same-named installed bundles. The
        # is_under_root guard keeps rmtree confined to the profile skill root.
        stale_bundle = _find_installed_bundle(install_root, name)
        if stale_bundle is not None and is_under_root(stale_bundle, install_root):
            await asyncio.to_thread(shutil.rmtree, stale_bundle)

        if target.exists():
            raise SkillRuntimeError(SKILL_INSTALL_CONFLICT, "bundle already installed")

        # symlinks=True copies any symlink verbatim instead of dereferencing it;
        # _compute_source_hash already rejected symlinked bundles above, so this
        # is defense-in-depth against a TOCTOU race between hashing and copying.
        await asyncio.to_thread(
            shutil.copytree,
            source_path,
            target,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
        )

        install_payload = {
            "bundle_name": name,
            "source_hash": source_hash,
            "source": "profile",
            "source_path": str(source_path),
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "manifest_status": manifest_status,
            "enabled": True,
            # SkillMetadata._install_summary() (Task 2) reads this exact key
            # to report "install.installed" in the redacted catalog view; it
            # must be present and true for a freshly installed bundle.
            "installed": True,
        }
        await asyncio.to_thread(
            (target / _INSTALL_METADATA_FILENAME).write_text,
            json.dumps(install_payload, indent=2),
            encoding="utf-8",
        )

        await self._registry.refresh()
        await self._refresh_runtime_bridge_catalogs_if_connected()

        return {
            "name": name,
            "install_id": target.name,
            "source_hash": source_hash,
            "manifest_status": manifest_status,
        }

    async def uninstall(self, name: str) -> dict:
        """Remove a previously installed bundle by its skill name."""
        install_root = self._resolve_install_root()
        bundle_dir = _find_installed_bundle(install_root, name)
        if bundle_dir is None:
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID, f"no installed skill named '{name}'"
            )
        if not is_under_root(bundle_dir, install_root):
            raise SkillRuntimeError(
                UNSAFE_BUNDLE_PATH, "installed bundle path escapes the profile skill root"
            )

        await asyncio.to_thread(shutil.rmtree, bundle_dir)

        await self._registry.refresh()
        await self._refresh_runtime_bridge_catalogs_if_connected()

        return {"name": name, "removed": True}

    def list_installed(self) -> list[dict]:
        """Return install metadata for every installed bundle (device-local view)."""
        user_id = self._registry._resolve_current_user_id()
        if not user_id:
            return []

        install_root = get_installed_skills_root(user_id)
        if not install_root.exists():
            return []

        installed: list[dict] = []
        for entry in sorted(install_root.iterdir()):
            if not entry.is_dir():
                continue
            metadata = _read_install_metadata(entry)
            if metadata is not None:
                installed.append(metadata)

        return installed

    @staticmethod
    def _resolve_bundle_name(raw_skill_md: str, source_path: Path) -> str:
        parsed = parse_skill_front_matter(raw_skill_md)
        if parsed is None:
            return sanitize_filename(source_path.name)

        if not parsed.name or not parsed.name.strip():
            raise SkillRuntimeError(
                SKILL_INSTALL_INVALID, "malformed front matter: missing name"
            )

        return parsed.name

    @staticmethod
    async def _validate_manifest(source_path: Path) -> str:
        """Validate an optional ``skill.json``. Returns the manifest status string."""
        skill_json_path = source_path / "skill.json"
        if not skill_json_path.exists():
            return "instruction_only"

        try:
            raw_manifest = await asyncio.to_thread(skill_json_path.read_text, encoding="utf-8")
            load_manifest(json.loads(raw_manifest))
        except Exception as exc:
            raise SkillRuntimeError(
                SKILL_MANIFEST_INVALID, f"invalid skill.json: {exc}"
            ) from exc

        return "manifest_present"

    def _resolve_install_root(self) -> Path:
        user_id = self._registry._resolve_current_user_id()
        if not user_id:
            raise SkillRuntimeError(SKILL_INSTALL_INVALID, "no active user profile")

        install_root = get_installed_skills_root(user_id)
        install_root.mkdir(parents=True, exist_ok=True)
        return install_root

    async def _refresh_runtime_bridge_catalogs_if_connected(self) -> None:
        bridge = get_runtime_bridge()
        if not bridge.is_connected() or not bridge.get_registered_device_id():
            return
        await bridge.refresh_catalogs()
