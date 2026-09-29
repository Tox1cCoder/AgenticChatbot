"""
Path handling utilities for the client backend.

Provides secure path normalization, validation, and sandboxing.
"""

import hashlib
import os
import re
from pathlib import Path

from client_backend.core.config import client_settings


class PathSecurityError(Exception):
    """Raised when a path operation violates security constraints."""

    pass


def normalize_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve ``path`` to an absolute path; relative paths are taken from ``base_dir`` or cwd."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        base = Path(base_dir).expanduser() if base_dir is not None else Path.cwd()
        p = base / p
    return p.resolve()


def is_under_root(path: Path, root: Path) -> bool:
    """Report whether ``path`` resolves inside ``root`` (both resolved first)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_workspace_path(path: str | Path) -> Path:
    """Normalize a local path; when workspace roots are configured, require it inside one."""
    normalized = normalize_path(path)
    workspace_roots = client_settings.workspace_roots

    if not workspace_roots:
        return normalized

    for root in workspace_roots:
        root_path = normalize_path(root)
        if is_under_root(normalized, root_path):
            return normalized

    raise PathSecurityError(
        f"Path '{path}' is not within any configured workspace root. "
        f"Allowed roots: {workspace_roots}"
    )


def sanitize_filename(filename: str) -> str:
    """Reduce ``filename`` to one safe, non-hidden component of at most 255 characters."""
    name = os.path.basename(filename)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)

    if len(name) > 255:
        base, ext = os.path.splitext(name)
        name = base[: 255 - len(ext)] + ext

    if not name or name.startswith("."):
        name = "_" + name

    return name


def make_relative_to_root(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` for messages, or unchanged if it is outside."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _validate_profile_component(value: str, label: str) -> str:
    normalized = str(value or "").strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
        or Path(normalized).name != normalized
    ):
        raise ValueError(f"{label} must be a non-empty path component")
    return normalized


def profile_subdir_path(user_id: str, subdir: str) -> Path:
    """
    Resolve a user-specific profile subdirectory without touching the filesystem.

    The structure follows: {profile_root}/{server_hash}/{user_id}/{subdir}

    Use this from read paths. Callers that are about to write should use
    :func:`get_profile_subdir`, which also creates the directory.

    ``user_id`` reaches here from a query string (``/auth/restore``) and from the
    unverified ``sub`` claim of a presented bearer token, so it is validated as a
    single path component: ``..\\..\\x`` would otherwise create, read, and delete
    ``session/credentials.json`` anywhere the user can write.
    """
    safe_user_id = _validate_profile_component(user_id, "user_id")
    safe_subdir = _validate_profile_component(subdir, "subdir")
    server_hash = hashlib.sha256(client_settings.server_api_base_url.encode()).hexdigest()[:12]

    return Path(client_settings.profile_root) / server_hash / safe_user_id / safe_subdir


def get_profile_subdir(user_id: str, subdir: str) -> Path:
    """Like :func:`profile_subdir_path`, but create the directory first."""
    path = profile_subdir_path(user_id, subdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device_profile_subdir(
    user_id: str,
    device_identifier: str,
    subdir: str,
) -> Path:
    """Return a profile directory isolated to one installation identity."""

    safe_device_id = _validate_profile_component(device_identifier, "device_identifier")
    safe_subdir = _validate_profile_component(subdir, "subdir")
    path = get_profile_subdir(user_id, "devices") / safe_device_id / safe_subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_skills_root(user_id: str) -> Path:
    """Return the one directory this profile reads and writes skill bundles in.

    Single source of truth for the bundle location so the installer (which
    writes here) and the registry scanner (which reads here) cannot drift.

    ``CLIENT_SKILLS_ROOT`` wins when the operator sets it: they named the folder
    they want to see their skills in, so the sidecar scans *and* installs there.
    Unset, bundles live under the profile, where the ``{server_hash}/{user_id}``
    prefix keeps two users on one machine from sharing a catalog.

    Resolution only: the directory legitimately does not exist until the first
    bundle is installed, and the installer creates it then.
    """
    configured = str(client_settings.skills_root or "").strip()
    if configured:
        return Path(configured)
    return _user_skills_root(user_id) / "installed"


def is_promotable_bundle(bundle_root: Path, skills_root: Path) -> bool:
    """Report whether the installer can swap ``bundle_root`` in place.

    The installer promotes, backs up, and rolls back bundles as direct children
    of the skills root, so that is the only shape it can replace. A bundle nested
    deeper -- a cloned repository of skills dropped into the root, say -- would
    have to be moved to be replaced, and a rollback could not put it back where
    it came from. Those are the operator's folders to reorganize, not ours.
    """
    try:
        return bundle_root.resolve().parent == skills_root.resolve()
    except OSError:
        return False


def get_skill_runtimes_root(user_id: str) -> Path:
    """Return the profile directory that holds prepared skill runtimes.

    Resolution only; the runtime preparer creates it when it first needs it.
    """
    return profile_subdir_path(user_id, "skills") / "runtimes"


def _user_skills_root(user_id: str) -> Path:
    return profile_subdir_path(user_id, "skills")


def get_skill_uploads_root(user_id: str) -> Path:
    """Return the profile directory that holds staged skill archive uploads."""
    return _user_skills_root(user_id) / "uploads"


def get_skill_operations_root(user_id: str) -> Path:
    """Return the profile directory that holds persisted installation receipts."""
    return _user_skills_root(user_id) / "operations"


def get_skill_locks_root(user_id: str) -> Path:
    """Return the profile directory that holds cross-process skill lock files."""
    return _user_skills_root(user_id) / "locks"


def get_skill_catalog_state_path(user_id: str) -> Path:
    """Return the file that persists this profile's catalog generation state."""
    return _user_skills_root(user_id) / "catalog_state.json"
