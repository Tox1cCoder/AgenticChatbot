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
    """
    Normalize a path to an absolute, resolved form.

    Args:
        path: The path to normalize.

    Returns:
        The normalized absolute path.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        base = Path(base_dir).expanduser() if base_dir is not None else Path.cwd()
        p = base / p
    return p.resolve()


def is_under_root(path: Path, root: Path) -> bool:
    """
    Check if a path is under a given root directory.

    Args:
        path: The path to check.
        root: The root directory.

    Returns:
        True if path is under root, False otherwise.
    """
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_workspace_path(path: str | Path) -> Path:
    """
    Normalize a local path and optionally enforce workspace roots.

    Args:
        path: The path to validate.

    Returns:
        The normalized path if valid.

    Raises:
        PathSecurityError: If workspace roots are configured and the path is outside them.
    """
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
    """
    Sanitize a filename for safe storage.

    Args:
        filename: The filename to sanitize.

    Returns:
        A sanitized filename safe for filesystem storage.
    """
    # Remove path separators
    name = os.path.basename(filename)

    # Replace dangerous characters
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)

    # Limit length
    if len(name) > 255:
        base, ext = os.path.splitext(name)
        name = base[: 255 - len(ext)] + ext

    # Handle empty or dot-only names
    if not name or name.startswith("."):
        name = "_" + name

    return name


def to_posix_path(path: str | Path) -> str:
    """
    Convert a path to POSIX format for cross-platform compatibility.

    Args:
        path: The path to convert.

    Returns:
        The path in POSIX format (forward slashes).
    """
    return str(Path(path).as_posix())


def make_relative_to_root(path: Path, root: Path) -> str:
    """
    Make a path relative to a root for safe audit logging.

    Args:
        path: The path to make relative.
        root: The root directory.

    Returns:
        The relative path string, or the original path if not under root.
    """
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def profile_subdir_path(user_id: str, subdir: str) -> Path:
    """
    Resolve a user-specific profile subdirectory without touching the filesystem.

    The structure follows: {profile_root}/{server_hash}/{user_id}/{subdir}

    Use this from read paths. Callers that are about to write should use
    :func:`get_profile_subdir`, which also creates the directory.

    Args:
        user_id: The user's ID.
        subdir: The subdirectory name (e.g., "session", "mcp", "skills").

    Returns:
        The path to the subdirectory, whether or not it exists.
    """
    server_hash = hashlib.sha256(client_settings.server_api_base_url.encode()).hexdigest()[:12]

    return Path(client_settings.profile_root) / server_hash / user_id / subdir


def get_profile_subdir(user_id: str, subdir: str) -> Path:
    """
    Get a user-specific profile subdirectory, creating it if needed.

    Args:
        user_id: The user's ID.
        subdir: The subdirectory name (e.g., "session", "mcp", "skills").

    Returns:
        The path to the subdirectory, created if it doesn't exist.
    """
    path = profile_subdir_path(user_id, subdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def get_device_profile_subdir(
    user_id: str,
    device_identifier: str,
    subdir: str,
) -> Path:
    """Return a profile directory isolated to one installation identity."""

    safe_user_id = _validate_profile_component(user_id, "user_id")
    safe_device_id = _validate_profile_component(device_identifier, "device_identifier")
    safe_subdir = _validate_profile_component(subdir, "subdir")
    path = (
        get_profile_subdir(safe_user_id, "devices")
        / safe_device_id
        / safe_subdir
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_installed_skills_root(user_id: str) -> Path:
    """Return the profile directory that holds profile-installed skill bundles.

    Single source of truth for the installed-bundle location so the installer
    (which writes here) and the registry scanner (which reads here) cannot
    drift. Resolution only: the directory legitimately does not exist until the
    first bundle is installed, and the installer creates it then.
    """
    return profile_subdir_path(user_id, "skills") / "installed"


def get_skill_runtimes_root(user_id: str) -> Path:
    """Return the profile directory that holds prepared skill runtimes.

    Resolution only; the runtime preparer creates it when it first needs it.
    """
    return profile_subdir_path(user_id, "skills") / "runtimes"
