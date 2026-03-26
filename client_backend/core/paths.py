"""
Path handling utilities for the client backend.

Provides secure path normalization, validation, and sandboxing.
"""

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


def get_profile_subdir(user_id: str, subdir: str) -> Path:
    """
    Get a user-specific subdirectory within the profile root.

    The structure follows: {profile_root}/{server_hash}/{user_id}/{subdir}

    Args:
        user_id: The user's ID.
        subdir: The subdirectory name (e.g., "session", "mcp", "skills").

    Returns:
        The path to the subdirectory, created if it doesn't exist.
    """
    import hashlib

    server_hash = hashlib.sha256(client_settings.server_api_base_url.encode()).hexdigest()[:12]

    path = Path(client_settings.profile_root) / server_hash / user_id / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path
