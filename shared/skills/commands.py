"""Shared command-asset safety predicates for portable skill bundles."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def _has_windows_reparse_point(
    path: Path,
    *,
    platform_name: str | None = None,
) -> bool:
    """Detect junctions and other reparse points on Python 3.10 and newer."""
    if (platform_name or os.name) != "nt":
        return False
    try:
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def is_link_like(path: Path) -> bool:
    """Return whether a path is a symlink or Windows junction/reparse point."""
    if path.is_symlink():
        return True
    if _has_windows_reparse_point(path):
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def is_supported_bundle_command(path: Path) -> bool:
    """Return whether the current platform can directly run this bundle asset."""
    if not path.is_file() or is_link_like(path):
        return False
    suffix = path.suffix.lower()
    if suffix == ".py":
        return True
    if os.name == "nt":
        return suffix == ".exe"
    return not suffix and os.access(path, os.X_OK)
