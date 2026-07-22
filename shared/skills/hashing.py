"""Canonical, collision-safe hashing for standard Agent Skill bundles."""

from __future__ import annotations

import hashlib
import os
import stat
import struct
from collections.abc import Callable
from pathlib import Path

from shared.skills.commands import is_link_like

_HASH_FORMAT = b"agent-skill-bundle-v2\0"
_IGNORED_DIR_NAMES = frozenset({".git", "__pycache__", ".venv"})
_INSTALL_METADATA_FILENAME = "install.json"


class UnsafeSkillBundleError(ValueError):
    """Raised when a bundle cannot be hashed without following an unsafe path."""


def _add_record_field(hasher, value: bytes) -> None:
    hasher.update(struct.pack(">Q", len(value)))
    hasher.update(value)


def compute_skill_bundle_hash(
    bundle_root: Path,
    *,
    link_checker: Callable[[Path], bool] = is_link_like,
) -> str:
    """Hash bundle paths, modes, sizes, and contents with explicit boundaries."""
    root = bundle_root.resolve()
    if not root.is_dir() or link_checker(bundle_root):
        raise UnsafeSkillBundleError("skill bundle root is missing or linked")

    entries: list[tuple[str, Path]] = []
    for current_root, dir_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        for dir_name in dir_names:
            directory = current / dir_name
            if link_checker(directory):
                raise UnsafeSkillBundleError(f"skill bundle contains linked directory: {dir_name}")
        dir_names[:] = sorted(name for name in dir_names if name not in _IGNORED_DIR_NAMES)
        for file_name in file_names:
            path = current / file_name
            if link_checker(path):
                raise UnsafeSkillBundleError(f"skill bundle contains linked file: {file_name}")
            relative = path.relative_to(root).as_posix()
            if relative == _INSTALL_METADATA_FILENAME or path.suffix == ".pyc":
                continue
            try:
                path.resolve().relative_to(root)
            except ValueError as exc:
                raise UnsafeSkillBundleError(
                    f"skill bundle path escapes its root: {relative}"
                ) from exc
            entries.append((relative, path))

    hasher = hashlib.sha256(_HASH_FORMAT)
    for relative, path in sorted(entries):
        descriptor: int | None = None
        try:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(path, flags)
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise UnsafeSkillBundleError(f"skill bundle path is not a regular file: {relative}")
            relative_bytes = relative.encode("utf-8")
            _add_record_field(hasher, relative_bytes)
            hasher.update(struct.pack(">I", stat.S_IMODE(before.st_mode)))
            hasher.update(struct.pack(">Q", before.st_size))
            bytes_read = 0
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = None
                while chunk := stream.read(1024 * 1024):
                    bytes_read += len(chunk)
                    hasher.update(chunk)
                after_open = os.fstat(stream.fileno())
            after = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise UnsafeSkillBundleError(f"skill bundle changed while hashing: {relative}") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if (
            bytes_read != before.st_size
            or not stat.S_ISREG(after.st_mode)
            or (after_open.st_dev, after_open.st_ino) != (before.st_dev, before.st_ino)
            or after_open.st_size != before.st_size
            or after_open.st_mtime_ns != before.st_mtime_ns
            or stat.S_IMODE(after_open.st_mode) != stat.S_IMODE(before.st_mode)
            or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or link_checker(path)
        ):
            raise UnsafeSkillBundleError(f"skill bundle changed while hashing: {relative}")
    return hasher.hexdigest()
