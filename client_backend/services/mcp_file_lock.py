"""Process-local path locks for atomic MCP read-modify-write transactions."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}


def _lock_for(path: Path) -> threading.RLock:
    key = str(Path(path).resolve()).casefold()
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


@contextmanager
def mcp_path_lock(directory: Path) -> Iterator[None]:
    """Serialize all MCP file transactions targeting one profile directory."""

    with _lock_for(directory):
        yield
