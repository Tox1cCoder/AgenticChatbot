"""Crash-safe JSON state for the skill installation workflow.

Uploads, installation operations, cleanup receipts, and the catalog generation
are all resumed from disk after a restart, so a half-written record is worse than
no record: it turns a recoverable interruption into an unparseable state the
service has to guess about. Every write here is therefore all-or-nothing -- the
reader either sees the previous complete document or the next complete one.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class SkillStateError(Exception):
    """Raised when persisted skill state is unreadable or not a JSON object."""


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace ``path`` with ``payload`` atomically, or leave it untouched.

    The document is written to a uniquely named sibling, flushed all the way to
    the platter, and only then renamed over the target. ``os.replace`` is atomic
    on both POSIX and Windows, so a concurrent reader never observes a truncated
    file and a crash mid-write loses the new document rather than the old one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        with contextlib.suppress(OSError):
            # Best-effort owner-only mode; a no-op against Windows ACLs.
            os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        # Reached both when replace() failed and when it succeeded (the name is
        # gone by then), so the staging file never accumulates.
        with contextlib.suppress(OSError):
            temporary.unlink()


def _sync_directory(directory: Path) -> None:
    """Best-effort fsync of a directory so the rename itself is durable."""
    if os.name != "posix":
        return
    descriptor = None
    try:
        descriptor = os.open(directory, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        return
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Read a JSON object written by :func:`atomic_write_json`.

    Returns ``None`` when the file does not exist, which is the ordinary
    "nothing persisted yet" case. Anything present but unreadable raises, because
    silently treating corruption as absence would drop a real upload or
    installation record.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SkillStateError(f"skill state at {path.name} could not be read: {exc}") from exc

    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise SkillStateError(f"skill state at {path.name} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise SkillStateError(
            f"skill state at {path.name} must be a JSON object, got {type(payload).__name__}"
        )
    return payload
