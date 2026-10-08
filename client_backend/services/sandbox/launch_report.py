"""Why the last sandboxed launch failed, for the sidecar to report.

The launcher sits between the MCP SDK and the sandboxed program. Its stderr,
and the program's, reach only the sidecar's console: the MCP adapter offers no
way to capture them, so a failed start reaches the sidecar as nothing more
than "Connection closed". The launcher writes the reason here instead, and the
manager reads it back when a sandboxed start fails.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from client_backend.core.config import client_settings

__all__ = ["LaunchFailure", "launch_report_path", "read_launch_failure", "write_launch_failure"]

STDERR_TAIL_LINES = 20
_MAX_LINE_LENGTH = 500


@dataclass(frozen=True)
class LaunchFailure:
    message: str
    stderr: tuple[str, ...]

    def summary(self) -> str:
        """The reason in one line, with the program's last words when it left any."""

        last_line = next((line for line in reversed(self.stderr) if line.strip()), "")
        return f"{self.message}: {last_line.strip()}" if last_line else self.message


def launch_report_path() -> Path:
    return Path(client_settings.profile_root) / "sandbox" / "last-launch-failure.json"


def write_launch_failure(message: str, stderr: Sequence[str] = ()) -> None:
    tail = [line[:_MAX_LINE_LENGTH] for line in list(stderr)[-STDERR_TAIL_LINES:]]
    payload = {"message": message, "stderr": tail, "written_at": time.time()}
    path = launch_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def read_launch_failure(since: float) -> LaunchFailure | None:
    """The failure recorded at or after ``since``; an older one is another launch's."""

    try:
        payload = json.loads(launch_report_path().read_text(encoding="utf-8"))
        written_at = float(payload["written_at"])
        message = str(payload["message"])
        stderr = tuple(str(line) for line in payload.get("stderr") or [])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if written_at < since:
        return None
    return LaunchFailure(message, stderr)
