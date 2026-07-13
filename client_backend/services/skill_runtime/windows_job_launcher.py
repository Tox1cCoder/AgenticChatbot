"""Trusted Windows gate that starts a skill command only after Job attachment."""

from __future__ import annotations

import json
import subprocess
import sys

_MAX_REQUEST_BYTES = 1024 * 1024


def main() -> int:
    raw = sys.stdin.buffer.readline(_MAX_REQUEST_BYTES + 1)
    if not raw or len(raw) > _MAX_REQUEST_BYTES:
        print("invalid command request", file=sys.stderr)
        return 125
    try:
        argv = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        print("invalid command request", file=sys.stderr)
        return 125
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(value, str) or "\0" in value for value in argv)
    ):
        print("invalid command request", file=sys.stderr)
        return 125
    try:
        return subprocess.run(argv, check=False, shell=False).returncode
    except OSError:
        print("unable to start command", file=sys.stderr)
        return 126


if __name__ == "__main__":
    raise SystemExit(main())
