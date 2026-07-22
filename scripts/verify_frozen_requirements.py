"""Resolve the complete frozen runtime requirements without installed packages."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_EXPECTED_PYTHON = (3, 11)


def main() -> int:
    if sys.version_info[:2] != _EXPECTED_PYTHON:
        expected = ".".join(map(str, _EXPECTED_PYTHON))
        actual = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(
            f"Run this resolver with Python {expected}; current interpreter is {actual}.",
            file=sys.stderr,
        )
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--disable-pip-version-check",
        "--quiet",
        "-r",
        "requirements.txt",
    ]
    return subprocess.run(command, cwd=repo_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
