"""The take100 skill bundle must run on its own, wherever it is unpacked."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BUNDLE = Path(__file__).resolve().parents[1] / "skills" / "take100"
CREDENTIAL_PREFIXES = ("TAKE100_", "ATTENDANCE_")


def _run_outside_repo(tmp_path: Path, argv: list[str]) -> subprocess.CompletedProcess:
    copy = tmp_path / "take100"
    shutil.copytree(BUNDLE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(CREDENTIAL_PREFIXES) and name != "PYTHONPATH"
    }
    return subprocess.run(
        [sys.executable, *argv],
        cwd=copy,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )


@pytest.mark.parametrize(
    ("argv", "missing"),
    [
        (["scripts/take100.py", "--action", "list"], "TAKE100_EMAIL"),
        (["scripts/attendance.py"], "ATTENDANCE_USERNAME"),
    ],
)
def test_a_copy_outside_the_repo_reports_missing_credentials(tmp_path, argv, missing):
    result = _run_outside_repo(tmp_path, argv)

    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert missing in payload["error"]


def test_only_the_two_commands_live_under_scripts():
    # A skill runtime publishes every .py under scripts/ as a command.
    commands = sorted(
        path.relative_to(BUNDLE).as_posix() for path in (BUNDLE / "scripts").rglob("*.py")
    )

    assert commands == ["scripts/attendance.py", "scripts/take100.py"]
