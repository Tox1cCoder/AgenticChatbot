"""The command line the launcher hands to the sandbox account.

It runs through ``cmd.exe`` so a few variables can be added on top of the
account's own profile environment. These tests run the line through the real
``cmd.exe`` as the current user; starting it as the sandbox account is
verified on a machine where the account exists.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from client_backend.services.sandbox import launcher

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe is Windows-only")


def _run(command_line: str) -> str:
    completed = subprocess.run(command_line, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def test_program_runs_with_its_arguments_and_the_added_variables(tmp_path):
    script = tmp_path / "show env.py"
    script.write_text(
        "import os, sys\nprint(os.environ['KANI_PROBE'], os.environ['KANI_OTHER'], sys.argv[1:])\n",
        encoding="utf-8",
    )

    line = launcher.build_command_line(
        [sys.executable, str(script), "two words", "--flag"],
        {"KANI_PROBE": "0", "KANI_OTHER": "safe.directory"},
    )

    assert _run(line) == "0 safe.directory ['two words', '--flag']"


def test_the_programs_exit_code_comes_back(tmp_path):
    line = launcher.build_command_line([sys.executable, "-c", "raise SystemExit(7)"], {})

    completed = subprocess.run(line, capture_output=True, text=True)

    assert completed.returncode == 7


@pytest.mark.parametrize(
    ("argv", "env"),
    [
        (["node", "index.js"], {"TOKEN": "a&calc"}),
        (["node", "index.js"], {"TOKEN": "%PATH%"}),
        (["node", "index.js & calc"], {}),
        (["node", "index.js"], {"BAD NAME": "1"}),
    ],
)
def test_text_cmd_would_interpret_is_refused(argv, env):
    with pytest.raises(ValueError):
        launcher.build_command_line(argv, env)
