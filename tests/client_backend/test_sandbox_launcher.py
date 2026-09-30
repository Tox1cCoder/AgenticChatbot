"""The command line the launcher hands to the sandbox account.

It runs through ``cmd.exe`` so a few variables can be added on top of the
account's own profile environment. These tests run the line through the real
``cmd.exe`` as the current user; starting it as the sandbox account is
verified on a machine where the account exists.
"""

from __future__ import annotations

import base64
import os
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


def _run_main(monkeypatch, encoded: str | None) -> tuple[int, list[str]]:
    from client_backend.services.sandbox import account, windows

    started: list[str] = []

    def run_as_user(username, password, command_line, cwd):
        started.append(command_line)
        return 0

    credentials = type("Credentials", (), {"username": "kani-sandbox", "password": "pw"})()
    monkeypatch.setattr(account, "load_credentials", lambda: credentials)
    monkeypatch.setattr(windows, "run_as_user", run_as_user)
    if encoded is None:
        monkeypatch.delenv(launcher.ENV_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(launcher.ENV_VARIABLE, encoded)
    code = launcher.main(["--cwd", r"C:\work", "--", "node", "index.js"])
    return code, started


def test_the_programs_variables_come_from_the_environment_not_arguments(monkeypatch):
    encoded = launcher.encode_environment({"GIT_TERMINAL_PROMPT": "0", "TOKEN": "${NOT_EXPANDED}"})

    code, started = _run_main(monkeypatch, encoded)

    assert code == 0
    assert started == [
        'cmd.exe /d /s /c "set "GIT_TERMINAL_PROMPT=0"&&set "TOKEN=${NOT_EXPANDED}"&&node index.js"'
    ]
    # Consumed: the sandboxed program gets its variables through cmd.exe only.
    assert launcher.ENV_VARIABLE not in os.environ


def test_without_variables_the_program_runs_with_none_added(monkeypatch):
    code, started = _run_main(monkeypatch, None)

    assert code == 0
    assert started == ['cmd.exe /d /s /c "node index.js"']


@pytest.mark.parametrize(
    "payload",
    [b"not json", b'["A=1"]', b'{"A": 1}'],
    ids=["not-json", "not-a-map", "non-string-value"],
)
def test_a_malformed_variable_payload_is_refused(monkeypatch, payload):
    code, started = _run_main(monkeypatch, base64.b64encode(payload).decode("ascii"))

    assert code == 126
    assert started == []


def test_a_payload_that_is_not_base64_is_refused(monkeypatch):
    code, started = _run_main(monkeypatch, "not base64!")

    assert code == 126
    assert started == []


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
