"""How the launcher starts a program as the sandbox account.

The program is started directly, with no shell in between, and its variables
travel in the environment block handed to ``CreateProcessWithLogonW``, never on
a command line. Command lines here run as the current user through the real
``CreateProcess``; starting one as the sandbox account is verified by
``tests/integration/test_sandbox_windows.py`` on a machine where it exists.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys

import pytest

from client_backend.services.sandbox import launcher

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows sandbox only")

PROFILE = r"C:\Users\KaniSbTest"
# What CreateEnvironmentBlock returns while the account's hive is not loaded.
WINDOWS_ENV = {
    "Path": r"C:\WINDOWS\system32;C:\WINDOWS",
    "SystemRoot": r"C:\WINDOWS",
    "USERPROFILE": r"C:\Users\Default",
    "TEMP": r"C:\WINDOWS\TEMP",
    "TMP": r"C:\WINDOWS\TEMP",
    "USERNAME": "KaniSbTest",
}


def test_program_runs_with_its_arguments_unchanged(tmp_path):
    script = tmp_path / "show args.py"
    script.write_text("import sys\nprint(sys.argv[1:])\n", encoding="utf-8")

    line = launcher.build_command_line(
        [sys.executable, str(script), "two words", "a&b", "%PATH%", 'say "hi"']
    )
    completed = subprocess.run(line, capture_output=True, text=True, check=True)

    assert completed.stdout.strip() == "['two words', 'a&b', '%PATH%', 'say \"hi\"']"


def test_the_programs_exit_code_comes_back():
    line = launcher.build_command_line([sys.executable, "-c", "raise SystemExit(7)"])

    assert subprocess.run(line, capture_output=True, text=True).returncode == 7


def test_an_argument_with_a_nul_is_refused():
    with pytest.raises(ValueError):
        launcher.build_command_line(["node", "a\0b"])


def test_profile_variables_come_from_the_accounts_own_folder():
    assert launcher.profile_variables(PROFILE) == {
        "USERPROFILE": PROFILE,
        "HOMEDRIVE": "C:",
        "HOMEPATH": r"\Users\KaniSbTest",
        "APPDATA": PROFILE + r"\AppData\Roaming",
        "LOCALAPPDATA": PROFILE + r"\AppData\Local",
        "TEMP": PROFILE + r"\AppData\Local\Temp",
        "TMP": PROFILE + r"\AppData\Local\Temp",
    }


def test_added_variables_replace_existing_ones_whatever_their_case():
    merged = launcher.overlay_environment({"Path": "old", "KEEP": "1"}, {"PATH": "new"})

    assert merged == {"PATH": "new", "KEEP": "1"}


@pytest.mark.parametrize(
    "added",
    [{"BAD NAME": "1"}, {"A=B": "1"}, {"": "1"}, {"TOKEN": "a\0b"}],
    ids=["space", "equals", "empty", "nul-value"],
)
def test_a_variable_windows_could_not_carry_is_refused(added):
    with pytest.raises(ValueError):
        launcher.overlay_environment({}, added)


def test_the_environment_block_is_sorted_and_double_nul_terminated():
    block = launcher.environment_block({"b": "2", "A": "1", "C": "x=y"})

    assert block == "A=1\0b=2\0C=x=y\0\0"


def _run_main(monkeypatch, encoded: str | None) -> tuple[int, list[tuple[str, str]]]:
    from client_backend.services.sandbox import account, windows

    started: list[tuple[str, str]] = []

    def run_as_user(username, password, command_line, cwd, environment):
        started.append((command_line, environment))
        return 0

    credentials = type("Credentials", (), {"username": "KaniSbTest", "password": "pw"})()
    monkeypatch.setattr(account, "load_credentials", lambda: credentials)
    monkeypatch.setattr(windows, "account_environment", lambda *_: (dict(WINDOWS_ENV), PROFILE))
    monkeypatch.setattr(windows, "run_as_user", run_as_user)
    if encoded is None:
        monkeypatch.delenv(launcher.ENV_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(launcher.ENV_VARIABLE, encoded)
    code = launcher.main(["--cwd", r"C:\work", "--", "node", "index.js"])
    return code, started


def _variables(block: str) -> dict[str, str]:
    assert block.endswith("\0\0")
    return dict(entry.split("=", 1) for entry in block[:-2].split("\0"))


def test_the_programs_variables_travel_in_its_environment_block(monkeypatch):
    added = {"GIT_TERMINAL_PROMPT": "0", "TOKEN": "${NOT_EXPANDED}&%PATH%\"quoted\""}

    code, started = _run_main(monkeypatch, launcher.encode_environment(added))

    assert code == 0
    [(command_line, block)] = started
    assert command_line == "node index.js"
    variables = _variables(block)
    assert variables["GIT_TERMINAL_PROMPT"] == "0"
    assert variables["TOKEN"] == added["TOKEN"]
    # The account's own environment is kept, with its real profile folders.
    assert variables["SystemRoot"] == r"C:\WINDOWS"
    assert variables["USERPROFILE"] == PROFILE
    assert variables["TEMP"] == PROFILE + r"\AppData\Local\Temp"
    # Consumed: the payload is not handed on, and never reaches the program.
    assert launcher.ENV_VARIABLE not in os.environ
    assert launcher.ENV_VARIABLE not in variables


def test_without_variables_the_program_gets_the_accounts_environment(monkeypatch):
    code, started = _run_main(monkeypatch, None)

    assert code == 0
    [(command_line, block)] = started
    assert command_line == "node index.js"
    assert _variables(block)["USERNAME"] == "KaniSbTest"


@pytest.mark.parametrize(
    "payload",
    [b"not json", b'["A=1"]', b'{"A": 1}', b'{"BAD NAME": "1"}'],
    ids=["not-json", "not-a-map", "non-string-value", "bad-name"],
)
def test_a_malformed_variable_payload_is_refused(monkeypatch, payload):
    code, started = _run_main(monkeypatch, base64.b64encode(payload).decode("ascii"))

    assert code == 126
    assert started == []


def test_a_payload_that_is_not_base64_is_refused(monkeypatch):
    code, started = _run_main(monkeypatch, "not base64!")

    assert code == 126
    assert started == []
