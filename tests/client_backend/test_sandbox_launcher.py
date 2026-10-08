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


@pytest.fixture(autouse=True)
def profile(tmp_path, monkeypatch):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path))
    return tmp_path


def _run_main(
    monkeypatch, encoded: str | None, run=None
) -> tuple[int, list[tuple[str, str]]]:
    from client_backend.services.sandbox import account, windows

    started: list[tuple[str, str]] = []

    def run_as_user(username, password, command_line, cwd, environment, *, stderr_handle=None):
        started.append((command_line, environment))
        return run(stderr_handle) if run else 0

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


def _failure():
    from client_backend.services.sandbox.launch_report import read_launch_failure

    return read_launch_failure(since=0)


def _write_to_handle(handle: int, data: bytes) -> None:
    import ctypes
    from ctypes import wintypes

    written = wintypes.DWORD()
    assert ctypes.windll.kernel32.WriteFile(
        wintypes.HANDLE(handle), data, len(data), ctypes.byref(written), None
    )


def test_a_program_that_fails_leaves_its_stderr_tail_in_the_report(monkeypatch, capfd):
    """Its stderr reaches only the sidecar's console, which the sidecar cannot
    read; the report is how "Connection closed" gets its real reason."""

    def crash(stderr_handle):
        _write_to_handle(stderr_handle, b"line one\r\nError: Cannot find module 'x'\r\n")
        return 1

    code, _ = _run_main(monkeypatch, None, run=crash)

    assert code == 1
    failure = _failure()
    assert failure.message == "node exited with code 1 as KaniSbTest"
    assert failure.stderr == ("line one", "Error: Cannot find module 'x'")
    assert failure.summary().endswith("Error: Cannot find module 'x'")
    # Still passed through to the console, as before.
    assert "Cannot find module" in capfd.readouterr().err


def test_a_program_that_exits_cleanly_leaves_no_report(monkeypatch):
    code, _ = _run_main(monkeypatch, None)

    assert code == 0
    assert _failure() is None


def test_missing_credentials_are_reported(monkeypatch):
    from client_backend.services.sandbox import account

    monkeypatch.setattr(account, "load_credentials", lambda: None)

    assert launcher.main(["--cwd", r"C:\work", "--", "node", "index.js"]) == 127
    assert "not set up" in _failure().message


def test_a_disabled_secondary_logon_service_says_how_to_enable_it(monkeypatch):
    import ctypes

    def refuse(_stderr_handle):
        raise ctypes.WinError(1058)

    code, _ = _run_main(monkeypatch, None, run=refuse)

    assert code == 126
    message = _failure().message
    assert "Could not start the program as KaniSbTest" in message
    assert "seclogon" in message


def test_a_refused_sign_in_is_reported_with_its_reason(monkeypatch):
    import ctypes

    from client_backend.services.sandbox import account, windows

    def refuse(*_):
        raise ctypes.WinError(1385)

    credentials = type("Credentials", (), {"username": "KaniSbTest", "password": "pw"})()
    monkeypatch.setattr(account, "load_credentials", lambda: credentials)
    monkeypatch.setattr(windows, "account_environment", refuse)

    assert launcher.main(["--cwd", r"C:\work", "--", "node", "index.js"]) == 126
    message = _failure().message
    assert "Could not sign in as KaniSbTest" in message
    assert "log on locally" in message


def test_profile_root_argument_names_the_profile_before_settings_load(monkeypatch):
    """Settings are built on first use, from the environment, so the argument
    has to be in the environment before anything reads them."""
    from client_backend.services.sandbox import account

    monkeypatch.setenv("CLIENT_PROFILE_ROOT", "inherited")
    monkeypatch.setattr(account, "load_credentials", lambda: None)

    launcher.main(["--cwd", r"C:\work", "--profile-root", r"D:\profiles\kani", "--", "node"])

    assert os.environ["CLIENT_PROFILE_ROOT"] == r"D:\profiles\kani"
