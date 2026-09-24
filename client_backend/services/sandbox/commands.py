"""`python -m client_backend sandbox setup | remove | status`.

``setup`` and ``remove`` need administrator rights. Run from an ordinary
prompt, they relaunch themselves through the Windows administrator prompt and
wait; the elevated copy has no visible console, so it leaves a result file
(which never contains the password) for this one to report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from client_backend.core.config import client_settings
from client_backend.core.privileges import is_process_elevated
from client_backend.services.sandbox.account import load_credentials
from client_backend.services.sandbox.setup import (
    SandboxSetupError,
    provision_account,
    remove_account,
)
from client_backend.services.sandbox.windows import (
    ElevationDeclinedError,
    logon_problem,
    run_elevated,
)

__all__ = ["ElevationDeclinedError", "read_result", "run", "write_result"]

_DONE = {"setup": "The sandbox account is set up.", "remove": "The sandbox account is removed."}


def run(action: str, *, elevated: bool = False) -> int:
    if action == "status":
        return _status()
    if elevated or is_process_elevated():
        return _perform(action)
    return _perform_elevated(action)


def _perform(action: str) -> int:
    step = provision_account if action == "setup" else remove_account
    try:
        step()
    except SandboxSetupError as exc:
        write_result(action, ok=False, message=str(exc))
        print(f"Sandbox {action} failed: {exc}", file=sys.stderr)
        return 1
    write_result(action, ok=True)
    print(_DONE[action])
    return 0


def _perform_elevated(action: str) -> int:
    print("Windows will ask for administrator permission to change local accounts.")
    # A result left by an earlier run must not read as this run's success.
    _result_path().unlink(missing_ok=True)
    try:
        exit_code = run_elevated(["sandbox", action, "--elevated"])
    except ElevationDeclinedError:
        print("Administrator permission was declined; nothing was changed.", file=sys.stderr)
        return 1
    result = read_result()
    if exit_code == 0 and result.get("ok"):
        print(_DONE[action])
        return 0
    detail = result.get("message") or f"the elevated step exited with {exit_code}"
    print(f"Sandbox {action} failed: {detail}", file=sys.stderr)
    return 1


def _status() -> int:
    credentials = load_credentials()
    if credentials is None:
        print("The sandbox account is not set up. Run: python -m client_backend sandbox setup")
        return 1
    problem = logon_problem(credentials.username, credentials.password)
    if problem is not None:
        print(f"The sandbox account {credentials.username} cannot be used: {problem}")
        return 1
    print(f"The sandbox account {credentials.username} is set up and usable.")
    return 0


def _result_path() -> Path:
    return Path(client_settings.profile_root) / "sandbox" / "last-result.json"


def write_result(action: str, *, ok: bool, message: str = "") -> None:
    path = _result_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"action": action, "ok": ok, "message": message}), encoding="utf-8"
    )


def read_result() -> dict:
    try:
        return json.loads(_result_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
