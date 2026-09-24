"""`sandbox setup` / `remove`: the elevated half does the work, the caller reports it.

Windows shows the elevated process no console, so it leaves a result file the
unelevated caller reads back. Elevation itself (the UAC prompt) and the
account change are replaced here; they are verified on a real machine.
"""

from __future__ import annotations

import pytest

from client_backend import cli
from client_backend.services.sandbox import account, commands


@pytest.fixture
def profile(tmp_path, monkeypatch):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path))
    return tmp_path


def test_elevated_setup_provisions_and_records_success(profile, monkeypatch):
    provisioned = []
    monkeypatch.setattr(commands, "is_process_elevated", lambda: True)
    monkeypatch.setattr(
        commands, "provision_account", lambda: provisioned.append(True) or object()
    )

    assert cli.main(["sandbox", "setup", "--elevated"]) == 0
    assert provisioned == [True]
    assert commands.read_result() == {"action": "setup", "ok": True, "message": ""}


def test_elevated_setup_records_what_windows_refused(profile, monkeypatch):
    from client_backend.services.sandbox.setup import SandboxSetupError

    def refuse():
        raise SandboxSetupError("Access is denied.")

    monkeypatch.setattr(commands, "is_process_elevated", lambda: True)
    monkeypatch.setattr(commands, "provision_account", refuse)

    assert cli.main(["sandbox", "setup", "--elevated"]) == 1
    assert commands.read_result()["message"] == "Access is denied."


def test_unelevated_setup_asks_windows_for_elevation_and_reports(profile, monkeypatch, capsys):
    relaunched = []

    def run_elevated(arguments):
        relaunched.append(arguments)
        commands.write_result("setup", ok=True)
        return 0

    monkeypatch.setattr(commands, "is_process_elevated", lambda: False)
    monkeypatch.setattr(commands, "run_elevated", run_elevated)

    assert cli.main(["sandbox", "setup"]) == 0
    assert relaunched == [["sandbox", "setup", "--elevated"]]
    assert "set up" in capsys.readouterr().out.lower()


def test_declined_elevation_changes_nothing(profile, monkeypatch, capsys):
    def decline(_arguments):
        raise commands.ElevationDeclinedError()

    monkeypatch.setattr(commands, "is_process_elevated", lambda: False)
    monkeypatch.setattr(commands, "run_elevated", decline)

    assert cli.main(["sandbox", "setup"]) == 1
    assert account.load_credentials() is None
    assert "declined" in capsys.readouterr().err.lower()


def test_status_before_setup_says_it_is_not_set_up(profile, capsys):
    assert cli.main(["sandbox", "status"]) == 1
    assert "not set up" in capsys.readouterr().out.lower()


def _pretend_account_is_usable(monkeypatch) -> None:
    from client_backend.services.sandbox.account import SandboxCredentials

    creds = SandboxCredentials("KaniSandbox", "x")
    monkeypatch.setattr(commands, "load_credentials", lambda: creds)
    monkeypatch.setattr(commands, "logon_problem", lambda *args: None)


def test_status_shows_the_managed_workspace_when_no_root_is_configured(
    profile, monkeypatch, capsys
):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "workspace_roots", [])
    _pretend_account_is_usable(monkeypatch)

    assert cli.main(["sandbox", "status"]) == 0
    out = capsys.readouterr().out
    assert "Managed workspace" in out
    assert str(profile / "workspace") in out
    assert (profile / "workspace").is_dir()


def test_status_lists_configured_roots_instead(profile, monkeypatch, capsys):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "workspace_roots", [r"C:\work\project"])
    _pretend_account_is_usable(monkeypatch)

    assert cli.main(["sandbox", "status"]) == 0
    out = capsys.readouterr().out
    assert r"C:\work\project" in out
    assert "Managed workspace" not in out
