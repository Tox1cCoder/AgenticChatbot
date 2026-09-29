"""Provisioning the sandbox account never exposes its password.

A command line is readable by every process on the machine (Task Manager,
``Get-CimInstance Win32_Process``), so the password must reach PowerShell on
stdin. Only ``subprocess.run`` is replaced: creating a Windows account needs
administrator rights and is verified on a real machine.
"""

from __future__ import annotations

import base64
import subprocess

import pytest

from client_backend.services.sandbox import account, setup


@pytest.fixture
def profile(tmp_path, monkeypatch):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path))
    return tmp_path


@pytest.fixture
def powershell(monkeypatch):
    calls: list[dict] = []

    def fake_run(argv, **kwargs):
        calls.append({"argv": argv, "input": kwargs.get("input")})
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(setup.subprocess, "run", fake_run)
    return calls


def test_password_reaches_powershell_only_on_stdin(profile, powershell):
    credentials = setup.provision_account()

    assert len(powershell) == 1
    call = powershell[0]
    assert not [part for part in call["argv"] if credentials.password in part]
    assert credentials.password in call["input"]


def test_provisioned_password_is_the_one_stored(profile, powershell):
    credentials = setup.provision_account()

    assert account.load_credentials() == credentials


def test_setup_from_two_windows_profiles_uses_separate_accounts(tmp_path, monkeypatch, powershell):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path / "alice"))
    alice = setup.provision_account()
    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path / "bob"))
    bob = setup.provision_account()

    assert alice.username != bob.username
    assert len(alice.username) <= 20
    assert len(bob.username) <= 20
    first_script = base64.b64decode(powershell[0]["argv"][-1]).decode("utf-16-le")
    second_script = base64.b64decode(powershell[1]["argv"][-1]).decode("utf-16-le")
    assert alice.username in first_script
    assert bob.username in second_script
    assert "{{" not in first_script + second_script
    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path / "alice"))
    assert account.load_credentials() == alice


def test_failed_provisioning_stores_no_credentials(profile, monkeypatch):
    def failing_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="Access is denied.")

    monkeypatch.setattr(setup.subprocess, "run", failing_run)

    with pytest.raises(setup.SandboxSetupError, match="Access is denied"):
        setup.provision_account()

    assert account.load_credentials() is None


def test_removing_the_account_forgets_its_credentials(profile, powershell):
    setup.provision_account()

    setup.remove_account()

    assert account.load_credentials() is None


def test_folder_entries_are_revoked_before_the_account_is_deleted(profile, monkeypatch):
    """Once the account is deleted its entries point at an orphan SID; they are
    taken out first, while everything about them is still known."""
    order = []

    def fake_run(argv, **kwargs):
        order.append("delete account")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(setup.subprocess, "run", fake_run)
    monkeypatch.setattr(setup, "revoke_path_resolution", lambda: order.append("revoke"))

    setup.remove_account()

    assert order == ["revoke", "delete account"]
