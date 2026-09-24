"""Provisioning the sandbox account never exposes its password.

A command line is readable by every process on the machine (Task Manager,
``Get-CimInstance Win32_Process``), so the password must reach PowerShell on
stdin. Only ``subprocess.run`` is replaced: creating a Windows account needs
administrator rights and is verified on a real machine.
"""

from __future__ import annotations

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
