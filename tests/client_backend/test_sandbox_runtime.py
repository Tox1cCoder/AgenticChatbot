"""What the sandbox account may reach: its workspaces, and a read-only runtime.

Access grants run the real ``icacls`` against a temporary folder, using the
built-in Users group's SID as a stand-in principal (the sandbox account exists
only after setup). The npm install is observed at the process boundary: a real
one needs the network.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from client_backend.services.sandbox import runtime

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="icacls is Windows-only")

USERS_GROUP = "*S-1-5-32-545"


def _acl(folder: Path) -> list[str]:
    """The folder's access entries, one per line, without the path icacls prints first."""
    completed = subprocess.run(["icacls", str(folder)], capture_output=True, text=True)
    lines = [line.replace(str(folder), "").strip() for line in completed.stdout.splitlines()]
    return [line for line in lines if ":(" in line]


def test_workspace_grant_lets_the_account_change_everything_below(tmp_path):
    before = _acl(tmp_path)

    runtime.grant_workspace_access(tmp_path, principal=USERS_GROUP)

    added = [line for line in _acl(tmp_path) if line not in before]
    assert any("(OI)(CI)(M)" in line for line in added)


@pytest.fixture
def program_data(tmp_path, monkeypatch):
    folder = tmp_path / "ProgramData"
    folder.mkdir()
    monkeypatch.setenv("PROGRAMDATA", str(folder))
    return folder


def test_runtime_folder_is_outside_the_profile_and_read_only_for_the_account(
    profile, program_data
):
    """Node reads every folder's attributes on the way to its script, and folders
    in the user's profile are closed to the account, so the runtime lives in
    ProgramData -- writable by the user, only readable by the account."""
    folder = runtime.ensure_runtime_root(principal=USERS_GROUP)

    assert folder.parent == program_data
    entries = _acl(folder)
    assert not any("(I)" in line for line in entries), "nothing may be inherited"
    account = [line for line in entries if "(RX)" in line and "(OI)(CI)" in line]
    assert account, entries
    assert not any(line.endswith(("(M)", "(W)")) for line in entries)


def test_runtime_folder_is_created_once_and_reused(profile, program_data):
    first = runtime.ensure_runtime_root(principal=USERS_GROUP)

    assert runtime.ensure_runtime_root(principal=USERS_GROUP) == first


def test_runtime_folder_name_cannot_be_guessed_in_advance(profile, program_data, tmp_path):
    """Any account can create folders in ProgramData; one made in advance by the
    sandbox account would be the account's to rewrite. A fresh random name,
    created exclusively, cannot have been squatted."""
    other_profile = tmp_path / "other"
    first = runtime.ensure_runtime_root(principal=USERS_GROUP)
    from client_backend.core.config import client_settings

    client_settings.profile_root = str(other_profile)
    second = runtime.ensure_runtime_root(principal=USERS_GROUP)

    assert first != second


def test_folders_the_user_runs_unseen_are_protected(tmp_path):
    """Git hooks and environments run as the user, and git review never shows
    them: a changed hook or a planted .pth file is a way out of the sandbox."""
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pyvenv.cfg").write_text("home = C:/Python", encoding="utf-8")
    (tmp_path / "envs" / "conda-meta").mkdir(parents=True)
    (tmp_path / "src").mkdir()

    protected = runtime.protected_paths(tmp_path)

    assert set(protected) == {tmp_path / ".git", tmp_path / ".venv", tmp_path / "envs"}


def test_the_sidecars_own_interpreter_is_protected_wherever_it_sits(tmp_path, monkeypatch):
    interpreter = tmp_path / "tools" / "python-runtime"
    interpreter.mkdir(parents=True)
    monkeypatch.setattr(runtime.sys, "prefix", str(interpreter))

    assert interpreter in runtime.protected_paths(tmp_path)


def test_a_workspace_holding_a_protected_folder_is_refused_not_granted(tmp_path, monkeypatch):
    """Granting is inherited by everything below, so a workspace containing .git
    or an environment cannot be opened without also opening those. It fails
    closed: refuse, and never run icacls on it."""
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    grants = []
    monkeypatch.setattr(runtime, "grant_workspace_access", lambda *a, **k: grants.append(a))

    with pytest.raises(runtime.SandboxRuntimeError, match=r"\.git"):
        runtime.ensure_workspace_access(tmp_path, principal=USERS_GROUP)

    assert grants == []


def test_granting_a_missing_folder_says_which_folder(tmp_path):
    missing = tmp_path / "gone"

    with pytest.raises(runtime.SandboxRuntimeError, match="gone"):
        runtime.grant_workspace_access(missing, principal=USERS_GROUP)


@pytest.fixture
def profile(tmp_path, monkeypatch):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path / "KaniDesktop"))
    return tmp_path


def test_install_skips_package_scripts_and_pins_the_version(profile, monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    monkeypatch.setattr(runtime, "ensure_runtime_root", lambda principal=None: profile / "rt")

    runtime.install_desktop_commander()

    npm = calls[0]
    # Scripts would send the vendor's install telemetry and download Chromium.
    assert "--ignore-scripts" in npm
    assert f"@wonderwhy-er/desktop-commander@{runtime.PINNED_VERSION}" in npm


def test_node_inside_the_users_profile_is_copied_where_the_account_can_run_it(
    profile, monkeypatch
):
    home = profile / "home"
    private_node = home / "fnm" / "node.exe"
    private_node.parent.mkdir(parents=True)
    private_node.write_bytes(b"MZ-fake-node")
    monkeypatch.setattr(runtime.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(runtime.shutil, "which", lambda name: str(private_node))
    monkeypatch.setattr(runtime, "ensure_runtime_root", lambda principal=None: profile / "rt")

    node = runtime.node_executable()

    assert node != private_node
    assert not node.is_relative_to(home)
    assert node.read_bytes() == b"MZ-fake-node"


def _sandbox_account_exists() -> bool:
    completed = subprocess.run(["net", "user", "KaniSandbox"], capture_output=True, text=True)
    return completed.returncode == 0


needs_account = pytest.mark.skipif(
    not _sandbox_account_exists(), reason="the KaniSandbox account is not set up here"
)


@needs_account
def test_missing_workspace_grant_is_applied(tmp_path):
    runtime.ensure_workspace_access(tmp_path)

    assert any("kanisandbox:(oi)(ci)(m)" in line.lower() for line in _acl(tmp_path))


@needs_account
def test_existing_workspace_grant_is_not_applied_again(tmp_path, monkeypatch):
    runtime.grant_workspace_access(tmp_path)
    commands = []
    real_run = subprocess.run

    def spy(argv, **kwargs):
        commands.append(argv)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(runtime.subprocess, "run", spy)

    runtime.ensure_workspace_access(tmp_path)

    assert not [argv for argv in commands if "/grant" in argv]


def test_node_the_account_can_already_run_is_used_in_place(profile, monkeypatch):
    shared_node = profile / "Program Files" / "nodejs" / "node.exe"
    shared_node.parent.mkdir(parents=True)
    shared_node.write_bytes(b"MZ")
    monkeypatch.setattr(runtime.Path, "home", classmethod(lambda cls: profile / "home"))
    monkeypatch.setattr(runtime.shutil, "which", lambda name: str(shared_node))

    assert runtime.node_executable() == shared_node
