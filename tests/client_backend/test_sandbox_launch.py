"""How the sidecar launches Desktop Commander once the sandbox account exists.

Once the account is set up, Desktop Commander always runs as it; if the
sandbox cannot be prepared, Desktop Commander stays off rather than falling
back to the user's full rights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from client_backend.schemas.mcp_config import MCPProfileScope
from client_backend.services.local_mcp_manager import LocalMCPManager
from client_backend.services.mcp_config_store import MCPConfigStore
from client_backend.services.sandbox import launch
from client_backend.services.sandbox.launch import SandboxLaunch


def _store(tmp_path: Path, servers: dict[str, dict]) -> MCPConfigStore:
    application_root = tmp_path / "application"
    registry_path = application_root / "app" / "ai" / "mcp_config.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps({"schemaVersion": 2, "servers": {}}), encoding="utf-8")
    store = MCPConfigStore(
        MCPProfileScope(user_id="user-1", device_identifier="device-a"),
        registry_path=registry_path,
        application_root=application_root,
        profile_root=tmp_path / "profiles",
    )
    for name, spec in servers.items():
        store.save_custom_server(name, {"transport": "stdio", **spec}, env={}, headers={})
    return store


_DESKTOP_COMMANDER = {"command": "npx", "args": ["-y", "@wonderwhy-er/desktop-commander@latest"]}
_OTHER = {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"]}

_LAUNCH = SandboxLaunch(
    node=Path(r"C:\Program Files\nodejs\node.exe"),
    desktop_commander=Path(r"C:\ProgramData\KaniDesktop-runtime-x\dc\dist\index.js"),
    cwd=Path(r"C:\work\project"),
)


def _entries(tmp_path, sandbox):
    store = _store(tmp_path, {"desktop-commander": dict(_DESKTOP_COMMANDER), "other": _OTHER})
    return LocalMCPManager._build_server_config(store.list_effective_servers(), sandbox=sandbox)


def test_desktop_commander_runs_as_the_sandbox_account_from_the_pinned_install(tmp_path):
    entry = _entries(tmp_path, _LAUNCH)["desktop-commander"]

    assert entry["command"] == sys.executable
    args = entry["args"]
    assert args[:2] == ["-m", "client_backend.services.sandbox.launcher"]
    assert args[args.index("--cwd") + 1] == str(_LAUNCH.cwd)
    program = args[args.index("--") + 1 :]
    assert program == [str(_LAUNCH.node), str(_LAUNCH.desktop_commander), "--no-onboarding"]


def test_sandboxed_desktop_commander_keeps_its_quiet_non_interactive_environment(tmp_path):
    args = _entries(tmp_path, _LAUNCH)["desktop-commander"]["args"]
    settings = {args[i + 1] for i, arg in enumerate(args) if arg == "--env"}

    assert "DESKTOP_COMMANDER_DISABLE_TELEMETRY=1" in settings
    assert "GIT_TERMINAL_PROMPT=0" in settings
    # Workspace files belong to the user; git refuses "dubious ownership" otherwise.
    assert {"GIT_CONFIG_KEY_0=safe.directory", "GIT_CONFIG_VALUE_0=*"} <= settings


def test_other_servers_are_not_sandboxed(tmp_path):
    entry = _entries(tmp_path, _LAUNCH)["other"]

    assert entry["args"] == _OTHER["args"]


def test_without_the_sandbox_desktop_commander_launches_as_configured(tmp_path):
    entry = _entries(tmp_path, None)["desktop-commander"]

    assert entry["command"] == "npx"


@pytest.fixture
def workspace_mode(monkeypatch):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "sandbox_mode", "workspace")


async def _desktop_commander_state(tmp_path):
    manager = LocalMCPManager(store=_store(tmp_path, {"desktop-commander": _DESKTOP_COMMANDER}))
    await manager.initialize()
    try:
        runtime = manager.servers["desktop-commander"]
        return runtime.is_running(), runtime.error_message or ""
    finally:
        await manager.shutdown()


async def test_desktop_commander_stays_off_when_the_sandbox_cannot_be_prepared(
    tmp_path, monkeypatch, workspace_mode
):
    def broken():
        raise launch.SandboxRuntimeError("npm is not installed")

    monkeypatch.setattr(launch, "sandbox_is_set_up", lambda: True)
    monkeypatch.setattr(launch, "prepare_sandbox_launch", broken)

    running, message = await _desktop_commander_state(tmp_path)

    assert not running
    assert "npm is not installed" in message
    assert "full" in message.lower()


async def test_workspace_mode_without_the_account_keeps_desktop_commander_off(
    tmp_path, workspace_mode
):
    running, message = await _desktop_commander_state(tmp_path)

    assert not running
    assert "sandbox setup" in message


async def test_sandbox_mode_off_ignores_an_existing_account(tmp_path, monkeypatch):
    prepared = []
    monkeypatch.setattr(launch, "prepare_sandbox_launch", lambda: prepared.append(True))

    store = _store(tmp_path, {"desktop-commander": dict(_DESKTOP_COMMANDER)})
    sandbox, problem = await LocalMCPManager._prepare_sandbox(store.list_effective_servers())

    assert (sandbox, problem, prepared) == (None, None, [])


def test_workspace_inside_the_user_profile_is_refused(tmp_path, monkeypatch):
    """PowerShell cannot start in a folder whose parents the account cannot read,
    and falls back to the drive root: commands meant for the project would run
    in C:\\ instead."""
    from client_backend.core.config import client_settings

    home = tmp_path / "home"
    (home / "Documents" / "project").mkdir(parents=True)
    monkeypatch.setattr(launch.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(client_settings, "workspace_roots", [str(home / "Documents" / "project")])
    monkeypatch.setattr(launch, "install_desktop_commander", lambda: tmp_path / "dc.js")
    monkeypatch.setattr(launch, "node_executable", lambda: tmp_path / "node.exe")

    with pytest.raises(launch.SandboxRuntimeError, match="outside"):
        launch.prepare_sandbox_launch()
