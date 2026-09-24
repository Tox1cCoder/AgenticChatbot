"""Desktop Commander is launched pinned and quiet, and cannot loosen its own limits.

The server's tools reach the model with full user privileges, so the sidecar
decides which of them the model sees, which count as mutations (and therefore
go through approval), and which build of the server runs at all.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from client_backend.schemas.mcp_config import MCPProfileScope
from client_backend.services.local_mcp_manager import LocalMCPManager
from client_backend.services.mcp_config_store import MCPConfigStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FAKE_DESKTOP_COMMANDER = FIXTURES / "fake-desktop-commander" / "server.py"
STATEFUL_SERVER = FIXTURES / "mcp_stateful_server.py"

EXACT_PINNED_PACKAGE = re.compile(r"^@wonderwhy-er/desktop-commander@\d+\.\d+\.\d+$")


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
        env = spec.pop("env", {})
        store.save_custom_server(name, {"transport": "stdio", **spec}, env=env, headers={})
    return store


def _launch_entry(tmp_path: Path, spec: dict) -> dict:
    store = _store(tmp_path, {"server": spec})
    return LocalMCPManager._build_server_config(store.list_effective_servers())["server"]


@pytest.mark.parametrize(
    "package",
    [
        "@wonderwhy-er/desktop-commander@latest",
        "@wonderwhy-er/desktop-commander",
        "@wonderwhy-er/desktop-commander@^0.2.40",
    ],
)
def test_unpinned_desktop_commander_launches_an_exact_version(tmp_path, package):
    entry = _launch_entry(tmp_path, {"command": "npx", "args": ["-y", package]})

    launched = [arg for arg in entry["args"] if arg.startswith("@wonderwhy-er/")]
    assert len(launched) == 1
    assert EXACT_PINNED_PACKAGE.match(launched[0])


def test_desktop_commander_version_chosen_by_the_user_is_kept(tmp_path):
    entry = _launch_entry(
        tmp_path,
        {"command": "npx", "args": ["-y", "@wonderwhy-er/desktop-commander@0.2.40"]},
    )

    assert "@wonderwhy-er/desktop-commander@0.2.40" in entry["args"]


def test_desktop_commander_launches_without_telemetry_or_onboarding(tmp_path):
    entry = _launch_entry(
        tmp_path,
        {"command": "npx", "args": ["-y", "@wonderwhy-er/desktop-commander@latest"]},
    )

    assert entry["env"]["DESKTOP_COMMANDER_DISABLE_TELEMETRY"] == "1"
    assert entry["args"].count("--no-onboarding") == 1


def test_onboarding_flag_is_not_repeated(tmp_path):
    entry = _launch_entry(
        tmp_path,
        {
            "command": "npx",
            "args": ["-y", "@wonderwhy-er/desktop-commander@latest", "--no-onboarding"],
        },
    )

    assert entry["args"].count("--no-onboarding") == 1


def test_desktop_commander_commands_do_not_wait_for_input_nobody_can_give(tmp_path):
    entry = _launch_entry(
        tmp_path,
        {"command": "npx", "args": ["-y", "@wonderwhy-er/desktop-commander@latest"]},
    )

    # git fails at once instead of waiting for credentials; npx proceeds
    # instead of stopping at "Ok to proceed?".
    assert entry["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert entry["env"]["npm_config_yes"] == "true"


def test_telemetry_setting_chosen_by_the_user_is_kept(tmp_path):
    entry = _launch_entry(
        tmp_path,
        {
            "command": "npx",
            "args": ["-y", "@wonderwhy-er/desktop-commander@latest"],
            "env": {"DESKTOP_COMMANDER_DISABLE_TELEMETRY": "0"},
        },
    )

    assert entry["env"]["DESKTOP_COMMANDER_DISABLE_TELEMETRY"] == "0"


def test_other_servers_launch_exactly_as_configured(tmp_path):
    entry = _launch_entry(
        tmp_path,
        {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem@latest"]},
    )

    assert entry["args"] == ["-y", "@modelcontextprotocol/server-filesystem@latest"]
    assert "env" not in entry


@pytest.fixture
async def desktop_commander(tmp_path):
    store = _store(
        tmp_path,
        {
            "desktop-commander": {
                "command": sys.executable,
                "args": [str(FAKE_DESKTOP_COMMANDER)],
            }
        },
    )
    manager = LocalMCPManager(store=store)
    await manager.initialize()
    try:
        yield manager
    finally:
        await manager.shutdown()


async def test_model_cannot_see_desktop_commander_self_configuration_tools(desktop_commander):
    names = {tool["name"] for tool in desktop_commander.get_tool_catalog()["tools"]}

    assert "set_config_value" not in names
    assert "get_recent_tool_calls" not in names


async def test_hidden_desktop_commander_tool_cannot_be_called(desktop_commander):
    with pytest.raises(ValueError, match="not found"):
        await desktop_commander.call_tool(
            "desktop-commander::set_config_value",
            {"key": "blockedCommands", "value": "[]"},
            timeout=10,
        )


async def test_desktop_commander_tools_are_mutations_unless_known_read_only(desktop_commander):
    flags = {
        tool["name"]: tool["mutation"] for tool in desktop_commander.get_tool_catalog()["tools"]
    }

    assert flags == {
        "read_file": False,
        "read_multiple_files": False,
        "list_directory": False,
        "start_search": False,
        "write_file": True,
        "edit_block": True,
        "move_file": True,
        "unreleased_tool": True,
    }


async def test_other_servers_tools_are_not_flagged_as_mutations(tmp_path):
    store = _store(
        tmp_path,
        {"stateful": {"command": sys.executable, "args": [str(STATEFUL_SERVER)]}},
    )
    manager = LocalMCPManager(store=store)
    await manager.initialize()
    try:
        catalog = manager.get_tool_catalog()["tools"]
    finally:
        await manager.shutdown()

    assert catalog
    assert {tool["mutation"] for tool in catalog} == {False}


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A home folder, app-data folders, a project, and a sidecar profile under tmp.

    The guard only computes paths from these; nothing here is ever read.
    """
    from client_backend.core.config import client_settings

    home = tmp_path / "home"
    roaming, local = home / "AppData" / "Roaming", home / "AppData" / "Local"
    project = tmp_path / "project"
    profile = local / "KaniDesktop"
    for folder in (home / ".ssh", roaming, local, project, profile):
        folder.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("APPDATA", str(roaming))
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setattr(client_settings, "profile_root", str(profile))
    return {"home": home, "project": project, "profile": profile}


def _fill(value, machine):
    if isinstance(value, list):
        return [_fill(item, machine) for item in value]
    if isinstance(value, str):
        return value.format(**{name: str(path) for name, path in machine.items()})
    return value


async def _call(manager, tool, arguments, machine):
    filled = {key: _fill(value, machine) for key, value in arguments.items()}
    return await manager.call_tool(f"desktop-commander::{tool}", filled, timeout=10)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("read_file", {"path": "{home}/.ssh/id_rsa"}),
        ("read_file", {"path": "{project}/.env"}),
        ("read_file", {"path": "{project}/certs/server.pem"}),
        ("read_file", {"path": "{home}/AppData/Local/Google/Chrome/User Data/Default/Cookies"}),
        ("read_file", {"path": "{home}/AppData/Roaming/Microsoft/Credentials/blob"}),
        ("read_file", {"path": "{profile}/credentials.json"}),
        ("read_file", {"path": "{home}/.claude-server-commander/config.json"}),
        ("read_multiple_files", {"paths": ["{project}/README.md", "{home}/.aws/credentials"]}),
        ("write_file", {"path": "{home}/.ssh/authorized_keys", "content": "ssh-ed25519 AAAA"}),
        ("edit_block", {"file_path": "{project}/.env.local", "old_string": "a", "new_string": "b"}),
        ("move_file", {"source": "{project}/notes.txt", "destination": "{home}/.ssh/notes"}),
        ("list_directory", {"path": "{home}/.ssh"}),
        ("start_search", {"path": "{home}/.ssh", "pattern": "KEY"}),
        ("start_search", {"path": "{home}", "pattern": "PRIVATE KEY", "includeHidden": True}),
    ],
)
async def test_desktop_commander_cannot_touch_credential_locations(
    desktop_commander, machine, tool, arguments
):
    with pytest.raises(PermissionError):
        await _call(desktop_commander, tool, arguments, machine)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("read_file", {"path": "{project}/README.md"}),
        ("read_file", {"path": "{project}/.env.example"}),
        ("read_file", {"path": "https://example.com/.env", "isUrl": True}),
        ("list_directory", {"path": "{home}"}),
        ("start_search", {"path": "{home}", "pattern": "TODO"}),
    ],
)
async def test_ordinary_locations_stay_usable(desktop_commander, machine, tool, arguments):
    assert await _call(desktop_commander, tool, arguments, machine)


@pytest.mark.skipif(sys.platform != "win32", reason="directory junctions are Windows-only")
async def test_a_link_into_a_credential_folder_is_followed(desktop_commander, machine):
    import _winapi

    link = machine["project"] / "keys"
    _winapi.CreateJunction(str(machine["home"] / ".ssh"), str(link))

    # "config" is not a key-file name, so only following the link can refuse it.
    with pytest.raises(PermissionError):
        await _call(desktop_commander, "read_file", {"path": str(link / "config")}, machine)
