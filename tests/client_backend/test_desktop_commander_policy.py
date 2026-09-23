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

    assert flags == {"read_file": False, "write_file": True, "unreleased_tool": True}


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
