"""FR-3: enabled server MCP servers ARE the global-default toolset. Keep it minimal."""

import json
from pathlib import Path

GLOBAL_DEFAULT_SERVERS = {"widgets", "tavily", "time", "brave_image_search"}
MACHINE_SPECIFIC_SERVERS = ("desktop-commander", "mcp-server-for-revit", "excel")


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "app" / "ai" / "mcp_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def test_enabled_server_mcp_is_exactly_the_global_default_set():
    servers = _load_config()["servers"]
    enabled = {
        name for name, spec in servers.items() if spec.get("enabledByDefault")
    }
    assert enabled == GLOBAL_DEFAULT_SERVERS


def test_machine_specific_servers_are_not_in_the_server_config():
    servers = _load_config()["servers"]
    for name in MACHINE_SPECIFIC_SERVERS:
        assert name not in servers


def test_brave_image_search_is_a_reserved_managed_server():
    from app.ai.mcp_integration import MCPManager

    assert "brave_image_search" in MCPManager.DEFAULT_SERVERS


def test_tavily_remains_one_global_server_with_multiple_tools():
    servers = _load_config()["servers"]

    assert servers["tavily"]["enabledByDefault"] is True
    assert "tavily_server.py" in " ".join(servers["tavily"]["args"])
