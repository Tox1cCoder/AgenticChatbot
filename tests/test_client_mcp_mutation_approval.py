"""A client MCP tool the sidecar marks as a mutation is gated for approval.

The sidecar flags Desktop Commander's machine-changing tools in the catalog it
publishes; this is the server half of that contract. A read stays ungated, and
a user's explicit per-server rule still wins over the default.
"""

from app.ai.client_runtime_tools import _build_tool, _parse_tool_specs
from app.ai.hitl_config import calls_requiring_approval


def _entry(name: str, *, mutation: bool) -> dict:
    return {
        "name": name,
        "description": f"Desktop Commander {name}",
        "origin": "mcp",
        "server_name": "desktop-commander",
        "qualified_id": f"desktop-commander::{name}",
        "input_schema": {"type": "object", "properties": {}},
        "mutation": mutation,
    }


def _tools_by_source_name() -> tuple[dict, dict[str, str]]:
    catalog = {
        "tools": [
            _entry("start_process", mutation=True),
            _entry("read_file", mutation=False),
        ]
    }
    tools = [
        _build_tool(
            spec=spec,
            bound_user_id="00000000-0000-0000-0000-000000000001",
            bound_device_id="00000000-0000-0000-0000-000000000002",
            bound_session_id="session-1",
            bound_catalog_version=1,
        )
        for spec in _parse_tool_specs(catalog)
    ]
    tool_map = {tool.name: tool for tool in tools}
    exposed = {tool.metadata["source_tool_name"]: tool.name for tool in tools}
    return tool_map, exposed


def _policy(servers: dict[str, bool] | None = None) -> dict:
    return {
        "master_enabled": True,
        "client_rules": {
            "client_mcp": {"servers": servers or {}, "tools": {}},
            "client_skill": {"servers": {}, "tools": {}},
        },
        "global_tools": [],
    }


def _calls(exposed: dict[str, str]) -> list[dict]:
    return [
        {"name": exposed["start_process"], "args": {"command": "dir"}, "id": "call-start"},
        {"name": exposed["read_file"], "args": {"path": "notes.txt"}, "id": "call-read"},
    ]


def test_desktop_commander_mutation_needs_approval_without_any_rule():
    tool_map, exposed = _tools_by_source_name()

    gated = calls_requiring_approval(_calls(exposed), policy=_policy(), tool_map=tool_map)

    assert gated == {"call-start"}


def test_user_rule_for_the_server_overrides_the_mutation_default():
    tool_map, exposed = _tools_by_source_name()

    gated = calls_requiring_approval(
        _calls(exposed),
        policy=_policy(servers={"desktop-commander": False}),
        tool_map=tool_map,
    )

    assert gated == set()
