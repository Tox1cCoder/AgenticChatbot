"""Restricted-tool / restricted-skill tests for custom agents (Task 7)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.ai import skill_resolver
from app.ai.custom_agent_runtime import (
    build_custom_agent_runtime_spec,
    filter_tools_for_custom_agent,
    is_custom_runtime_id,
    parse_custom_agent_id,
)
from app.ai.skill_resolver import ResolvedSkill, filter_skills_by_refs
from app.ai.skills_tool import create_activate_skill_tool

CLIENT_TOOL_REF = {
    "type": "client",
    "device_id": "desktop-1",
    "session_id": "session-1",
    "catalog_version": "v1",
    "tool_instance_id": "csv-profile-instance",
    "server_name": "csv",
    "qualified_tool_id": "client__csv__profile",
    "tool_name": "profile",
}

SERVER_MCP_TOOL_REF = {
    "type": "server_mcp",
    "server_name": "calculator",
    "tool_name": "calculate",
    "qualified_tool_id": "calculator::calculate",
}


class _FakeTool:
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata


def _spec(tool_refs=None, skill_refs=None):
    cid = uuid4()
    return build_custom_agent_runtime_spec(
        {
            "id": str(cid),
            "runtime_agent_id": f"custom_agent:{cid}",
            "model_agent_key": "custom",
            "name": "Analyst",
            "prompt": "p",
            "model_request": {"provider_type": "openai", "model": "gpt-4.1-mini"},
            "tool_refs": tool_refs or [],
            "skill_refs": skill_refs or [],
        }
    )


def test_runtime_id_helpers():
    cid = uuid4()
    rid = f"custom_agent:{cid}"
    assert is_custom_runtime_id(rid)
    assert not is_custom_runtime_id("chat_agent")
    assert parse_custom_agent_id(rid) == cid
    assert parse_custom_agent_id("chat_agent") is None


def test_spec_does_not_inject_example_time_tool():
    spec = _spec(tool_refs=[])
    assert spec.allowed_server_tool_refs == []
    assert spec.allow_all_server_tools is True


def test_custom_agent_receives_all_backend_server_tools_and_selected_client_tools():
    spec = _spec(tool_refs=[CLIENT_TOOL_REF])
    calculator_tool = _FakeTool(
        "calculate",
        {
            "qualified_tool_id": "calculator::calculate",
            "server_name": "calculator",
            "tool_origin": "server_mcp",
        },
    )
    spreadsheet_tool = _FakeTool(
        "read_sheet",
        {
            "qualified_tool_id": "spreadsheet::read_sheet",
            "server_name": "spreadsheet",
            "tool_origin": "server_mcp",
        },
    )
    selected_client = _FakeTool(
        "client__csv__profile",
        {
            "tool_origin": "client_mcp",
            "device_id": "desktop-1",
            "session_id": "session-1",
            "catalog_version": "v1",
            "tool_instance_id": "csv-profile-instance",
            "qualified_tool_id": "client__csv__profile",
        },
    )
    unselected_client = _FakeTool(
        "client__fs__delete",
        {
            "tool_origin": "client_mcp",
            "device_id": "desktop-1",
            "session_id": "session-1",
            "catalog_version": "v1",
            "tool_instance_id": "fs-delete-instance",
            "qualified_tool_id": "client__fs__delete",
        },
    )
    another_server = _FakeTool(
        "search_web", {"qualified_tool_id": "search::web", "tool_origin": "server"}
    )

    allowed, warnings = filter_tools_for_custom_agent(
        [calculator_tool, spreadsheet_tool, selected_client, unselected_client, another_server],
        spec,
        request_device_id="desktop-1",
    )
    names = {t.name for t in allowed}
    assert names == {"calculate", "read_sheet", "search_web", "client__csv__profile"}
    assert "client__fs__delete" not in names
    assert warnings == []


def test_base_runtime_can_still_restrict_server_tools_by_refs():
    spec = _spec(tool_refs=[SERVER_MCP_TOOL_REF])
    spec.allow_all_server_tools = False
    selected_server = _FakeTool(
        "calculate",
        {
            "tool_origin": "server_mcp",
            "server_name": "calculator",
            "qualified_tool_id": "calculator::calculate",
        },
    )
    same_name_other_server = _FakeTool(
        "calculate",
        {
            "tool_origin": "server_mcp",
            "server_name": "other",
            "qualified_tool_id": "other::calculate",
        },
    )

    allowed, warnings = filter_tools_for_custom_agent(
        [selected_server, same_name_other_server], spec
    )

    assert allowed == [selected_server]
    assert warnings == []


def test_filter_warns_when_selected_client_tool_unavailable():
    spec = _spec(tool_refs=[CLIENT_TOOL_REF])
    # Catalog has no matching tool.
    allowed, warnings = filter_tools_for_custom_agent([], spec, request_device_id="desktop-1")
    assert allowed == []
    assert any("client__csv__profile" in w for w in warnings)


def test_filter_blocks_client_tool_from_other_device():
    spec = _spec(tool_refs=[CLIENT_TOOL_REF])
    other_device_tool = _FakeTool(
        "client__csv__profile",
        {
            "tool_origin": "client_mcp",
            "device_id": "desktop-1",
            "session_id": "session-1",
            "catalog_version": "v1",
            "tool_instance_id": "csv-profile-instance",
            "qualified_tool_id": "client__csv__profile",
        },
    )
    # Request comes from a different device than the stored ref.
    allowed, warnings = filter_tools_for_custom_agent(
        [other_device_tool], spec, request_device_id="desktop-2"
    )
    assert allowed == []
    assert warnings  # the selected tool was unavailable for this device


def test_restricted_skill_filtering_is_exact_and_no_op_without_refs():
    skills = [
        ResolvedSkill(
            name="data-analysis", description="", source="server", lookup_name="data-analysis"
        ),
        ResolvedSkill(name="browser", description="", source="server", lookup_name="browser"),
    ]
    refs = [{"source": "server", "lookup_name": "data-analysis", "name": "data-analysis"}]
    assert [s.name for s in filter_skills_by_refs(skills, refs)] == ["data-analysis"]
    # None means base-agent behavior: no filtering.
    assert filter_skills_by_refs(skills, None) == skills


@pytest.mark.asyncio
async def test_custom_agent_cannot_activate_unselected_skill(monkeypatch):
    monkeypatch.setattr(
        skill_resolver,
        "_list_server_skills",
        lambda: [
            ResolvedSkill(
                name="data-analysis", description="d", source="server", lookup_name="data-analysis"
            ),
            ResolvedSkill(name="browser", description="b", source="server", lookup_name="browser"),
        ],
    )
    monkeypatch.setattr(skill_resolver, "_list_client_skills", lambda **kwargs: [])

    allowed_refs = [{"source": "server", "lookup_name": "data-analysis", "name": "data-analysis"}]
    tool = create_activate_skill_tool(user_id=None, device_id=None, allowed_skill_refs=allowed_refs)

    # "browser" is real but not in the agent's allowlist -> rejected as unavailable.
    result = await tool.ainvoke({"skill_name": "browser"})
    assert "not found" in result.lower()
    assert "data-analysis" in result  # only the allowed skill is listed as available


def test_selected_client_tool_cannot_be_substituted_by_name_from_another_session():
    """A tool with the same qualified id but a different session/instance is rejected."""
    spec = _spec(tool_refs=[CLIENT_TOOL_REF])
    # Same qualified_tool_id + device, but a DIFFERENT session_id and
    # tool_instance_id (i.e. a tool from another client session masquerading
    # under the same name).
    substitute = _FakeTool(
        "client__csv__profile",
        {
            "tool_origin": "client_mcp",
            "device_id": "desktop-1",
            "session_id": "session-OTHER",
            "catalog_version": "v1",
            "tool_instance_id": "different-instance",
            "qualified_tool_id": "client__csv__profile",
        },
    )
    allowed, warnings = filter_tools_for_custom_agent(
        [substitute], spec, request_device_id="desktop-1"
    )
    assert allowed == []  # exact-identity mismatch -> not bound
    assert warnings  # the genuinely-selected tool is reported unavailable


def test_two_custom_agents_do_not_share_deferred_tool_state():
    from app.ai.agents.custom_agent import CustomAgent

    spec_a = _spec()
    spec_b = _spec()
    agent_a = CustomAgent(spec_a)
    agent_b = CustomAgent(spec_b)

    # Deferred/loaded tool state is keyed by the runtime id, not the shared
    # "custom" model key, so two custom agents never collide.
    assert agent_a.tool_state_key == spec_a.runtime_agent_id
    assert agent_b.tool_state_key == spec_b.runtime_agent_id
    assert agent_a.tool_state_key != agent_b.tool_state_key
    assert agent_a.agent_config_key == agent_b.agent_config_key == "custom"


def test_tool_search_allowlist_scopes_to_selected():
    spec = _spec(
        tool_refs=[
            CLIENT_TOOL_REF,
        ]
    )
    assert spec.server_tool_search_allowlist() is None
    client_allowlist = set(spec.client_tool_search_allowlist())
    assert "csv-profile-instance" in client_allowlist
    assert "client__csv__profile" not in client_allowlist
    assert "profile" not in client_allowlist
