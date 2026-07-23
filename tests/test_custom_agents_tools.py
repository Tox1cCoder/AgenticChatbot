"""Restricted-tool / restricted-skill tests for custom agents (Task 7)."""

from __future__ import annotations

from types import SimpleNamespace
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
    # Full backend catalog is discoverable even with no pre-selected server refs.
    assert spec.allow_all_server_tools is True


def test_custom_agent_server_tool_refs_are_recorded_but_not_restrictive():
    spec = _spec(tool_refs=[SERVER_MCP_TOOL_REF])

    # Selected server refs are still recorded (UI hint / future pinning) ...
    assert spec.allowed_server_tool_refs == [SERVER_MCP_TOOL_REF]
    # ... but they do NOT restrict discovery: the full backend catalog is searched.
    assert spec.allow_all_server_tools is True
    assert spec.server_tool_search_allowlist() is None


def test_custom_agent_receives_all_server_tools_but_only_selected_client_tools():
    spec = _spec(tool_refs=[SERVER_MCP_TOOL_REF, CLIENT_TOOL_REF])
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
    # Full server catalog: every backend MCP tool passes (selected or not).
    assert {"calculate", "read_sheet", "search_web"} <= names
    # Client tools stay strictly scoped: only the exact selected instance binds.
    assert "client__csv__profile" in names
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
            name="data-analysis", description="", source="client", lookup_name="data-analysis"
        ),
        ResolvedSkill(name="browser", description="", source="client", lookup_name="browser"),
    ]
    refs = [{"source": "client", "lookup_name": "data-analysis", "name": "data-analysis"}]
    assert [s.name for s in filter_skills_by_refs(skills, refs)] == ["data-analysis"]
    # None means base-agent behavior: no filtering.
    assert filter_skills_by_refs(skills, None) == skills


@pytest.mark.asyncio
async def test_custom_agent_cannot_activate_unselected_skill(monkeypatch):
    device_id = str(uuid4())
    monkeypatch.setattr(
        skill_resolver.ClientDeviceService,
        "lookup_active_session",
        lambda _device_uuid: SimpleNamespace(
            user_id="user-1",
            session_id="session-1",
            device_id=device_id,
            skill_catalog={
                "skills": [
                    {"name": "data-analysis", "description": "d", "enabled": True},
                    {"name": "browser", "description": "b", "enabled": True},
                ]
            },
        ),
    )

    allowed_refs = [{"source": "client", "lookup_name": "data-analysis", "name": "data-analysis"}]
    tool = create_activate_skill_tool(
        user_id="user-1", device_id=device_id, allowed_skill_refs=allowed_refs
    )

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


def test_custom_agent_initial_binding_defers_selected_server_tools():
    from app.ai.agents.custom_agent import CustomAgent
    from app.ai.deferred_tool_state import reset_deferred_tool_state

    reset_deferred_tool_state()
    spec = _spec(tool_refs=[SERVER_MCP_TOOL_REF])
    agent = CustomAgent(spec)
    agent.tools = [
        _FakeTool(
            "calculate",
            {
                "tool_origin": "server_mcp",
                "server_name": "calculator",
                "qualified_tool_id": "calculator::calculate",
            },
        ),
        _FakeTool(
            "search_web",
            {
                "tool_origin": "server_mcp",
                "server_name": "search",
                "qualified_tool_id": "search::search_web",
            },
        ),
    ]

    try:
        tools = agent._get_tools_for_binding(conversation_id="conv-1")
        names = {tool.name for tool in tools}

        assert "tool_search" in names
        assert "calculate" not in names
        assert "search_web" not in names
    finally:
        reset_deferred_tool_state()


def test_custom_agent_binds_only_loaded_tools_not_unloaded_catalog():
    """Deferred discovery still gates binding on LOADING, not selection: a tool
    the agent has autoloaded binds; a backend catalog tool it has not loaded does
    not (otherwise the whole catalog would be bound at once, defeating deferral).
    """
    from app.ai.agents.custom_agent import CustomAgent
    from app.ai.deferred_tool_state import get_deferred_tool_state, reset_deferred_tool_state
    from app.ai.mcp_tool_catalog import ToolReference

    reset_deferred_tool_state()
    spec = _spec(tool_refs=[SERVER_MCP_TOOL_REF])
    agent = CustomAgent(spec)
    loaded_server = _FakeTool(
        "calculate",
        {
            "tool_origin": "server_mcp",
            "server_name": "calculator",
            "qualified_tool_id": "calculator::calculate",
        },
    )
    unloaded_server = _FakeTool(
        "search_web",
        {
            "tool_origin": "server_mcp",
            "server_name": "search",
            "qualified_tool_id": "search::search_web",
        },
    )

    class FakeManager:
        _tool_index = {
            "calculate": [loaded_server],
            "search_web": [unloaded_server],
        }
        _server_tools = {
            "calculator": [loaded_server],
            "search": [unloaded_server],
        }

        def get_server_for_tool(self, tool):
            return "calculator" if tool is loaded_server else "search"

    agent.tools = [loaded_server, unloaded_server]
    agent.mcp_manager = FakeManager()

    try:
        # Only "calculate" is loaded. "search_web" stays in the catalog, unloaded.
        get_deferred_tool_state().autoload(
            "conv-1",
            spec.runtime_agent_id,
            [ToolReference("calculate", "calculator")],
        )

        tools = agent._get_tools_for_binding(conversation_id="conv-1")
        names = {tool.name for tool in tools}

        assert "tool_search" in names
        assert "calculate" in names
        assert "search_web" not in names
    finally:
        reset_deferred_tool_state()


def test_custom_agent_spec_enables_full_server_catalog_discovery():
    """Custom agents discover the full backend MCP catalog (not just selected
    refs). Server discovery is unfiltered; selected refs become an optional hint,
    not a restriction. Client tools remain strictly scoped (see isolation tests).
    """
    spec = _spec(tool_refs=[SERVER_MCP_TOOL_REF])

    assert spec.allow_all_server_tools is True
    assert spec.server_tool_search_allowlist() is None


def test_custom_agent_binds_loaded_server_tool_under_full_catalog_discovery():
    """Any backend MCP tool that tool_search autoloads is bindable, even when it
    is NOT in the agent's selected tool_refs and even when the raw MCP tool object
    carries no server_name/qualified_tool_id metadata (real MCP tools track server
    identity in the manager's id-map, not on ``tool.metadata``)."""
    from app.ai.agents.custom_agent import CustomAgent
    from app.ai.deferred_tool_state import get_deferred_tool_state, reset_deferred_tool_state
    from app.ai.mcp_tool_catalog import ToolReference

    reset_deferred_tool_state()
    spec = _spec(tool_refs=[])  # nothing pre-selected
    agent = CustomAgent(spec)
    # Real-MCP-like tool: no server_name / qualified_tool_id on the tool object.
    discovered = _FakeTool("tavily_search", {})

    class FakeManager:
        _tool_index = {"tavily_search": [discovered]}
        _server_tools = {"tavily": [discovered]}

        def get_server_for_tool(self, tool):
            return "tavily"

    agent.tools = [discovered]
    agent.mcp_manager = FakeManager()

    try:
        get_deferred_tool_state().autoload(
            "conv-1",
            spec.runtime_agent_id,
            [ToolReference("tavily_search", "tavily")],
        )
        tools = agent._get_tools_for_binding(conversation_id="conv-1")
        names = {tool.name for tool in tools}
        assert "tavily_search" in names
    finally:
        reset_deferred_tool_state()


def test_full_server_catalog_does_not_loosen_client_tool_isolation():
    """Full server-catalog discovery must NOT let a client tool from a different
    sidecar session be substituted for the selected one. Client tools stay
    exact-matched on device/session/catalog_version/tool_instance_id."""
    spec = _spec(tool_refs=[CLIENT_TOOL_REF])  # selected on desktop-1 / session-1
    foreign_session_client = _FakeTool(
        "client__csv__profile",
        {
            "tool_origin": "client_mcp",
            "device_id": "desktop-1",
            "session_id": "session-2",  # different live sidecar session
            "catalog_version": "v1",
            "tool_instance_id": "csv-profile-instance-2",
            "qualified_tool_id": "client__csv__profile",
        },
    )

    allowed, warnings = filter_tools_for_custom_agent(
        [foreign_session_client], spec, request_device_id="desktop-1"
    )

    assert allowed == []
    assert warnings  # selected client tool reported unavailable, not substituted


@pytest.mark.asyncio
async def test_custom_agent_tool_search_refresh_binds_loaded_tools(monkeypatch):
    from app.ai.agents.custom_agent import CustomAgent
    from app.ai.deferred_tool_state import get_deferred_tool_state, reset_deferred_tool_state
    from app.ai.mcp_tool_catalog import ToolReference
    from app.ai.tool_execution import _refresh_tool_map_after_search

    reset_deferred_tool_state()
    spec = _spec(tool_refs=[SERVER_MCP_TOOL_REF])
    agent = CustomAgent(spec)
    selected_server = _FakeTool(
        "calculate",
        {
            "tool_origin": "server_mcp",
            "server_name": "calculator",
            "qualified_tool_id": "calculator::calculate",
        },
    )
    unselected_server = _FakeTool(
        "search_web",
        {
            "tool_origin": "server_mcp",
            "server_name": "search",
            "qualified_tool_id": "search::search_web",
        },
    )

    class FakeManager:
        _tool_index = {
            "calculate": [selected_server],
            "search_web": [unselected_server],
        }
        _server_tools = {
            "calculator": [selected_server],
            "search": [unselected_server],
        }

        async def get_tools(self):
            return [selected_server, unselected_server]

        def get_server_for_tool(self, tool):
            return "calculator" if tool is selected_server else "search"

    async def fake_get_global_mcp_manager():
        return FakeManager()

    agent.tools = [selected_server, unselected_server]
    agent.mcp_manager = FakeManager()

    try:
        # Only "calculate" is loaded; "search_web" stays in the catalog, unloaded.
        get_deferred_tool_state().autoload(
            "conv-1",
            spec.runtime_agent_id,
            [ToolReference("calculate", "calculator")],
        )
        monkeypatch.setattr(
            "app.ai.mcp_registry.get_global_mcp_manager",
            fake_get_global_mcp_manager,
        )

        tool_map = {"tool_search": object()}
        await _refresh_tool_map_after_search(
            tool_map=tool_map,
            agent=agent,
            conversation_id="conv-1",
            user_id="user-1",
            device_id=None,
        )

        assert "calculate" in tool_map
        assert "search_web" not in tool_map
    finally:
        reset_deferred_tool_state()


def test_tool_search_allowlist_full_server_scoped_client():
    spec = _spec(
        tool_refs=[
            SERVER_MCP_TOOL_REF,
            CLIENT_TOOL_REF,
        ]
    )
    # Server discovery spans the full backend catalog (no allowlist filter) ...
    assert spec.server_tool_search_allowlist() is None
    # ... while client discovery stays scoped to the exact selected instance id,
    # never the public tool name (which a foreign sidecar could also expose).
    client_allowlist = set(spec.client_tool_search_allowlist())
    assert "csv-profile-instance" in client_allowlist
    assert "client__csv__profile" not in client_allowlist
    assert "profile" not in client_allowlist


def test_missing_selected_skill_populates_runtime_warning(monkeypatch):
    from app.ai.agents import custom_agent as custom_agent_module
    from app.ai.agents.custom_agent import CustomAgent

    kobo = {"source": "client", "lookup_name": "kobo-library", "name": "kobo-library"}
    agent = CustomAgent(_spec(skill_refs=[kobo]))

    monkeypatch.setattr(custom_agent_module, "list_resolved_skills", lambda **kwargs: [])
    monkeypatch.setattr(
        custom_agent_module,
        "get_active_client_runtime_session",
        lambda **kwargs: SimpleNamespace(
            session_id="session-b",
            tool_catalog_version=1,
            skill_catalog_version=1,
        ),
    )

    agent._get_tools_for_binding(user_id="user-1", device_id="desktop-2")

    assert agent._runtime_warnings == [
        "Selected skill 'kobo-library' is not available on this device."
    ]
    assert "kobo-library" not in agent._build_skills_suffix(
        user_id="user-1", device_id="desktop-2"
    )
