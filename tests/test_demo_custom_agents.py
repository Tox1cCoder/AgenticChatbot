from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest


class _SessionState(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class _CacheDecorator:
    def __call__(self, *args: Any, **kwargs: Any):
        return lambda func: func

    def clear(self) -> None:
        return None


class _StreamlitStub(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.query_params: dict[str, str] = {}
        self.cache_data = _CacheDecorator()
        self.cache_resource = _CacheDecorator()

    def set_page_config(self, *args: Any, **kwargs: Any) -> None:
        return None

    def markdown(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __getattr__(self, name: str):
        def _noop(*args: Any, **kwargs: Any):
            return None

        return _noop


def _import_demo_with_ui_stubs(monkeypatch: pytest.MonkeyPatch):
    streamlit_stub = _StreamlitStub()
    components_module = types.ModuleType("streamlit.components")
    components_v1_module = types.ModuleType("streamlit.components.v1")
    components_v1_module.html = lambda *args, **kwargs: None
    components_module.v1 = components_v1_module
    streamlit_stub.components = components_module
    markdown_stub = types.ModuleType("markdown")
    markdown_stub.markdown = lambda text, **_kwargs: text

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(sys.modules, "streamlit.components", components_module)
    monkeypatch.setitem(sys.modules, "streamlit.components.v1", components_v1_module)
    monkeypatch.setitem(sys.modules, "markdown", markdown_stub)
    sys.modules.pop("demo", None)
    return importlib.import_module("demo")


def test_custom_agent_tool_refs_do_not_collide_between_server_and_client(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    server_tool = {
        "type": "server_mcp",
        "server_name": "desktop_commander",
        "tool_name": "start_process",
        "qualified_tool_id": "desktop_commander::start_process",
    }
    client_tool = {
        "type": "client",
        "device_id": "device-1",
        "session_id": "session-1",
        "catalog_version": 3,
        "tool_instance_id": "instance-1",
        "server_name": "desktop_commander",
        "qualified_tool_id": "desktop_commander::start_process",
        "tool_name": "start_process",
    }

    refs = demo._build_tool_refs(
        [
            demo._custom_agent_tool_option_key(server_tool),
            demo._custom_agent_tool_option_key(client_tool),
        ],
        [server_tool],
        [client_tool],
    )

    assert [ref["type"] for ref in refs] == ["server_mcp", "client"]
    assert refs[0]["server_name"] == "desktop_commander"
    assert refs[1]["device_id"] == "device-1"
    assert refs[1]["catalog_version"] == "3"


def test_custom_agent_edit_defaults_include_current_tools_and_skills(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    server_tool = {
        "type": "server_mcp",
        "server_name": "calculator",
        "tool_name": "calculate",
        "qualified_tool_id": "calculator::calculate",
    }
    client_tool = {
        "type": "client",
        "device_id": "device-1",
        "session_id": "session-1",
        "catalog_version": "3",
        "tool_instance_id": "instance-1",
        "server_name": "csv",
        "qualified_tool_id": "client__csv__profile",
        "tool_name": "profile",
    }
    stale_client_tool = {
        **client_tool,
        "session_id": "old-session",
        "tool_instance_id": "old-instance",
    }
    skills = [
        {"source": "server", "lookup_name": "data-analysis", "name": "data-analysis"},
        {"source": "client", "lookup_name": "desktop", "name": "desktop"},
    ]

    selected_tool_keys = demo._custom_agent_selected_tool_keys(
        [server_tool, client_tool, stale_client_tool],
        [server_tool],
        [client_tool],
    )
    selected_skill_keys = demo._custom_agent_selected_skill_keys(
        [
            {"source": "server", "lookup_name": "data-analysis", "name": "data-analysis"},
            {"source": "server", "lookup_name": "missing", "name": "missing"},
        ],
        skills,
    )

    assert selected_tool_keys == [
        demo._custom_agent_tool_option_key(server_tool),
        demo._custom_agent_tool_option_key(client_tool),
    ]
    assert selected_skill_keys == [("server", "data-analysis")]


def test_build_tool_refs_expands_selected_mcp_server_to_all_server_tools(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    server_tools = [
        {
            "type": "server_mcp",
            "server_name": "calculator",
            "tool_name": "add",
            "qualified_tool_id": "calculator::add",
        },
        {
            "type": "server_mcp",
            "server_name": "calculator",
            "tool_name": "subtract",
            "qualified_tool_id": "calculator::subtract",
        },
        {
            "type": "server_mcp",
            "server_name": "search",
            "tool_name": "web",
            "qualified_tool_id": "search::web",
        },
    ]

    refs = demo._build_tool_refs(
        [],
        server_tools,
        [],
        selected_server_group_keys=["server::calculator"],
    )

    assert [ref["qualified_tool_id"] for ref in refs] == [
        "calculator::add",
        "calculator::subtract",
    ]
    assert all(ref["type"] == "server_mcp" for ref in refs)


def test_build_tool_refs_deduplicates_server_group_and_individual_tool(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    calculate_tool = {
        "type": "server_mcp",
        "server_name": "calculator",
        "tool_name": "calculate",
        "qualified_tool_id": "calculator::calculate",
    }
    server_tools = [
        calculate_tool,
        {
            "type": "server_mcp",
            "server_name": "calculator",
            "tool_name": "explain",
            "qualified_tool_id": "calculator::explain",
        },
    ]

    refs = demo._build_tool_refs(
        [demo._custom_agent_tool_option_key(calculate_tool)],
        server_tools,
        [],
        selected_server_group_keys=["server::calculator"],
    )

    assert [ref["qualified_tool_id"] for ref in refs] == [
        "calculator::calculate",
        "calculator::explain",
    ]


def test_custom_agent_edit_defaults_detect_all_tools_from_server(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    server_tools = [
        {
            "type": "server_mcp",
            "server_name": "calculator",
            "tool_name": "add",
            "qualified_tool_id": "calculator::add",
        },
        {
            "type": "server_mcp",
            "server_name": "calculator",
            "tool_name": "subtract",
            "qualified_tool_id": "calculator::subtract",
        },
        {
            "type": "server_mcp",
            "server_name": "search",
            "tool_name": "web",
            "qualified_tool_id": "search::web",
        },
    ]
    tool_refs = [server_tools[0], server_tools[1], server_tools[2]]

    selected_servers = demo._custom_agent_selected_server_group_keys(
        tool_refs, server_tools, []
    )
    selected_tool_keys = demo._custom_agent_selected_tool_keys(
        tool_refs,
        server_tools,
        [],
        excluded_group_keys=selected_servers,
    )

    assert selected_servers == ["server::calculator"]
    assert selected_tool_keys == [demo._custom_agent_tool_option_key(server_tools[2])]


def _desktop_commander_client_tools(count: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "type": "client",
            "device_id": "device-1",
            "session_id": "session-1",
            "catalog_version": "5",
            "tool_instance_id": f"instance-{i}",
            "server_name": "desktop_commander",
            "qualified_tool_id": f"desktop_commander::tool_{i}",
            "tool_name": f"tool_{i}",
        }
        for i in range(count)
    ]


def _desktop_commander_client_servers(count: int = 3) -> list[dict[str, Any]]:
    return [
        {"server_name": "desktop_commander", "device_id": "device-1", "tool_count": count}
    ]


def test_server_groups_surface_sidecar_server_distinct_from_backend(monkeypatch):
    """A sidecar MCP server is pickable as its own group, namespaced so it never
    collides with a backend server of the same name."""
    demo = _import_demo_with_ui_stubs(monkeypatch)
    backend_tools = [
        {
            "type": "server_mcp",
            "server_name": "desktop_commander",
            "tool_name": "backend_a",
            "qualified_tool_id": "desktop_commander::backend_a",
        },
        {
            "type": "server_mcp",
            "server_name": "desktop_commander",
            "tool_name": "backend_b",
            "qualified_tool_id": "desktop_commander::backend_b",
        },
    ]
    client_tools = _desktop_commander_client_tools(3)
    client_servers = _desktop_commander_client_servers(3)

    groups = demo._custom_agent_server_groups(backend_tools, client_tools, client_servers)

    assert set(groups) == {"server::desktop_commander", "client::device-1::desktop_commander"}
    assert groups["client::device-1::desktop_commander"]["kind"] == "client"
    assert (
        demo._custom_agent_server_group_label("client::device-1::desktop_commander", groups)
        == "[client] desktop_commander (all 3 tools)"
    )


def test_server_groups_fall_back_to_client_tools_without_client_servers(monkeypatch):
    """If the backend omits clientServers (older build), groups are still derived
    from the client tools themselves."""
    demo = _import_demo_with_ui_stubs(monkeypatch)
    client_tools = _desktop_commander_client_tools(3)

    groups = demo._custom_agent_server_groups([], client_tools, [])

    assert "client::device-1::desktop_commander" in groups


def test_server_groups_omit_single_tool_sidecar_server(monkeypatch):
    """A one-tool sidecar server is attached via the individual-tool picker, not
    the 'all tools from server' group — matching backend behavior."""
    demo = _import_demo_with_ui_stubs(monkeypatch)
    client_tools = _desktop_commander_client_tools(1)
    client_servers = _desktop_commander_client_servers(1)

    groups = demo._custom_agent_server_groups([], client_tools, client_servers)

    assert groups == {}


def test_build_tool_refs_expands_selected_sidecar_server_to_all_client_tools(monkeypatch):
    """Selecting a sidecar server group expands to one fully-scoped client ref per
    tool (device/session/catalog/instance preserved)."""
    demo = _import_demo_with_ui_stubs(monkeypatch)
    client_tools = _desktop_commander_client_tools(3)
    client_servers = _desktop_commander_client_servers(3)

    refs = demo._build_tool_refs(
        [],
        [],
        client_tools,
        selected_server_group_keys=["client::device-1::desktop_commander"],
        client_servers=client_servers,
    )

    assert [r["qualified_tool_id"] for r in refs] == [
        "desktop_commander::tool_0",
        "desktop_commander::tool_1",
        "desktop_commander::tool_2",
    ]
    assert all(r["type"] == "client" for r in refs)
    assert {r["device_id"] for r in refs} == {"device-1"}
    assert {r["catalog_version"] for r in refs} == {"5"}
    assert [r["tool_instance_id"] for r in refs] == ["instance-0", "instance-1", "instance-2"]


def test_build_tool_refs_dedups_sidecar_group_and_individual_client_tool(monkeypatch):
    """A client tool chosen both via its server group and individually appears once."""
    demo = _import_demo_with_ui_stubs(monkeypatch)
    client_tools = _desktop_commander_client_tools(3)
    client_servers = _desktop_commander_client_servers(3)

    refs = demo._build_tool_refs(
        [demo._custom_agent_tool_option_key(client_tools[0])],
        [],
        client_tools,
        selected_server_group_keys=["client::device-1::desktop_commander"],
        client_servers=client_servers,
    )

    qids = [r["qualified_tool_id"] for r in refs]
    assert qids == [
        "desktop_commander::tool_0",
        "desktop_commander::tool_1",
        "desktop_commander::tool_2",
    ]
    assert len(qids) == len(set(qids))


def test_selected_server_group_keys_detect_fully_selected_sidecar_server(monkeypatch):
    """Editing an agent that already has every sidecar tool pre-selects the group."""
    demo = _import_demo_with_ui_stubs(monkeypatch)
    client_tools = _desktop_commander_client_tools(3)
    client_servers = _desktop_commander_client_servers(3)

    selected = demo._custom_agent_selected_server_group_keys(
        list(client_tools), [], client_tools, client_servers
    )

    assert selected == ["client::device-1::desktop_commander"]


def test_grouped_tool_keys_cover_selected_server_tools(monkeypatch):
    """Selecting a whole server yields exactly that server's individual-tool
    option keys, so the individual-tool list can drop them."""
    demo = _import_demo_with_ui_stubs(monkeypatch)
    server_tools = [
        {
            "type": "server_mcp",
            "server_name": "calculator",
            "tool_name": "add",
            "qualified_tool_id": "calculator::add",
        },
        {
            "type": "server_mcp",
            "server_name": "calculator",
            "tool_name": "subtract",
            "qualified_tool_id": "calculator::subtract",
        },
    ]
    client_tools = _desktop_commander_client_tools(3)
    client_servers = _desktop_commander_client_servers(3)
    server_groups = demo._custom_agent_server_groups(server_tools, client_tools, client_servers)

    assert demo._custom_agent_grouped_tool_keys(server_groups, []) == set()

    backend_covered = demo._custom_agent_grouped_tool_keys(server_groups, ["server::calculator"])
    assert backend_covered == {demo._custom_agent_tool_option_key(t) for t in server_tools}

    both_covered = demo._custom_agent_grouped_tool_keys(
        server_groups, ["server::calculator", "client::device-1::desktop_commander"]
    )
    expected = {demo._custom_agent_tool_option_key(t) for t in (*server_tools, *client_tools)}
    assert both_covered == expected


def test_individual_tool_list_excludes_selected_server_tools(monkeypatch):
    """End-to-end of the picker rule: with a server group selected, none of that
    server's tools remain in the individual-tool option list."""
    demo = _import_demo_with_ui_stubs(monkeypatch)
    client_tools = _desktop_commander_client_tools(3)
    client_servers = _desktop_commander_client_servers(3)
    server_groups = demo._custom_agent_server_groups([], client_tools, client_servers)
    tool_labels = {demo._custom_agent_tool_option_key(t): "x" for t in client_tools}

    excluded = demo._custom_agent_grouped_tool_keys(
        server_groups, ["client::device-1::desktop_commander"]
    )
    remaining = [key for key in tool_labels if key not in excluded]

    assert remaining == []


def test_retain_session_options_prunes_unavailable_picks(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    demo.st.session_state["ca_tools"] = ["a", "b", "c"]
    demo._ca_retain_session_options("ca_tools", ["a", "c"])
    assert demo.st.session_state["ca_tools"] == ["a", "c"]
    # Missing key is a no-op (must not raise).
    demo._ca_retain_session_options("ca_missing", ["x"])


def test_custom_agent_edit_matches_reconnected_client_tool_by_stable_identity(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    current_client_tool = {
        "type": "client",
        "device_id": "device-1",
        "session_id": "session-2",
        "catalog_version": "2",
        "tool_instance_id": "instance-2",
        "server_name": "csv",
        "qualified_tool_id": "client__csv__profile",
        "tool_name": "profile",
    }
    saved_client_tool = {
        **current_client_tool,
        "session_id": "session-1",
        "catalog_version": "1",
        "tool_instance_id": "instance-1",
    }

    assert demo._custom_agent_tool_refs_available([saved_client_tool], [], [current_client_tool])
    selected_keys = demo._custom_agent_selected_tool_keys(
        [saved_client_tool], [], [current_client_tool]
    )

    assert selected_keys == [demo._custom_agent_tool_option_key(current_client_tool)]


def test_custom_agent_edit_matches_legacy_server_skill_to_client_skill(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    current_client_skill = {
        "source": "client",
        "lookup_name": "data-analysis",
        "name": "data-analysis",
    }
    legacy_server_skill_ref = {
        "source": "server",
        "lookup_name": "data-analysis",
        "name": "data-analysis",
    }

    selected = demo._custom_agent_selected_skill_keys(
        [legacy_server_skill_ref], [current_client_skill]
    )

    assert demo._custom_agent_skill_refs_available(
        [legacy_server_skill_ref], [current_client_skill]
    )
    assert selected == [("client", "data-analysis")]
    assert demo._build_skill_refs(selected, [current_client_skill]) == [current_client_skill]


def test_custom_agent_build_skill_refs_filters_stale_selection(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    skills = [
        {"source": "server", "lookup_name": "data-analysis", "name": "data-analysis"},
        {"source": "client", "lookup_name": "desktop", "name": "desktop"},
    ]

    refs = demo._build_skill_refs(
        [("server", "data-analysis"), ("server", "missing")],
        skills,
    )

    assert refs == [{"source": "server", "lookup_name": "data-analysis", "name": "data-analysis"}]


def test_custom_agent_edit_detects_unavailable_existing_refs(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    server_tool = {
        "type": "server_mcp",
        "server_name": "calculator",
        "tool_name": "calculate",
        "qualified_tool_id": "calculator::calculate",
    }
    stale_client_tool = {
        "type": "client",
        "device_id": "device-1",
        "session_id": "old-session",
        "catalog_version": "2",
        "tool_instance_id": "old-instance",
        "server_name": "csv",
        "qualified_tool_id": "client__csv__profile",
        "tool_name": "profile",
    }
    skill = {"source": "server", "lookup_name": "data-analysis", "name": "data-analysis"}
    stale_skill = {"source": "client", "lookup_name": "missing", "name": "missing"}

    assert demo._custom_agent_tool_refs_available([server_tool], [server_tool], [])
    assert not demo._custom_agent_tool_refs_available(
        [server_tool, stale_client_tool], [server_tool], []
    )
    assert demo._custom_agent_skill_refs_available([skill], [skill])
    assert not demo._custom_agent_skill_refs_available([skill, stale_skill], [skill])


def test_message_agent_label_prefers_canonical_agent_metadata(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    metadata = {
        "agent": {
            "id": "custom_agent:abc",
            "kind": "custom",
            "name": "Data Analyst",
            "custom_agent_id": "abc",
            "source": "response",
        }
    }

    assert demo.get_message_agent_label(metadata) == "Data Analyst"


def test_message_agent_label_falls_back_to_legacy_custom_fields(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    metadata = {
        "runtime_agent_id": "custom_agent:abc",
        "custom_agent_name": "Data Analyst",
    }

    assert demo.get_message_agent_label(metadata) == "Data Analyst"
