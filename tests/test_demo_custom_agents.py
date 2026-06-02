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
