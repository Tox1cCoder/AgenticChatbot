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
    components_v1_module.declare_component = lambda *args, **kwargs: (
        lambda **_component_kwargs: _component_kwargs.get("default")
    )
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


def test_group_mcp_tools_by_server_scopes_each_tool_to_its_own_server(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    tools = [
        {"name": "get_current_time", "serverName": "time", "qualifiedId": "time::get_current_time"},
        {"name": "widget_create", "serverName": "widgets", "qualifiedId": "widgets::widget_create"},
        {"name": "widget_close", "serverName": "widgets", "qualifiedId": "widgets::widget_close"},
        {"name": "tavily_search", "serverName": "tavily", "qualifiedId": "tavily::tavily_search"},
    ]

    grouped = demo._group_mcp_tools_by_server(tools)

    assert set(grouped.keys()) == {"time", "widgets", "tavily"}
    assert [t["name"] for t in grouped["time"]] == ["get_current_time"]
    assert [t["name"] for t in grouped["widgets"]] == ["widget_create", "widget_close"]
    assert [t["name"] for t in grouped["tavily"]] == ["tavily_search"]
    # No cross-server leakage: widgets/tavily tools never appear under "time".
    assert all(t["serverName"] == "time" for t in grouped["time"])


def test_group_mcp_tools_by_server_buckets_missing_server_name(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    grouped = demo._group_mcp_tools_by_server([{"name": "orphan", "qualifiedId": "x"}])
    assert list(grouped.keys()) == ["(unknown)"]


def test_only_custom_mcp_servers_are_removable(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    assert demo._mcp_server_is_removable({"name": "my-server", "source": "custom"}) is True
    assert demo._mcp_server_is_removable({"name": "time", "source": "bundled"}) is False
    # Unknown/absent source is treated as non-removable (fail safe for built-ins).
    assert demo._mcp_server_is_removable({"name": "legacy"}) is False


def test_scoped_tool_fetch_uses_the_dedicated_route_with_an_encoded_segment(monkeypatch):
    """T007: a scoped fetch hits ``/mcp/servers/{name}/tools`` with the server
    name URL-encoded, never string-concatenated into a query."""
    demo = _import_demo_with_ui_stubs(monkeypatch)
    calls: list[tuple[str, str]] = []

    def _fake_request(method: str, endpoint: str, *_args, **_kwargs):
        calls.append((method, endpoint))
        return {"data": {"tools": [], "scope": {"kind": "server"}}}

    monkeypatch.setattr(demo, "make_api_request", _fake_request)

    demo.get_mcp_tools("weird/name space")

    assert calls == [("GET", "/mcp/servers/weird%2Fname%20space/tools")]


def test_unscoped_tool_fetch_uses_the_compatibility_route(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        demo,
        "make_api_request",
        lambda method, endpoint, *a, **k: (calls.append((method, endpoint)), {"data": {}})[1],
    )

    demo.get_mcp_tools()

    assert calls == [("GET", "/mcp/tools")]


def test_scoped_tool_fetch_returns_none_when_the_server_is_unknown(monkeypatch):
    """The sidecar/canonical scoped route 404s an unknown server; the UI helper
    surfaces that as "no data" rather than raising."""
    demo = _import_demo_with_ui_stubs(monkeypatch)
    monkeypatch.setattr(demo, "make_api_request", lambda *a, **k: {})

    assert demo.get_mcp_tools("nope") is None
