"""Streamlit project helpers and sidebar wiring."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from typing import Any
from uuid import uuid4

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


def test_project_client_helpers_exist(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)

    for name in (
        "list_projects",
        "create_project",
        "update_project",
        "delete_project",
        "get_project",
        "set_project_custom_agents",
        "attach_conversation_to_project",
        "detach_conversation_from_project",
        "render_project_view",
    ):
        assert hasattr(demo, name), f"demo.{name} is missing"


def test_get_conversations_accepts_a_project_filter(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)

    assert "project_id" in inspect.signature(demo.get_conversations).parameters


def test_list_projects_calls_the_endpoint(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    calls = []

    def _fake_request(method, path, **kwargs):
        calls.append((method, path))
        return {"success": True, "data": [{"id": str(uuid4()), "name": "Roadmap"}]}

    monkeypatch.setattr(demo, "make_api_request", _fake_request)

    result = demo.list_projects()

    assert calls == [("GET", "/projects")]
    assert result[0]["name"] == "Roadmap"


def test_attach_calls_put_on_the_membership_route(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    project_id, conversation_id = uuid4(), uuid4()
    calls = []

    monkeypatch.setattr(
        demo,
        "make_api_request",
        lambda method, path, **kwargs: calls.append((method, path)) or {"success": True},
    )

    demo.attach_conversation_to_project(str(project_id), str(conversation_id))

    assert calls == [("PUT", f"/projects/{project_id}/conversations/{conversation_id}")]


def test_project_settings_are_not_rendered_inside_tabs(monkeypatch):
    """Streamlit garbage-collects widget state for tabs that are not open, so an
    unsaved 8000-character instruction edit would vanish on a tab switch."""
    demo = _import_demo_with_ui_stubs(monkeypatch)

    source = inspect.getsource(demo.render_project_view)

    assert "st.tabs" not in source


def test_sidebar_renders_a_projects_section(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)

    source = inspect.getsource(demo.render_sidebar)

    assert "Projects" in source
    assert "render_project_view" in inspect.getsource(demo.main)
