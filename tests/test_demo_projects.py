"""Streamlit project helpers and sidebar wiring."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from contextlib import nullcontext
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
        if "page=1" in path:
            return {
                "success": True,
                "data": {
                    "items": [{"id": str(uuid4()), "name": "Roadmap"}],
                    "meta": {"currentPage": 1, "lastPage": 2},
                },
            }
        return {
            "success": True,
            "data": {
                "items": [{"id": str(uuid4()), "name": "Notes"}],
                "meta": {"currentPage": 2, "lastPage": 2},
            },
        }

    monkeypatch.setattr(demo, "make_api_request", _fake_request)

    result = demo.list_projects()

    assert calls == [
        ("GET", "/projects?page=1&limit=100"),
        ("GET", "/projects?page=2&limit=100"),
    ]
    assert result[0]["name"] == "Roadmap"
    assert result[1]["name"] == "Notes"


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


def test_set_project_custom_agents_sends_camel_case_body(monkeypatch):
    """The backend contract requires ``customAgentIds`` (camelCase); a stray
    snake_case rename here would pass silently against every mocked test
    unless the body itself is asserted."""
    demo = _import_demo_with_ui_stubs(monkeypatch)
    project_id = uuid4()
    calls = []

    def _fake_request(method, path, data=None, **kwargs):
        calls.append((method, path, data))
        return {"success": True}

    monkeypatch.setattr(demo, "make_api_request", _fake_request)

    demo.set_project_custom_agents(str(project_id), ["agent-1", "agent-2"])

    assert calls == [
        (
            "PUT",
            f"/projects/{project_id}/custom-agents",
            {"customAgentIds": ["agent-1", "agent-2"]},
        )
    ]


def test_create_project_sends_name_description_instructions(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    calls = []

    def _fake_request(method, path, data=None, **kwargs):
        calls.append((method, path, data))
        return {"success": True, "data": {"id": str(uuid4())}}

    monkeypatch.setattr(demo, "make_api_request", _fake_request)

    demo.create_project("Roadmap", "Q3 planning conversations.", "Be brief.")

    assert calls == [
        (
            "POST",
            "/projects",
            {
                "name": "Roadmap",
                "description": "Q3 planning conversations.",
                "instructions": "Be brief.",
            },
        )
    ]


def test_update_project_passes_fields_through_unchanged(monkeypatch):
    """``update_project`` must not rename or reshape the caller's fields dict —
    the backend's ``exclude_unset`` semantics depend on only the keys the
    caller actually set being present."""
    demo = _import_demo_with_ui_stubs(monkeypatch)
    project_id = uuid4()
    calls = []

    def _fake_request(method, path, data=None, **kwargs):
        calls.append((method, path, data))
        return {"success": True, "data": {}}

    monkeypatch.setattr(demo, "make_api_request", _fake_request)

    fields = {"name": "Renamed", "description": None}
    demo.update_project(str(project_id), fields)

    assert calls == [("PATCH", f"/projects/{project_id}", fields)]


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


def test_sidebar_search_filters_project_names_and_descriptions(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    streamlit_stub = sys.modules["streamlit"]
    state = streamlit_stub.session_state
    state.projects_list = [
        {"id": "roadmap", "name": "Roadmap", "description": "Planning"},
        {"id": "research", "name": "Notes", "description": "Research backlog"},
        {"id": "misc", "name": "Misc", "description": "Other"},
    ]
    state.conversations_list = []
    state.current_user_id = None
    state.active_view = "chat"
    state.current_project_id = None
    streamlit_stub.sidebar = nullcontext()
    streamlit_stub.text_input = lambda label, **_kwargs: (
        "RESEARCH" if label == "Search projects" else ""
    )
    shown_buttons = []
    streamlit_stub.button = lambda label, **_kwargs: shown_buttons.append(label) or False

    demo.render_sidebar()

    assert "Notes" in shown_buttons
    assert "Roadmap" not in shown_buttons
    assert "Misc" not in shown_buttons


def _stub_existing_project_view(monkeypatch, *, search: str = "", clicked: str | None = None):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    streamlit_stub = sys.modules["streamlit"]
    state = streamlit_stub.session_state
    state.current_user_id = "user-1"
    state.auth_token = "token"
    state.current_project_id = "project-1"
    state.active_view = "project"
    monkeypatch.setattr(
        demo, "get_project", lambda _id: {"id": "project-1", "name": "Roadmap", "customAgents": []}
    )
    monkeypatch.setattr(demo, "list_custom_agents", lambda: [])
    streamlit_stub.text_input = lambda label, **kwargs: (
        search if label == "Search conversations" else kwargs.get("value", "")
    )
    streamlit_stub.text_area = lambda _label, **kwargs: kwargs.get("value", "")
    streamlit_stub.multiselect = lambda _label, _options, **kwargs: kwargs.get("default", [])
    streamlit_stub.button = lambda label, **_kwargs: label == clicked
    streamlit_stub.columns = lambda _spec: (nullcontext(), nullcontext())
    return demo, streamlit_stub


def test_project_conversation_search_resets_page_and_uses_server_filter(monkeypatch):
    demo, streamlit_stub = _stub_existing_project_view(monkeypatch, search=" budget ")
    state = streamlit_stub.session_state
    state["project_conversation_page_project-1"] = 3
    state["project_conversation_last_search_project-1"] = "old"
    calls = []

    def _get_conversations(**kwargs):
        calls.append(kwargs)
        return {
            "success": True,
            "data": {"items": [], "meta": {"total": 0, "currentPage": 1, "lastPage": 1}},
        }

    monkeypatch.setattr(demo, "get_conversations", _get_conversations)

    demo.render_project_view()

    assert calls == [{"project_id": "project-1", "search": "budget", "page": 1, "limit": 20}]
    assert state["project_conversation_page_project-1"] == 1


def test_project_conversation_next_page_uses_next_server_page(monkeypatch):
    demo, streamlit_stub = _stub_existing_project_view(monkeypatch, clicked="Next")
    calls = []

    def _get_conversations(**kwargs):
        calls.append(kwargs)
        return {
            "success": True,
            "data": {
                "items": [],
                "meta": {"total": 25, "currentPage": kwargs["page"], "lastPage": 2},
            },
        }

    monkeypatch.setattr(demo, "get_conversations", _get_conversations)

    demo.render_project_view()
    streamlit_stub.button = lambda _label, **_kwargs: False
    demo.render_project_view()

    assert [call["page"] for call in calls] == [1, 2]


def test_project_delete_requires_confirmation(monkeypatch):
    demo, streamlit_stub = _stub_existing_project_view(monkeypatch, clicked="Delete project")
    monkeypatch.setattr(
        demo,
        "get_conversations",
        lambda **_kwargs: {
            "success": True,
            "data": {"items": [], "meta": {"total": 0, "currentPage": 1, "lastPage": 1}},
        },
    )
    deleted = []
    monkeypatch.setattr(
        demo, "delete_project", lambda project_id: deleted.append(project_id) or True
    )

    demo.render_project_view()

    assert deleted == []
    assert streamlit_stub.session_state["project_delete_pending_id"] == "project-1"

    streamlit_stub.button = lambda label, **_kwargs: label == "Confirm delete"
    monkeypatch.setattr(streamlit_stub, "rerun", lambda: None)
    demo.render_project_view()

    assert deleted == ["project-1"]
    assert streamlit_stub.session_state["project_delete_pending_id"] is None


def test_project_delete_confirmation_can_be_cancelled(monkeypatch):
    demo, streamlit_stub = _stub_existing_project_view(monkeypatch, clicked="Cancel")
    streamlit_stub.session_state.project_delete_pending_id = "project-1"
    monkeypatch.setattr(
        demo,
        "get_conversations",
        lambda **_kwargs: {
            "success": True,
            "data": {"items": [], "meta": {"total": 0, "currentPage": 1, "lastPage": 1}},
        },
    )
    deleted = []
    monkeypatch.setattr(demo, "delete_project", lambda project_id: deleted.append(project_id))

    demo.render_project_view()

    assert deleted == []
    assert streamlit_stub.session_state.project_delete_pending_id is None


def test_back_to_chat_clears_current_project_id(monkeypatch):
    """A browsed project must not silently capture the next new conversation.

    Reachable sequence: "New Chat" (clears it), click a project in the
    sidebar to look at it, "Back to chat", then type a message — without this
    clear the conversation is silently created inside the browsed project.
    """
    demo = _import_demo_with_ui_stubs(monkeypatch)
    streamlit_stub = sys.modules["streamlit"]
    streamlit_stub.session_state.current_user_id = "user-1"
    streamlit_stub.session_state.auth_token = "token"
    streamlit_stub.session_state.current_project_id = "project-1"
    streamlit_stub.session_state.project_delete_pending_id = "project-1"
    streamlit_stub.button = lambda label, *args, **kwargs: (
        label == (":material/arrow_back: Back to chat")
    )

    def _raise_rerun():
        raise RuntimeError("rerun")

    streamlit_stub.rerun = _raise_rerun

    with pytest.raises(RuntimeError, match="rerun"):
        demo.render_project_view()

    assert streamlit_stub.session_state.current_project_id is None
    assert streamlit_stub.session_state.active_view == "chat"
    assert streamlit_stub.session_state.project_delete_pending_id is None


def test_new_chat_in_project_button_keeps_current_project_id(monkeypatch):
    """The counterpart to ``test_back_to_chat_clears_current_project_id``:
    starting a new chat from inside a project must keep ``current_project_id``
    so the conversation is created in that project."""
    demo = _import_demo_with_ui_stubs(monkeypatch)

    source = inspect.getsource(demo.render_project_view)
    new_chat_block = source.split('"New chat in this project"', 1)[1].split("st.rerun()", 1)[0]

    assert "current_project_id" not in new_chat_block
