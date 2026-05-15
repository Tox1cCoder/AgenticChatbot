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

    def info(self, *args: Any, **kwargs: Any) -> None:
        return None

    def divider(self, *args: Any, **kwargs: Any) -> None:
        return None

    def button(self, *args: Any, **kwargs: Any) -> bool:
        return False

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
    return importlib.import_module("demo"), streamlit_stub


def _seed_chat_state(stub: _StreamlitStub, conversation_id: str) -> None:
    stub.session_state.clear()
    stub.session_state.current_conversation_id = conversation_id
    stub.session_state.conversation_messages_page = 1
    stub.session_state.has_more_messages = False
    stub.session_state.messages = [
        {
            "id": "msg-1",
            "sender": "user",
            "content": "hello before the plan existed",
            "metadata": {},
        }
    ]
    stub.session_state.pending_image_attachments = []
    stub.session_state.pending_file_attachments = []
    stub.session_state.conversation_messages_meta = {}


def test_chat_view_renders_plan_widget_when_conversation_cache_is_stale(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    _seed_chat_state(streamlit_stub, "conv-1")
    streamlit_stub.session_state.conversations_list = []

    calls: list[dict[str, Any] | None] = []

    def record_plan_widget(conversation_id: str, *, current_conv: dict[str, Any] | None) -> None:
        calls.append(current_conv)
        raise RuntimeError("stop after plan widget")

    monkeypatch.setattr(demo, "_render_plan_progress_widget", record_plan_widget)

    with pytest.raises(RuntimeError, match="stop after plan widget"):
        demo.render_chat_view()

    assert calls == [None]


def test_chat_view_renders_plan_widget_when_cached_flag_is_false(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    _seed_chat_state(streamlit_stub, "conv-2")
    cached_conversation = {
        "id": "conv-2",
        "title": "Existing chat",
        "planningModeEnabled": False,
    }
    streamlit_stub.session_state.conversations_list = [cached_conversation]

    calls: list[dict[str, Any] | None] = []

    def record_plan_widget(conversation_id: str, *, current_conv: dict[str, Any] | None) -> None:
        calls.append(current_conv)
        raise RuntimeError("stop after plan widget")

    monkeypatch.setattr(demo, "_render_plan_progress_widget", record_plan_widget)

    with pytest.raises(RuntimeError, match="stop after plan widget"):
        demo.render_chat_view()

    assert calls == [cached_conversation]
