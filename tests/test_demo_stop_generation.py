"""Stop-generation targets the conversation that was streaming, not the current view.

Regression for: POST /messages/stop returning 422 because the stop trigger
passed the view's ``current_conversation_id`` — which is the ``pending_new``
sentinel (not a UUID) when the user clicks "New Chat" mid-generation, or the
wrong conversation after switching mid-generation. The streaming conversation
id is recorded in ``st.session_state.stream_conversation_id`` at stream start
and is the only valid stop target.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
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


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False


class _StreamlitStub(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.query_params: dict[str, str] = {}
        self.cache_data = _CacheDecorator()
        self.cache_resource = _CacheDecorator()
        self.sidebar = _Context()

    def set_page_config(self, *args: Any, **kwargs: Any) -> None:
        return None

    def chat_message(self, *args: Any, **kwargs: Any) -> _Context:
        return _Context()

    def rerun(self) -> None:
        raise RuntimeError("rerun")

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
    return importlib.import_module("demo"), streamlit_stub


_STREAM_CONVERSATION_ID = "827faf55-1041-4357-9e1a-0f7d031fa546"
_USER_MESSAGE_ID = "11111111-2222-3333-4444-555555555555"


def _set_inflight_stream(state: _SessionState) -> None:
    state.stream_inflight = True
    state.stream_conversation_id = _STREAM_CONVERSATION_ID
    state.stream_user_message_id = _USER_MESSAGE_ID
    state.stream_partial_text = ""
    state.pending_image_attachments = [{"name": "already-sent.png", "data": "YWJj"}]
    state.show_attachment_uploader = True


def test_stoppable_id_is_the_streaming_conversation_even_on_pending_new_view(monkeypatch):
    demo, stub = _import_demo_with_ui_stubs(monkeypatch)
    _set_inflight_stream(stub.session_state)
    stub.session_state.current_conversation_id = "pending_new"

    assert demo._stoppable_stream_conversation_id() == _STREAM_CONVERSATION_ID


def test_stoppable_id_empty_without_inflight_stream(monkeypatch):
    demo, stub = _import_demo_with_ui_stubs(monkeypatch)
    stub.session_state.stream_inflight = False
    stub.session_state.stream_conversation_id = _STREAM_CONVERSATION_ID
    stub.session_state.stream_user_message_id = _USER_MESSAGE_ID

    assert demo._stoppable_stream_conversation_id() == ""


def test_stoppable_id_empty_when_no_user_message_was_captured(monkeypatch):
    demo, stub = _import_demo_with_ui_stubs(monkeypatch)
    _set_inflight_stream(stub.session_state)
    stub.session_state.stream_user_message_id = ""

    assert demo._stoppable_stream_conversation_id() == ""


def test_stop_trigger_no_longer_uses_the_view_conversation():
    demo_source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")
    assert "_handle_stop_rerun(str(conversation_id))" not in demo_source
    assert "_handle_stop_rerun(_stoppable_stream_conversation_id" not in demo_source
    assert demo_source.count("_stoppable_stream_conversation_id()") >= 2  # def + trigger


def test_stop_rerun_sends_the_streaming_conversation_payload(monkeypatch):
    demo, stub = _import_demo_with_ui_stubs(monkeypatch)
    _set_inflight_stream(stub.session_state)
    # User switched to another conversation while the stream was inflight.
    stub.session_state.current_conversation_id = "99999999-8888-7777-6666-555555555555"
    stub.session_state.messages = []

    calls: list[dict[str, Any]] = []

    def _record(method: str, endpoint: str, data: dict | None = None) -> dict:
        calls.append({"method": method, "endpoint": endpoint, "data": data})
        return {
            "success": True,
            "data": {"status": "cancelled", "message": {"id": "m1", "content": "partial"}},
        }

    monkeypatch.setattr(demo, "make_api_request", _record)

    with pytest.raises(RuntimeError, match="rerun"):
        demo._handle_stop_rerun(_STREAM_CONVERSATION_ID)

    assert calls == [
        {
            "method": "POST",
            "endpoint": "/messages/stop",
            "data": {
                "conversationId": _STREAM_CONVERSATION_ID,
                "userMessageId": _USER_MESSAGE_ID,
            },
        }
    ]
    # The stopped conversation is not the one on screen: its partial message
    # must not leak into the visible message list.
    assert stub.session_state.messages == []
    assert stub.session_state.stream_inflight is False
    assert stub.session_state.pending_image_attachments == []
    assert stub.session_state.show_attachment_uploader is False


def test_stop_rerun_appends_partial_message_when_view_matches(monkeypatch):
    demo, stub = _import_demo_with_ui_stubs(monkeypatch)
    _set_inflight_stream(stub.session_state)
    stub.session_state.current_conversation_id = _STREAM_CONVERSATION_ID
    stub.session_state.messages = []

    bot_msg = {"id": "m1", "content": "partial"}

    monkeypatch.setattr(
        demo,
        "make_api_request",
        lambda method, endpoint, data=None: {
            "success": True,
            "data": {"status": "cancelled", "message": bot_msg},
        },
    )

    with pytest.raises(RuntimeError, match="rerun"):
        demo._handle_stop_rerun(_STREAM_CONVERSATION_ID)

    assert stub.session_state.messages == [bot_msg]
