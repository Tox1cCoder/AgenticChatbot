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
        self.pressed_buttons: set[str] = set()

    def set_page_config(self, *args: Any, **kwargs: Any) -> None:
        return None

    def markdown(self, *args: Any, **kwargs: Any) -> None:
        return None

    def divider(self, *args: Any, **kwargs: Any) -> None:
        return None

    def button(self, label: str, *args: Any, **kwargs: Any) -> bool:
        return label in self.pressed_buttons

    def toast(self, *args: Any, **kwargs: Any) -> None:
        return None

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
    components_v1_module.declare_component = (
        lambda *args, **kwargs: (lambda **_component_kwargs: _component_kwargs.get("default"))
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


def test_sidebar_sign_out_logs_out_sidecar_before_clearing_local_state(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    streamlit_stub.pressed_buttons.add("Sign Out")
    streamlit_stub.session_state.current_user_id = "user-1"
    streamlit_stub.session_state.auth_token = "local-session-token"
    streamlit_stub.session_state.current_user_profile = {
        "id": "user-1",
        "username": "Ada",
    }
    streamlit_stub.session_state.current_conversation_id = "conversation-1"
    streamlit_stub.session_state.conversations_loaded = True
    streamlit_stub.session_state.conversations_list = []
    streamlit_stub.session_state.conversations_last_fetch_params = None

    monkeypatch.setattr(demo, "close_conversation_manager", lambda: None)
    monkeypatch.setattr(demo, "reset_conversation_state", lambda: None)

    calls: list[dict[str, Any]] = []

    def _record_api_call(method: str, endpoint: str, data: dict | None = None) -> dict:
        calls.append(
            {
                "method": method,
                "endpoint": endpoint,
                "data": data,
                "auth_token": streamlit_stub.session_state.get("auth_token"),
            }
        )
        return {"success": True}

    monkeypatch.setattr(demo, "make_api_request", _record_api_call)

    with pytest.raises(RuntimeError, match="rerun"):
        demo.render_sidebar()

    assert calls == [
        {
            "method": "POST",
            "endpoint": "/auth/logout",
            "data": None,
            "auth_token": "local-session-token",
        }
    ]
