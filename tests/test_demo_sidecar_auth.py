from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest
import requests


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


def test_sidebar_sign_out_clears_project_state(monkeypatch):
    """A stale ``projects_loaded=True`` after sign-out would make the next
    sign-in's sidebar keep showing the previous user's projects, since
    ``_load_projects_if_needed`` only fetches when the flag is falsy — and a
    stale ``current_project_id`` would carry into that user's next new
    conversation."""
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
    streamlit_stub.session_state.projects_list = [{"id": "project-1", "name": "Roadmap"}]
    streamlit_stub.session_state.projects_loaded = True
    streamlit_stub.session_state.current_project_id = "project-1"

    monkeypatch.setattr(demo, "close_conversation_manager", lambda: None)
    monkeypatch.setattr(demo, "reset_conversation_state", lambda: None)
    monkeypatch.setattr(demo, "make_api_request", lambda *args, **kwargs: {"success": True})

    with pytest.raises(RuntimeError, match="rerun"):
        demo.render_sidebar()

    assert streamlit_stub.session_state.projects_list == []
    assert streamlit_stub.session_state.projects_loaded is False
    assert streamlit_stub.session_state.current_project_id is None


# ── Launch token: demo.py proves it is the sidecar's own client ────────────


class _RecordingAdapter(requests.adapters.BaseAdapter):
    """Answers each send with the next queued status and records the headers."""

    def __init__(self, statuses: list[int]) -> None:
        super().__init__()
        self.statuses = list(statuses)
        self.sent: list[dict[str, str]] = []
        self.bodies: list[bytes | str | None] = []

    def send(self, request, **_kwargs):
        self.sent.append(dict(request.headers))
        self.bodies.append(request.body)
        response = requests.Response()
        response.status_code = self.statuses.pop(0) if self.statuses else 200
        response.url = request.url
        response.request = request
        response._content = b"{}"
        return response

    def close(self) -> None:
        return None


@pytest.fixture
def token_file(monkeypatch, tmp_path):
    from client_backend.core import launch_token
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path))
    return launch_token.launch_token_path()


def _session_with(demo, adapter: _RecordingAdapter):
    session = demo.get_http_session()
    session.mount(demo.API_BASE_URL, adapter)
    return session


def test_demo_session_sends_the_launch_token_to_the_sidecar(monkeypatch, token_file):
    demo, _ = _import_demo_with_ui_stubs(monkeypatch)
    token_file.write_text("launch-one", encoding="utf-8")
    adapter = _RecordingAdapter([200])

    response = _session_with(demo, adapter).get(f"{demo.API_BASE_URL}/mcp/sandbox")

    assert response.status_code == 200
    assert adapter.sent[0]["X-Kani-Client"] == "launch-one"


def test_demo_session_rereads_a_rotated_token_and_retries_once(monkeypatch, token_file):
    demo, _ = _import_demo_with_ui_stubs(monkeypatch)
    token_file.write_text("launch-one", encoding="utf-8")
    adapter = _RecordingAdapter([200, 401, 200])
    session = _session_with(demo, adapter)
    session.get(f"{demo.API_BASE_URL}/health")

    token_file.write_text("launch-two", encoding="utf-8")  # the sidecar restarted
    response = session.post(f"{demo.API_BASE_URL}/auth/login", json={"email": "a"})

    assert response.status_code == 200
    assert [sent["X-Kani-Client"] for sent in adapter.sent] == [
        "launch-one",
        "launch-one",
        "launch-two",
    ]
    assert adapter.bodies[1] == adapter.bodies[2]


def test_demo_session_does_not_retry_a_401_with_an_unchanged_token(monkeypatch, token_file):
    """An expired bearer session is a real 401; resending would not change it."""

    demo, _ = _import_demo_with_ui_stubs(monkeypatch)
    token_file.write_text("launch-one", encoding="utf-8")
    adapter = _RecordingAdapter([401, 401])

    response = _session_with(demo, adapter).get(f"{demo.API_BASE_URL}/mcp/sandbox")

    assert response.status_code == 401
    assert len(adapter.sent) == 1


def test_demo_session_without_a_token_file_sends_no_header(monkeypatch, token_file):
    demo, _ = _import_demo_with_ui_stubs(monkeypatch)
    adapter = _RecordingAdapter([200])

    _session_with(demo, adapter).get(f"{demo.API_BASE_URL}/health")

    assert "X-Kani-Client" not in adapter.sent[0]


def test_demo_session_never_sends_the_token_to_another_host(monkeypatch, token_file):
    demo, _ = _import_demo_with_ui_stubs(monkeypatch)
    token_file.write_text("launch-one", encoding="utf-8")
    adapter = _RecordingAdapter([200])
    session = demo.get_http_session()
    session.mount("https://elsewhere.example", adapter)

    session.get("https://elsewhere.example/x", headers={"X-Kani-Client": "leak"})

    assert "X-Kani-Client" not in adapter.sent[0]


def test_upload_support_uses_the_sidecar_session(monkeypatch):
    _import_demo_with_ui_stubs(monkeypatch)
    import upload_support
    from app.ui.sidecar_session import SidecarSession

    assert isinstance(upload_support.get_http_session(), SidecarSession)
