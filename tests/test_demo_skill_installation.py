"""Streamlit transport and session-state rules for skill ZIP installation.

Two things this covers that the sidecar cannot enforce for us: the browser-side
transport must reuse the same auth, error, and cache contract as every other
call, and an upload id plus its archive bytes must never outlive the session that
selected them.
"""

from __future__ import annotations

import ast
import importlib
import io
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import pytest

SOURCE_HASH = "a" * 64


class _SessionState(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        self.pop(name, None)


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
        self.toasts: list[tuple[str, str]] = []

    def set_page_config(self, *args: Any, **kwargs: Any) -> None:
        return None

    def toast(self, message: str = "", icon: str = "", *args: Any, **kwargs: Any) -> None:
        self.toasts.append((message, icon))

    def rerun(self) -> None:
        raise RuntimeError("rerun")

    def __getattr__(self, name: str):
        def _noop(*args: Any, **kwargs: Any):
            return None

        return _noop


def _import_demo(monkeypatch: pytest.MonkeyPatch):
    streamlit_stub = _StreamlitStub()
    components_module = types.ModuleType("streamlit.components")
    components_v1_module = types.ModuleType("streamlit.components.v1")
    components_v1_module.html = lambda *args, **kwargs: None
    components_v1_module.declare_component = lambda *args, **kwargs: (
        lambda **kwargs_inner: kwargs_inner.get("default")
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
    module = importlib.import_module("demo")
    return module, streamlit_stub


@pytest.fixture()
def demo_module(monkeypatch):
    module, streamlit_stub = _import_demo(monkeypatch)
    module.st.session_state.clear()
    module.st.session_state.auth_token = "local-token"
    module.st.session_state.api_cache_version = 0
    yield module
    sys.modules.pop("demo", None)


def _valid_skill_zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("demo/SKILL.md", "---\nname: demo\ndescription: Demo\n---\nBody")
    return buffer.getvalue()


class _Response:
    def __init__(self, status_code: int, payload: dict | None, *, invalid_json: bool = False):
        self.status_code = status_code
        self._payload = payload
        self._invalid_json = invalid_json

    def json(self):
        if self._invalid_json:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(response=self)


class _RecordingSession:
    def __init__(self, *responses: _Response):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def _record(self, method: str, url: str, **kwargs) -> _Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self._responses:
            return _Response(200, {"success": True, "data": {}})
        return self._responses.pop(0)

    def post(self, url, **kwargs):
        return self._record("POST", url, **kwargs)

    def request(self, method, url, **kwargs):
        return self._record(method, url, **kwargs)


def _success_upload_response() -> _Response:
    return _Response(
        201,
        {
            "success": True,
            "message": "Skill archive staged",
            "data": {
                "uploadId": "upload-a",
                "state": "staged",
                "preview": {"name": "demo", "sourceHash": SOURCE_HASH},
                "archive": {"filename": "demo.zip", "compressedBytes": 10},
            },
            "error": None,
        },
    )


def _error_response(status_code: int, message: str) -> _Response:
    return _Response(
        status_code,
        {
            "success": False,
            "code": "SKILL_ARCHIVE_TOO_LARGE",
            "message": message,
            "data": None,
            "error": {"retryable": False},
        },
    )


def test_stage_skill_zip_uses_authenticated_multipart_without_json(demo_module, monkeypatch):
    session = _RecordingSession(_success_upload_response())
    monkeypatch.setattr(demo_module, "get_http_session", lambda: session)

    result = demo_module.stage_skill_zip(
        "demo.zip",
        _valid_skill_zip_bytes(),
        "application/zip",
    )

    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/skills/uploads")
    assert call["files"]["file"][0] == "demo.zip"
    assert call["files"]["file"][1] == _valid_skill_zip_bytes()
    assert "json" not in call or call["json"] is None
    assert call["headers"]["Authorization"] == "Bearer local-token"
    assert result["uploadId"] == "upload-a"


def test_upload_uses_the_long_stream_timeout(demo_module, monkeypatch):
    """A 25 MiB archive does not upload inside the interactive request timeout."""
    session = _RecordingSession(_success_upload_response())
    monkeypatch.setattr(demo_module, "get_http_session", lambda: session)

    demo_module.stage_skill_zip("demo.zip", b"PK", "application/zip")

    assert session.calls[0]["timeout"] == demo_module.STREAM_REQUEST_TIMEOUT


def test_successful_upload_invalidates_the_api_cache(demo_module, monkeypatch):
    session = _RecordingSession(_success_upload_response())
    monkeypatch.setattr(demo_module, "get_http_session", lambda: session)
    before = demo_module.st.session_state.api_cache_version

    demo_module.stage_skill_zip("demo.zip", b"PK", "application/zip")

    assert demo_module.st.session_state.api_cache_version == before + 1


@pytest.mark.parametrize("status_code", [413, 415])
def test_upload_errors_use_shared_safe_message(demo_module, monkeypatch, status_code):
    session = _RecordingSession(_error_response(status_code, "safe message"))
    monkeypatch.setattr(demo_module, "get_http_session", lambda: session)

    assert demo_module.stage_skill_zip("demo.zip", b"PK", "application/zip") is None
    assert demo_module.st.session_state["_last_api_error_message"] == "safe message"


def test_upload_401_signs_the_user_out_like_any_other_call(demo_module, monkeypatch):
    """The multipart path must not be a way to stay signed in after a 401."""
    session = _RecordingSession(_Response(401, {"message": "Session expired"}))
    monkeypatch.setattr(demo_module, "get_http_session", lambda: session)
    transitions: list[bool] = []
    monkeypatch.setattr(demo_module, "_transition_to_login", lambda: transitions.append(True))

    assert demo_module.stage_skill_zip("demo.zip", b"PK", "application/zip") is None
    assert transitions == [True]


def test_upload_connection_error_is_reported_not_raised(demo_module, monkeypatch):
    import requests

    class _DeadSession:
        def post(self, *args, **kwargs):
            raise requests.exceptions.ConnectionError("sidecar down")

    monkeypatch.setattr(demo_module, "get_http_session", lambda: _DeadSession())

    assert demo_module.stage_skill_zip("demo.zip", b"PK", "application/zip") is None


def test_start_skill_install_sends_camel_case_confirmations(demo_module, monkeypatch):
    session = _RecordingSession(
        _Response(202, {"success": True, "data": {"operationId": "op-a", "state": "pending"}})
    )
    monkeypatch.setattr(demo_module, "get_http_session", lambda: session)

    result = demo_module.start_skill_install("upload-a", SOURCE_HASH, True, "b" * 64)

    call = session.calls[0]
    assert call["url"].endswith("/skills/uploads/upload-a/install")
    assert call["json"] == {
        "expectedSourceHash": SOURCE_HASH,
        "approveSetup": True,
        "replaceSourceHash": "b" * 64,
    }
    assert result["operationId"] == "op-a"


def test_start_skill_install_omits_replace_hash_for_a_new_install(demo_module, monkeypatch):
    """Sending replaceSourceHash unnecessarily would authorize an overwrite."""
    session = _RecordingSession(_Response(202, {"success": True, "data": {}}))
    monkeypatch.setattr(demo_module, "get_http_session", lambda: session)

    demo_module.start_skill_install("upload-a", SOURCE_HASH, False)

    assert "replaceSourceHash" not in session.calls[0]["json"]


@pytest.mark.parametrize(
    ("helper", "args", "expected"),
    [
        ("get_skill_installation", ("op/../a",), "/skills/installations/op%2F..%2Fa"),
        ("cancel_skill_upload", ("upload id",), "/skills/uploads/upload%20id"),
        ("cancel_skill_installation", ("op?x=1",), "/skills/installations/op%3Fx%3D1"),
    ],
)
def test_operation_helpers_url_encode_identifiers(
    demo_module, monkeypatch, helper, args, expected
):
    session = _RecordingSession(_Response(200, {"success": True, "data": {}}))
    monkeypatch.setattr(demo_module, "get_http_session", lambda: session)

    getattr(demo_module, helper)(*args)

    assert session.calls[0]["url"].endswith(expected)


def test_polling_never_serves_a_cached_status(demo_module, monkeypatch):
    """A cached poll would report a stale phase for the whole cache TTL."""
    session = _RecordingSession(
        _Response(200, {"success": True, "data": {"state": "running"}}),
        _Response(200, {"success": True, "data": {"state": "succeeded"}}),
    )
    monkeypatch.setattr(demo_module, "get_http_session", lambda: session)
    monkeypatch.setattr(
        demo_module,
        "_cached_get_request",
        lambda **_kwargs: pytest.fail("polling must not use the GET cache"),
    )

    first = demo_module.get_skill_installation("op-a")
    second = demo_module.get_skill_installation("op-a")

    assert first["state"] == "running"
    assert second["state"] == "succeeded"


def test_logout_discards_ids_and_bytes_before_remote_cleanup(demo_module, monkeypatch):
    demo_module.st.session_state.skill_upload_id = "upload-a"
    demo_module.st.session_state.skill_upload_bytes = b"must-not-persist"
    deleted: list[str] = []
    monkeypatch.setattr(
        demo_module, "cancel_skill_upload", lambda upload_id: deleted.append(upload_id)
    )

    demo_module._clear_skill_installation_session_state(cleanup_remote=True)

    assert "skill_upload_id" not in demo_module.st.session_state
    assert "skill_upload_bytes" not in demo_module.st.session_state
    assert deleted == ["upload-a"]


def test_local_state_is_cleared_even_when_remote_cleanup_fails(demo_module, monkeypatch):
    demo_module.st.session_state.skill_upload_id = "upload-a"
    demo_module.st.session_state.skill_operation_id = "op-a"

    def explode(_identifier):
        raise RuntimeError("sidecar unreachable")

    monkeypatch.setattr(demo_module, "cancel_skill_upload", explode)
    monkeypatch.setattr(demo_module, "cancel_skill_installation", explode)

    demo_module._clear_skill_installation_session_state(cleanup_remote=True)

    assert "skill_upload_id" not in demo_module.st.session_state
    assert "skill_operation_id" not in demo_module.st.session_state


def test_signing_out_clears_skill_installation_state(demo_module, monkeypatch):
    """The HITL logout path owns this; a leftover upload id would cross users."""
    demo_module.st.session_state.skill_upload_id = "upload-a"
    demo_module.st.session_state.skill_upload_preview = {"name": "demo"}

    demo_module._clear_skill_hitl_session_state()

    assert "skill_upload_id" not in demo_module.st.session_state
    assert "skill_upload_preview" not in demo_module.st.session_state


def test_every_declared_session_key_is_cleared(demo_module):
    for key in demo_module.SKILL_INSTALL_SESSION_KEYS:
        demo_module.st.session_state[key] = "value"

    demo_module._clear_skill_installation_session_state()

    assert not [key for key in demo_module.SKILL_INSTALL_SESSION_KEYS
                if key in demo_module.st.session_state]


def test_json_requests_still_bump_the_cache_version(demo_module, monkeypatch):
    """The extracted helper must not change behavior for existing callers."""
    session = _RecordingSession(_Response(200, {"success": True, "data": {}}))
    monkeypatch.setattr(demo_module, "get_http_session", lambda: session)
    before = demo_module.st.session_state.api_cache_version

    demo_module.make_api_request("POST", "/skills/reload")

    assert demo_module.st.session_state.api_cache_version == before + 1


def test_json_get_requests_do_not_bump_the_cache_version(demo_module, monkeypatch):
    session = _RecordingSession(_Response(200, {"success": True, "data": {}}))
    monkeypatch.setattr(demo_module, "get_http_session", lambda: session)
    before = demo_module.st.session_state.api_cache_version

    demo_module.make_api_request("GET", "/skills/demo", use_cache=False)

    assert demo_module.st.session_state.api_cache_version == before


def test_json_401_still_transitions_to_login(demo_module, monkeypatch):
    session = _RecordingSession(_Response(401, {"message": "Session expired"}))
    monkeypatch.setattr(demo_module, "get_http_session", lambda: session)
    transitions: list[bool] = []
    monkeypatch.setattr(demo_module, "_transition_to_login", lambda: transitions.append(True))

    assert demo_module.make_api_request("POST", "/skills/reload") == {}
    assert transitions == [True]


def test_auth_headers_are_omitted_when_signed_out(demo_module):
    demo_module.st.session_state.auth_token = None

    assert demo_module._auth_headers() == {}


# --- Render contract -------------------------------------------------------
#
# Asserted against the source rather than a rendered page: Streamlit widgets need
# a running script context, and what matters here is a security contract (the
# approvals exist, are separate, and default to off) that source inspection pins
# precisely.

REPO_ROOT = Path(__file__).resolve().parents[1]


def _demo_source() -> str:
    return REPO_ROOT.joinpath("demo.py").read_text(encoding="utf-8")


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in demo.py")


def test_streamlit_skill_panel_has_zip_preview_and_explicit_approvals():
    source = _demo_source()

    assert "st.file_uploader(" in source
    assert 'type=["zip"]' in source
    assert "Install only skills you trust" in source
    assert "skill_install_approve_setup" in source
    assert "skill_install_approve_replace" in source


def test_streamlit_polling_does_not_block_in_a_while_loop():
    """A sleep or loop here would freeze the page, including its own cancel button."""
    tree = ast.parse(_demo_source())
    polling = _function_node(tree, "_poll_skill_installation")

    assert not any(isinstance(node, (ast.While, ast.AsyncFor)) for node in ast.walk(polling))
    calls = [
        node.func.attr
        for node in ast.walk(polling)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "sleep" not in calls


def test_status_widget_is_a_fragment_so_the_page_stays_interactive():
    """Auto-refresh must be scoped to the status widget, not the whole page."""
    tree = ast.parse(_demo_source())
    status = _function_node(tree, "_render_skill_installation_status")
    decorators = [
        ast.unparse(decorator)
        for node in ast.walk(status)
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
    ]

    assert any("st.fragment" in decorator for decorator in decorators)
    assert any("run_every" in decorator for decorator in decorators)


def test_archive_metadata_is_never_rendered_as_raw_html():
    tree = ast.parse(_demo_source())
    preview = _function_node(tree, "_render_skill_preview")

    for node in ast.walk(preview):
        if isinstance(node, ast.keyword) and node.arg == "unsafe_allow_html":
            raise AssertionError("archive-derived text must not be rendered as HTML")


@pytest.mark.parametrize(
    ("current", "expected"),
    [(None, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 3.0), (99.0, 3.0)],
)
def test_poll_backoff_steps_then_holds(demo_module, current, expected):
    assert demo_module._next_skill_poll_delay(current) == expected


# --- Polling behavior ------------------------------------------------------


def _running(**overrides) -> dict:
    payload = {"operationId": "op-a", "state": "running", "phase": "copying"}
    payload.update(overrides)
    return payload


def _succeeded(sync_status: str = "synced", generation: int = 7) -> dict:
    return {
        "operationId": "op-a",
        "state": "succeeded",
        "phase": "syncingCatalog",
        "result": {
            "action": "installed",
            "name": "demo",
            "sourceHash": SOURCE_HASH,
            "runtimeStatus": "ready",
            "catalog": {
                "deviceId": "device-123",
                "catalogGeneration": generation,
                "catalogSyncStatus": sync_status,
                "skills": [],
                "totalCount": 0,
                "enabledCount": 0,
            },
        },
    }


def test_poll_issues_at_most_one_request_per_rerun(demo_module, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        demo_module,
        "get_skill_installation",
        lambda operation_id: calls.append(operation_id) or _running(),
    )
    demo_module.st.session_state.skill_operation_id = "op-a"

    demo_module._poll_skill_installation()
    demo_module._poll_skill_installation()

    assert calls == ["op-a"]


def test_poll_resumes_after_the_backoff_window(demo_module, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        demo_module,
        "get_skill_installation",
        lambda operation_id: calls.append(operation_id) or _running(),
    )
    demo_module.st.session_state.skill_operation_id = "op-a"

    demo_module._poll_skill_installation()
    demo_module.st.session_state.skill_install_next_poll_at = 0.0
    demo_module._poll_skill_installation()

    assert len(calls) == 2


def test_poll_stops_once_the_operation_is_terminal(demo_module, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        demo_module,
        "get_skill_installation",
        lambda operation_id: calls.append(operation_id) or _succeeded(),
    )
    demo_module.st.session_state.skill_operation_id = "op-a"

    demo_module._poll_skill_installation()
    demo_module.st.session_state.skill_install_next_poll_at = 0.0
    demo_module._poll_skill_installation()

    assert calls == ["op-a"]


def test_poll_without_an_operation_does_nothing(demo_module, monkeypatch):
    monkeypatch.setattr(
        demo_module,
        "get_skill_installation",
        lambda operation_id: pytest.fail("must not poll without an operation"),
    )

    assert demo_module._poll_skill_installation() is None


def test_network_failure_keeps_the_receipt_and_retries(demo_module, monkeypatch):
    """A failed poll is not a failed installation."""
    monkeypatch.setattr(demo_module, "get_skill_installation", lambda operation_id: None)
    demo_module.st.session_state.skill_operation_id = "op-a"
    demo_module.st.session_state.skill_operation_state = _running()

    result = demo_module._poll_skill_installation()

    assert result["state"] == "running"
    assert demo_module.st.session_state.skill_operation_id == "op-a"


def test_success_applies_the_returned_catalog_and_invalidates_the_cache(demo_module, monkeypatch):
    monkeypatch.setattr(demo_module, "get_skill_installation", lambda operation_id: _succeeded())
    demo_module.st.session_state.skill_operation_id = "op-a"
    before = demo_module.st.session_state.api_cache_version

    demo_module._poll_skill_installation()

    catalog = demo_module.st.session_state.skill_operation_catalog
    assert catalog["catalogGeneration"] == 7
    assert demo_module.st.session_state.api_cache_version == before + 1


def test_an_older_generation_never_replaces_a_newer_cached_catalog(demo_module):
    demo_module.st.session_state.skill_operation_catalog = {"catalogGeneration": 9}

    demo_module._apply_installed_skill_catalog(_succeeded(generation=4))

    assert demo_module.st.session_state.skill_operation_catalog["catalogGeneration"] == 9


@pytest.mark.parametrize("sync_status", ["synced", "pending", "disconnected"])
def test_each_sync_status_has_distinct_user_facing_copy(demo_module, sync_status):
    message, _icon = demo_module._SYNC_STATUS_COPY[sync_status]

    assert message
    others = {
        text for status, (text, _) in demo_module._SYNC_STATUS_COPY.items() if status != sync_status
    }
    assert message not in others


def test_pending_sync_is_never_described_as_a_failure(demo_module):
    message, _icon = demo_module._SYNC_STATUS_COPY["pending"]

    assert "fail" not in message.lower()
    assert "error" not in message.lower()
    assert "Installed locally" in message


def test_configured_root_collision_is_not_replaceable(demo_module, monkeypatch):
    rendered: list[str] = []
    monkeypatch.setattr(
        demo_module.st,
        "error",
        lambda message, **_kwargs: rendered.append(message),
        raising=False,
    )

    replaceable = demo_module._render_existing_skill_notice(
        {"name": "demo", "replaceable": False, "sourceHash": SOURCE_HASH}
    )

    assert replaceable is False
    assert "configured skills folder" in rendered[0]


def test_profile_installed_collision_offers_an_update(demo_module, monkeypatch):
    monkeypatch.setattr(demo_module.st, "info", lambda *args, **kwargs: None, raising=False)

    replaceable = demo_module._render_existing_skill_notice(
        {"name": "demo", "replaceable": True, "sourceHash": SOURCE_HASH}
    )

    assert replaceable is True


def test_no_collision_needs_no_update_confirmation(demo_module):
    assert demo_module._render_existing_skill_notice(None) is False
