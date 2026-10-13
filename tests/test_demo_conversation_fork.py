"""The demo's "Fork conversation" action.

The server owns correctness (cutoff validation, copying, idempotent replay). What
the UI owns, and what these tests pin, is navigation: one request id per action,
a clean hand-off to the child with none of the source's Continue/Stop/Approve or
widget state, and a source that stays fully usable when the fork fails.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest
import requests

from app.ui import conversation_fork as fork

SOURCE_ID = "827faf55-1041-4357-9e1a-0f7d031fa546"
CHILD_ID = "11111111-1111-4111-8111-111111111111"
MESSAGE_ID = "22222222-2222-4222-8222-222222222222"


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
        self.buttons: list[dict[str, Any]] = []
        self.captions: list[str] = []
        self.errors: list[str] = []
        self.toasts: list[str] = []
        self.button_returns: dict[str, bool] = {}

    def set_page_config(self, *args: Any, **kwargs: Any) -> None:
        return None

    def button(self, label: str, *args: Any, **kwargs: Any) -> bool:
        self.buttons.append({"label": label, **kwargs})
        return bool(self.button_returns.get(kwargs.get("key"), False))

    def caption(self, text: str, *args: Any, **kwargs: Any) -> None:
        self.captions.append(text)

    def error(self, text: str, *args: Any, **kwargs: Any) -> None:
        self.errors.append(text)

    def toast(self, text: str, *args: Any, **kwargs: Any) -> None:
        self.toasts.append(text)

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
        lambda **kwargs_: kwargs_.get("default")
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


def _child() -> dict[str, Any]:
    return {
        "id": CHILD_ID,
        "title": "Source (fork)",
        "forkedFromConversationId": SOURCE_ID,
        "forkedFromMessageId": MESSAGE_ID,
        "forkedThroughSequence": 4,
    }


class _Api:
    """Records requests; replies come from ``responses`` in order."""

    def __init__(self, responses: list[dict[str, Any]], st: _StreamlitStub) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.responses = responses
        self.st = st
        self.error_code: str | None = None

    def __call__(self, method: str, endpoint: str, data: dict | None = None, **_: Any) -> dict:
        self.calls.append((method, endpoint, data))
        self.st.session_state["_last_api_error_code"] = self.error_code
        return self.responses.pop(0) if self.responses else {}

    def fork_calls(self) -> list[dict[str, Any] | None]:
        return [d for _, endpoint, d in self.calls if endpoint.endswith("/fork")]


@pytest.fixture
def harness(monkeypatch):
    demo, st = _import_demo(monkeypatch)
    state = st.session_state
    state.conversations_list = [{"id": SOURCE_ID, "title": "Source"}]
    state.current_conversation_id = SOURCE_ID
    state.messages = [{"id": MESSAGE_ID, "content": "UI page only"}]
    state.conversation_messages_page = 3
    state.generation_snapshot = {
        "generation_id": "gen-1",
        "conversation_id": SOURCE_ID,
        "status": "continuable",
        "version": 5,
        "continuation_available": True,
        "continuation_id": "cont-1",
    }
    state.live_widget_mounts = {f"{MESSAGE_ID}:w1": True}
    state.pending_interrupt = {"interrupt_id": "i-1"}
    state.interrupt_conversation_id = SOURCE_ID
    state.pending_image_attachments = [{"name": "a.png"}]
    state.message_image_thumbnails = {MESSAGE_ID: [{"name": "a.png"}]}
    state.stream_inflight = False
    state.stream_conversation_id = ""
    state.stream_user_message_id = ""
    refreshed: list[bool] = []
    monkeypatch.setattr(demo, "refresh_conversations_list", lambda: refreshed.append(True))
    api = _Api([{"success": True, "data": _child()}], st)
    real_make_api_request = demo.make_api_request
    monkeypatch.setattr(demo, "make_api_request", api)
    return types.SimpleNamespace(
        demo=demo,
        st=st,
        api=api,
        refreshed=refreshed,
        state=state,
        real_make_api_request=real_make_api_request,
    )


def test_success_navigates_to_the_child_and_reloads_server_history(harness):
    assert harness.demo.fork_conversation_from_message(SOURCE_ID, MESSAGE_ID) is True

    state = harness.state
    method, endpoint, body = harness.api.calls[0]
    assert (method, endpoint) == ("POST", f"/conversations/{SOURCE_ID}/fork")
    assert body["messageId"] == MESSAGE_ID and body["requestId"]
    assert "title" not in body
    assert state.current_conversation_id == CHILD_ID
    assert state.conversations_list[0]["id"] == CHILD_ID
    assert state.messages == []
    assert state.conversation_messages_page == 0
    assert harness.refreshed == [True]
    assert state.pending_image_attachments == []
    assert state.message_image_thumbnails == {}


def test_child_inherits_no_source_generation_widget_or_approval_state(harness):
    harness.demo.fork_conversation_from_message(SOURCE_ID, MESSAGE_ID)

    state = harness.state
    assert harness.demo.generation_snapshot_for(CHILD_ID) == {}
    assert harness.demo.generation_controls(harness.demo.generation_snapshot_for(CHILD_ID)) == {
        "stop": "hidden",
        "continue": "hidden",
    }
    assert harness.demo.build_continue_request(state.generation_snapshot, CHILD_ID) is None
    assert state.live_widget_mounts == {}
    assert "pending_interrupt" not in state
    assert "interrupt_conversation_id" not in state


def test_returning_to_the_source_still_offers_its_paused_turn_continue(harness):
    harness.demo.fork_conversation_from_message(SOURCE_ID, MESSAGE_ID)
    harness.state.current_conversation_id = SOURCE_ID
    harness.demo._navigate_away_from_source()

    snapshot = harness.demo.generation_snapshot_for(SOURCE_ID)
    assert snapshot["generation_id"] == "gen-1"
    assert harness.demo.generation_controls(snapshot)["continue"] == "enabled"
    body = harness.demo.build_continue_request(snapshot, SOURCE_ID)
    assert body is not None and body["generationId"] == "gen-1"


def test_an_ordinary_switch_does_not_carry_continue_into_another_conversation(harness):
    harness.state.current_conversation_id = CHILD_ID
    harness.demo.reset_conversation_state()

    assert harness.demo.generation_controls(harness.demo.generation_snapshot_for(CHILD_ID)) == {
        "stop": "hidden",
        "continue": "hidden",
    }
    assert harness.demo.generation_snapshot_for(SOURCE_ID)["continuation_id"] == "cont-1"


def test_the_continue_button_renders_only_in_the_paused_turns_conversation(harness):
    def continue_buttons() -> list[dict[str, Any]]:
        return [b for b in harness.st.buttons if b.get("key") == "continue_generation_btn"]

    harness.demo._render_continue_control(CHILD_ID)
    assert continue_buttons() == []

    harness.demo._render_continue_control(SOURCE_ID)
    assert len(continue_buttons()) == 1


def test_a_snapshot_without_a_conversation_belongs_to_none(harness):
    harness.state.generation_snapshot = {"generation_id": "gen-1", "continuation_id": "c"}

    assert harness.demo.generation_snapshot_for(SOURCE_ID) == {}
    assert harness.demo.build_continue_request(harness.state.generation_snapshot, SOURCE_ID) is None


def test_failure_leaves_the_source_selection_and_state_usable(harness):
    harness.api.responses = [{}]
    snapshot = dict(harness.state.generation_snapshot)

    assert harness.demo.fork_conversation_from_message(SOURCE_ID, MESSAGE_ID) is False

    state = harness.state
    assert state.current_conversation_id == SOURCE_ID
    assert state.generation_snapshot == snapshot
    assert state.live_widget_mounts == {f"{MESSAGE_ID}:w1": True}
    assert state.pending_interrupt == {"interrupt_id": "i-1"}
    assert state.messages == [{"id": MESSAGE_ID, "content": "UI page only"}]
    assert state.conversation_messages_page == 3
    assert harness.refreshed == []


def test_a_retry_reuses_the_request_id_and_success_retires_it(harness):
    harness.api.responses = [{}, {"success": True, "data": _child()}]

    harness.demo.fork_conversation_from_message(SOURCE_ID, MESSAGE_ID)
    harness.demo.fork_conversation_from_message(SOURCE_ID, MESSAGE_ID)

    first, second = harness.api.fork_calls()
    assert first["requestId"] == second["requestId"]
    assert harness.state.fork_attempts == {}


def test_a_terminal_error_forces_a_fresh_request_id_next_time(harness):
    harness.api.responses = [{}, {"success": True, "data": _child()}]
    harness.api.error_code = "FORK_RESULT_DELETED"

    harness.demo.fork_conversation_from_message(SOURCE_ID, MESSAGE_ID)
    harness.demo.fork_conversation_from_message(SOURCE_ID, MESSAGE_ID)

    first, second = harness.api.fork_calls()
    assert first["requestId"] != second["requestId"]


def test_a_second_submit_while_pending_sends_nothing(harness):
    attempts = harness.state.setdefault(fork.FORK_ATTEMPTS_KEY, {})
    assert fork.acquire_fork_request(attempts, SOURCE_ID, MESSAGE_ID) is not None

    assert harness.demo.fork_conversation_from_message(SOURCE_ID, MESSAGE_ID) is False

    assert harness.api.calls == []


def test_a_paused_source_with_stale_stream_state_is_never_stopped(harness):
    """The server forks only when no turn is active, so a stop could only hit a paused
    (continuable) source generation and mutate it."""
    state = harness.state
    state.stream_inflight = True
    state.stream_conversation_id = SOURCE_ID
    state.stream_user_message_id = "user-msg-1"

    assert harness.demo.fork_conversation_from_message(SOURCE_ID, MESSAGE_ID) is True

    assert [e for _, e, _ in harness.api.calls] == [f"/conversations/{SOURCE_ID}/fork"]
    assert state.current_conversation_id == CHILD_ID
    assert state.stream_inflight is False
    assert harness.demo.generation_snapshot_for(CHILD_ID) == {}
    assert state.live_widget_mounts == {}
    assert "pending_interrupt" not in state


def test_same_conversation_inflight_clear_still_keeps_the_snapshot(harness):
    harness.demo._clear_inflight_state()

    assert harness.state.generation_snapshot["generation_id"] == "gen-1"


def test_lineage_links_to_a_source_still_in_the_list():
    lineage = fork.fork_lineage(_child(), [{"id": SOURCE_ID, "title": "Source"}])

    assert lineage == fork.ForkLineage(SOURCE_ID, "Source", linked=True)
    assert fork.lineage_label(lineage) == "Forked from Source"


def test_lineage_for_a_deleted_source_is_a_plain_historical_label():
    child = {**_child(), "forkedFromConversationId": None, "forkedFromMessageId": None}

    lineage = fork.fork_lineage(child, [])

    assert lineage is not None and lineage.linked is False
    assert fork.lineage_label(lineage) == "Forked from a deleted conversation"


def test_a_conversation_that_is_not_a_fork_has_no_lineage():
    assert fork.fork_lineage({"id": SOURCE_ID}, []) is None
    assert fork.fork_lineage(None, []) is None


def test_lineage_label_renders_a_switch_button_only_when_linked(harness):
    harness.st.button_returns[f"fork_lineage_open_{CHILD_ID}"] = True
    harness.state.conversations_list = [{"id": SOURCE_ID, "title": "Source"}, _child()]

    with pytest.raises(RuntimeError, match="rerun"):
        harness.demo.render_fork_lineage(_child())

    assert harness.state.current_conversation_id == SOURCE_ID
    assert harness.demo.generation_snapshot_for(SOURCE_ID)["generation_id"] == "gen-1"

    harness.st.buttons.clear()
    harness.state.conversations_list = [_child()]
    harness.demo.render_fork_lineage({**_child(), "forkedFromConversationId": None})
    assert harness.st.buttons == []
    assert "Forked from a deleted conversation" in harness.st.captions


@pytest.mark.parametrize(
    ("message", "metadata", "is_user", "expected"),
    [
        ({"id": MESSAGE_ID, "content": "answer"}, {}, False, True),
        ({"id": MESSAGE_ID, "content": "question"}, {}, True, False),
        ({"content": "unsaved streamed text"}, {}, False, False),
        ({"id": MESSAGE_ID, "content": ""}, {}, False, False),
        ({"id": MESSAGE_ID, "content": "x"}, {"paused": True}, False, False),
        # A copied reply the server recorded as no valid fork point answers 409.
        ({"id": MESSAGE_ID, "content": "x"}, {"fork_cutoff_eligible": False}, False, False),
        ({"id": MESSAGE_ID, "content": "x"}, {"fork_cutoff_eligible": True}, False, True),
        ({"id": MESSAGE_ID, "content": "x"}, {"fork_history": True}, False, True),
    ],
)
def test_control_visibility_rules(message, metadata, is_user, expected):
    assert fork.can_fork_message(message, metadata, is_user=is_user) is expected


def test_the_control_renders_only_for_forkable_rows(harness):
    msg = {"id": MESSAGE_ID, "content": "answer"}

    harness.demo.render_fork_control(msg, {}, SOURCE_ID, is_user=True)
    assert harness.st.buttons == []

    harness.demo.render_fork_control(msg, {}, SOURCE_ID, is_user=False)
    assert [b["label"] for b in harness.st.buttons] == ["Fork conversation"]

    harness.st.buttons.clear()
    copied = {"fork_history": True, "fork_cutoff_eligible": False}
    harness.demo.render_fork_control(msg, copied, SOURCE_ID, is_user=False)
    assert harness.st.buttons == []


class _Response:
    """What ``requests`` hands back for one HTTP status and JSON body."""

    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self.payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self) -> dict[str, Any]:
        return self.payload


def test_a_failed_fork_shows_one_message(harness, monkeypatch):
    """The shared request helper already reports the server's message; no second one."""
    payload = {
        "success": False,
        "code": "FORK_POINT_NOT_COMPLETED",
        "message": "Only a completed assistant reply can be forked",
    }
    session = types.SimpleNamespace(request=lambda *_args, **_kwargs: _Response(409, payload))
    monkeypatch.setattr(harness.demo, "make_api_request", harness.real_make_api_request)
    monkeypatch.setattr(harness.demo, "get_http_session", lambda: session)
    harness.st.button_returns[f"fork_conversation_{MESSAGE_ID}"] = True

    harness.demo.render_fork_control(
        {"id": MESSAGE_ID, "content": "answer"}, {}, SOURCE_ID, is_user=False
    )

    assert harness.st.toasts + harness.st.errors == [
        "Only a completed assistant reply can be forked"
    ]
    assert harness.state.current_conversation_id == SOURCE_ID


def test_the_control_is_disabled_while_its_fork_is_pending(harness):
    attempts = harness.state.setdefault(fork.FORK_ATTEMPTS_KEY, {})
    fork.acquire_fork_request(attempts, SOURCE_ID, MESSAGE_ID)

    harness.demo.render_fork_control(
        {"id": MESSAGE_ID, "content": "answer"}, {}, SOURCE_ID, is_user=False
    )

    assert harness.st.buttons[0]["disabled"] is True
