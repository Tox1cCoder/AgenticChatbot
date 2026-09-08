"""The Streamlit Continue/Stop state machine.

What the buttons show is a pure function of the last canonical snapshot the
stream published. That is the whole design, and these tests are written against
it, because the alternative the demo used to implement — inferring state from
whether an HTTP connection was still open — cannot distinguish "the worker
stopped" from "the socket closed", and those need different answers.

Three properties carry the rest:

* **A pending Stop is not a stop.** ``stop_requested`` disables the button and
  says so; it does not report "Generation stopped", because the worker may be
  mid-provider-call in another process.
* **Continue is offered only when the server would honour it.** The snapshot
  has to carry a live continuation id, so an unavailable one shows a reason
  rather than a button that 409s.
* **Repeated clicks replay one command.** Streamlit reruns the whole script on
  every interaction, so an idempotency key held in a local is regenerated each
  time — which is exactly how a control that is "safe because idempotent"
  stops being idempotent.
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

    def update(self, *args: Any, **kwargs: Any) -> None:
        return None


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

    def chat_message(self, *args: Any, **kwargs: Any) -> _Context:
        return _Context()

    def status(self, *args: Any, **kwargs: Any) -> _Context:
        return _Context()

    def empty(self, *args: Any, **kwargs: Any) -> _Context:
        return _Context()

    def button(self, label: str, *args: Any, **kwargs: Any) -> bool:
        key = kwargs.get("key")
        self.buttons.append({"label": label, **kwargs})
        return bool(self.button_returns.get(key, False))

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


CONVERSATION_ID = "827faf55-1041-4357-9e1a-0f7d031fa546"
GENERATION_ID = "33333333-3333-3333-3333-333333333333"
CONTINUATION_ID = "44444444-4444-4444-4444-444444444444"


def _snapshot(status: str, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generation_id": GENERATION_ID,
        "logical_turn_id": "turn-1",
        "conversation_id": CONVERSATION_ID,
        "status": status,
        "version": 3,
        "execution_epoch": 0,
        "continuation_id": None,
        "continuation_available": False,
        "continuation_block_reason": None,
        "assistant_message_id": None,
        "terminal_reason": None,
    }
    payload.update(overrides)
    return payload


def _resumable(status: str, **overrides: Any) -> dict[str, Any]:
    return _snapshot(
        status,
        continuation_available=True,
        continuation_id=CONTINUATION_ID,
        **overrides,
    )


# ----------------------------------------------------------------------
# the status table, row by row
# ----------------------------------------------------------------------


@pytest.mark.parametrize("status", ["starting", "running", "continuing", "finalizing_after_limit"])
def test_an_active_turn_offers_stop_and_hides_continue(monkeypatch, status):
    demo, _ = _import_demo(monkeypatch)

    assert demo.generation_controls(_snapshot(status)) == {
        "stop": "enabled",
        "continue": "hidden",
    }


def test_a_pending_stop_disables_the_button_rather_than_hiding_it(monkeypatch):
    """Hiding would read as "the turn ended". Disabled says "asked, waiting"."""
    demo, _ = _import_demo(monkeypatch)

    assert demo.generation_controls(_snapshot("stop_requested")) == {
        "stop": "disabled",
        "continue": "hidden",
    }


def test_a_paused_turn_offers_both(monkeypatch):
    """Stop here accepts the validated partial; nothing is cancelled."""
    demo, _ = _import_demo(monkeypatch)

    assert demo.generation_controls(_resumable("continuable")) == {
        "stop": "enabled",
        "continue": "enabled",
    }


def test_a_stopped_but_resumable_turn_offers_only_continue(monkeypatch):
    demo, _ = _import_demo(monkeypatch)

    assert demo.generation_controls(_resumable("stopped")) == {
        "stop": "hidden",
        "continue": "enabled",
    }


def test_a_stopped_turn_that_cannot_resume_offers_nothing(monkeypatch):
    """An unknown mutation outcome blocks continuation, and this is where it shows."""
    demo, _ = _import_demo(monkeypatch)

    assert demo.generation_controls(
        _snapshot("stopped", continuation_block_reason="mutation_outcome_unknown")
    ) == {"stop": "hidden", "continue": "hidden"}


@pytest.mark.parametrize("status", ["completed", "completed_partial", "failed"])
def test_a_terminal_turn_shows_no_controls(monkeypatch, status):
    demo, _ = _import_demo(monkeypatch)

    assert demo.generation_controls(_snapshot(status)) == {
        "stop": "hidden",
        "continue": "hidden",
    }


def test_a_continuable_turn_whose_id_is_spent_does_not_offer_continue(monkeypatch):
    """``continuation_available`` alone is not enough — the id is the thing redeemed."""
    demo, _ = _import_demo(monkeypatch)

    assert (
        demo.generation_controls(
            _snapshot("continuable", continuation_available=True, continuation_id=None)
        )["continue"]
        == "hidden"
    )


def test_no_snapshot_at_all_shows_no_controls(monkeypatch):
    demo, _ = _import_demo(monkeypatch)

    assert demo.generation_controls(None) == {"stop": "hidden", "continue": "hidden"}
    assert demo.generation_controls({}) == {"stop": "hidden", "continue": "hidden"}


def test_every_declared_status_is_covered_by_the_table(monkeypatch):
    """A status the table forgets falls through to "no controls" silently.

    That is the failure mode worth guarding: a new lifecycle status would make
    both buttons disappear from a turn that is genuinely still running.
    """
    demo, _ = _import_demo(monkeypatch)
    from app.models.generation import GenerationStatus

    declared = {member.value for member in GenerationStatus}
    covered = (
        set(demo.GENERATION_ACTIVE_STATUSES)
        | set(demo.GENERATION_TERMINAL_STATUSES)
        | {"stop_requested", "continuable", "stopped"}
    )
    assert declared == covered, f"uncovered statuses: {declared - covered}"


# ----------------------------------------------------------------------
# reading the lifecycle off the stream
# ----------------------------------------------------------------------


def test_the_snapshot_is_built_from_the_lifecycle_events(monkeypatch):
    demo, stub = _import_demo(monkeypatch)

    demo._apply_generation_event(
        {
            "type": "generation_start",
            "generation_id": GENERATION_ID,
            "status": "running",
            "version": 2,
            "execution_epoch": 0,
        }
    )

    snapshot = stub.session_state.generation_snapshot
    assert snapshot["status"] == "running"
    assert snapshot["version"] == 2


def test_a_status_event_does_not_erase_a_continuation_it_omits(monkeypatch):
    """Partial updates must merge, not replace.

    A ``generation_status`` carrying only a status would otherwise wipe the
    continuation id an earlier ``continuation_available`` established, and the
    Continue button would vanish for no reason the user could see.
    """
    demo, stub = _import_demo(monkeypatch)
    demo._apply_generation_event(
        {
            "generation_id": GENERATION_ID,
            "status": "continuable",
            "version": 4,
            "continuation_id": CONTINUATION_ID,
            "continuation_available": True,
        }
    )

    demo._apply_generation_event({"status": "continuable", "version": 4})

    snapshot = stub.session_state.generation_snapshot
    assert snapshot["continuation_id"] == CONTINUATION_ID
    assert demo.generation_controls(snapshot)["continue"] == "enabled"


def test_a_closed_stream_changes_no_status(monkeypatch):
    """The stream simply ending is not a lifecycle transition."""
    demo, stub = _import_demo(monkeypatch)
    demo._apply_generation_event(
        {"generation_id": GENERATION_ID, "status": "running", "version": 2}
    )

    outcome = demo.consume_continue_stream(iter([]))

    assert outcome["message"] is None
    assert stub.session_state.generation_snapshot["status"] == "running"


def test_clearing_inflight_state_keeps_the_snapshot(monkeypatch):
    """A paused turn is not in flight but is still continuable.

    Clearing the snapshot alongside the stream state is what would take the
    Continue button away with it.
    """
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.generation_snapshot = _resumable("continuable")

    demo._clear_inflight_state()

    assert stub.session_state.stream_inflight is False
    assert stub.session_state.generation_snapshot["continuation_id"] == CONTINUATION_ID


# ----------------------------------------------------------------------
# the Continue request
# ----------------------------------------------------------------------


def test_the_continue_request_carries_the_server_issued_identity(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.generation_snapshot = _resumable("continuable")

    body = demo.build_continue_request(stub.session_state.generation_snapshot, CONVERSATION_ID)

    assert body["generationId"] == GENERATION_ID
    assert body["continuationId"] == CONTINUATION_ID
    assert body["expectedVersion"] == 3
    assert body["conversationId"] == CONVERSATION_ID
    assert body["idempotencyKey"]


def test_a_continue_request_without_a_continuation_is_refused_not_guessed(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.generation_snapshot = _snapshot("continuable")

    assert (
        demo.build_continue_request(stub.session_state.generation_snapshot, CONVERSATION_ID) is None
    )


def test_a_continue_request_for_the_pending_new_sentinel_is_refused(monkeypatch):
    """``pending_new`` is not a UUID; the server answers 422."""
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.generation_snapshot = _resumable("continuable")

    assert (
        demo.build_continue_request(stub.session_state.generation_snapshot, "pending_new") is None
    )


def test_repeated_clicks_reuse_one_idempotency_key(monkeypatch):
    """Streamlit reruns the script on every click.

    A key generated per render would make each click a new command, and the
    ledger could not collapse them.
    """
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.generation_snapshot = _resumable("continuable")

    first = demo.build_continue_request(stub.session_state.generation_snapshot, CONVERSATION_ID)
    second = demo.build_continue_request(stub.session_state.generation_snapshot, CONVERSATION_ID)

    assert first["idempotencyKey"] == second["idempotencyKey"]


def test_a_new_version_earns_a_new_key(monkeypatch):
    """The fence moved, so this is a different command, not a replay."""
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.generation_snapshot = _resumable("continuable")
    first = demo.build_continue_request(stub.session_state.generation_snapshot, CONVERSATION_ID)

    stub.session_state.generation_snapshot = _resumable("continuable", version=4)
    second = demo.build_continue_request(stub.session_state.generation_snapshot, CONVERSATION_ID)

    assert first["idempotencyKey"] != second["idempotencyKey"]


def test_stop_and_continue_never_share_a_key(monkeypatch):
    """One key, one action: the ledger refuses a key reused for another command."""
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.generation_snapshot = _resumable("continuable")

    assert demo._generation_command_key("stop") != demo._generation_command_key("continue")


# ----------------------------------------------------------------------
# consuming a continued epoch
# ----------------------------------------------------------------------


def test_a_continued_epoch_creates_no_user_message(monkeypatch):
    """Continue asks nothing new, so it must not look like a new turn."""
    demo, _ = _import_demo(monkeypatch)

    outcome = demo.consume_continue_stream(
        iter(
            [
                {
                    "type": "generation_start",
                    "generation_id": GENERATION_ID,
                    "status": "continuing",
                },
                {"type": "token", "content": "more "},
                {"type": "token", "content": "answer"},
                {"type": "complete", "message": {"id": "m2", "content": "more answer"}},
            ]
        )
    )

    assert outcome["text"] == "more answer"
    assert outcome["message"] == {"id": "m2", "content": "more answer"}
    assert outcome["error"] is None


def test_a_second_pause_is_reported_rather_than_treated_as_completion(monkeypatch):
    demo, stub = _import_demo(monkeypatch)

    outcome = demo.consume_continue_stream(
        iter(
            [
                {"type": "token", "content": "still partial"},
                {"type": "message_end", "message": {"id": "m3", "content": "still partial"}},
                {
                    "type": "continuation_available",
                    "generation_id": GENERATION_ID,
                    "status": "continuable",
                    "version": 6,
                    "continuation_id": CONTINUATION_ID,
                    "continuation_available": True,
                },
            ]
        )
    )

    assert outcome["paused"] is True
    assert outcome["message"] == {"id": "m3", "content": "still partial"}
    # And the new continuation is immediately offerable.
    assert demo.generation_controls(stub.session_state.generation_snapshot)["continue"] == "enabled"


def test_a_refused_continuation_surfaces_its_error(monkeypatch):
    demo, _ = _import_demo(monkeypatch)

    outcome = demo.consume_continue_stream(
        iter([{"type": "error", "error": "the continuation has already been used"}])
    )

    assert outcome["error"] == "the continuation has already been used"
    assert outcome["message"] is None


# ----------------------------------------------------------------------
# reconciling an unsettled Stop
# ----------------------------------------------------------------------


def test_a_pending_stop_is_reported_as_pending_not_as_stopped(monkeypatch):
    """The claim this control must never make."""
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.stream_inflight = True
    stub.session_state.stream_conversation_id = CONVERSATION_ID
    stub.session_state.stream_user_message_id = "11111111-2222-3333-4444-555555555555"
    stub.session_state.stream_partial_text = ""
    stub.session_state.messages = []
    stub.session_state.pending_image_attachments = []
    stub.session_state.generation_snapshot = _snapshot("running")

    monkeypatch.setattr(
        demo,
        "make_api_request",
        lambda *a, **k: {
            "success": True,
            "data": {
                "status": "stop_requested",
                "message": None,
                "generation": _snapshot("stop_requested", version=4),
            },
        },
    )

    with pytest.raises(RuntimeError, match="rerun"):
        demo._handle_stop_rerun(CONVERSATION_ID)

    assert any("Stop requested" in toast for toast in stub.toasts)
    assert not any("Generation stopped" in toast for toast in stub.toasts)
    assert stub.session_state.generation_snapshot["status"] == "stop_requested"


def test_a_stop_sends_the_fence_when_the_stream_published_one(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.stream_inflight = True
    stub.session_state.stream_conversation_id = CONVERSATION_ID
    stub.session_state.stream_user_message_id = "11111111-2222-3333-4444-555555555555"
    stub.session_state.stream_partial_text = ""
    stub.session_state.messages = []
    stub.session_state.pending_image_attachments = []
    stub.session_state.generation_snapshot = _snapshot("running", version=5)

    calls: list[dict[str, Any]] = []

    def _record(method: str, endpoint: str, data: dict | None = None, **_kwargs) -> dict:
        calls.append({"endpoint": endpoint, "data": data})
        return {"success": True, "data": {"status": "cancelled", "message": None}}

    monkeypatch.setattr(demo, "make_api_request", _record)

    with pytest.raises(RuntimeError, match="rerun"):
        demo._handle_stop_rerun(CONVERSATION_ID)

    body = calls[0]["data"]
    assert body["generationId"] == GENERATION_ID
    assert body["expectedVersion"] == 5
    assert body["idempotencyKey"]
    # The turn-scoped fallback is not sent alongside a generation id.
    assert "userMessageId" not in body


def test_a_stop_falls_back_to_the_user_message_before_run_start(monkeypatch):
    """A turn that started before the lifecycle event arrived is still stoppable."""
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.stream_inflight = True
    stub.session_state.stream_conversation_id = CONVERSATION_ID
    stub.session_state.stream_user_message_id = "11111111-2222-3333-4444-555555555555"
    stub.session_state.stream_partial_text = ""
    stub.session_state.messages = []
    stub.session_state.pending_image_attachments = []
    stub.session_state.generation_snapshot = {}

    calls: list[dict[str, Any]] = []

    def _record(method: str, endpoint: str, data: dict | None = None, **_kwargs) -> dict:
        calls.append({"endpoint": endpoint, "data": data})
        return {"success": True, "data": {"status": "cancelled", "message": None}}

    monkeypatch.setattr(demo, "make_api_request", _record)

    with pytest.raises(RuntimeError, match="rerun"):
        demo._handle_stop_rerun(CONVERSATION_ID)

    body = calls[0]["data"]
    assert body["userMessageId"] == "11111111-2222-3333-4444-555555555555"
    assert "generationId" not in body


def test_refreshing_the_snapshot_asks_the_server_rather_than_assuming(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.generation_snapshot = _snapshot("stop_requested")

    calls: list[dict[str, Any]] = []

    def _record(method: str, endpoint: str, data: dict | None = None, **kwargs) -> dict:
        calls.append({"method": method, "endpoint": endpoint, "kwargs": kwargs})
        return {
            "success": True,
            "data": {
                "generationId": GENERATION_ID,
                "status": "stopped",
                "version": 5,
                "continuationAvailable": True,
                "continuationId": CONTINUATION_ID,
            },
        }

    monkeypatch.setattr(demo, "make_api_request", _record)

    refreshed = demo.refresh_generation_snapshot(CONVERSATION_ID)

    assert calls[0]["method"] == "GET"
    assert f"/messages/generations/{GENERATION_ID}" in calls[0]["endpoint"]
    assert f"conversation_id={CONVERSATION_ID}" in calls[0]["endpoint"]
    # A polled value must never come from cache.
    assert calls[0]["kwargs"].get("use_cache") is False
    assert refreshed["status"] == "stopped"
    assert demo.generation_controls(refreshed)["continue"] == "enabled"


def test_the_snapshot_refresh_needs_no_guessing_when_there_is_nothing_to_read(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.generation_snapshot = {}

    monkeypatch.setattr(
        demo,
        "make_api_request",
        lambda *a, **k: pytest.fail("refresh called the API with no generation to read"),
    )

    assert demo.refresh_generation_snapshot(CONVERSATION_ID) is None


# ----------------------------------------------------------------------
# the rendered control
# ----------------------------------------------------------------------


def test_the_continue_button_appears_only_for_a_resumable_turn(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.generation_snapshot = _resumable("continuable")

    demo._render_continue_control(CONVERSATION_ID)

    assert any(button.get("key") == "continue_generation_btn" for button in stub.buttons)


def test_no_continue_button_while_the_turn_is_still_running(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.generation_snapshot = _snapshot("running")

    demo._render_continue_control(CONVERSATION_ID)

    assert stub.buttons == []


def test_a_blocked_continuation_explains_itself_instead_of_hiding(monkeypatch):
    """An absent button with no reason is indistinguishable from a bug."""
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.generation_snapshot = _snapshot(
        "stopped", continuation_block_reason="mutation_outcome_unknown"
    )

    demo._render_continue_control(CONVERSATION_ID)

    assert stub.buttons == []
    assert any("mutation_outcome_unknown" in caption for caption in stub.captions)


def test_the_continue_control_is_rendered_outside_the_message_form():
    """Inside the form, clicking it would submit a draft the user was typing."""
    source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")
    trigger = source.index("_render_continue_control(str(conversation_id))")
    form_start = source.index("# Message form")
    # The call site sits in the same block as the stop-rerun handler, which is
    # itself outside the form.
    assert trigger > form_start
    assert "_handle_stop_rerun(stoppable_conversation_id)" in source[form_start:trigger]
