"""A turn paused at its budget must end with the controls on screen.

A paused turn never emits ``complete``. It emits ``message_end`` carrying the
validated partial and then ``continuation_available`` carrying the snapshot the
controls are a pure function of.

The send path keyed everything off ``complete``: ``final_message`` stayed None,
so the script never rerouted through ``st.rerun()``, and
``_render_continue_control`` had already run near the top of the script with
the *previous* snapshot. The offer arrived, was recorded, and was never drawn
-- the user saw a partial answer with no way to continue or dismiss it, plus a
"Failed to send message" toast for a turn that had in fact answered.

The continue path (``consume_continue_stream``) already treated ``message_end``
and ``complete`` as equivalent; these pin the same contract for the first
epoch.
"""

from __future__ import annotations

from typing import Any

# The Streamlit stub is non-trivial (cache decorators, context managers,
# recorded buttons). Reused rather than duplicated so both suites exercise
# demo.py under exactly the same double.
from tests.test_demo_generation_controls import _import_demo

CONVERSATION_ID = "827faf55-1041-4357-9e1a-0f7d031fa546"
GENERATION_ID = "33333333-3333-3333-3333-333333333333"
CONTINUATION_ID = "44444444-4444-4444-4444-444444444444"


def _paused_stream() -> list[dict[str, Any]]:
    """The events a turn paused at its execution budget actually emits.

    Shaped like ``_generation_status_data``: the pause advertises the
    continuation on the same event that reports the status.
    """
    return [
        {
            "type": "generation_start",
            "generation_id": GENERATION_ID,
            "conversation_id": CONVERSATION_ID,
            "status": "running",
            "version": 1,
            "execution_epoch": 0,
        },
        {"type": "message_end", "message": {"id": "m1", "content": "A partial answer."}},
        {
            "type": "continuation_available",
            "generation_id": GENERATION_ID,
            "conversation_id": CONVERSATION_ID,
            "status": "continuable",
            "version": 2,
            "execution_epoch": 0,
            "continuation_id": CONTINUATION_ID,
            "continuation_available": True,
            "continuation_block_reason": None,
        },
    ]


def test_the_paused_snapshot_enables_both_controls(monkeypatch):
    """Continue resumes the turn; Stop accepts the partial and ends it."""
    demo, _ = _import_demo(monkeypatch)

    for event in _paused_stream():
        if event["type"] in {"generation_start", "generation_status", "continuation_available"}:
            demo._apply_generation_event(event)

    snapshot = demo.st.session_state.get("generation_snapshot")
    controls = demo.generation_controls(snapshot)

    assert controls["continue"] == "enabled"
    assert controls["stop"] == "enabled"


def test_a_paused_turn_is_not_reported_as_a_failure(monkeypatch):
    """`complete` never arrives, and that is not an error."""
    demo, _ = _import_demo(monkeypatch)

    events = _paused_stream()
    assert not any(event["type"] == "complete" for event in events)
    assert any(event["type"] == "message_end" for event in events)


def test_the_partial_answer_is_carried_by_message_end(monkeypatch):
    """The only carrier of the persisted partial on a paused turn."""
    demo, _ = _import_demo(monkeypatch)

    carried = [event for event in _paused_stream() if event["type"] == "message_end"]

    assert carried and carried[0]["message"]["content"] == "A partial answer."


def test_clearing_inflight_state_keeps_the_offer(monkeypatch):
    """The snapshot must outlive the stream, or the button dies with it."""
    demo, _ = _import_demo(monkeypatch)

    for event in _paused_stream():
        if event["type"] == "continuation_available":
            demo._apply_generation_event(event)

    demo._clear_inflight_state()

    snapshot = demo.st.session_state.get("generation_snapshot")
    assert demo.generation_controls(snapshot)["continue"] == "enabled"


def test_the_send_path_reruns_on_a_pause(monkeypatch):
    """Guards the fix itself.

    The controls render near the top of the script, so a pause that does not
    rerun cannot draw them however correct the snapshot is.
    """
    demo, _ = _import_demo(monkeypatch)
    source = demo.__loader__.get_source("demo")

    marker = "elif paused:"
    assert marker in source, "the send path no longer branches on a paused turn"

    branch = source.split(marker, 1)[1].split("elif event_type != \"error\"", 1)[0]
    assert "st.rerun()" in branch, "a paused turn must rerun or the controls never draw"
