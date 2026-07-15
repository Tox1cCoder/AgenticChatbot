from __future__ import annotations

from typing import Any

from tests.test_demo_plan_widget import _import_demo_with_ui_stubs


def test_live_tool_trace_renders_queued_start_and_retains_end_presentation(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    demo._reset_stream_trace_state()

    demo._upsert_stream_tool_trace(
        {
            "type": "tool",
            "phase": "start",
            "state": "queued",
            "tool_call_id": "call-1",
            "name": "make_chart",
            "args": {"metric": "sales"},
        }
    )

    started = streamlit_stub.session_state.stream_trace_items[0]
    assert started["state"] == "queued"
    assert demo._trace_status_meta(started["state"]) == (
        "Queued",
        "hourglass_top",
        "running",
    )

    render = {
        "type": "chart",
        "structured_content": {"rows": [{"label": "Q1", "sales": 10}]},
    }
    demo._upsert_stream_tool_trace(
        {
            "type": "tool",
            "phase": "end",
            "state": "error",
            "tool_call_id": "call-1",
            "name": "make_chart",
            "result": {"rows": [["Q1", 10]]},
            "render": render,
            "error": "Chart partially failed",
            "hint": "Try a smaller date range.",
        }
    )

    ended = streamlit_stub.session_state.stream_trace_items[0]
    assert ended["render"] == render
    assert ended["error"] == "Chart partially failed"
    assert ended["hint"] == "Try a smaller date range."

    rendered: list[tuple[dict[str, Any], Any]] = []
    errors: list[str] = []
    hints: list[str] = []
    monkeypatch.setattr(
        demo,
        "render_tool_render_payload",
        lambda payload, fallback_output=None: rendered.append((payload, fallback_output)) or True,
    )
    streamlit_stub.error = errors.append
    streamlit_stub.info = hints.append

    demo._render_trace_tool_card(ended, 1)

    assert rendered == [(render, {"rows": [["Q1", 10]]})]
    assert errors == ["Chart partially failed"]
    assert hints == ["Try a smaller date range."]


def test_live_tool_end_without_id_reuses_queued_start(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    demo._reset_stream_trace_state()

    demo._upsert_stream_tool_trace(
        {"type": "tool", "phase": "start", "state": "queued", "name": "lookup"}
    )
    demo._upsert_stream_tool_trace(
        {"type": "tool", "phase": "end", "state": "completed", "name": "lookup", "result": "ok"}
    )

    assert len(streamlit_stub.session_state.stream_trace_items) == 1
    assert streamlit_stub.session_state.stream_trace_items[0]["state"] == "completed"


def test_successful_hitl_resume_clears_original_pending_images(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)

    class Status:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self, **_kwargs):
            return None

    class Renderer:
        def __init__(self, *_args, **_kwargs):
            pass

        def finalize(self, _message):
            return None

    streamlit_stub.session_state.clear()
    streamlit_stub.session_state.update(
        {
            "pending_interrupt": {"interrupt_id": "interrupt-1"},
            "interrupt_conversation_id": "conversation-1",
            "pending_decisions_interrupt_1": {"tool-1": {"type": "approve"}},
            "pending_image_attachments": [{"token": "image-1", "data": "YWJj"}],
        }
    )
    streamlit_stub.session_state[demo._hitl_resume_lock_key("interrupt-1")] = True
    reruns: list[bool] = []
    streamlit_stub.status = lambda *_args, **_kwargs: Status()
    streamlit_stub.empty = lambda: object()
    streamlit_stub.rerun = lambda: reruns.append(True)

    monkeypatch.setattr(demo, "_StreamingRichResponseRenderer", Renderer)
    monkeypatch.setattr(
        demo,
        "make_streaming_request",
        lambda *_args, **_kwargs: iter(
            [{"type": "complete", "message": {"id": "assistant-1", "content": "done"}}]
        ),
    )

    demo._submit_interrupt_decisions(
        "thread-1",
        "interrupt-1",
        [{"task_id": "tool-1"}],
        "pending_decisions_interrupt_1",
    )

    assert streamlit_stub.session_state.pending_image_attachments == []
    assert reruns == [True]


def test_attachment_toggle_draft_is_preserved_for_exactly_one_rerun(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)

    demo._preserve_message_draft_for_attachment_toggle("conversation-1", "draft text")

    assert demo._consume_preserved_message_draft("conversation-1") == "draft text"
    assert demo._consume_preserved_message_draft("conversation-1") == ""
