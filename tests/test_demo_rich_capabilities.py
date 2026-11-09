"""Streamlit advertises and renders answer visuals and source figures.

Three things are pinned here:

* every request that produces an assistant message advertises the same three
  rich flags (Send, approval Resume, Continue), so a stream and its reload are
  projected the same way;
* every canonical fixture's ``full`` projection renders through demo's real
  dispatch with no raw marker left and every value, label and caption visible;
* a continued epoch handles the same live vocabulary as a first epoch -- rich
  items, sources, persisted partials and approval interrupts -- and never
  leaves a spent Continue offer behind.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from app.ui.rich_response import build_rich_response_view
from tests.test_demo_generation_controls import _import_demo

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "rich_answer_visuals"
CONVERSATION_ID = "827faf55-1041-4357-9e1a-0f7d031fa546"
GENERATION_ID = "33333333-3333-3333-3333-333333333333"
CONTINUATION_ID = "44444444-4444-4444-4444-444444444444"
NEXT_CONTINUATION_ID = "55555555-5555-5555-5555-555555555555"

ALL_FLAGS = {"inlineRichResponseV1": True, "answerVisualsV1": True, "sourceFiguresV1": True}


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _full_projection(name: str) -> dict[str, Any]:
    data = _fixture(name)
    if "projections" in data:
        return data["projections"]["full"]
    return data["degraded_projections"]["full"]


def _resumable_snapshot(**overrides: Any) -> dict[str, Any]:
    snapshot = {
        "generation_id": GENERATION_ID,
        "conversation_id": CONVERSATION_ID,
        "status": "continuable",
        "version": 3,
        "execution_epoch": 0,
        "continuation_id": CONTINUATION_ID,
        "continuation_available": True,
    }
    snapshot.update(overrides)
    return snapshot


# ---------------------------------------------------------------------------
# advertising the flags
# ---------------------------------------------------------------------------


def test_the_client_capabilities_name_all_three_flags(monkeypatch):
    demo, _ = _import_demo(monkeypatch)

    assert demo.RICH_RESPONSE_CAPABILITIES == ALL_FLAGS


def test_continue_advertises_the_base_flag_and_both_families(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.generation_snapshot = _resumable_snapshot()

    body = demo.build_continue_request(stub.session_state.generation_snapshot, CONVERSATION_ID)

    for flag, value in ALL_FLAGS.items():
        assert body[flag] is value
    # The fence and the single-use id are unchanged by the flags.
    assert body["continuationId"] == CONTINUATION_ID
    assert body["expectedVersion"] == 3


def test_the_send_path_advertises_the_shared_capabilities(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    source = demo.__loader__.get_source("demo")

    send_body = source.split('"conversationId": st.session_state.current_conversation_id,', 1)[1]
    send_body = send_body.split("}", 1)[0]
    assert "**RICH_RESPONSE_CAPABILITIES" in send_body


def test_approval_resume_advertises_all_three_flags(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.update(
        {
            "interrupt_conversation_id": CONVERSATION_ID,
            "pending_decisions_i1": {"t1": {"type": "approve"}},
            "pending_image_attachments": [],
        }
    )
    sent: list[tuple[str, dict]] = []

    class _Renderer:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def finalize(self, _message: Any) -> None:
            return None

    def _stream(endpoint: str, payload: dict) -> Any:
        sent.append((endpoint, payload))
        return iter([{"type": "complete", "message": {"id": "a1", "content": "done"}}])

    monkeypatch.setattr(demo, "_StreamingRichResponseRenderer", _Renderer)
    monkeypatch.setattr(demo, "make_streaming_request", _stream)

    with pytest.raises(RuntimeError, match="rerun"):
        demo._submit_interrupt_decisions(
            "thread-1", "i1", [{"task_id": "t1"}], "pending_decisions_i1"
        )

    endpoint, payload = sent[0]
    assert endpoint == "/messages/resume-interrupt"
    for flag, value in ALL_FLAGS.items():
        assert payload[flag] is value


# ---------------------------------------------------------------------------
# rendering the canonical fixtures through demo's dispatch
# ---------------------------------------------------------------------------


def _render_projection(monkeypatch, projection: dict[str, Any]) -> tuple[str, list[str], Any]:
    demo, stub = _import_demo(monkeypatch)
    rendered: list[str] = []
    stub.markdown = lambda body, **_kwargs: rendered.append(str(body))
    monkeypatch.setattr(
        demo, "_resolve_displayable_image_src", lambda url=None, data=None, mime=None: url
    )
    metadata = {"rich_items_version": 1, "rich_items": projection["rich_items"]}
    view = build_rich_response_view(
        projection["content"], metadata, answer_visuals=True, source_figures=True
    )
    demo._render_rich_segments(
        view.segments, message_metadata=metadata, message_key="m1", auto_mount=False
    )
    return "\n".join(rendered), stub.captions, view


ANSWER_FIXTURES = [
    "edge-google-steps",
    "game-edition-markdown-table",
    "overclock-bars",
    "tweak-assessment-table",
    "graphics-screenshot-figure",
    "mixed-static-and-images",
    "partial-figure-failure",
]


@pytest.mark.parametrize("name", ANSWER_FIXTURES)
def test_every_fixture_renders_without_a_raw_marker_or_unavailable_block(monkeypatch, name):
    output, captions, _view = _render_projection(monkeypatch, _full_projection(name))

    assert "<!--rich:" not in output
    assert not any("unavailable" in caption for caption in captions)


def test_steps_fixture_renders_numbered_steps_with_note_and_citation(monkeypatch):
    output, _, _ = _render_projection(monkeypatch, _full_projection("edge-google-steps"))

    assert "<ol" in output
    assert "Open Settings from the ... menu" in output
    assert "Pick Google under Search engine used in the address bar" in output
    assert "Google appears only after you have searched with it once." in output
    assert '<a href="https://support.example/edge-search"' in output
    assert ">support.example</a>" in output
    assert "Restart Edge if it does not apply." in output


def test_bars_fixture_renders_values_units_basis_and_assumptions(monkeypatch):
    output, _, _ = _render_projection(monkeypatch, _full_projection("overclock-bars"))

    for text in ("4700 MHz", "4750 MHz", "1.25 V", "1.32 V", "100 FPS", "101 FPS"):
        assert text in output
    assert output.count("Basis: measured") == 2
    assert output.count("Basis: calculated") == 1
    assert "Assumes FPS scales exactly with core clock" in output
    assert ">bench.example</a>" in output


def test_table_fixture_renders_badges_as_text_with_reasons(monkeypatch):
    output, _, _ = _render_projection(monkeypatch, _full_projection("tweak-assessment-table"))

    assert "<table" in output
    assert '<th scope="col">Claimed effect</th>' in output
    assert ">Skip</span>" in output
    assert ">Do it</span>" in output
    assert "No measurable change in testing; can break timers." in output
    assert "Measured gains in CPU-bound games." in output


def test_figure_fixture_renders_labels_sources_and_one_caption(monkeypatch):
    output, _, _ = _render_projection(monkeypatch, _full_projection("graphics-screenshot-figure"))

    assert "Shadows: High" in output
    assert "Shadows: Low" in output
    assert "https://screens1.example/graphics" in output
    assert "https://screens2.example/graphics" in output
    caption = "Low shadows remove the contact shadow, so the car looks like it floats."
    assert output.count(caption) == 1 + output.count(f'aria-label="{caption}"')
    assert 'alt="Car with a soft contact shadow under the chassis"' in output


def test_mixed_fixture_keeps_authored_order(monkeypatch):
    output, _, view = _render_projection(monkeypatch, _full_projection("mixed-static-and-images"))

    kinds = [(s.kind, (s.item or {}).get("type")) for s in view.segments if s.kind == "rich"]
    assert kinds == [("rich", "image"), ("rich", "answer_visual"), ("rich", "image_group")]
    assert output.index("/web-images/00000000-0000-4000-8000-000000000009") < output.index(
        "<ol"
    )
    assert output.index("<ol") < output.index("Shadows: High")


def test_partial_figure_failure_shows_the_note_and_no_caption(monkeypatch):
    output, _, _ = _render_projection(monkeypatch, _full_projection("partial-figure-failure"))

    # Visible text, not only the group's aria-label.
    assert 'data-role="comparison-unavailable"' in output
    assert 'data-role="cell-label"' in output
    assert "Shadows: High" in output
    assert "https://screens1.example/graphics" in output
    assert output.count("<img") == 1
    original_caption = _fixture("partial-figure-failure")["original"]["payload"]["caption"]
    assert original_caption not in output


def test_unknown_visual_variant_renders_its_fallback_markdown_without_raw_html(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    calls: list[tuple[str, dict]] = []
    stub.markdown = lambda body, **kwargs: calls.append((str(body), kwargs))
    item = {
        "id": "visual:future",
        "type": "answer_visual",
        "display_policy": "inline_only",
        "payload": {"variant": "radar", "fallback_markdown": "**Future** <b>x</b>"},
    }

    demo._render_inline_rich_item(item, message_metadata={}, message_key="m", auto_mount=False)

    assert len(calls) == 1
    body, kwargs = calls[0]
    assert "**Future**" in body
    assert not kwargs.get("unsafe_allow_html")


def test_history_view_renders_visuals_and_figures_natively(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    projection = _full_projection("mixed-static-and-images")
    metadata = {"rich_items_version": 1, "rich_items": projection["rich_items"]}

    view = demo._build_rich_response_view_for_msg(projection["content"], metadata)

    types = [(s.item or {}).get("type") for s in view.segments if s.kind == "rich"]
    assert "answer_visual" in types
    figure = next(s.item for s in view.segments if (s.item or {}).get("type") == "image_group")
    assert figure["payload"].get("caption")


def test_live_renderer_renders_finalized_visuals_natively(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    rendered: list[str] = []

    class _Slot:
        def markdown(self, _text: str) -> None:
            return None

    class _Root:
        def container(self) -> Any:
            return nullcontext()

    stub.empty = lambda: _Slot()
    monkeypatch.setattr(
        demo,
        "_render_inline_rich_item",
        lambda item, **_kwargs: rendered.append(item.get("type")),
    )
    projection = _full_projection("edge-google-steps")

    renderer = demo._StreamingRichResponseRenderer(_Root(), message_key="live")
    renderer.finalize(
        {
            "id": "a1",
            "content": projection["content"],
            "messageMetadata": {"rich_items_version": 1, "rich_items": projection["rich_items"]},
        }
    )

    assert rendered == ["answer_visual"]


# ---------------------------------------------------------------------------
# consuming a continued epoch
# ---------------------------------------------------------------------------


class _RecordingRenderer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def append_text(self, delta: str) -> None:
        self.calls.append(("text", delta))

    def apply_rich_items_upsert(self, items: list[dict[str, Any]]) -> None:
        self.calls.append(("rich_items", items))

    def finalize(self, message: dict[str, Any] | None) -> None:
        self.calls.append(("finalize", message))


def test_continue_stream_applies_live_rich_items_to_the_renderer(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    renderer = _RecordingRenderer()
    items = [{"id": "widget:w-1", "type": "live_widget", "payload": {}}]

    demo.consume_continue_stream(iter([{"type": "rich_items", "items": items}]), renderer)

    assert ("rich_items", items) in renderer.calls


def test_continue_stream_records_sources(monkeypatch):
    demo, stub = _import_demo(monkeypatch)

    demo.consume_continue_stream(
        iter([{"type": "sources", "sources": [{"source_id": "S1", "url": "https://a.example"}]}])
    )

    assert stub.session_state.stream_web_sources["S1"]["url"] == "https://a.example"


@pytest.mark.parametrize("terminal", ["message_end", "complete"])
def test_continue_stream_finalizes_the_renderer_with_the_persisted_row(monkeypatch, terminal):
    demo, _ = _import_demo(monkeypatch)
    renderer = _RecordingRenderer()
    message = {"id": "m2", "content": "answer"}

    outcome = demo.consume_continue_stream(
        iter([{"type": terminal, "message": message}]), renderer
    )

    assert outcome["message"] == message
    assert ("finalize", message) in renderer.calls


def test_continue_stream_surfaces_an_approval_interrupt_with_its_partial(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    renderer = _RecordingRenderer()
    partial = {"id": "m3", "content": "Checking the file"}

    outcome = demo.consume_continue_stream(
        iter(
            [
                {"type": "thinking", "content": "need approval"},
                {"type": "token", "content": "Checking the file"},
                {
                    "type": "interrupt",
                    "interrupt": {"interrupt_id": "i9", "action_requests": []},
                    "message": partial,
                },
                {"type": "token", "content": "never consumed"},
            ]
        ),
        renderer,
    )

    assert outcome["interrupt"]["interrupt_id"] == "i9"
    assert outcome["interrupt"]["agent_partial_content"] == "Checking the file"
    assert outcome["interrupt"]["agent_thinking"] == "need approval"
    assert outcome["message"] == partial
    assert ("finalize", partial) in renderer.calls
    assert ("text", "never consumed") not in renderer.calls


def test_continue_stream_without_an_interrupt_reports_none(monkeypatch):
    demo, _ = _import_demo(monkeypatch)

    outcome = demo.consume_continue_stream(iter([]))

    assert outcome["interrupt"] is None


# ---------------------------------------------------------------------------
# running a continued epoch
# ---------------------------------------------------------------------------


def _run_continue(monkeypatch, events: list[dict[str, Any]]):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.generation_snapshot = _resumable_snapshot()
    stub.session_state.messages = []
    monkeypatch.setattr(
        demo, "_StreamingRichResponseRenderer", lambda *_a, **_k: _RecordingRenderer()
    )
    sent: list[tuple[str, dict]] = []

    def _stream(endpoint: str, payload: dict) -> Any:
        sent.append((endpoint, payload))
        return iter(events)

    monkeypatch.setattr(demo, "make_streaming_request", _stream)
    body = demo.build_continue_request(stub.session_state.generation_snapshot, CONVERSATION_ID)
    with pytest.raises(RuntimeError, match="rerun"):
        demo._handle_continue(body, CONVERSATION_ID)
    return demo, stub, sent


def test_continue_stays_on_the_internal_endpoint(monkeypatch):
    _demo, _stub, sent = _run_continue(
        monkeypatch, [{"type": "complete", "message": {"id": "m2", "content": "x"}}]
    )

    assert sent[0][0] == "/messages/continue"


def test_an_interrupted_continuation_hands_off_to_the_approval_panel(monkeypatch):
    demo, stub, _ = _run_continue(
        monkeypatch,
        [
            {"type": "token", "content": "partial"},
            {"type": "interrupt", "interrupt": {"interrupt_id": "i9"}},
        ],
    )

    assert stub.session_state.pending_interrupt["interrupt_id"] == "i9"
    assert stub.session_state.pending_interrupt["agent_partial_content"] == "partial"
    assert stub.session_state.interrupt_conversation_id == CONVERSATION_ID
    snapshot = stub.session_state.generation_snapshot
    assert demo.generation_controls(snapshot)["continue"] == "hidden"


def test_a_refused_continuation_retires_the_spent_offer(monkeypatch):
    demo, stub, _ = _run_continue(
        monkeypatch, [{"type": "error", "error": "the continuation has already been used"}]
    )

    snapshot = stub.session_state.generation_snapshot
    assert demo.generation_controls(snapshot)["continue"] == "hidden"
    # The fence is kept: the version and generation are untouched.
    assert snapshot["version"] == 3
    assert snapshot["generation_id"] == GENERATION_ID
    # The refusal is still shown on the next render.
    demo._render_continue_control(CONVERSATION_ID)
    assert "the continuation has already been used" in stub.errors
    assert not any(b.get("key") == "continue_generation_btn" for b in stub.buttons)


def test_a_fresh_offer_issued_during_the_epoch_survives_the_refusal(monkeypatch):
    demo, stub, _ = _run_continue(
        monkeypatch,
        [
            {
                "type": "continuation_available",
                "generation_id": GENERATION_ID,
                "conversation_id": CONVERSATION_ID,
                "status": "continuable",
                "version": 5,
                "continuation_id": NEXT_CONTINUATION_ID,
                "continuation_available": True,
            },
            {"type": "error", "error": "late failure"},
        ],
    )

    snapshot = stub.session_state.generation_snapshot
    assert snapshot["continuation_id"] == NEXT_CONTINUATION_ID
    assert demo.generation_controls(snapshot)["continue"] == "enabled"
