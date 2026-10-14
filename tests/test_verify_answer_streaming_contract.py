"""The live verifier's wire analysis, checked against real adapter output."""

from __future__ import annotations

import pytest

from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
from app.services.event_streaming.events import make_event
from scripts.verify_answer_streaming_contract import analyze_wire, compare_with_history


async def _wire(events, **state_kwargs) -> str:
    async def source():
        for event in events:
            yield event

    adapter = AISDKV6StreamAdapter(
        source,
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1", **state_kwargs),
        heartbeat_interval_seconds=30,
    )
    return "".join([frame async for frame in adapter.iter_sse()])


@pytest.mark.asyncio
async def test_a_reconciled_web_answer_has_no_problems():
    wire = await _wire(
        [
            make_event("answer_preview", sequence=1, data={"operation": "replace", "text": "D"}),
            make_event("answer_preview", sequence=2, data={"operation": "replace", "text": "R"}),
            make_event("complete", sequence=3, data={"message": {"id": "m-1", "content": "Final"}}),
        ],
        stream_web_answer_v1=True,
    )

    report = analyze_wire(wire)

    assert report.problems == []
    assert report.message_id == "m-1"
    assert report.visible_answer == "Final"
    assert report.reconciles == 1
    assert report.finish_reason is None
    assert report.first_answer_index is not None


@pytest.mark.asyncio
async def test_an_incomplete_stream_is_reported_with_its_code():
    wire = await _wire([make_event("message_delta", sequence=1, data={"text": "partial"})])

    report = analyze_wire(wire)

    assert report.finish_reason == "error"
    assert report.error_code == "stream_incomplete"
    assert report.problems == []  # well-formed; the outcome is the caller's to judge


@pytest.mark.parametrize(
    ("wire", "problem"),
    [
        ('data: {"type":"heartbeat"}\n\ndata: [DONE]\n\n', "heartbeat"),
        (
            'data: {"type":"error","errorText":"x","statusCode":409}\n\ndata: [DONE]\n\n',
            "error",
        ),
        (
            'data: {"type":"tool-output-available","toolCallId":"c","output":1,"render":{}}\n\n'
            "data: [DONE]\n\n",
            "tool-output-available",
        ),
        ('data: {"type":"finish"}\n\ndata: [DONE]\n\ndata: [DONE]\n\n', "[DONE]"),
        ('data: {"type":"finish"}\n\n', "[DONE]"),
        ("data: {nope\n\ndata: [DONE]\n\n", "JSON"),
        (
            'data: {"type":"data-answer-reconcile","data":{"messageId":"m","content":"a"}}\n\n'
            'data: {"type":"data-answer-reconcile","data":{"messageId":"m","content":"b"}}\n\n'
            'data: {"type":"finish"}\n\ndata: [DONE]\n\n',
            "reconcile",
        ),
    ],
)
def test_contract_violations_are_reported(wire, problem):
    report = analyze_wire(wire)

    assert any(problem in item for item in report.problems), report.problems


@pytest.mark.parametrize(
    "chunks",
    [
        # The direct server writes compact JSON.
        ['data: {"type":"start"}\n\n', 'data: {"type":"text-delta","id":"t","delta":"a"}\n\n'],
        # The sidecar re-serialises with default separators.
        [
            'data: {"type": "start"}\n\n',
            'data: {"type": "text-delta", "id": "t", "delta": "a"}\n\n',
        ],
        # A frame split across network chunks.
        ['data: {"type":"start"}\n\ndata: {"type":"te', 'xt-delta","id":"t","delta":"a"}\n\n'],
    ],
)
def test_first_answer_is_detected_on_parsed_frames(chunks):
    from scripts.verify_answer_streaming_contract import FirstAnswerWatch

    watch = FirstAnswerWatch()
    seen = [watch.feed(chunk) for chunk in chunks]

    assert seen == [False, True]


def test_keepalives_and_reasoning_are_not_a_first_answer():
    from scripts.verify_answer_streaming_contract import FirstAnswerWatch

    watch = FirstAnswerWatch()

    assert watch.feed(': keepalive\n\ndata: {"type":"reasoning-delta","delta":"x"}\n\n') is False


def test_history_comparison_requires_the_streamed_message_and_its_text():
    history = {
        "data": {
            "messages": [
                {"id": "m-0", "role": "assistant", "content": "older"},
                {"id": "m-1", "role": "assistant", "content": "Final"},
            ]
        }
    }

    assert compare_with_history("m-1", "Final", history) == []
    assert compare_with_history("m-1", "Draft", history) != []
    assert compare_with_history("m-9", "Final", history) != []
