"""Answer-streaming metrics: bounded labels, recorded by the real adapter.

Each test uses an isolated registry so counts are exact. Labels are closed
sets; message ids, prompts, content, and URLs never become a series.
"""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from app.observability.streaming import StreamingMetrics
from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
from app.services.event_streaming.events import make_event


def _metrics() -> StreamingMetrics:
    return StreamingMetrics(CollectorRegistry())


def _sample(metrics: StreamingMetrics, name: str, **labels) -> float:
    return metrics.registry.get_sample_value(name, labels) or 0.0


async def _drain(source, metrics, *, operation="chat", **state_kwargs):
    state = AISDKV6StreamState(
        message_id="m-1", text_id="t-1", reasoning_id="r-1", operation=operation, **state_kwargs
    )
    adapter = AISDKV6StreamAdapter(source, state, heartbeat_interval_seconds=30, metrics=metrics)
    return [frame async for frame in adapter.iter_sse()]


def test_labels_are_bounded():
    metrics = _metrics()
    metrics.record_end(operation="https://tenant.example/x", outcome="raw exception text")
    metrics.record_first_answer(0.5, operation="msg-123")
    metrics.record_reconcile(changed=True)

    rendered = metrics.render().decode()

    assert 'operation="other"' in rendered
    assert 'outcome="other"' in rendered
    assert "tenant.example" not in rendered
    assert "msg-123" not in rendered


@pytest.mark.parametrize(
    ("operation", "outcome"),
    [
        ("chat", "complete"),
        ("resume", "paused"),
        ("continue", "interrupt"),
        ("unknown", "error"),
        ("chat", "disconnect"),
        ("chat", "stopped"),
        ("chat", "unknown"),
    ],
)
def test_every_documented_label_is_its_own_series(operation, outcome):
    metrics = _metrics()
    metrics.record_end(operation=operation, outcome=outcome)

    assert _sample(
        metrics, "ai_sdk_stream_end_total", operation=operation, outcome=outcome
    ) == 1.0


@pytest.mark.asyncio
async def test_a_completed_stream_records_first_answer_and_end_once():
    metrics = _metrics()

    async def source():
        yield make_event("message_delta", sequence=1, data={"text": "hello"})
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

    await _drain(source, metrics, operation="resume")

    assert _sample(metrics, "ai_sdk_stream_first_answer_seconds_count", operation="resume") == 1
    assert _sample(metrics, "ai_sdk_stream_delta_to_wire_seconds_count", operation="resume") == 1
    assert _sample(
        metrics, "ai_sdk_stream_end_total", operation="resume", outcome="complete"
    ) == 1


@pytest.mark.asyncio
async def test_keepalives_reasoning_tools_and_sources_are_not_a_first_answer():
    metrics = _metrics()

    async def source():
        yield make_event("heartbeat", sequence=0)
        yield make_event("reasoning_delta", sequence=1, data={"text": "thinking"})
        yield make_event(
            "tool_call_available", sequence=2, tool_call_id="c", tool_name="t", data={"args": {}}
        )
        yield make_event(
            "sources", sequence=3, data={"sources": [{"url": "https://a.test", "title": "A"}]}
        )
        yield make_event("complete", sequence=4, data={"message": {"id": "m-1"}})

    await _drain(source, metrics)

    assert _sample(metrics, "ai_sdk_stream_first_answer_seconds_count", operation="chat") == 0


@pytest.mark.asyncio
async def test_a_web_preview_counts_as_the_first_answer_text():
    metrics = _metrics()

    async def source():
        yield make_event("answer_preview", sequence=1, data={"operation": "replace", "text": "d"})
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1", "content": "d"}})

    await _drain(source, metrics)

    assert _sample(metrics, "ai_sdk_stream_first_answer_seconds_count", operation="chat") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("final", "changed"), [("Draft", "false"), ("Cited final", "true")])
async def test_reconciliation_records_whether_the_draft_changed(final, changed):
    metrics = _metrics()

    async def source():
        yield make_event(
            "answer_preview", sequence=1, data={"operation": "replace", "text": "Draft"}
        )
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1", "content": final}})

    await _drain(source, metrics, stream_web_answer_v1=True)

    assert _sample(metrics, "ai_sdk_stream_reconcile_total", changed=changed) == 1
    other = "true" if changed == "false" else "false"
    assert _sample(metrics, "ai_sdk_stream_reconcile_total", changed=other) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "outcome"),
    [
        ([make_event("message_delta", sequence=1, data={"text": "partial"})], "unknown"),
        ([make_event("error", sequence=1, data={"error": "x"})], "error"),
        (
            [make_event("interrupt", sequence=1, data={"thread_id": "t", "message": "Approve?"})],
            "interrupt",
        ),
        (
            [
                make_event(
                    "message_end",
                    sequence=1,
                    data={"message": {"id": "m-1", "content": "Saved partial"}},
                )
            ],
            "paused",
        ),
        (
            [make_event("generation_status", sequence=1, data={"status": "stopped"})],
            "stopped",
        ),
    ],
)
async def test_terminal_outcomes_are_recorded(events, outcome):
    metrics = _metrics()

    async def source():
        for event in events:
            yield event

    await _drain(source, metrics)

    assert _sample(metrics, "ai_sdk_stream_end_total", operation="chat", outcome=outcome) == 1


@pytest.mark.asyncio
async def test_a_disconnect_is_recorded_once_and_never_as_an_outcome():
    import asyncio

    metrics = _metrics()

    async def source():
        yield make_event("message_delta", sequence=1, data={"text": "partial"})
        await asyncio.Event().wait()

    state = AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1")
    adapter = AISDKV6StreamAdapter(source, state, heartbeat_interval_seconds=30, metrics=metrics)
    stream = adapter.iter_sse()
    async for frame in stream:
        if '"text-delta"' in frame:
            break
    await stream.aclose()

    rendered = metrics.render().decode()
    assert _sample(
        metrics, "ai_sdk_stream_end_total", operation="unknown", outcome="disconnect"
    ) == 1
    assert rendered.count("ai_sdk_stream_end_total{") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["chat", "resume", "continue"])
async def test_routes_label_their_operation(operation):
    from uuid import UUID, uuid4

    from app.api import ai_sdk
    from app.observability.streaming import streaming_metrics

    async def source():
        yield make_event("complete", sequence=1, data={"message": {"id": "m-1"}})

    class _Service:
        def create_message_stream(self, *_a, **_k):
            return source()

        def resume_message_creation_stream(self, **_k):
            return source()

        def continue_message_generation_stream(self, **_k):
            return source()

    before = _sample(
        streaming_metrics, "ai_sdk_stream_end_total", operation=operation, outcome="complete"
    )
    conversation = "11111111-1111-4111-8111-111111111111"
    if operation == "chat":
        response = await ai_sdk.chat_ui_message_stream(
            conversation_id=UUID(conversation),
            payload=ai_sdk.AISDKChatRequest(
                messages=[{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
            ),
            message_service=_Service(),
            current_user_id=uuid4(),
        )
    elif operation == "resume":
        response = await ai_sdk.resume_interrupt_ai_sdk(
            resume_request=ai_sdk.InterruptResumeRequest(
                thread_id="t", interrupt_id="i", conversation_id=conversation, decisions=[]
            ),
            message_service=_Service(),
            current_user_id=uuid4(),
        )
    else:
        response = await ai_sdk.continue_generation_ai_sdk(
            continue_request=ai_sdk.ContinueGenerationRequest(
                conversation_id=conversation,
                generation_id="22222222-2222-4222-8222-222222222222",
                continuation_id="33333333-3333-4333-8333-333333333333",
                idempotency_key="continue-key-1",
                expected_version=1,
            ),
            message_service=_Service(),
            current_user_id=uuid4(),
        )
    [chunk async for chunk in response.body_iterator]

    after = _sample(
        streaming_metrics, "ai_sdk_stream_end_total", operation=operation, outcome="complete"
    )
    assert after - before == 1


def test_streaming_metrics_are_exposed_for_scraping():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.health import create_health_router

    metrics = _metrics()
    metrics.record_end(operation="chat", outcome="complete")
    app = FastAPI()
    app.include_router(create_health_router(streaming_metrics=metrics))

    response = TestClient(app).get("/metrics/streaming")

    assert response.status_code == 200
    assert 'ai_sdk_stream_end_total{operation="chat",outcome="complete"} 1.0' in response.text
