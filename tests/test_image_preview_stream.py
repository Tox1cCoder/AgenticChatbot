"""End-to-end contract for early-delivery `image_preview` stream events.

Covers every hop: publisher policy → sink → graph public projection →
AIService canonicalization → both wire adapters (internal SSE, AI SDK v6).
"""

from __future__ import annotations

import json

import pytest

from app.ai.image_generation import ImagePreviewPublisher, use_image_preview_emitter
from app.services.ai_service import AIService
from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
from app.services.event_streaming.events import make_event
from app.services.event_streaming.graph_public_projection import (
    GraphPublicStreamProjector,
    StreamProjectionContext,
)
from app.services.event_streaming.internal_sse import legacy_event_from_v3
from app.services.event_streaming.subagents import SubagentEventSink

# ---------------------------------------------------------------------------
# Publisher policy
# ---------------------------------------------------------------------------


def _publish(publisher: ImagePreviewPublisher, **overrides) -> bool:
    payload = {
        "image_index": 0,
        "status": "final",
        "mime": "image/png",
        "data_b64": "QUJD",
        "seq": 0,
    }
    payload.update(overrides)
    return publisher.publish(**payload)


def test_publisher_noop_without_emitter():
    publisher = ImagePreviewPublisher(enabled=True, max_b64_chars=100)
    assert _publish(publisher) is False


def test_publisher_noop_when_disabled_even_with_emitter():
    emitted: list[dict] = []
    with use_image_preview_emitter(emitted.append):
        publisher = ImagePreviewPublisher(enabled=False, max_b64_chars=100)
        assert _publish(publisher) is False
    assert emitted == []


def test_publisher_emits_payload_shape():
    emitted: list[dict] = []
    with use_image_preview_emitter(emitted.append):
        publisher = ImagePreviewPublisher(enabled=True, max_b64_chars=100)
        assert _publish(publisher, status="partial", seq=3) is True
    assert emitted == [
        {
            "item_id": "image-preview-0",
            "image_index": 0,
            "status": "partial",
            "mime": "image/png",
            "data_b64": "QUJD",
            "seq": 3,
        }
    ]


def test_publisher_emits_final_once_per_index():
    emitted: list[dict] = []
    with use_image_preview_emitter(emitted.append):
        publisher = ImagePreviewPublisher(enabled=True, max_b64_chars=100)
        assert _publish(publisher) is True
        assert _publish(publisher) is False
        assert _publish(publisher, image_index=1) is True
    assert len(emitted) == 2


def test_publisher_drops_oversized_payloads():
    emitted: list[dict] = []
    with use_image_preview_emitter(emitted.append):
        publisher = ImagePreviewPublisher(enabled=True, max_b64_chars=3)
        assert _publish(publisher, data_b64="QUJDRA") is False
    assert emitted == []


def test_publisher_swallows_emitter_failures():
    def _boom(_payload: dict) -> None:
        raise RuntimeError("sink gone")

    with use_image_preview_emitter(_boom):
        publisher = ImagePreviewPublisher(enabled=True, max_b64_chars=100)
        assert _publish(publisher) is False


# ---------------------------------------------------------------------------
# Sink → projector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sink_emit_event_enqueues_resequenced_event():
    sink = SubagentEventSink()
    sink.emit_event(make_event("image_preview", sequence=0, data={"image_index": 0}))
    events = await sink.drain()
    assert len(events) == 1
    assert events[0].type == "image_preview"
    assert events[0].sequence == 1


def test_projector_maps_image_preview_despite_token_suppression():
    projector = GraphPublicStreamProjector(
        tool_end_events_from_node_state=lambda **_kwargs: iter(()),
        suppress_internal_stream_chunks=True,
    )
    ctx = StreamProjectionContext(suppress_tokens=True)
    event = make_event(
        "image_preview",
        sequence=1,
        data={"item_id": "image-preview-0", "image_index": 0, "status": "final"},
    )

    public = list(projector.map_event(event, ctx))

    assert public == [
        {
            "type": "image_preview",
            "item_id": "image-preview-0",
            "image_index": 0,
            "status": "final",
        }
    ]


# ---------------------------------------------------------------------------
# AIService canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ai_service_maps_image_preview_dict_to_canonical_event():
    service = object.__new__(AIService)

    async def _stream():
        yield {
            "type": "image_preview",
            "item_id": "image-preview-0",
            "image_index": 0,
            "status": "final",
            "mime": "image/png",
            "data_b64": "QUJD",
            "seq": 0,
        }

    events = [event async for event in service._map_workflow_stream(_stream())]

    preview_events = [event for event in events if event.type == "image_preview"]
    assert len(preview_events) == 1
    assert preview_events[0].data == {
        "item_id": "image-preview-0",
        "image_index": 0,
        "status": "final",
        "mime": "image/png",
        "data_b64": "QUJD",
        "seq": 0,
    }
    # the stream still terminates with a complete event
    assert events[-1].type == "complete"


# ---------------------------------------------------------------------------
# Wire adapters
# ---------------------------------------------------------------------------


def test_internal_sse_projects_image_preview():
    event = make_event(
        "image_preview",
        sequence=1,
        data={"item_id": "image-preview-0", "image_index": 0, "data_b64": "QUJD"},
    )
    assert legacy_event_from_v3(event) == {
        "type": "image_preview",
        "item_id": "image-preview-0",
        "image_index": 0,
        "data_b64": "QUJD",
    }


async def _collect_ai_sdk_payloads(source):
    adapter = AISDKV6StreamAdapter(
        lambda: source(),
        AISDKV6StreamState(message_id="m-1", text_id="t-1", reasoning_id="r-1"),
    )
    chunks = [chunk async for chunk in adapter.iter_sse()]
    payloads = []
    for line in "".join(chunks).splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        payloads.append(raw if raw == "[DONE]" else json.loads(raw))
    return payloads


@pytest.mark.asyncio
async def test_ai_sdk_v6_projects_image_preview_as_transient_data_part():
    async def source():
        yield make_event(
            "image_preview",
            sequence=1,
            data={
                "item_id": "image-preview-0",
                "image_index": 0,
                "status": "partial",
                "mime": "image/png",
                "data_b64": "QUJD",
                "seq": 2,
            },
        )
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

    payloads = await _collect_ai_sdk_payloads(source)

    previews = [
        payload
        for payload in payloads
        if isinstance(payload, dict) and payload.get("type") == "data-image-preview"
    ]
    assert previews == [
        {
            "type": "data-image-preview",
            "id": "image-preview-0",
            "data": {
                "imageIndex": 0,
                "status": "partial",
                "mediaType": "image/png",
                "url": "data:image/png;base64,QUJD",
                "seq": 2,
            },
            "transient": True,
        }
    ]


@pytest.mark.asyncio
async def test_ai_sdk_v6_drops_preview_without_payload():
    async def source():
        yield make_event(
            "image_preview",
            sequence=1,
            data={"item_id": "image-preview-0", "image_index": 0, "status": "final"},
        )
        yield make_event("complete", sequence=2, data={"message": {"id": "m-1"}})

    payloads = await _collect_ai_sdk_payloads(source)
    assert not any(
        isinstance(payload, dict) and payload.get("type") == "data-image-preview"
        for payload in payloads
    )


# ---------------------------------------------------------------------------
# Graph node emitter binding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_binds_emitter_to_registered_sink(monkeypatch):
    from app.ai.graph import MultiAgentWorkflow
    from app.core.config import settings
    from app.services.event_streaming.subagents import register_subagent_event_sink

    monkeypatch.setattr(settings, "enable_image_streaming", True)
    sink = SubagentEventSink()
    token = register_subagent_event_sink(sink)
    state = {"context": {"subagent_event_sink_token": token}}

    emitter = MultiAgentWorkflow._build_image_preview_emitter(None, state)
    assert emitter is not None

    emitter({"item_id": "image-preview-0", "image_index": 0})
    events = await sink.drain()
    assert len(events) == 1
    assert events[0].type == "image_preview"
    assert events[0].agent == "image_generator_agent"
    assert events[0].data == {"item_id": "image-preview-0", "image_index": 0}


def test_graph_emitter_disabled_by_flag_or_missing_sink(monkeypatch):
    from app.ai.graph import MultiAgentWorkflow
    from app.core.config import settings

    monkeypatch.setattr(settings, "enable_image_streaming", False)
    assert MultiAgentWorkflow._build_image_preview_emitter(None, {"context": {}}) is None

    monkeypatch.setattr(settings, "enable_image_streaming", True)
    assert MultiAgentWorkflow._build_image_preview_emitter(None, {"context": {}}) is None
    assert MultiAgentWorkflow._build_image_preview_emitter(None, {}) is None
