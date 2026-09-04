"""End-to-end contract for early-delivery `image_preview` stream events.

Covers every hop: publisher policy → sink → graph public projection →
AIService canonicalization → both wire adapters (internal SSE, AI SDK v6).
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ai.image_generation import (
    ImagePreviewPublisher,
    MediaDeliveryService,
    use_image_preview_emitter,
)
from app.services.ai_service import AIService
from app.services.chat_image_service import ChatImageStorageService
from app.services.event_streaming.ai_sdk_v6 import AISDKV6StreamAdapter, AISDKV6StreamState
from app.services.event_streaming.events import make_event
from app.services.event_streaming.graph_public_projection import (
    GraphPublicStreamProjector,
    StreamProjectionContext,
)
from app.services.event_streaming.internal_sse import legacy_event_from_v3

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


def test_publisher_skips_oversized_inline_with_structured_status():
    """An oversized inline preview is NOT delivered inline (publish returns
    False) but is never a silent drop: a structured ``preview_skipped`` status
    is emitted instead, carrying no base64 (FR-IMG-007)."""
    emitted: list[dict] = []
    with use_image_preview_emitter(emitted.append):
        publisher = ImagePreviewPublisher(enabled=True, max_b64_chars=3)
        assert _publish(publisher, data_b64="QUJDRA") is False
    assert emitted == [
        {
            "schema_version": 2,
            "item_id": "image-preview-0",
            "image_index": 0,
            "status": "preview_skipped",
            "seq": 0,
            "media_type": "image/png",
            "reason": "oversized_inline_preview",
        }
    ]
    assert "QUJDRA" not in json.dumps(emitted)


def test_publisher_swallows_emitter_failures():
    def _boom(_payload: dict) -> None:
        raise RuntimeError("sink gone")

    with use_image_preview_emitter(_boom):
        publisher = ImagePreviewPublisher(enabled=True, max_b64_chars=100)
        assert _publish(publisher) is False


class _StubImageStorage:
    """Deterministic storage stand-in returning a fixed protected reference."""

    def __init__(self, url: str):
        self._url = url
        self._image_id = url.rsplit("/", 1)[-1]

    def store(self, *, conversation_id, user_id, mime, data_b64, name):
        return {
            "image_id": self._image_id,
            "url": self._url,
            "mime": mime,
            "name": name,
            "content_hash": "hash",
        }


class _InMemoryChatImageRepo:
    """Row store with the (user_id, sha256) lookup the storage dedup needs."""

    def __init__(self):
        self.rows = {}
        self.created = []

    def create(self, data):
        row = SimpleNamespace(deleted_at=None, **data)
        self.rows[data["id"]] = row
        self.created.append(data)
        return row

    def get_by_user_and_sha(self, user_id, sha256):
        for row in self.rows.values():
            if row.user_id == user_id and row.sha256 == sha256 and row.deleted_at is None:
                return row
        return None


def test_resume_repersist_reuses_ownership_row_across_runs(tmp_path):
    """A resumed run persisting the SAME generated image as the original run
    must NOT create a second ownership row. The resumed MediaDeliveryService is
    a fresh instance whose per-instance idempotency cache is empty, so dedup has
    to hold at the storage row level (cross-run idempotency, FR-IMG-008)."""
    repo = _InMemoryChatImageRepo()
    storage = ChatImageStorageService(
        repo, storage_root=str(tmp_path / "imgs"), max_bytes=10_000
    )
    conversation_id = uuid4()
    user_id = uuid4()
    run_id = str(conversation_id)
    data_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"g" * 64).decode()

    def _persist_once():
        service = MediaDeliveryService(
            storage=storage,
            conversation_id=conversation_id,
            user_id=user_id,
            preview_publisher=ImagePreviewPublisher(enabled=True, max_b64_chars=10),
            request_id=run_id,
        )
        return service.persist_final(image_index=0, mime="image/png", data_b64=data_b64)

    first = _persist_once()  # original run
    second = _persist_once()  # resumed run — fresh instance, empty cache

    assert first is not None and second is not None
    assert len(repo.created) == 1
    assert first["image_id"] == second["image_id"]
    assert first["url"] == second["url"]


def test_persist_final_emits_v2_reference_event_by_reference():
    """persist_final publishes the FINAL image as an early schema-v2 REFERENCE
    event (delivery.kind=reference) even when the base64 exceeds the inline
    preview cap; the emitted event carries the protected URL and no base64."""
    emitted: list[dict] = []
    with use_image_preview_emitter(emitted.append):
        publisher = ImagePreviewPublisher(enabled=True, max_b64_chars=3)
        service = MediaDeliveryService(
            storage=_StubImageStorage("/chat-images/ref-1"),
            conversation_id="c-1",
            user_id="u-1",
            preview_publisher=publisher,
        )
        # publish one small partial first so the final reference seq follows it
        service.publish_partial(image_index=0, mime="image/png", data_b64="QUJD", seq=1)
        descriptor = service.persist_final(
            image_index=0, mime="image/png", data_b64="QUJDRA"  # oversized for the inline cap
        )

    assert descriptor is not None
    assert descriptor["url"] == "/chat-images/ref-1"
    reference_events = [e for e in emitted if e.get("status") == "final"]
    assert reference_events == [
        {
            "schema_version": 2,
            "item_id": "image-preview-0",
            "image_index": 0,
            "status": "final",
            "seq": 2,
            "media_type": "image/png",
            "delivery": {
                "kind": "reference",
                "image_id": "ref-1",
                "url": "/chat-images/ref-1",
            },
        }
    ]
    assert "QUJDRA" not in json.dumps(emitted)


class _PreviewCollector:
    """Stand-in for the deleted event sink, in its only remaining role: a test
    collector. Production previews now go straight to the graph's custom
    channel, so nothing but these tests ever needed the queue."""

    def __init__(self) -> None:
        self.events: list = []

    def emit_event(self, event) -> None:
        self.events.append(event)

    async def drain(self) -> list:
        drained, self.events = self.events, []
        return drained


# ---------------------------------------------------------------------------
# Sink → projector
# ---------------------------------------------------------------------------


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

    assert len(public) == 1
    assert public[0].type == "image_preview"
    assert public[0].data == {
        "item_id": "image-preview-0",
        "image_index": 0,
        "status": "final",
    }


# ---------------------------------------------------------------------------
# AIService canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ai_service_passes_image_preview_event_through():
    service = object.__new__(AIService)

    async def _stream():
        yield make_event(
            "image_preview",
            sequence=0,
            data={
                "item_id": "image-preview-0",
                "image_index": 0,
                "status": "final",
                "mime": "image/png",
                "data_b64": "QUJD",
                "seq": 0,
            },
        )

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


def test_internal_sse_projects_v2_reference_final_with_protected_url():
    """A schema-v2 reference-delivery FINAL image_preview surfaces on the
    internal SSE wire with its status and protected relative URL intact and no
    base64 anywhere."""
    event = make_event(
        "image_preview",
        sequence=1,
        data={
            "schema_version": 2,
            "item_id": "image-preview-0",
            "image_index": 0,
            "status": "final",
            "seq": 2,
            "media_type": "image/png",
            "delivery": {
                "kind": "reference",
                "image_id": "abc",
                "url": "/chat-images/abc",
            },
        },
    )
    projected = legacy_event_from_v3(event)
    assert projected["type"] == "image_preview"
    assert projected["status"] == "final"
    assert projected["delivery"] == {
        "kind": "reference",
        "image_id": "abc",
        "url": "/chat-images/abc",
    }
    assert "data_b64" not in projected


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
async def test_graph_binds_emitter_to_the_runs_custom_channel(monkeypatch):
    """The emitter writes to the run, not to a registry entry keyed by state."""
    from app.ai.graph import MultiAgentWorkflow
    from app.core.config import settings

    monkeypatch.setattr(settings, "enable_image_streaming", True)
    written: list[dict] = []
    monkeypatch.setattr("app.ai.graph._graph_stream_writer", lambda: written.append)

    emitter = MultiAgentWorkflow._build_image_preview_emitter(None, {})
    assert emitter is not None

    emitter({"item_id": "image-preview-0", "image_index": 0})

    assert len(written) == 1
    assert written[0]["type"] == "image_preview"
    assert written[0]["item_id"] == "image-preview-0"

    # Agent attribution is the projection's job now, not the emitter's.
    from app.services.event_streaming.langchain_v3 import V3ProtocolTranslator

    [projected] = V3ProtocolTranslator().translate(
        {
            "type": "event",
            "method": "custom",
            "params": {"namespace": [], "timestamp": 0, "data": written[0]},
        }
    )
    assert projected.type == "image_preview"
    assert projected.agent == "image_generator_agent"
    assert projected.data == {"item_id": "image-preview-0", "image_index": 0}


def test_graph_emitter_disabled_by_flag_or_outside_a_run(monkeypatch):
    from app.ai.graph import MultiAgentWorkflow
    from app.core.config import settings

    monkeypatch.setattr("app.ai.graph._graph_stream_writer", lambda: (lambda _e: None))
    monkeypatch.setattr(settings, "enable_image_streaming", False)
    assert MultiAgentWorkflow._build_image_preview_emitter(None, {}) is None

    # Enabled, but there is no run to write into.
    monkeypatch.setattr(settings, "enable_image_streaming", True)
    monkeypatch.setattr("app.ai.graph._graph_stream_writer", lambda: None)
    assert MultiAgentWorkflow._build_image_preview_emitter(None, {}) is None


# ---------------------------------------------------------------------------
# Phase-0 RED characterization (T001): desired-but-unmet contracts
#
# These pin the *target* image-delivery contracts that Phase 1 (T002-T005)
# will make GREEN. They intentionally FAIL on current source. The existing
# ``test_publisher_drops_oversized_payloads`` above documents TODAY'S behavior;
# the tests below document the intended behavior and must be flipped, not
# deleted, when the fix lands.
# ---------------------------------------------------------------------------


def test_oversized_final_image_is_delivered_not_silently_dropped_CHARACTERIZATION():
    """RED: an oversized FINAL image must still be delivered early (by
    reference), never silently dropped.

    Current ``ImagePreviewPublisher`` drops any payload whose base64 exceeds
    ``max_b64_chars`` with only a ``logger.info`` — no metric, no event
    (``emitter.py:70-78``). A real generated image commonly exceeds the cap, so
    no early image event exists at all (FR-IMG-002/FR-IMG-003/FR-IMG-007).
    """
    emitted: list[dict] = []
    with use_image_preview_emitter(emitted.append):
        publisher = ImagePreviewPublisher(enabled=True, max_b64_chars=3)
        oversized_b64 = "QUJDRA"  # 6 chars > cap of 3
        publisher.publish(
            image_index=0,
            status="final",
            mime="image/png",
            data_b64=oversized_b64,
            seq=1,
        )
    assert emitted, (
        "DEFECT (emitter.py:70-78): oversized FINAL image was dropped with only "
        "a logger.info; no stream event and no reference delivery were produced. "
        f"[sizes] final_b64_len={len('QUJDRA')} chars > cap=3 chars"
    )


@pytest.mark.asyncio
async def test_ai_sdk_v6_terminal_file_part_preserves_protected_reference_CHARACTERIZATION():
    """RED: a terminal ``file`` part sourced from a protected
    ``/chat-images/{id}`` reference must preserve that URL verbatim so the
    client can fetch it with credentials.

    Current ``_normalize_image_item_to_file_part`` only special-cases ``data:``
    and ``http/https/blob:`` URLs; a protected relative URL falls through to
    ``base64.b64decode(..., validate=False)`` and is reinterpreted as loose
    base64 into a corrupt ``data:`` URL (``ai_sdk_projection.py:104-154``).
    """
    protected_url = "/chat-images/11111111-1111-4111-8111-111111111111"

    async def source():
        yield make_event(
            "complete",
            sequence=1,
            data={
                "message": {
                    "id": "m-1",
                    "message_metadata": {
                        "images": [{"url": protected_url, "mime": "image/png"}]
                    },
                }
            },
        )

    payloads = await _collect_ai_sdk_payloads(source)
    file_parts = [
        payload
        for payload in payloads
        if isinstance(payload, dict) and payload.get("type") == "file"
    ]
    urls = [part.get("url") for part in file_parts]
    assert protected_url in urls, (
        "DEFECT (ai_sdk_projection.py:104-154): the protected relative image URL "
        "was not preserved as a `file` url; it was reinterpreted as loose base64 "
        f"and mangled (FR-IMG-006). expected {protected_url!r}; got file urls {urls}"
    )
