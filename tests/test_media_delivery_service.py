"""T002: per-run media delivery service — early final-image persistence.

The service persists a FINAL generated image to storage the moment it is
produced (before the next narrative delta / terminal event), is idempotent per
(run, item, content), and records a typed failure instead of raising when
storage is unavailable. Final bytes are governed by the STORAGE byte cap, not
the transient SSE preview character cap.
"""

from __future__ import annotations

import base64
import hashlib
from uuid import uuid4

import pytest

from app.ai.agents.image_generator_agent import ImageGeneratorAgent
from app.ai.image_generation import (
    ImageFinal,
    ImageGenerationRequest,
    ImageUsage,
    MediaDeliveryError,
    MediaDeliveryService,
    NarrativeDelta,
    use_media_delivery_service,
)
from app.ai.image_generation.emitter import ImagePreviewPublisher, use_image_preview_emitter
from app.usage.types import NormalizedUsage


class _RecordingStorage:
    """Minimal ChatImageStorageService stand-in that records store() calls."""

    def __init__(self, *, max_bytes: int = 10_000_000, timeline: list | None = None):
        self.max_bytes = max_bytes
        self.calls: list[dict] = []
        self._timeline = timeline

    def store(self, *, conversation_id, user_id, mime, data_b64, name):
        if self._timeline is not None:
            self._timeline.append("store")
        raw = base64.b64decode(data_b64)
        if len(raw) > self.max_bytes:
            raise ValueError("too big")
        sha = hashlib.sha256(raw).hexdigest()
        image_id = str(uuid4())
        record = {
            "name": name or "image",
            "mime": mime,
            "image_id": image_id,
            "url": f"/chat-images/{image_id}",
            "content_hash": sha,
        }
        self.calls.append(record)
        return record


class _BoomStorage:
    max_bytes = 10_000_000

    def store(self, **_kwargs):
        raise RuntimeError("storage down")


def _service(storage, *, preview_enabled=False, preview_cap=100):
    return MediaDeliveryService(
        storage=storage,
        conversation_id=uuid4(),
        user_id=uuid4(),
        preview_publisher=ImagePreviewPublisher(enabled=preview_enabled, max_b64_chars=preview_cap),
    )


def _agent() -> ImageGeneratorAgent:
    agent = ImageGeneratorAgent.__new__(ImageGeneratorAgent)
    agent.model_name = "gemini-3-image"
    agent.default_aspect_ratio = "1:1"
    return agent


@pytest.mark.asyncio
async def test_final_image_persisted_before_next_narrative_delta():
    """The final image must be stored before the next narrative delta/terminal
    event is even pulled from the provider stream."""
    timeline: list[str] = []
    storage = _RecordingStorage(timeline=timeline)
    final_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"final-bytes" * 4).decode()

    class _Provider:
        async def stream_generate(self, _request):
            yield ImageFinal(index=0, data_b64=final_b64, mime="image/png")
            timeline.append("narrative")
            yield NarrativeDelta(text="here is your image")
            yield ImageUsage(usage=NormalizedUsage(source="test"))

    service = _service(storage)
    request = ImageGenerationRequest(prompt="p", model="gemini-3-image")

    with use_media_delivery_service(service):
        outcome = await _agent()._consume_image_stream(
            _Provider(), request, "draw a cat", handle=None
        )

    assert timeline == ["store", "narrative"]
    assert len(storage.calls) == 1
    ref = outcome.images[0]["stored_ref"]
    assert ref["url"].startswith("/chat-images/")
    assert ref["image_id"]
    # the persisted descriptor must never carry inline bytes
    assert "data" not in ref and "b64_data" not in ref


def test_persist_final_is_idempotent_per_item_and_content():
    storage = _RecordingStorage()
    service = _service(storage)
    b64 = base64.b64encode(b"same-bytes").decode()

    first = service.persist_final(image_index=0, mime="image/png", data_b64=b64)
    second = service.persist_final(image_index=0, mime="image/png", data_b64=b64)

    assert len(storage.calls) == 1
    assert first == second
    assert first["image_id"]


def test_persist_final_stores_distinct_items_separately():
    storage = _RecordingStorage()
    service = _service(storage)

    service.persist_final(image_index=0, mime="image/png", data_b64=base64.b64encode(b"a").decode())
    service.persist_final(image_index=1, mime="image/png", data_b64=base64.b64encode(b"b").decode())

    assert len(storage.calls) == 2


def test_persist_final_records_typed_error_on_storage_failure():
    service = _service(_BoomStorage())
    b64 = base64.b64encode(b"bytes").decode()

    result = service.persist_final(image_index=0, mime="image/png", data_b64=b64)

    assert result is None
    assert len(service.failures) == 1
    failure = service.failures[0]
    assert isinstance(failure, MediaDeliveryError)
    assert failure.code == "media_delivery_persist_failed"
    assert failure.item_id == "image-final-0"
    # a typed failure must never leak image bytes
    assert b64 not in failure.detail


def test_persist_final_uses_storage_cap_not_preview_char_cap():
    """A final larger than the transient preview char cap must still be stored,
    because final persistence is bounded by the storage byte cap."""
    storage = _RecordingStorage(max_bytes=10_000_000)
    # preview char cap of 3 would drop this payload from the transient preview
    service = _service(storage, preview_enabled=True, preview_cap=3)
    big_b64 = base64.b64encode(b"x" * 4096).decode()

    result = service.persist_final(image_index=0, mime="image/png", data_b64=big_b64)

    assert result is not None
    assert len(storage.calls) == 1
    assert result["url"].startswith("/chat-images/")


def test_publish_partial_delegates_to_preview_publisher():
    emitted: list[dict] = []
    with use_image_preview_emitter(emitted.append):
        service = _service(None, preview_enabled=True, preview_cap=100)
        assert (
            service.publish_partial(image_index=0, mime="image/png", data_b64="QUJD", seq=2) is True
        )
    assert emitted == [
        {
            "item_id": "image-preview-0",
            "image_index": 0,
            "status": "partial",
            "mime": "image/png",
            "data_b64": "QUJD",
            "seq": 2,
        }
    ]
