"""Per-request escape hatch for streaming image previews out of a graph node.

``_image_generator_node`` installs an emitter (bound to the run's
``SubagentEventSink``) around the agent invocation; the agent publishes
previews through :class:`ImagePreviewPublisher` without knowing anything
about graph state. ``ContextVar`` scoping keeps concurrent requests isolated
and makes non-streaming contexts (direct ``invoke_model``, tests, resumed
runs) a silent no-op.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

from app.services.event_streaming.events import (
    build_image_preview_reference_data,
    build_image_preview_skipped_data,
)

from .models import MediaDeliveryError

logger = logging.getLogger(__name__)

PreviewEmitter = Callable[[dict[str, Any]], None]

_image_preview_emitter: ContextVar[PreviewEmitter | None] = ContextVar(
    "image_preview_emitter", default=None
)

PREVIEW_ITEM_ID_PREFIX = "image-preview-"
FINAL_ITEM_ID_PREFIX = "image-final-"


@contextmanager
def use_image_preview_emitter(emitter: PreviewEmitter | None) -> Iterator[None]:
    token = _image_preview_emitter.set(emitter)
    try:
        yield
    finally:
        _image_preview_emitter.reset(token)


def current_image_preview_emitter() -> PreviewEmitter | None:
    return _image_preview_emitter.get()


class ImagePreviewPublisher:
    """Applies the emission policy before handing payloads to the emitter.

    - disabled entirely when streaming is off or no emitter is installed
    - drops payloads whose base64 body exceeds ``max_b64_chars`` (the image
      still arrives with the terminal ``complete`` event)
    - emits ``final`` at most once per image index; ``partial`` payloads are
      keyed by a monotonically increasing ``seq`` so clients replace in place
    - never lets a publish failure break generation
    """

    def __init__(self, *, enabled: bool, max_b64_chars: int) -> None:
        self._emitter = current_image_preview_emitter() if enabled else None
        self._max_b64_chars = max(0, int(max_b64_chars))
        self._final_indexes: set[int] = set()
        self._counters: dict[str, int] = {
            "emitted": 0,
            "coalesced": 0,
            "oversized": 0,
        }

    @property
    def counters(self) -> dict[str, int]:
        return dict(self._counters)

    def _emit(self, payload: dict[str, Any]) -> bool:
        """Hand a payload to the bound emitter, swallowing sink failures.

        Never logs the payload — image base64 must never reach logs/traces.
        """
        if self._emitter is None:
            return False
        try:
            self._emitter(payload)
        except Exception as err:
            logger.warning(
                "Image preview emission failed (index=%s): %s",
                payload.get("image_index"),
                err,
            )
            return False
        return True

    def publish(
        self,
        *,
        image_index: int,
        status: str,
        mime: str,
        data_b64: str,
        seq: int = 0,
    ) -> bool:
        if self._emitter is None or not data_b64:
            return False
        if len(data_b64) > self._max_b64_chars:
            self._counters["oversized"] += 1
            # First-defense budget: never a silent drop. Surface a structured
            # ``preview_skipped`` status carrying no base64 (FR-IMG-007). The
            # authoritative image still arrives (finals by reference through
            # ``MediaDeliveryService.persist_final``; else with ``complete``).
            self._emit(
                build_image_preview_skipped_data(
                    image_index=image_index,
                    item_id=f"{PREVIEW_ITEM_ID_PREFIX}{image_index}",
                    media_type=mime or "image/png",
                    seq=seq,
                    reason="oversized_inline_preview",
                )
            )
            return False
        if status == "final":
            if image_index in self._final_indexes:
                self._counters["coalesced"] += 1
                return False
            self._final_indexes.add(image_index)
        payload = {
            "item_id": f"{PREVIEW_ITEM_ID_PREFIX}{image_index}",
            "image_index": image_index,
            "status": status,
            "mime": mime or "image/png",
            "data_b64": data_b64,
            "seq": seq,
        }
        if not self._emit(payload):
            return False
        self._counters["emitted"] += 1
        return True

    def emit_reference(
        self,
        *,
        image_index: int,
        image_id: Any,
        url: str,
        media_type: str,
        seq: int,
    ) -> bool:
        """Emit a schema-v2 FINAL image_preview delivered by protected reference.

        Bypasses the inline base64 budget entirely — a final is always by
        reference, so its early delivery does not depend on the SSE cap
        (FR-IMG-003). No-op when no emitter is bound (non-streaming/resume).
        """
        if self._emitter is None or not url:
            return False
        payload = build_image_preview_reference_data(
            image_index=image_index,
            item_id=f"{PREVIEW_ITEM_ID_PREFIX}{image_index}",
            image_id=image_id,
            url=url,
            media_type=media_type,
            seq=seq,
        )
        if not self._emit(payload):
            return False
        self._counters["emitted"] += 1
        return True


_media_delivery_service: ContextVar[MediaDeliveryService | None] = ContextVar(
    "media_delivery_service", default=None
)


@contextmanager
def use_media_delivery_service(
    service: MediaDeliveryService | None,
) -> Iterator[None]:
    token = _media_delivery_service.set(service)
    try:
        yield
    finally:
        _media_delivery_service.reset(token)


def current_media_delivery_service() -> MediaDeliveryService | None:
    return _media_delivery_service.get()


class MediaDeliveryService:
    """Per-run delivery of generated images: transient previews + durable storage.

    Bound once at the graph boundary with the run's user/conversation context
    and storage backend, then read by the image generator agent through
    :func:`current_media_delivery_service`. It owns two concerns:

    - ``publish_partial`` — stream an in-progress preview (transient, bounded by
      the SSE character cap the preview publisher enforces).
    - ``persist_final`` — publish the final transient preview *and* persist the
      final bytes to storage the moment the image is ready, returning a
      reference descriptor. Storage is bounded by the storage byte cap, not the
      transient preview character cap, so a final too large to preview is still
      persisted.

    ``persist_final`` is idempotent per ``(run id, item id, content hash)`` and
    returns the existing descriptor on repeat, so terminal persistence can reuse
    it instead of decoding and writing the same bytes again. Storage failures do
    not raise: they are recorded as a typed :class:`MediaDeliveryError` and
    ``persist_final`` returns ``None`` so the caller preserves its current
    behavior (the image keeps its inline bytes for the terminal fallback).
    """

    def __init__(
        self,
        *,
        storage: Any,
        conversation_id: Any,
        user_id: Any,
        preview_publisher: ImagePreviewPublisher,
        request_id: str | None = None,
    ) -> None:
        self._storage = storage
        self._conversation_id = conversation_id
        self._user_id = user_id
        self._preview = preview_publisher
        self._run_id = request_id or uuid4().hex
        self._descriptors: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._failures: list[MediaDeliveryError] = []
        self._max_seq = 0
        self._referenced_item_ids: set[str] = set()
        self._reference_emitted = 0

    @property
    def failures(self) -> list[MediaDeliveryError]:
        return list(self._failures)

    @property
    def counters(self) -> dict[str, int]:
        counters = {
            "reference_emitted": self._reference_emitted,
            "storage_failed": len(self._failures),
        }
        counters.update({f"preview_{k}": v for k, v in self._preview.counters.items()})
        return counters

    def publish_partial(
        self,
        *,
        image_index: int,
        mime: str,
        data_b64: str,
        seq: int = 0,
    ) -> bool:
        self._max_seq = max(self._max_seq, int(seq or 0))
        return self._preview.publish(
            image_index=image_index,
            status="partial",
            mime=mime,
            data_b64=data_b64,
            seq=seq,
        )

    def persist_final(
        self,
        *,
        image_index: int,
        mime: str,
        data_b64: str,
    ) -> dict[str, Any] | None:
        # Persist the bytes first (bounded by the storage byte cap, not the
        # transient SSE preview char cap). When a durable reference exists, the
        # final is delivered EARLY and ALWAYS by protected reference, so its
        # early delivery does not depend on the base64 SSE cap (FR-IMG-002/003).
        descriptor = self._store_final(image_index=image_index, mime=mime, data_b64=data_b64)
        if descriptor is None:
            # No durable reference (storage-less fallback for
            # non-streaming/resume/tests, or a storage failure): preserve the
            # transient inline-preview behavior — the terminal complete still
            # carries the bytes. Oversized inline previews surface a structured
            # ``preview_skipped`` status rather than being silently dropped.
            self._preview.publish(
                image_index=image_index,
                status="final",
                mime=mime,
                data_b64=data_b64,
            )
            return None
        item_id = f"{PREVIEW_ITEM_ID_PREFIX}{image_index}"
        if item_id not in self._referenced_item_ids:
            self._referenced_item_ids.add(item_id)
            self._max_seq += 1
            if self._preview.emit_reference(
                image_index=image_index,
                image_id=descriptor.get("image_id"),
                url=descriptor.get("url"),
                media_type=descriptor.get("mime") or mime,
                seq=self._max_seq,
            ):
                self._reference_emitted += 1
        return descriptor

    def _store_final(
        self,
        *,
        image_index: int,
        mime: str,
        data_b64: str,
    ) -> dict[str, Any] | None:
        if self._storage is None or not data_b64 or not self._user_id:
            return None

        item_id = f"{FINAL_ITEM_ID_PREFIX}{image_index}"
        content_hash = hashlib.sha256(data_b64.encode("ascii", "ignore")).hexdigest()
        key = (self._run_id, item_id, content_hash)
        cached = self._descriptors.get(key)
        if cached is not None:
            return dict(cached)

        try:
            ref = self._storage.store(
                conversation_id=self._conversation_id,
                user_id=self._user_id,
                mime=mime or "image/png",
                data_b64=data_b64,
                name="generated-image",
            )
        except Exception as err:
            self._record_failure(item_id, err)
            return None

        descriptor = {
            "image_id": ref["image_id"],
            "url": ref["url"],
            "mime": ref.get("mime") or mime or "image/png",
            "name": ref.get("name") or "generated-image",
        }
        if ref.get("content_hash"):
            descriptor["content_hash"] = ref["content_hash"]
        self._descriptors[key] = descriptor
        return dict(descriptor)

    def _record_failure(self, item_id: str, err: Exception) -> None:
        failure = MediaDeliveryError(
            code="media_delivery_persist_failed",
            item_id=item_id,
            detail=type(err).__name__,
        )
        self._failures.append(failure)
        logger.warning(
            "Final image persistence failed code=%s item_id=%s detail=%s",
            failure.code,
            failure.item_id,
            failure.detail,
        )
