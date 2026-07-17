"""Per-request escape hatch for streaming image previews out of a graph node.

``_image_generator_node`` installs an emitter (bound to the run's
``SubagentEventSink``) around the agent invocation; the agent publishes
previews through :class:`ImagePreviewPublisher` without knowing anything
about graph state. ``ContextVar`` scoping keeps concurrent requests isolated
and makes non-streaming contexts (direct ``invoke_model``, tests, resumed
runs) a silent no-op.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

PreviewEmitter = Callable[[dict[str, Any]], None]

_image_preview_emitter: ContextVar[PreviewEmitter | None] = ContextVar(
    "image_preview_emitter", default=None
)

PREVIEW_ITEM_ID_PREFIX = "image-preview-"


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
            logger.info(
                "Skipping image preview %s/%s: payload %d chars exceeds cap %d",
                image_index,
                status,
                len(data_b64),
                self._max_b64_chars,
            )
            return False
        if status == "final":
            if image_index in self._final_indexes:
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
        try:
            self._emitter(payload)
        except Exception as err:
            logger.warning("Image preview emission failed (index=%s): %s", image_index, err)
            return False
        return True
