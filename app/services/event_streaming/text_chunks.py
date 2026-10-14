"""Explicit text chunk semantics for streamed model output.

Callers declare what a chunk is instead of the normalizer guessing from its
content. Guessing is what lost text: a model that wrote ``ha`` twice looked
like a provider resending its accumulated answer.

- ``delta``: the chunk is new text. It is returned unchanged, always.
- ``snapshot``: the chunk restates the stream's whole text so far. Only the
  suffix that extends what the stream already produced is returned.

State is scoped by a caller-chosen stream key (run, message, block), so a
repair run that begins with the same words as the draft is its own stream.
"""

from __future__ import annotations

from typing import Literal

ChunkMode = Literal["delta", "snapshot"]
StreamKey = tuple[str, ...]


class TextNormalizationError(ValueError):
    """A snapshot rewrote text the stream had already published."""


class TextChunkAccumulator:
    def __init__(self) -> None:
        self._text: dict[StreamKey, str] = {}

    @property
    def active_streams(self) -> int:
        return len(self._text)

    def consume(self, text: str, *, mode: ChunkMode, stream_key: StreamKey) -> str:
        """Return the text to publish for one chunk of ``stream_key``."""
        if mode == "delta":
            if text:
                self._text[stream_key] = self._text.get(stream_key, "") + text
            return text
        if mode != "snapshot":
            raise ValueError(f"Unknown text chunk mode {mode!r}; expected 'delta' or 'snapshot'.")

        published = self._text.get(stream_key, "")
        if not text.startswith(published):
            raise TextNormalizationError(
                f"Snapshot for stream {stream_key!r} does not extend its published text "
                f"({len(published)} chars); refusing to splice a rewrite into the answer."
            )
        self._text[stream_key] = text
        return text[len(published) :]

    def discard(self, stream_key: StreamKey) -> None:
        """Forget one finished stream."""
        self._text.pop(stream_key, None)

    def clear(self) -> None:
        """Forget every stream, at the end of an invocation."""
        self._text.clear()
