"""A turn-scoped channel for candidates that passed visual verification.

Approved candidates must reach the rich-item inventory without appearing in the
model-visible tool result, and rejected candidates must be unreachable rather
than merely unmentioned. The sink is a list owned by the tool-execution layer
and mutated by the tool, so no candidate is ever serialized into the text the
model reads.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_sink: ContextVar[list[dict[str, Any]] | None] = ContextVar("verified_image_sink", default=None)


@contextmanager
def verified_image_sink() -> Iterator[list[dict[str, Any]]]:
    """Collect verified candidates offered while the block is active."""

    collected: list[dict[str, Any]] = []
    token = _sink.set(collected)
    try:
        yield collected
    finally:
        _sink.reset(token)


def offer_verified_images(candidates: Sequence[Mapping[str, Any]]) -> None:
    """Offer approved candidates to the active sink, if any."""

    sink = _sink.get()
    if sink is None:
        return
    sink.extend(dict(candidate) for candidate in candidates)
