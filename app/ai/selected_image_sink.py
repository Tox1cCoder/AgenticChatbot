"""A turn-scoped channel for provider-selected rich image candidates.

Selected candidates reach the rich-item inventory without appearing in the
model-visible tool result. The sink is owned by the tool-execution layer and
mutated by the tool, so image selection remains out-of-band from model text.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_sink: ContextVar[list[dict[str, Any]] | None] = ContextVar("selected_image_sink", default=None)


@contextmanager
def selected_image_sink() -> Iterator[list[dict[str, Any]]]:
    """Collect provider-selected candidates offered while the block is active."""

    collected: list[dict[str, Any]] = []
    token = _sink.set(collected)
    try:
        yield collected
    finally:
        _sink.reset(token)


def offer_selected_images(candidates: Sequence[Mapping[str, Any]]) -> None:
    """Offer provider-selected candidates to the active sink, if any."""

    sink = _sink.get()
    if sink is not None:
        sink.extend(dict(candidate) for candidate in candidates)
