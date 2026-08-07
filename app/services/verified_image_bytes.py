"""Bounded hand-off of verified image bytes to the registration that persists them.

Visual verification already downloads, validates and decodes every candidate it
approves. Registration happens later — in ``message_service``, after the graph
has finished — so by then every ContextVar scope around the tool call has
closed and the bytes are gone. Without this, an approved image is fetched a
second time at render, and can fail then, after it has already been placed.

The store is deliberately small rather than a general cache:

- only approved candidates are remembered, never the rejected majority;
- a reader *takes* its entry, so the common path frees itself immediately;
- a total-byte budget evicts the oldest entries, because nothing guarantees a
  reader ever arrives — an answer can drop an image after it was approved.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from ..core.config import settings

_lock = threading.Lock()
_entries: OrderedDict[tuple[str, str], Any] = OrderedDict()
_total_bytes = 0


def _key(conversation_id: str | None, url: str) -> tuple[str, str]:
    return (str(conversation_id or "__no_conversation__"), str(url or ""))


def _size(image: Any) -> int:
    return len(getattr(image, "content", b"") or b"")


def _budget(explicit: int | None) -> int:
    if explicit is not None:
        return max(0, int(explicit))
    return max(0, int(getattr(settings, "verified_image_cache_max_bytes", 0)))


def remember_verified_bytes(
    conversation_id: str | None,
    url: str,
    image: Any,
    *,
    budget: int | None = None,
) -> None:
    """Hold an approved image's validated bytes until registration takes them."""

    global _total_bytes
    size = _size(image)
    limit = _budget(budget)
    if size <= 0 or size > limit:
        # A single image that cannot fit the whole budget would evict every
        # other entry and still not be storable; refuse it outright.
        return

    key = _key(conversation_id, url)
    with _lock:
        existing = _entries.pop(key, None)
        if existing is not None:
            _total_bytes -= _size(existing)
        _entries[key] = image
        _total_bytes += size
        while _total_bytes > limit and _entries:
            _, evicted = _entries.popitem(last=False)
            _total_bytes -= _size(evicted)


def take_verified_bytes(conversation_id: str | None, url: str) -> Any | None:
    """Return and release the bytes held for ``url``, or None if there are none."""

    global _total_bytes
    key = _key(conversation_id, url)
    with _lock:
        image = _entries.pop(key, None)
        if image is not None:
            _total_bytes -= _size(image)
        return image


def forget_conversation_bytes(conversation_id: str | None) -> None:
    """Drop everything held for one conversation."""

    global _total_bytes
    bucket = _key(conversation_id, "")[0]
    with _lock:
        for key in [key for key in _entries if key[0] == bucket]:
            _total_bytes -= _size(_entries.pop(key))
