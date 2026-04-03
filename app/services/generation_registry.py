"""
In-flight generation registry for tracking and cancelling active streaming responses.

Provides a module-level singleton backed by cachetools.TTLCache to prevent unbounded
growth if cleanup fails. Entries are keyed by the persisted user message ID created
at stream start.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from uuid import UUID

from cachetools import TTLCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entry shape
# ---------------------------------------------------------------------------


@dataclass
class InflightEntry:
    """Tracks a single in-flight streaming generation."""

    conversation_id: UUID
    user_id: UUID
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    partial_text: str = ""
    partial_thinking: str = ""
    selected_agent: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    last_event_at: float = field(default_factory=time.monotonic)

    # Resolved when generation ends (complete / interrupt / cancel / error).
    # Value: the final/partial assistant message dict, or None.
    done: asyncio.Future = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.done is None:
            loop = asyncio.get_event_loop()
            self.done = loop.create_future()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def touch(self) -> None:
        """Update last_event_at to now."""
        self.last_event_at = time.monotonic()

    def request_cancel(self) -> None:
        """Signal cancellation to the producer."""
        self.cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def resolve(self, result=None) -> None:
        """Resolve the ``done`` future if not already resolved."""
        if not self.done.done():
            self.done.set_result(result)


# ---------------------------------------------------------------------------
# Registry singleton
# ---------------------------------------------------------------------------


class GenerationRegistry:
    """
    Best-effort in-memory registry of in-flight streaming generations.

    Keyed by the *user_message_id* (UUID) emitted in the ``user_message_created``
    SSE event. Backed by a TTLCache so orphan entries expire automatically.
    """

    def __init__(self, maxsize: int = 1000, ttl: int = 600) -> None:
        self._store: TTLCache[str, InflightEntry] = TTLCache(maxsize=maxsize, ttl=ttl)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        user_message_id: UUID,
        conversation_id: UUID,
        user_id: UUID,
    ) -> InflightEntry:
        """Register a new in-flight generation. Returns the entry."""
        key = str(user_message_id)
        entry = InflightEntry(conversation_id=conversation_id, user_id=user_id)
        self._store[key] = entry
        logger.debug("Registered in-flight generation for user_message_id=%s", key)
        return entry

    def get(self, user_message_id: UUID) -> InflightEntry | None:
        """Look up an in-flight entry (returns ``None`` if expired/missing)."""
        return self._store.get(str(user_message_id))

    def remove(self, user_message_id: UUID) -> InflightEntry | None:
        """Remove and return an entry (idempotent)."""
        key = str(user_message_id)
        entry = self._store.pop(key, None)
        if entry is not None:
            logger.debug("Removed in-flight entry for user_message_id=%s", key)
        return entry

    def cancel(self, user_message_id: UUID) -> InflightEntry | None:
        """Signal cancellation for a given user_message_id. Returns the entry or None."""
        entry = self.get(user_message_id)
        if entry is not None:
            entry.request_cancel()
            logger.info("Cancellation requested for user_message_id=%s", user_message_id)
        return entry

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, user_message_id: UUID) -> bool:
        return str(user_message_id) in self._store


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: GenerationRegistry | None = None


def get_generation_registry() -> GenerationRegistry:
    """Return the module-level singleton registry (created on first call)."""
    global _registry
    if _registry is None:
        _registry = GenerationRegistry()
    return _registry
