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
    active_agent_id: str | None = None
    # True once the run has paused on a HITL interrupt and can resume with the
    # same ``active_agent_id``. Paused entries remain in the registry so custom
    # agent edit/delete/detach stay blocked until the run resolves.
    paused: bool = False
    started_at: float = field(default_factory=time.monotonic)
    last_event_at: float = field(default_factory=time.monotonic)

    # Resolved when generation ends (complete / interrupt / cancel / error).
    # Value: the final/partial assistant message dict, or None.
    done: asyncio.Future = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.done is None:
            # Created inside the running event loop in production. Outside a
            # loop (e.g. lock-only unit tests) we leave ``done`` unset; the
            # lock queries never touch it.
            try:
                self.done = asyncio.get_running_loop().create_future()
            except RuntimeError:
                self.done = None  # type: ignore[assignment]

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
        if self.done is not None and not self.done.done():
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
        *,
        active_agent_id: str | None = None,
        paused: bool = False,
    ) -> InflightEntry:
        """Register a new in-flight generation. Returns the entry."""
        key = str(user_message_id)
        entry = InflightEntry(
            conversation_id=conversation_id,
            user_id=user_id,
            active_agent_id=active_agent_id,
            paused=paused,
        )
        self._store[key] = entry
        logger.debug("Registered in-flight generation for user_message_id=%s", key)
        return entry

    def mark_paused(self, user_message_id: UUID) -> InflightEntry | None:
        """Flag an entry as paused (HITL) so it remains a lock token until resume."""
        entry = self.get(user_message_id)
        if entry is not None:
            entry.paused = True
            entry.touch()
            logger.debug("Marked generation paused for user_message_id=%s", user_message_id)
        return entry

    def set_active_agent_id(self, user_message_id: UUID, agent_id: str | None) -> None:
        """Record the currently selected runtime agent for an entry."""
        entry = self.get(user_message_id)
        if entry is not None:
            entry.active_agent_id = agent_id
            entry.touch()

    # ------------------------------------------------------------------
    # Lock queries (custom-agent edit/delete/detach gating)
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(left, right) -> bool:
        return left is not None and right is not None and str(left) == str(right)

    def find_by_user(self, user_id) -> list[InflightEntry]:
        return [e for e in list(self._store.values()) if self._matches(e.user_id, user_id)]

    def find_by_conversation(self, conversation_id) -> list[InflightEntry]:
        return [
            e
            for e in list(self._store.values())
            if self._matches(e.conversation_id, conversation_id)
        ]

    def is_runtime_agent_in_use(
        self,
        owner_id,
        runtime_agent_id: str,
        conversation_id=None,
    ) -> bool:
        """True if any active or paused generation is running ``runtime_agent_id``.

        When ``conversation_id`` is supplied, also returns True if a generation
        is active in that conversation with no resolved selected agent yet
        (conservative gate while selected-agent metadata is unavailable).
        """
        for entry in list(self._store.values()):
            if not self._matches(entry.user_id, owner_id):
                continue
            if self._matches(entry.active_agent_id, runtime_agent_id):
                return True
            if (
                conversation_id is not None
                and self._matches(entry.conversation_id, conversation_id)
                and not entry.active_agent_id
            ):
                return True
        return False

    def has_active_unknown_agent_in_conversation(self, owner_id, conversation_id) -> bool:
        """True if a generation is active in the conversation with no selected agent."""
        for entry in list(self._store.values()):
            if (
                self._matches(entry.user_id, owner_id)
                and self._matches(entry.conversation_id, conversation_id)
                and not entry.active_agent_id
            ):
                return True
        return False

    def clear_paused_for_conversation(self, owner_id, conversation_id) -> int:
        """Remove paused entries for a conversation once a resume resolves them."""
        removed = 0
        for key, entry in list(self._store.items()):
            if (
                entry.paused
                and self._matches(entry.user_id, owner_id)
                and self._matches(entry.conversation_id, conversation_id)
            ):
                self._store.pop(key, None)
                removed += 1
        return removed

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
