"""Revocation state for issued access and refresh tokens.

Tokens are stateless, so a soft-deleted user's token kept authenticating until it
expired. Each token now carries the user's ``token_version`` as the ``ver`` claim,
and the auth dependencies compare it with the user's current version and
soft-delete state, read through :class:`TokenStateCache`.

A token with no ``ver`` claim counts as version 0: tokens issued before the claim
existed stay valid until they expire, and every user starts at version 0.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

TOKEN_VERSION_CLAIM = "ver"
DEFAULT_TTL_SECONDS = 30.0
DEFAULT_MAX_ENTRIES = 10_000


@dataclass(frozen=True)
class TokenState:
    """What a token check needs about its user."""

    version: int
    deleted: bool


TokenStateLoader = Callable[[UUID], "TokenState | None"]


def token_version_claim(payload: dict) -> int | None:
    """The token's ``ver``, 0 when absent, None when present but not an integer."""
    version = payload.get(TOKEN_VERSION_CLAIM, 0)
    if isinstance(version, bool) or not isinstance(version, int):
        return None
    return version


def with_token_version(data: dict, version: int) -> dict:
    """Token claims ``data`` stamped with the user's current version."""
    return {**data, TOKEN_VERSION_CLAIM: int(version)}


class TokenStateCache:
    """A bounded, per-process TTL cache of :class:`TokenState` by user id.

    A revocation made in another process is seen here within ``ttl_seconds``;
    one made in this process is seen at once through :meth:`invalidate`. A
    missing user is cached too (as ``None``), so a stream of tokens for a
    deleted id does not reach the database on every request.
    """

    def __init__(
        self,
        loader: TokenStateLoader,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0 or max_entries <= 0:
            raise ValueError("ttl_seconds and max_entries must be positive")
        self._loader = loader
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[UUID, tuple[float, TokenState | None]] = OrderedDict()
        self._generation = 0
        self._lock = threading.Lock()

    def peek(self, user_id: UUID) -> tuple[bool, TokenState | None]:
        """``(hit, state)`` without loading. Cheap enough for the event loop."""
        with self._lock:
            entry = self._entries.get(user_id)
            if entry is None:
                return False, None
            expires_at, state = entry
            if self._clock() >= expires_at:
                del self._entries[user_id]
                return False, None
            self._entries.move_to_end(user_id)
            return True, state

    def get(self, user_id: UUID) -> TokenState | None:
        """The cached state, loading it on a miss. The load may block."""
        hit, state = self.peek(user_id)
        if hit:
            return state
        with self._lock:
            generation = self._generation
        state = self._loader(user_id)
        with self._lock:
            if generation != self._generation:
                # An invalidation raced the load, so the row read may predate it.
                return state
            self._entries[user_id] = (self._clock() + self._ttl, state)
            self._entries.move_to_end(user_id)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return state

    def invalidate(self, user_id: UUID) -> None:
        with self._lock:
            self._generation += 1
            self._entries.pop(user_id, None)

    def clear(self) -> None:
        with self._lock:
            self._generation += 1
            self._entries.clear()


def _load_from_repository(user_id: UUID) -> TokenState | None:
    from app.core.container import container

    return container.user_repository().get_token_state(user_id)


token_state_cache = TokenStateCache(_load_from_repository)
