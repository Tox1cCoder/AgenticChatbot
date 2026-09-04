"""Serializes turns within one conversation, and nothing wider.

Two turns in the same conversation that overlap each snapshot a history the
other is about to change, so the second answers from a view that no longer
exists by the time it writes. The lock is held from context snapshot through
response persistence — releasing at the end of generation would leave exactly
the window that matters unprotected.

Different conversations share nothing, so they are never serialized against
each other: a global lock would trade a real correctness bug for a real
latency one.

Acquisition is bounded. Waiting forever turns a contended conversation into a
hung request; a bounded wait turns it into a retriable
``conversation_turn_conflict`` the caller can act on.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol

from app.ai.workflow.contracts import WorkflowRoutingException
from app.ai.workflow.errors import workflow_error

logger = logging.getLogger(__name__)

__all__ = [
    "ConversationTurnCoordinator",
    "InProcessTurnLockBackend",
    "PostgresAdvisoryLockBackend",
    "TurnLockBackend",
]


class TurnLockBackend(Protocol):
    """A lock that can be held across the whole turn."""

    #: Whether this backend coordinates across processes. An in-process lock
    #: cannot, which is why production refuses one.
    durable: bool

    async def acquire(self, key: str, *, timeout_seconds: float) -> bool: ...

    async def release(self, key: str) -> None: ...


class InProcessTurnLockBackend:
    """An asyncio lock per conversation. Test and single-process use only.

    Two workers each hold their own copy of this, so it coordinates nothing
    between them — hence ``durable = False``.
    """

    durable = False

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    async def acquire(self, key: str, *, timeout_seconds: float) -> bool:
        lock = self._locks.setdefault(key, asyncio.Lock())
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout_seconds)
        except TimeoutError:
            return False
        return True

    async def release(self, key: str) -> None:
        lock = self._locks.get(key)
        if lock is not None and lock.locked():
            lock.release()


class PostgresAdvisoryLockBackend:
    """A cross-process PostgreSQL advisory lock keyed on the conversation.

    Advisory locks are session-scoped and take a bigint, so the conversation ID
    is hashed into one deterministically. ``pg_try_advisory_lock`` is polled
    rather than blocking, so the caller's timeout is the one that applies.
    """

    durable = True

    def __init__(self, session_factory: Any, *, poll_interval_seconds: float = 0.05) -> None:
        self._session_factory = session_factory
        self._poll_interval_seconds = max(0.001, float(poll_interval_seconds))
        self._sessions: dict[str, Any] = {}

    @staticmethod
    def _require_raw_session(candidate: Any) -> Any:
        """Reject a ``@contextmanager`` session factory at the wiring boundary.

        A context-managed session is closed when its block exits, which would
        release the advisory lock the moment it was taken. The failure mode
        without this check is an ``AttributeError`` on the first query, which
        names neither the cause nor the fix.
        """
        if hasattr(candidate, "execute"):
            return candidate
        raise TypeError(
            "the advisory-lock session factory must return a Session held open "
            "until release, not a context manager; got "
            f"{type(candidate).__name__}"
        )

    @staticmethod
    def lock_key(conversation_id: str) -> int:
        """Hash a conversation ID into a signed 64-bit advisory-lock key."""
        digest = hashlib.sha256(str(conversation_id).encode("utf-8")).digest()
        unsigned = int.from_bytes(digest[:8], "big", signed=False)
        return unsigned - (2**64) if unsigned >= 2**63 else unsigned

    async def acquire(self, key: str, *, timeout_seconds: float) -> bool:
        from sqlalchemy import text

        deadline = asyncio.get_running_loop().time() + timeout_seconds
        lock_key = self.lock_key(key)

        # The lock lives on the session that took it, so that exact session is
        # held open until release rather than returned to the pool.
        session = self._require_raw_session(self._session_factory())
        while True:
            acquired = await asyncio.to_thread(
                lambda: session.execute(
                    text("SELECT pg_try_advisory_lock(:key)"), {"key": lock_key}
                ).scalar()
            )
            if acquired:
                self._sessions[key] = session
                return True
            if asyncio.get_running_loop().time() >= deadline:
                await asyncio.to_thread(session.close)
                return False
            await asyncio.sleep(self._poll_interval_seconds)

    async def release(self, key: str) -> None:
        from sqlalchemy import text

        session = self._sessions.pop(key, None)
        if session is None:
            return
        lock_key = self.lock_key(key)
        try:
            await asyncio.to_thread(
                lambda: session.execute(
                    text("SELECT pg_advisory_unlock(:key)"), {"key": lock_key}
                ).scalar()
            )
        finally:
            await asyncio.to_thread(session.close)


class ConversationTurnCoordinator:
    """Holds one conversation's turn lock for the life of a turn."""

    def __init__(
        self,
        *,
        backend: TurnLockBackend,
        timeout_seconds: float,
        production: bool = False,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if production and not getattr(backend, "durable", False):
            raise ValueError(
                "an in-process turn lock coordinates nothing between workers; "
                "production requires a durable backend"
            )
        self._backend = backend
        self._timeout_seconds = float(timeout_seconds)
        self._active: dict[str, int] = {}

    def max_active_for(self, conversation_id: str) -> int:
        """Peak concurrent holders observed for a conversation (diagnostics)."""
        return self._active.get(str(conversation_id), 0)

    @contextlib.asynccontextmanager
    async def hold(self, conversation_id: str | None, *, request_id: str) -> AsyncIterator[None]:
        """Hold the conversation's turn lock for the body of the turn.

        A turn with no conversation takes no lock: it shares no history with
        anything, and a shared fallback key would serialize unrelated turns.
        """
        key = str(conversation_id or "").strip()
        if not key:
            yield
            return

        acquired = await self._backend.acquire(key, timeout_seconds=self._timeout_seconds)
        if not acquired:
            logger.info(
                "Conversation %s is already running a turn; refusing after %.2fs",
                key,
                self._timeout_seconds,
            )
            raise WorkflowRoutingException(
                workflow_error(
                    "conversation_turn_conflict",
                    request_id=request_id,
                    details={"reason": "turn_already_running"},
                )
            )

        self._active[key] = self._active.get(key, 0) + 1
        try:
            yield
        finally:
            # Released in a finally so a failed, cancelled, or timed-out turn
            # cannot strand the conversation.
            self._active[key] = max(0, self._active.get(key, 1) - 1)
            await self._backend.release(key)
