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
from app.utils.threads import run_to_completion

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


def _close_session(session: Any) -> None:
    """Close the lock's session, ending its transaction and so the lock."""
    try:
        session.close()
    except Exception:
        # Not re-raised: this runs on a turn's way out, and a failed close
        # invalidates the connection, which ends the lock server-side anyway.
        logger.warning("Closing a conversation turn-lock session failed", exc_info=True)


def _close_after_attempt(attempt: asyncio.Future[bool], session: Any) -> None:
    """Close the session once an abandoned lock attempt has finished in its thread."""
    if not attempt.cancelled() and attempt.exception() is not None:
        logger.warning(
            "A conversation turn-lock attempt failed: %s", type(attempt.exception()).__name__
        )
    attempt.get_loop().run_in_executor(None, _close_session, session)


class PostgresAdvisoryLockBackend:
    """A cross-process PostgreSQL advisory lock keyed on the conversation.

    Advisory locks take a bigint, so the conversation ID is hashed into one
    deterministically. ``pg_try_advisory_xact_lock`` is polled rather than
    blocking, so the caller's timeout is the one that applies.

    The lock is transaction-scoped, not session-scoped. A session-scoped lock
    survives ``Session.close()``: the pool's reset on return is a ROLLBACK,
    which does not release it, so any path that missed the explicit unlock (a
    client disconnect cancelling the release) parked the lock on a pooled
    connection. Every other worker was then refused that conversation, and the
    next borrower of that connection re-entered the lock and stacked it. Tied
    to the transaction, ending the transaction is the release, and every way a
    session can end — close, pool reset, garbage collection, a dropped
    connection — ends it.
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

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        lock_key = self.lock_key(key)

        # The lock lives in the open transaction of the session that took it,
        # so that exact session is held open until release rather than
        # returned to the pool.
        session = self._require_raw_session(self._session_factory())

        def try_lock() -> bool:
            # The transaction sits idle for the whole turn by design. A server
            # or role idle-in-transaction timeout would kill it mid-turn and
            # drop the lock with nobody told. SET LOCAL ends with the
            # transaction, so the pooled connection keeps the server default.
            session.execute(text("SET LOCAL idle_in_transaction_session_timeout = 0"))
            return bool(
                session.execute(
                    text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": lock_key}
                ).scalar()
            )

        while True:
            attempt = loop.run_in_executor(None, try_lock)
            try:
                # Shielded so a cancelled caller leaves ``attempt`` itself
                # running: its completion is what makes closing safe below.
                acquired = await asyncio.shield(attempt)
            except BaseException:
                # The attempt may still take the lock in its thread after the
                # caller is gone. Closing the session alongside it would use
                # one Session from two threads, so close once it has finished.
                attempt.add_done_callback(lambda done: _close_after_attempt(done, session))
                raise
            if acquired:
                self._sessions[key] = session
                return True
            if loop.time() >= deadline:
                await run_to_completion(_close_session, session)
                return False
            await asyncio.sleep(self._poll_interval_seconds)

    async def release(self, key: str) -> None:
        session = self._sessions.pop(key, None)
        if session is None:
            return
        # Ending the transaction is the release, and it is one job on one
        # thread: a release cancelled by a client disconnect still runs it to
        # completion, and nothing touches the session concurrently.
        await run_to_completion(_close_session, session)


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
