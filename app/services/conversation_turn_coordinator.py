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
import inspect
import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol

from app.ai.workflow.contracts import WorkflowRoutingException
from app.ai.workflow.errors import workflow_error
from app.utils.cancellation import await_to_completion

logger = logging.getLogger(__name__)

__all__ = [
    "ConversationTurnCoordinator",
    "InProcessTurnLockBackend",
    "PostgresAdvisoryLockBackend",
    "TurnLockBackend",
    "TurnLockUnavailable",
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


class TurnLockUnavailable(Exception):
    """No connection was free to take the lock on: the worker is at capacity.

    Distinct from a refused lock, which means another turn holds this
    conversation. Both are worth a retry; only the cause differs.
    """


async def _close_session(session: Any) -> None:
    """Close the lock's session, ending its transaction and so the lock."""
    try:
        await session.close()
    except Exception:
        # Not re-raised: this runs on a turn's way out, and a failed close
        # invalidates the connection, which ends the lock server-side anyway.
        logger.warning("Closing a conversation turn-lock session failed", exc_info=True)


async def _close_to_completion(session: Any) -> None:
    """Close ``session`` even if the caller is cancelled while it closes.

    The same pattern ``AsyncSession.__aexit__`` uses. Under anyio's
    level-triggered cancellation — a client disconnecting from a streamed turn
    — every await on the way out is cancelled at once, and an unshielded close
    would be abandoned with the transaction, and so the lock, still open.
    """
    await await_to_completion(_close_session(session))


class PostgresAdvisoryLockBackend:
    """A cross-process PostgreSQL advisory lock keyed on the conversation.

    Advisory locks take a bigint, so the conversation ID is hashed into one
    deterministically. ``pg_try_advisory_xact_lock`` is polled rather than
    blocking, so the caller's timeout is the one that applies.

    The lock is transaction-scoped, not session-scoped. A session-scoped lock
    survives closing its session: the pool's reset on return is a ROLLBACK,
    which does not release it, so any path that missed the explicit unlock (a
    client disconnect cancelling the release) parked the lock on a pooled
    connection. Tied to the transaction, ending the transaction is the
    release, and every way a session can end — close, pool reset, garbage
    collection, a dropped connection — ends it.

    It runs on an ``AsyncSession``, so neither the lock nor a wait for a pool
    connection occupies a thread. On a sync session in a worker thread, turns
    waiting for a full pool each held a default-executor thread for up to the
    pool timeout, stalling every other ``to_thread`` call in the process.
    """

    durable = True

    def __init__(self, session_factory: Any, *, poll_interval_seconds: float = 0.05) -> None:
        self._session_factory = session_factory
        self._poll_interval_seconds = max(0.001, float(poll_interval_seconds))
        self._sessions: dict[str, Any] = {}

    @staticmethod
    def _require_async_session(candidate: Any) -> Any:
        """Reject a factory that does not hand back an open ``AsyncSession``.

        A context-managed session (``db.provided.async_session``) is closed
        when its block exits, which would release the lock the moment it was
        taken; a sync ``Session`` would block the event loop on every query.
        Without this check either fails on the first query with an error that
        names neither the cause nor the fix.
        """
        if inspect.iscoroutinefunction(getattr(candidate, "execute", None)):
            return candidate
        raise TypeError(
            "the advisory-lock session factory must return an AsyncSession held "
            "open until release, not a context manager or a sync Session; got "
            f"{type(candidate).__name__}"
        )

    @staticmethod
    def lock_key(conversation_id: str) -> int:
        """Hash a conversation ID into a signed 64-bit advisory-lock key."""
        digest = hashlib.sha256(str(conversation_id).encode("utf-8")).digest()
        unsigned = int.from_bytes(digest[:8], "big", signed=False)
        return unsigned - (2**64) if unsigned >= 2**63 else unsigned

    async def acquire(self, key: str, *, timeout_seconds: float) -> bool:
        from sqlalchemy.exc import TimeoutError as PoolTimeoutError

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        lock_key = self.lock_key(key)

        # The lock lives in the open transaction of the session that took it,
        # so that exact session is held open until release rather than
        # returned to the pool.
        session = self._require_async_session(self._session_factory())
        try:
            while True:
                if await self._try_lock(session, lock_key):
                    self._sessions[key] = session
                    return True
                if loop.time() >= deadline:
                    break
                await asyncio.sleep(self._poll_interval_seconds)
        except PoolTimeoutError as exc:
            await _close_to_completion(session)
            raise TurnLockUnavailable(
                "no turn-lock connection came free in time; raise "
                "conversation_turn_lock_pool_size or add workers"
            ) from exc
        except BaseException:
            # Cancelled or failed with the attempt possibly in flight: closing
            # ends its transaction, so a lock it may have taken goes with it.
            await _close_to_completion(session)
            raise
        await _close_to_completion(session)
        return False

    @staticmethod
    async def _try_lock(session: Any, lock_key: int) -> bool:
        from sqlalchemy import text

        # The transaction sits idle for the whole turn by design. A server or
        # role idle-in-transaction timeout would kill it mid-turn and drop the
        # lock with nobody told. SET LOCAL ends with the transaction, so the
        # pooled connection keeps the server default.
        await session.execute(text("SET LOCAL idle_in_transaction_session_timeout = 0"))
        result = await session.execute(
            text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": lock_key}
        )
        return bool(result.scalar())

    async def release(self, key: str) -> None:
        session = self._sessions.pop(key, None)
        if session is None:
            return
        # Ending the transaction is the release.
        await _close_to_completion(session)


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

        try:
            acquired = await self._backend.acquire(key, timeout_seconds=self._timeout_seconds)
        except TurnLockUnavailable:
            # Retriable like a busy conversation, so clients need no new
            # handling; the reason and the warning tell operators it is capacity.
            logger.warning(
                "No turn-lock connection for conversation %s within %.2fs; the "
                "worker is at its concurrent-turn capacity",
                key,
                self._timeout_seconds,
                exc_info=True,
            )
            raise WorkflowRoutingException(
                workflow_error(
                    "conversation_turn_conflict",
                    request_id=request_id,
                    details={"reason": "turn_lock_capacity"},
                )
            ) from None
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
