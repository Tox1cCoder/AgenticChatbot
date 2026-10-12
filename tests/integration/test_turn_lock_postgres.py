"""The conversation turn lock is released on every way out of a turn.

A session-scoped advisory lock outlived ``Session.close()``: the pool resets a
returned connection with ROLLBACK, which does not release it. A client
disconnect cancelled the explicit unlock, the lock stayed on the pooled
connection, and the conversation was refused to everyone else until that
connection happened to be recycled.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import anyio
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.services.conversation_turn_coordinator import (
    ConversationTurnCoordinator,
    PostgresAdvisoryLockBackend,
)

_HOLDERS = text(
    "SELECT pid FROM pg_locks WHERE locktype = 'advisory' "
    "AND ((classid::bigint << 32) | objid::bigint) = (:key & x'FFFFFFFFFFFFFFFF'::bigint)"
)


@pytest.fixture(scope="module")
def engine():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    # Pooled like production: a leaked lock is only visible when the
    # connection that holds it goes back to the pool instead of closing.
    engine = create_engine(database_url, pool_size=2, max_overflow=0)
    try:
        yield engine
    finally:
        engine.dispose()


def _holders(engine, conversation_id: str) -> list[int]:
    key = PostgresAdvisoryLockBackend.lock_key(conversation_id)
    with engine.connect() as connection:
        return [row.pid for row in connection.execute(_HOLDERS, {"key": key})]


async def _settled_holders(engine, conversation_id: str) -> list[int]:
    """Holders once any release still running in a thread has finished."""
    for _ in range(50):
        holders = _holders(engine, conversation_id)
        if not holders:
            return holders
        await asyncio.sleep(0.05)
    return holders


def _coordinator(engine, timeout_seconds: float = 0.5) -> ConversationTurnCoordinator:
    return ConversationTurnCoordinator(
        backend=PostgresAdvisoryLockBackend(sessionmaker(bind=engine), poll_interval_seconds=0.02),
        timeout_seconds=timeout_seconds,
    )


async def test_a_client_disconnect_mid_turn_releases_the_lock(engine):
    conversation_id = f"turn-lock-{uuid.uuid4()}"
    coordinator = _coordinator(engine)
    held: list[int] = []

    async def turn() -> None:
        async with coordinator.hold(conversation_id, request_id="streamed"):
            held.extend(_holders(engine, conversation_id))
            await anyio.sleep(10)

    # What StreamingResponse does on disconnect. anyio cancellation is
    # level-triggered, so every await in the release is cancelled as well.
    with anyio.CancelScope() as scope:
        async with anyio.create_task_group() as group:
            group.start_soon(turn)
            await anyio.sleep(0.3)
            scope.cancel()

    assert held, "the turn never held the lock, so the test proves nothing"
    assert await _settled_holders(engine, conversation_id) == []


async def test_a_server_idle_in_transaction_timeout_does_not_drop_a_held_lock(engine):
    """The lock's transaction is idle for the whole turn; a turn outlives the timeout."""
    short_timeout = create_engine(
        engine.url,
        pool_size=1,
        max_overflow=0,
        connect_args={"options": "-c idle_in_transaction_session_timeout=500"},
    )
    conversation_id = f"turn-lock-{uuid.uuid4()}"
    try:
        async with _coordinator(short_timeout).hold(conversation_id, request_id="slow"):
            await asyncio.sleep(1.2)
            assert _holders(engine, conversation_id), "the server ended the lock mid-turn"
    finally:
        short_timeout.dispose()


async def test_the_next_turn_after_a_disconnect_is_not_refused(engine):
    conversation_id = f"turn-lock-{uuid.uuid4()}"
    first = _coordinator(engine)

    with anyio.CancelScope() as scope:
        async with anyio.create_task_group() as group:

            async def turn() -> None:
                async with first.hold(conversation_id, request_id="streamed"):
                    await anyio.sleep(10)

            group.start_soon(turn)
            await anyio.sleep(0.3)
            scope.cancel()
    await _settled_holders(engine, conversation_id)

    # A different pool stands in for another worker: it cannot borrow the
    # connection a leaked lock would sit on, so it cannot re-enter one.
    other_worker = create_engine(engine.url, pool_size=1, max_overflow=0)
    try:
        async with _coordinator(other_worker).hold(conversation_id, request_id="next"):
            pass
    finally:
        other_worker.dispose()
