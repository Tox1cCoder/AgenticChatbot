"""The conversation turn lock is released on every way out of a turn.

A session-scoped advisory lock outlived closing its session: the pool resets a
returned connection with ROLLBACK, which does not release it. A client
disconnect cancelled the explicit unlock, the lock stayed on the pooled
connection, and the conversation was refused to everyone else until that
connection happened to be recycled.

The lock runs on the async engine, so a turn waiting for a lock connection
occupies no thread; on a sync session it held a default-executor thread for up
to the pool timeout and stalled every other ``to_thread`` call in the process.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

import anyio
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.ai.workflow.contracts import WorkflowRoutingException
from app.services.conversation_turn_coordinator import (
    ConversationTurnCoordinator,
    PostgresAdvisoryLockBackend,
)

pytestmark = pytest.mark.selector_event_loop

_HOLDERS = text(
    "SELECT pid FROM pg_locks WHERE locktype = 'advisory' "
    "AND ((classid::bigint << 32) | objid::bigint) = (:key & x'FFFFFFFFFFFFFFFF'::bigint)"
)


@pytest.fixture(scope="module")
def database_url() -> str:
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    return url


@pytest.fixture(scope="module")
def observer(database_url):
    """A separate sync connection for reading ``pg_locks`` from outside the lock."""
    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        engine.dispose()


def _lock_engine(database_url: str, *, pool_size: int = 2, **kwargs):
    # Pooled like production: a leaked lock is only visible when the
    # connection that holds it goes back to the pool instead of closing.
    url = make_url(database_url).set(drivername="postgresql+psycopg")
    return create_async_engine(url, pool_size=pool_size, max_overflow=0, **kwargs)


@pytest.fixture
async def engine(database_url):
    engine = _lock_engine(database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


def _holders(observer, conversation_id: str) -> list[int]:
    key = PostgresAdvisoryLockBackend.lock_key(conversation_id)
    with observer.connect() as connection:
        return [row.pid for row in connection.execute(_HOLDERS, {"key": key})]


async def _settled_holders(observer, conversation_id: str) -> list[int]:
    """Holders once any close still running in the background has finished."""
    for _ in range(50):
        holders = _holders(observer, conversation_id)
        if not holders:
            return holders
        await asyncio.sleep(0.05)
    return holders


async def _checked_out(engine) -> int:
    """Connections still out once a background close has had time to check in."""
    for _ in range(50):
        if engine.pool.checkedout() == 0:
            return 0
        await asyncio.sleep(0.05)
    return engine.pool.checkedout()


def _coordinator(engine, timeout_seconds: float = 0.5) -> ConversationTurnCoordinator:
    return ConversationTurnCoordinator(
        backend=PostgresAdvisoryLockBackend(
            async_sessionmaker(bind=engine), poll_interval_seconds=0.02
        ),
        timeout_seconds=timeout_seconds,
    )


async def _disconnect(turn) -> None:
    """Abandon ``turn`` the way StreamingResponse does when the client goes away.

    anyio cancellation is level-triggered, so every await on the turn's way
    out — the release included — is cancelled as well.
    """
    with anyio.CancelScope() as scope:
        async with anyio.create_task_group() as group:
            group.start_soon(turn)
            await anyio.sleep(0.3)
            scope.cancel()


async def test_a_client_disconnect_mid_turn_releases_the_lock(engine, observer):
    conversation_id = f"turn-lock-{uuid.uuid4()}"
    coordinator = _coordinator(engine)
    held: list[int] = []

    async def turn() -> None:
        async with coordinator.hold(conversation_id, request_id="streamed"):
            held.extend(_holders(observer, conversation_id))
            await anyio.sleep(10)

    await _disconnect(turn)

    assert held, "the turn never held the lock, so the test proves nothing"
    assert await _settled_holders(observer, conversation_id) == []
    assert await _checked_out(engine) == 0, "the lock's connection never went back"


async def test_a_disconnect_while_the_lock_query_runs_releases_it(engine, observer):
    """Cancelled mid-attempt, the attempt may still take the lock server-side."""
    conversation_id = f"turn-lock-{uuid.uuid4()}"
    key = PostgresAdvisoryLockBackend.lock_key(conversation_id)

    class SlowLockBackend(PostgresAdvisoryLockBackend):
        @staticmethod
        async def _try_lock(session, lock_key):
            result = await session.execute(
                text("SELECT pg_try_advisory_xact_lock(:key), pg_sleep(1)"), {"key": key}
            )
            return bool(result.scalar())

    coordinator = ConversationTurnCoordinator(
        backend=SlowLockBackend(async_sessionmaker(bind=engine)), timeout_seconds=5
    )

    async def turn() -> None:
        async with coordinator.hold(conversation_id, request_id="streamed"):
            pytest.fail("the turn body ran although the stream was abandoned")

    await _disconnect(turn)
    await asyncio.sleep(1.2)  # past the in-flight pg_sleep

    assert await _settled_holders(observer, conversation_id) == []
    assert await _checked_out(engine) == 0


async def test_a_server_idle_in_transaction_timeout_does_not_drop_a_held_lock(
    database_url, observer
):
    """The lock's transaction is idle for the whole turn; a turn outlives the timeout."""
    short_timeout = _lock_engine(
        database_url,
        pool_size=1,
        connect_args={"options": "-c idle_in_transaction_session_timeout=500"},
    )
    conversation_id = f"turn-lock-{uuid.uuid4()}"
    try:
        async with _coordinator(short_timeout).hold(conversation_id, request_id="slow"):
            await asyncio.sleep(1.2)
            assert _holders(observer, conversation_id), "the server ended the lock mid-turn"
    finally:
        await short_timeout.dispose()


async def test_the_next_turn_after_a_disconnect_is_not_refused(engine, database_url, observer):
    conversation_id = f"turn-lock-{uuid.uuid4()}"
    first = _coordinator(engine)

    async def turn() -> None:
        async with first.hold(conversation_id, request_id="streamed"):
            await anyio.sleep(10)

    await _disconnect(turn)
    await _settled_holders(observer, conversation_id)

    # A different pool stands in for another worker: it cannot borrow the
    # connection a leaked lock would sit on, so it cannot re-enter one.
    other_worker = _lock_engine(database_url, pool_size=1)
    try:
        async with _coordinator(other_worker).hold(conversation_id, request_id="next"):
            pass
    finally:
        await other_worker.dispose()


async def test_a_full_lock_pool_refuses_retriably_without_holding_threads(database_url):
    """Turns past the worker's capacity wait on the loop, then get a retriable code.

    On a sync session each waiting turn held a default-executor thread for the
    whole pool timeout, so a burst stalled every ``to_thread`` call in the
    process; and the raw pool ``TimeoutError`` broke the SSE stream.
    """
    from concurrent.futures import ThreadPoolExecutor

    # The default executor of a 2-vCPU server.
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=6))
    full = _lock_engine(database_url, pool_size=2, pool_timeout=1)
    holders = [_coordinator(full).hold(f"turn-lock-{uuid.uuid4()}", request_id="h") for _ in "ab"]
    try:
        for held in holders:
            await held.__aenter__()

        async def waiting_turn() -> WorkflowRoutingException:
            with pytest.raises(WorkflowRoutingException) as exc:
                async with _coordinator(full, timeout_seconds=1).hold(
                    f"turn-lock-{uuid.uuid4()}", request_id="w"
                ):
                    pytest.fail("a turn ran without a lock connection")
            return exc.value

        waiting = [asyncio.create_task(waiting_turn()) for _ in range(12)]
        await asyncio.sleep(0.2)
        started = time.monotonic()
        await asyncio.to_thread(lambda: None)
        stalled = time.monotonic() - started

        refusals = await asyncio.gather(*waiting)
    finally:
        for held in holders:
            await held.__aexit__(None, None, None)
        await full.dispose()

    assert stalled < 0.2, f"waiting turns starved the executor for {stalled:.2f}s"
    assert {(r.error.code, r.error.retriable) for r in refusals} == {
        ("conversation_turn_conflict", True)
    }
    assert {r.error.details["reason"] for r in refusals} == {"turn_lock_capacity"}
