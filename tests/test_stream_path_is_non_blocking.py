"""The event loop must stay responsive while the stream path hits the database.

This is the regression test for the whole migration. The contrast test
characterizes the blocking behavior being removed, so the first test's assertion
means something.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.selector_event_loop

_QUERY_SECONDS = 0.4
_TICK_SECONDS = 0.02


async def _count_ticks_during(work) -> int:
    """Run ``work`` while a heartbeat counts event-loop iterations."""
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(_TICK_SECONDS)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    try:
        await work()
    finally:
        beat.cancel()
        with pytest.raises(asyncio.CancelledError):
            await beat
    return ticks


async def test_async_query_does_not_stall_the_event_loop(require_async_db):
    from app.database.async_session import AsyncSessionLocal

    async def work():
        async with AsyncSessionLocal() as session:
            await session.execute(text(f"select pg_sleep({_QUERY_SECONDS})"))

    ticks = await _count_ticks_during(work)
    # ~20 ticks if the loop kept running; ~0 if the query blocked it.
    assert ticks >= 10, f"event loop stalled during the async query (only {ticks} ticks)"


async def test_sync_query_does_stall_the_event_loop_for_contrast(require_async_db):
    """Characterizes the behavior being removed from the request path."""
    from app.database.session import SessionLocal

    async def work():
        with SessionLocal() as session:
            session.execute(text(f"select pg_sleep({_QUERY_SECONDS})"))

    ticks = await _count_ticks_during(work)
    assert ticks <= 2, f"expected the sync session to block the loop, got {ticks} ticks"


async def test_run_sync_repository_work_does_not_stall_the_loop(require_async_db):
    """The transport repositories actually use, not just a raw async execute."""
    from app.database.async_session import AsyncSessionLocal
    from app.database.session import SessionLocal
    from app.repositories.session_transport import RepositorySessionMixin

    class _Probe(RepositorySessionMixin):
        pass

    probe = _Probe(session_factory=SessionLocal, async_session_factory=AsyncSessionLocal)

    async def work():
        await probe._arun(
            lambda session: session.execute(text(f"select pg_sleep({_QUERY_SECONDS})"))
        )

    ticks = await _count_ticks_during(work)
    assert ticks >= 10, f"run_sync blocked the event loop (only {ticks} ticks)"


async def test_two_concurrent_repository_calls_overlap(require_async_db):
    """One request's database work must not delay another's."""
    from app.database.async_session import AsyncSessionLocal
    from app.database.session import SessionLocal
    from app.repositories.session_transport import RepositorySessionMixin

    class _Probe(RepositorySessionMixin):
        pass

    probe = _Probe(session_factory=SessionLocal, async_session_factory=AsyncSessionLocal)

    async def one():
        await probe._arun(
            lambda session: session.execute(text(f"select pg_sleep({_QUERY_SECONDS})"))
        )

    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.gather(one(), one())
    elapsed = loop.time() - started
    assert elapsed < _QUERY_SECONDS * 1.8, (
        f"two concurrent repository calls serialized ({elapsed:.2f}s)"
    )
