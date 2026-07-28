"""The async engine must mirror the sync engine's semantics exactly.

Configuration assertions are hermetic, following
``tests/test_database_session_provider.py``. The behavioral tests need a live
PostgreSQL and skip when one is not reachable.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

from app.database.async_session import (
    AsyncSessionLocal,
    async_database_url,
    async_engine,
    get_async_engine,
    get_async_session_factory,
)
from app.database.session import engine as sync_engine

# Async psycopg cannot connect from a Windows ProactorEventLoop.
pytestmark = pytest.mark.selector_event_loop


# ── Configuration (no database required) ────────────────────────────────────


def test_async_url_is_derived_from_the_same_setting():
    """One database_url setting means the two engines can never drift apart."""
    url = async_database_url()
    assert url.drivername == "postgresql+psycopg"
    assert url.database == sync_engine.url.database
    assert url.host == sync_engine.url.host


def test_async_session_factory_matches_sync_semantics():
    """Repositories hand back detached ORM objects whose attributes callers
    still read; expire_on_commit=True would raise DetachedInstanceError."""
    from app.database.session import SessionLocal

    for option in ("expire_on_commit", "autoflush"):
        assert AsyncSessionLocal.kw[option] is False
        assert AsyncSessionLocal.kw[option] == SessionLocal.kw[option]


def test_async_session_factory_is_bound_to_the_async_engine():
    assert AsyncSessionLocal.kw["bind"] is async_engine


def test_providers_return_the_shared_objects():
    assert get_async_engine() is async_engine
    assert get_async_session_factory() is AsyncSessionLocal


def test_pool_is_explicitly_sized():
    """SQLAlchemy's default of 5 is a concurrency ceiling for streaming."""
    from app.core.config import settings

    assert async_engine.pool.size() == settings.db_pool_size
    assert settings.db_pool_size >= 10


# ── Behavior (live database) ────────────────────────────────────────────────


async def test_async_session_executes_a_query(require_async_db):
    async with AsyncSessionLocal() as session:
        assert (await session.execute(text("select 1"))).scalar_one() == 1


async def test_run_sync_executes_unchanged_sync_style_code(require_async_db):
    """The bridge the whole migration depends on."""

    def sync_style(sync_session):
        return sync_session.execute(text("select 42")).scalar_one()

    async with AsyncSessionLocal() as session:
        assert await session.run_sync(sync_style) == 42


async def test_concurrent_sessions_do_not_serialize(require_async_db):
    async def one():
        async with AsyncSessionLocal() as session:
            await session.execute(text("select pg_sleep(0.3)"))

    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.gather(*(one() for _ in range(4)))
    elapsed = loop.time() - started
    # Serialized would be ~1.2s; genuinely concurrent is ~0.3s.
    assert elapsed < 0.9, f"async sessions serialized ({elapsed:.2f}s for 4x0.3s)"
