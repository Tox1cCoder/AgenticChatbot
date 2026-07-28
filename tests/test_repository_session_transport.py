"""Both repository transports must execute the *same* sync-style callable."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.database.async_session import AsyncSessionLocal
from app.database.session import SessionLocal
from app.repositories.session_transport import RepositorySessionMixin

pytestmark = pytest.mark.selector_event_loop


class _Probe(RepositorySessionMixin):
    pass


def _work(session):
    """Ordinary sync-style query code — identical for both transports."""
    return session.execute(text("select 7")).scalar_one()


def test_sync_transport_runs_the_work(require_async_db):
    probe = _Probe(session_factory=SessionLocal)
    assert probe._run(_work) == 7


async def test_async_transport_runs_the_same_work(require_async_db):
    probe = _Probe(session_factory=SessionLocal, async_session_factory=AsyncSessionLocal)
    assert await probe._arun(_work) == 7


async def test_async_transport_without_a_factory_is_a_clear_error():
    probe = _Probe(session_factory=SessionLocal)
    with pytest.raises(RuntimeError, match="async_session_factory"):
        await probe._arun(_work)


async def test_async_transport_propagates_work_exceptions(require_async_db):
    def boom(session):
        raise ValueError("query failed")

    probe = _Probe(session_factory=SessionLocal, async_session_factory=AsyncSessionLocal)
    with pytest.raises(ValueError, match="query failed"):
        await probe._arun(boom)


def test_mixin_forwards_unused_kwargs_to_cooperative_bases():
    """Repositories mix this in ahead of their own __init__ work."""

    class _Base:
        def __init__(self, marker=None, **kwargs):
            self.marker = marker
            super().__init__(**kwargs)

    class _Combined(RepositorySessionMixin, _Base):
        pass

    combined = _Combined(session_factory=SessionLocal, marker="kept")
    assert combined.marker == "kept"
    assert combined.session_factory is SessionLocal
    assert combined.async_session_factory is None


async def test_database_provider_exposes_an_async_session(require_async_db):
    from app.database.database import Database

    database = Database()
    async with database.async_session() as session:
        assert (await session.execute(text("select 5"))).scalar_one() == 5
