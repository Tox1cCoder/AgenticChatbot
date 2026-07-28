"""Async engine and session factory for the FastAPI request path.

Mirrors :mod:`app.database.session` for asynchronous callers. The sync engine in
that module is deliberately kept: Celery workers and Alembic are synchronous and
stay that way.

The URL is derived from the single ``database_url`` setting rather than
configured separately, so the two engines can never drift onto different
databases.

psycopg's async mode requires a ``SelectorEventLoop``; ``app/main.py`` and the
Celery entrypoints set that policy, and ``tests/conftest.py`` does the same for
the suite. On Windows' default ``ProactorEventLoop`` every connect raises
``psycopg.InterfaceError``.
"""

from __future__ import annotations

from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

_ASYNC_DRIVER = "postgresql+psycopg"


def async_database_url() -> URL:
    """Return ``settings.database_url`` retargeted at the async psycopg driver."""
    return make_url(settings.database_url).set(drivername=_ASYNC_DRIVER)


async_engine: AsyncEngine = create_async_engine(
    async_database_url(),
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.api_debug,
)

# expire_on_commit=False and autoflush=False mirror SessionLocal: repositories
# hand back detached ORM objects whose loaded attributes callers still read.
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


def get_async_engine() -> AsyncEngine:
    """Return the shared async engine."""
    return async_engine


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the shared async session factory."""
    return AsyncSessionLocal


async def dispose_async_engine() -> None:
    """Close pooled async connections. Called on application shutdown."""
    await async_engine.dispose()
