"""Database provider for dependency-injector integration.

This is a thin adapter over the single application engine/session factory
defined in :mod:`app.database.session`. It exists only to expose a
context-manager ``session`` for the container's ``db.provided.session``
wiring; it does not own its own engine and never mutates schema (migrations
are the only schema mutation path).
"""

import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.database.async_session import AsyncSessionLocal
from app.database.session import SessionLocal

logger = logging.getLogger(__name__)


class Database:
    """Adapter exposing a context-manager session over the shared factory."""

    def __init__(self, db_url: str | None = None) -> None:
        # db_url is accepted for backward-compatible construction but ignored:
        # the single engine/session factory lives in app.database.session.
        self._session_factory = SessionLocal
        self._async_session_factory = AsyncSessionLocal

    @contextmanager
    def session(self) -> Iterator[Session]:
        """
        Provide a database session as a context manager.

        Yields:
            Session: SQLAlchemy database session
        """
        session: Session = self._session_factory()
        try:
            yield session
        except Exception:
            logger.exception("Session rollback because of exception")
            session.rollback()
            raise
        finally:
            session.close()

    @asynccontextmanager
    async def async_session(self) -> AsyncIterator[AsyncSession]:
        """
        Provide an async database session as a context manager.

        Mirrors :meth:`session` for the async request path.

        Yields:
            AsyncSession: SQLAlchemy async database session
        """
        session: AsyncSession = self._async_session_factory()
        try:
            yield session
        except Exception:
            logger.exception("Async session rollback because of exception")
            await session.rollback()
            raise
        finally:
            await session.close()
