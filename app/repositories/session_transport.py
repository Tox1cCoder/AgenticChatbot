"""Shared session transports for repositories.

A repository operation is a sync-style callable taking a ``Session`` — which is
exactly what the existing ``*CRUDStrategy`` methods already are. This mixin runs
such a callable over either transport:

* :meth:`RepositorySessionMixin._run` — the sync engine, for Celery workers,
  Alembic, and callers not yet migrated.
* :meth:`RepositorySessionMixin._arun` — an ``AsyncSession``, via
  ``run_sync``. SQLAlchemy drives the callable on a greenlet and performs the
  actual I/O asynchronously, so the query body needs no rewriting and the event
  loop is never blocked.

Keeping both here is what lets the query implementation exist exactly once.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from sqlalchemy.orm import Session

T = TypeVar("T")


class RepositorySessionMixin:
    """Provide sync and async execution of sync-style repository work."""

    def __init__(
        self,
        session_factory: Callable[[], Any],
        async_session_factory: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.session_factory = session_factory
        self.async_session_factory = async_session_factory
        super().__init__(**kwargs)

    def _run(self, work: Callable[[Session], T]) -> T:
        """Execute ``work`` on the sync engine."""
        with self.session_factory() as session:
            return work(session)

    async def _arun(self, work: Callable[[Session], T]) -> T:
        """Execute ``work`` on the async engine without blocking the event loop."""
        if self.async_session_factory is None:
            raise RuntimeError(
                f"{type(self).__name__} was constructed without an "
                "async_session_factory; wire it in app/core/container.py before "
                "calling an async repository method."
            )
        async with self.async_session_factory() as session:
            return await session.run_sync(work)
