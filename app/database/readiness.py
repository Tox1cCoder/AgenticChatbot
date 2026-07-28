"""Boot-time readiness checks for the database engines.

Two classes of misconfiguration are cheap to detect at startup and expensive to
discover in production:

* **An unusable async pool.** Wrong event loop, bad credentials, or an
  unreachable host make every request's database call fail while the process
  still answers ``/health``. Probing once at startup turns that into a single
  clear log line.
* **An over-subscribed connection budget.** The sync and async pools are
  independent, so their maxima add up, and multiply by worker count. Exceeding
  the server's ``max_connections`` shows up as intermittent "too many clients
  already" under load rather than as a configuration error.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text

from app.database.async_session import AsyncSessionLocal

# Connections the application must leave for everything that is not a web
# worker: Celery workers, the migration runner, monitoring, and an operator's
# psql session.
_RESERVED_CONNECTIONS = 10


@dataclass(frozen=True)
class ConnectionBudget:
    """Worst-case connection demand across both engines."""

    per_worker: int
    workers: int
    total: int

    def exceeds(self, *, max_connections: int) -> bool:
        """True when the budget leaves too little headroom for other clients."""
        return self.total > max_connections - _RESERVED_CONNECTIONS

    def describe(self, *, max_connections: int) -> str:
        return (
            f"database connection budget: {self.total} worst-case connections "
            f"({self.per_worker} per web worker x {self.workers}) against "
            f"max_connections={max_connections}, reserving {_RESERVED_CONNECTIONS} "
            "for Celery, migrations, and operators; multiply by the launch's "
            "--workers count"
        )


def resolve_connection_budget(
    *,
    sync_pool: int,
    sync_overflow: int,
    async_pool: int,
    async_overflow: int,
    workers: int,
) -> ConnectionBudget:
    """Compute worst-case demand. Both pools count: they are separate pools."""
    per_worker = sync_pool + sync_overflow + async_pool + async_overflow
    worker_count = max(1, workers)
    return ConnectionBudget(
        per_worker=per_worker,
        workers=worker_count,
        total=per_worker * worker_count,
    )


@dataclass(frozen=True)
class AsyncEngineReadiness:
    """Outcome of the async-engine probe."""

    ok: bool
    max_connections: int | None = None
    error: str | None = None


async def verify_async_engine_ready() -> AsyncEngineReadiness:
    """Open one async connection and read the server's ``max_connections``.

    Reports rather than raises: the caller decides whether an unreachable
    database should abort startup, and a probe should never be the source of an
    ambiguous crash.
    """
    try:
        async with AsyncSessionLocal() as session:
            max_connections = (await session.execute(text("show max_connections"))).scalar_one()
        return AsyncEngineReadiness(ok=True, max_connections=int(max_connections))
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return AsyncEngineReadiness(ok=False, error=f"{type(exc).__name__}: {exc}")
