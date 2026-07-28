"""Boot-time readiness for the async engine.

Two production failures this prevents:

* the async pool being unusable (wrong loop, bad credentials, unreachable host)
  is discovered on the first user request instead of at startup; and
* the sync and async pools together exceeding the server's ``max_connections``,
  which surfaces as intermittent "too many clients" under load rather than as a
  configuration error.
"""

from __future__ import annotations

import pytest

from app.database.readiness import (
    ConnectionBudget,
    resolve_connection_budget,
    verify_async_engine_ready,
)

pytestmark = pytest.mark.selector_event_loop


class TestConnectionBudget:
    def test_totals_both_pools(self):
        budget = resolve_connection_budget(
            sync_pool=5, sync_overflow=10, async_pool=20, async_overflow=10, workers=1
        )
        assert budget.per_worker == 45
        assert budget.total == 45

    def test_scales_with_worker_count(self):
        budget = resolve_connection_budget(
            sync_pool=5, sync_overflow=10, async_pool=20, async_overflow=10, workers=4
        )
        assert budget.total == 180

    def test_is_within_limit_when_headroom_remains(self):
        budget = resolve_connection_budget(
            sync_pool=5, sync_overflow=5, async_pool=10, async_overflow=5, workers=1
        )
        assert budget.exceeds(max_connections=100) is False

    def test_flags_a_budget_that_cannot_fit(self):
        budget = resolve_connection_budget(
            sync_pool=20, sync_overflow=20, async_pool=20, async_overflow=20, workers=4
        )
        assert budget.exceeds(max_connections=100) is True

    def test_reserves_headroom_for_other_clients(self):
        """A budget equal to max_connections leaves nothing for psql, Celery, or
        the migration runner, so it must be flagged."""
        budget = resolve_connection_budget(
            sync_pool=50, sync_overflow=0, async_pool=50, async_overflow=0, workers=1
        )
        assert budget.total == 100
        assert budget.exceeds(max_connections=100) is True

    def test_describe_names_both_pools_for_the_operator(self):
        budget = ConnectionBudget(per_worker=45, workers=2, total=90)
        description = budget.describe(max_connections=100)
        assert "90" in description
        assert "100" in description


class TestVerifyAsyncEngineReady:
    async def test_returns_the_server_max_connections(self, require_async_db):
        result = await verify_async_engine_ready()
        assert result.max_connections > 0
        assert result.ok is True

    async def test_reports_failure_instead_of_raising(self, monkeypatch):
        """A readiness probe must never be the thing that crashes startup
        ambiguously; it reports so the caller decides."""
        import app.database.readiness as readiness

        class _BrokenFactory:
            def __call__(self):
                raise OSError("connection refused")

        monkeypatch.setattr(readiness, "AsyncSessionLocal", _BrokenFactory())

        result = await verify_async_engine_ready()
        assert result.ok is False
        assert "connection refused" in (result.error or "")
        assert result.max_connections is None
