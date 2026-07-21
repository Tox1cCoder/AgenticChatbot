from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.health import create_health_router
from app.observability.model_usage import ModelUsageHealthService, ModelUsageMetrics


class SnapshotRepository:
    def __init__(self, snapshot=None, error: Exception | None = None):
        self.snapshot = snapshot or {}
        self.error = error

    def get_model_usage_health_snapshot(self, *, now, lookback_minutes):
        if self.error:
            raise self.error
        return dict(self.snapshot)


def snapshot(**overrides):
    value = {
        "raw_event_count": 100,
        "rollup_request_count": 100,
        "unattributed_event_count": 2,
        "latest_event_minute": datetime(2026, 7, 21, 12, 29, tzinfo=timezone.utc),
        "latest_rollup_minute": datetime(2026, 7, 21, 12, 29, tzinfo=timezone.utc),
    }
    value.update(overrides)
    return value


def service(repository, metrics=None):
    return ModelUsageHealthService(
        repository,
        metrics=metrics or ModelUsageMetrics(),
        lookback_minutes=60,
        unattributed_degraded_ratio=0.10,
        rollup_lag_degraded_minutes=2,
        rollup_lag_unhealthy_minutes=5,
        clock=lambda: datetime(2026, 7, 21, 12, 31, tzinfo=timezone.utc),
    )


def test_health_classifies_durable_gap_unattributed_ratio_and_rollup_lag():
    assert service(SnapshotRepository(snapshot())).get_health()["status"] == "healthy"
    unattributed = service(SnapshotRepository(snapshot(unattributed_event_count=11))).get_health()
    assert unattributed["status"] == "degraded"
    assert unattributed["unattributed_rate"] == 0.11
    lagged = service(
        SnapshotRepository(
            snapshot(latest_rollup_minute=datetime(2026, 7, 21, 12, 23, tzinfo=timezone.utc))
        )
    ).get_health()
    assert lagged["status"] == "unhealthy"
    gap = service(SnapshotRepository(snapshot(rollup_request_count=99))).get_health()
    assert gap["status"] == "unhealthy"
    assert gap["rollup_gap_count"] == 1


def test_old_matching_event_and_rollup_are_healthy_not_stale():
    old_minute = datetime(2026, 7, 21, 11, 40, tzinfo=timezone.utc)
    result = service(
        SnapshotRepository(
            snapshot(latest_event_minute=old_minute, latest_rollup_minute=old_minute)
        )
    ).get_health()

    assert result["status"] == "healthy"
    assert result["rollup_lag_minutes"] == 0


def test_health_classifies_local_recorder_persistence_failure():
    metrics = ModelUsageMetrics()
    metrics.record_persistence("retry_enqueued", failure_class="OperationalError")
    result = service(SnapshotRepository(snapshot()), metrics).get_health()
    assert result["status"] == "degraded"
    assert result["persistence_failure_count"] == 1
    assert result["persistence_failure_scope"] == "process"
    assert result["authoritative_health_scope"] == "database"


def test_recent_failure_buffer_is_bounded_and_reports_saturation():
    from app.observability.model_usage import _RECENT_FAILURE_BUFFER_MAX

    metrics = ModelUsageMetrics()
    for _ in range(_RECENT_FAILURE_BUFFER_MAX + 1):
        metrics.record_persistence("dropped", failure_class="ConnectionError")

    result = service(SnapshotRepository(snapshot()), metrics).get_health()
    assert result["persistence_failure_count"] == _RECENT_FAILURE_BUFFER_MAX
    assert result["persistence_failure_count_saturated"] is True
    rendered = metrics.render().decode("utf-8")
    assert (
        "# HELP model_usage_persistence_total Process-local ledger-write persistence outcomes"
        in rendered
    )


def test_health_and_metrics_are_content_free_and_fail_closed():
    secret = str(uuid4())
    metrics = ModelUsageMetrics()
    app = FastAPI()
    app.include_router(
        create_health_router(
            model_usage_service=service(
                SnapshotRepository(error=RuntimeError(f"database secret={secret}")), metrics
            ),
            model_usage_metrics=metrics,
        )
    )
    client = TestClient(app, raise_server_exceptions=False)

    health = client.get("/health/model-usage")
    metrics_response = client.get("/metrics/model-usage")

    assert health.status_code == 503
    assert health.json() == {"status": "unhealthy", "data_available": False}
    rendered = health.text + metrics_response.text
    assert secret not in rendered
    assert all(
        field not in rendered for field in ("user_id", "conversation_id", "model_id", "trace_id")
    )


def test_model_usage_health_routes_are_registered_once():
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/health/model-usage" in paths
    assert "/metrics/model-usage" in paths
