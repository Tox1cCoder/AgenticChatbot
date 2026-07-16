from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.health import create_health_router
from app.observability.conversation_compaction import (
    ConversationCompactionHealthService,
    ConversationCompactionMetrics,
)


class _SnapshotRepository:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def get_compaction_health_snapshot(self):
        return dict(self.snapshot)


def _snapshot(**overrides):
    value = {
        "job_counts": {
            "idle": 2,
            "pending": 0,
            "processing": 0,
            "retry": 0,
            "dead": 0,
        },
        "oldest_actionable_age_seconds": 0.0,
        "expired_lease_count": 0,
        "max_sequence_lag": 0,
        "total_sequence_lag": 0,
    }
    value.update(overrides)
    return value


def test_metrics_expose_required_content_free_measurements() -> None:
    metrics = ConversationCompactionMetrics()
    secret_id = str(uuid4())
    metrics.record_job_outcome("success", error_code=None)
    metrics.record_job_outcome(
        "retry",
        error_code=f"provider timeout transcript={secret_id}",
    )
    metrics.record_compaction(
        mode="background",
        outcome="success",
        provider="openai",
        model="gpt-4o-mini-2026-01-01",
        content_class="mixed",
        input_tokens=1_200,
        output_tokens=240,
        duration_seconds=0.4,
        cost_amount=0.012,
        cost_currency="USD",
    )
    metrics.record_token_calibration(
        provider="gemini",
        model="gemini-2.5-flash-preview-private-name",
        content_class="tools",
        estimated_tokens=900,
        actual_tokens=1_000,
    )
    metrics.record_deterministic_trim(removed_groups=2)
    metrics.record_provider_overflow_retry("success")

    payload = metrics.render().decode("utf-8")

    for metric_name in (
        "conversation_compaction_jobs_total",
        "conversation_compaction_operations_total",
        "conversation_compaction_duration_seconds",
        "conversation_compaction_input_tokens",
        "conversation_compaction_output_tokens",
        "conversation_compaction_estimate_delta_tokens",
        "conversation_compaction_cost_total",
        "conversation_compaction_deterministic_trim_total",
        "conversation_compaction_provider_overflow_retry_total",
    ):
        assert metric_name in payload
    assert 'provider="openai"' in payload
    assert 'model="gpt-4o"' in payload
    assert 'content_class="mixed"' in payload
    assert 'error_class="provider_timeout"' in payload
    assert secret_id not in payload
    assert "preview-private-name" not in payload
    assert "transcript" not in payload


def test_health_status_is_deterministic_for_queue_dead_lease_and_lag() -> None:
    healthy = ConversationCompactionHealthService(
        _SnapshotRepository(_snapshot()),
        queue_age_degraded_seconds=60,
        lag_degraded_sequences=50,
    ).get_health()
    degraded = ConversationCompactionHealthService(
        _SnapshotRepository(
            _snapshot(
                job_counts={
                    "idle": 0,
                    "pending": 2,
                    "processing": 0,
                    "retry": 1,
                    "dead": 0,
                },
                oldest_actionable_age_seconds=61,
                max_sequence_lag=75,
                total_sequence_lag=100,
            )
        ),
        queue_age_degraded_seconds=60,
        lag_degraded_sequences=50,
    ).get_health()
    dead = ConversationCompactionHealthService(
        _SnapshotRepository(
            _snapshot(
                job_counts={
                    "idle": 0,
                    "pending": 0,
                    "processing": 0,
                    "retry": 0,
                    "dead": 1,
                }
            )
        ),
        queue_age_degraded_seconds=60,
        lag_degraded_sequences=50,
    ).get_health()
    expired = ConversationCompactionHealthService(
        _SnapshotRepository(_snapshot(expired_lease_count=1)),
        queue_age_degraded_seconds=60,
        lag_degraded_sequences=50,
    ).get_health()

    assert healthy["status"] == "healthy"
    assert degraded["status"] == "degraded"
    assert dead["status"] == "unhealthy"
    assert expired["status"] == "unhealthy"


def test_health_router_is_aggregate_content_free_and_separate_from_celery() -> None:
    secret_id = str(uuid4())
    repository = _SnapshotRepository(_snapshot())
    service = ConversationCompactionHealthService(
        repository,
        queue_age_degraded_seconds=60,
        lag_degraded_sequences=50,
    )
    metrics = ConversationCompactionMetrics()
    app = FastAPI()
    app.include_router(create_health_router(service=service, metrics=metrics))
    client = TestClient(app)

    health_response = client.get("/health/conversation-compaction")
    metrics_response = client.get("/metrics/conversation-compaction")

    assert health_response.status_code == 200
    health_payload = health_response.json()
    assert health_payload["status"] == "healthy"
    assert set(health_payload) == {
        "status",
        "job_counts",
        "oldest_actionable_age_seconds",
        "expired_lease_count",
        "max_sequence_lag",
        "total_sequence_lag",
    }
    rendered = health_response.text + metrics_response.text
    assert secret_id not in rendered
    assert "conversation_id" not in rendered
    assert "summary_payload" not in rendered
    assert "/health/celery" not in app.openapi()["paths"]


def test_main_application_registers_compaction_health_separately() -> None:
    from app.main import app

    paths = set(app.openapi()["paths"])

    assert "/health/conversation-compaction" in paths
    assert "/metrics/conversation-compaction" in paths
    assert "/health/celery" in paths
