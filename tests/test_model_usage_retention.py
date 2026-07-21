from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings


def test_reconcile_chunk_setting_is_positive_and_configurable():
    configured = Settings(
        _env_file=None,
        secret_key="test-secret-key-with-at-least-32-bytes",
        model_usage_reconcile_chunk_minutes=17,
    )
    assert configured.model_usage_reconcile_chunk_minutes == 17
    with pytest.raises(ValidationError, match="positive"):
        Settings(
            _env_file=None,
            secret_key="test-secret-key-with-at-least-32-bytes",
            model_usage_reconcile_chunk_minutes=0,
        )


def test_shared_failure_ttl_covers_health_window_and_boundary_bucket():
    with pytest.raises(ValidationError, match="failure store TTL"):
        Settings(
            _env_file=None,
            secret_key="test-secret-key-with-at-least-32-bytes",
            model_usage_health_failure_window_seconds=300,
            model_usage_failure_store_ttl_seconds=300,
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {
                "model_usage_raw_retention_days": 1,
                "model_usage_reconcile_minutes": 1_441,
            },
            "reconcile window",
        ),
        (
            {
                "model_usage_raw_retention_days": 91,
                "model_usage_rollup_retention_days": 90,
            },
            "rollup retention",
        ),
        ({"model_usage_health_failure_window_seconds": 3_601}, "less than or equal"),
        ({"model_usage_failure_store_ttl_seconds": 86_401}, "less than or equal"),
    ],
)
def test_model_usage_startup_rejects_impossible_operational_bounds(overrides, message):
    with pytest.raises(ValidationError, match=message):
        Settings(
            _env_file=None,
            secret_key="test-secret-key-with-at-least-32-bytes",
            **overrides,
        )


def test_model_usage_startup_rejects_equal_reconcile_and_raw_retention_boundary():
    with pytest.raises(ValidationError, match="strictly shorter"):
        Settings(
            _env_file=None,
            secret_key="test-secret-key-with-at-least-32-bytes",
            model_usage_raw_retention_days=2,
            model_usage_rollup_retention_days=2,
            model_usage_reconcile_minutes=2 * 1_440,
        )


def test_model_usage_startup_accepts_maximum_safe_retention_boundary():
    configured = Settings(
        _env_file=None,
        secret_key="test-secret-key-with-at-least-32-bytes",
        model_usage_raw_retention_days=2,
        model_usage_rollup_retention_days=2,
        model_usage_reconcile_minutes=(2 * 1_440) - 1,
        model_usage_health_failure_window_seconds=3_600,
        model_usage_failure_store_ttl_seconds=3_660,
    )

    assert configured.model_usage_reconcile_minutes == (2 * 1_440) - 1
    assert configured.model_usage_rollup_retention_days == 2


def test_reconcile_partitions_full_window_into_bounded_chunks(monkeypatch):
    from app.repositories.model_usage import ModelUsageRepository

    repository = ModelUsageRepository(lambda: None)
    chunks = []

    def reconcile_chunk(*, start_inclusive, end_exclusive, insert_batch_size):
        chunks.append((start_inclusive, end_exclusive, insert_batch_size))
        return 1

    monkeypatch.setattr(repository, "_reconcile_minute_chunk", reconcile_chunk)
    start = datetime(2026, 7, 20, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=125)

    assert (
        repository.reconcile_minute_range(
            start_inclusive=start,
            end_exclusive=end,
            chunk_minutes=60,
            insert_batch_size=17,
        )
        == 3
    )
    assert chunks == [
        (start, start + timedelta(minutes=60), 17),
        (start + timedelta(minutes=60), start + timedelta(minutes=120), 17),
        (start + timedelta(minutes=120), end, 17),
    ]


def test_reconcile_partition_helper_never_exceeds_insert_batch_size():
    from app.repositories.model_usage import _partition_rows

    partitions = list(_partition_rows(iter(range(11)), batch_size=4))

    assert partitions == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
    assert max(map(len, partitions)) == 4


def test_reconcile_uses_exact_complete_minute_window(monkeypatch):
    from app.workers import model_usage as worker

    captured = {}

    class Repository:
        def reconcile_minute_range(self, *, start_inclusive, end_exclusive, chunk_minutes):
            captured.update(
                start=start_inclusive,
                end=end_exclusive,
                chunk_minutes=chunk_minutes,
            )
            return 4

    now = datetime(2026, 7, 21, 12, 34, 56, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(worker, "_build_repository", lambda: Repository())
    monkeypatch.setattr(settings, "model_usage_tracking_enabled", True)

    assert worker.reconcile_model_usage(now=now) == 4
    assert captured["end"] == datetime(2026, 7, 21, 12, 34, tzinfo=timezone.utc)
    assert captured["start"] == captured["end"] - timedelta(
        minutes=settings.model_usage_reconcile_minutes
    )
    assert captured["chunk_minutes"] == settings.model_usage_reconcile_chunk_minutes


def test_maintenance_short_circuits_without_database_when_tracking_disabled(monkeypatch):
    from app.workers import model_usage as worker

    monkeypatch.setattr(settings, "model_usage_tracking_enabled", False)
    monkeypatch.setattr(
        worker,
        "_build_repository",
        lambda: (_ for _ in ()).throw(AssertionError("database must not be opened")),
    )

    assert worker.reconcile_model_usage() == 0
    assert worker.cleanup_model_usage() == {
        "raw_events_deleted": 0,
        "rollups_deleted": 0,
        "tracking_enabled": False,
    }


def test_cleanup_uses_configured_retention_and_batch_size(monkeypatch):
    from app.workers import model_usage as worker

    captured = {}

    class Repository:
        def delete_raw_events_older_than(self, cutoff, *, batch_size):
            captured.update(raw_cutoff=cutoff, raw_batch=batch_size)
            return 5001

        def delete_rollups_older_than(self, cutoff, *, batch_size):
            captured.update(rollup_cutoff=cutoff, rollup_batch=batch_size)
            return 12

    now = datetime(2026, 7, 21, 4, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(worker, "_build_repository", lambda: Repository())
    monkeypatch.setattr(settings, "model_usage_tracking_enabled", True)

    assert worker.cleanup_model_usage(now=now) == {
        "raw_events_deleted": 5001,
        "rollups_deleted": 12,
    }
    assert captured["raw_cutoff"] == now - timedelta(days=90)
    assert captured["rollup_cutoff"] == now - timedelta(days=730)
    assert captured["raw_batch"] == 5000
    assert captured["rollup_batch"] == 5000


def test_worker_repository_uses_application_container(monkeypatch):
    from app.core import container as container_module
    from app.workers import model_usage as worker

    sentinel = object()

    class Container:
        def model_usage_repository(self):
            return sentinel

    monkeypatch.setattr(container_module, "get_container", lambda: Container())

    assert worker._build_repository() is sentinel
