from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.config import settings


def test_reconcile_uses_exact_complete_minute_window(monkeypatch):
    from app.workers import model_usage as worker

    captured = {}

    class Repository:
        def reconcile_minute_range(self, *, start_inclusive, end_exclusive):
            captured.update(start=start_inclusive, end=end_exclusive)
            return 4

    now = datetime(2026, 7, 21, 12, 34, 56, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(worker, "_build_repository", lambda: Repository())
    monkeypatch.setattr(settings, "model_usage_tracking_enabled", True)

    assert worker.reconcile_model_usage(now=now) == 4
    assert captured["end"] == datetime(2026, 7, 21, 12, 34, tzinfo=timezone.utc)
    assert captured["start"] == captured["end"] - timedelta(
        minutes=settings.model_usage_reconcile_minutes
    )


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
