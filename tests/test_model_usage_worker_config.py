"""Model-usage worker registration, routing, and beat-schedule preservation.

Covers Task 5 Step 5: ``app.workers.model_usage`` is registered in
``celery_app.conf.imports``; its retry/reconcile/cleanup tasks route to the
``summary`` queue; and importing ``app.workers.cleanup_tasks`` no longer
clobbers ``celery_app``'s two conversation-summary beat entries (plan finding
#7 -- cleanup_tasks must ``.update()`` the beat schedule, not reassign it).

Also verifies the reconcile/cleanup tasks delegate to the Task 3 repository
maintenance methods with the configured retention/reconcile settings.
"""

from __future__ import annotations

from datetime import datetime, timezone

from celery.schedules import crontab

from app.workers.celery_app import celery_app

_MODEL_USAGE_TASKS = (
    "app.workers.model_usage.retry_model_usage_write_task",
    "app.workers.model_usage.reconcile_model_usage_task",
    "app.workers.model_usage.cleanup_model_usage_task",
)


def test_model_usage_module_is_registered_in_imports():
    assert "app.workers.model_usage" in celery_app.conf.imports


def test_model_usage_tasks_route_to_summary_queue():
    routes = dict(celery_app.conf.task_routes)
    for task_name in _MODEL_USAGE_TASKS:
        assert routes[task_name] == {"queue": "summary"}, task_name


def test_conversation_summary_beat_entries_survive_cleanup_tasks_import():
    # Importing cleanup_tasks must merge (not replace) the beat schedule so the
    # conversation-summary entries defined in celery_app.py remain intact.
    from app.workers import cleanup_tasks  # noqa: F401

    schedule = dict(celery_app.conf.beat_schedule)
    assert "reconcile-conversation-summaries" in schedule
    assert "backfill-conversation-summaries" in schedule
    # The cleanup entries were still added by the same merge.
    assert "cleanup-temp-files" in schedule
    assert "cleanup-abandoned-interrupts" in schedule


def test_model_usage_tasks_are_registered_on_the_app():
    # Celery registers tasks when the worker imports each conf.imports module;
    # import it here to mirror that startup step.
    import app.workers.model_usage  # noqa: F401

    for task_name in _MODEL_USAGE_TASKS:
        assert task_name in celery_app.tasks


def test_reconcile_task_delegates_to_repository(monkeypatch):
    from app.workers import model_usage as worker

    captured: dict[str, object] = {}

    class FakeRepo:
        def reconcile_minute_range(self, *, start_inclusive, end_exclusive):
            captured["start"] = start_inclusive
            captured["end"] = end_exclusive
            return 7

    monkeypatch.setattr(worker, "_build_repository", lambda: FakeRepo())
    result = worker.reconcile_model_usage_task()

    assert result == 7
    span_minutes = (captured["end"] - captured["start"]).total_seconds() / 60
    from app.core.config import settings

    assert span_minutes == settings.model_usage_reconcile_minutes
    assert captured["start"].tzinfo is not None
    assert captured["end"].second == 0 and captured["end"].microsecond == 0


def test_cleanup_task_delegates_to_repository(monkeypatch):
    from app.core.config import settings
    from app.workers import model_usage as worker

    captured: dict[str, object] = {}

    class FakeRepo:
        def delete_raw_events_older_than(self, cutoff, *, batch_size):
            captured["raw_cutoff"] = cutoff
            captured["raw_batch"] = batch_size
            return 3

        def delete_rollups_older_than(self, cutoff, *, batch_size):
            captured["rollup_cutoff"] = cutoff
            captured["rollup_batch"] = batch_size
            return 5

    monkeypatch.setattr(worker, "_build_repository", lambda: FakeRepo())
    result = worker.cleanup_model_usage_task()

    assert result == {"raw_events_deleted": 3, "rollups_deleted": 5}
    assert captured["raw_batch"] == settings.model_usage_cleanup_batch_size
    assert captured["rollup_batch"] == settings.model_usage_cleanup_batch_size
    now = datetime.now(timezone.utc)
    raw_age_days = (now - captured["raw_cutoff"]).total_seconds() / 86400
    rollup_age_days = (now - captured["rollup_cutoff"]).total_seconds() / 86400
    assert round(raw_age_days) == settings.model_usage_raw_retention_days
    assert round(rollup_age_days) == settings.model_usage_rollup_retention_days


def test_retry_task_is_bound_and_bounded_by_max_attempts():
    from app.core.config import settings

    task = celery_app.tasks["app.workers.model_usage.retry_model_usage_write_task"]
    # bind=True exposes the task instance; max_retries reflects the setting.
    assert task.max_retries == settings.model_usage_retry_max_attempts


def test_usage_beat_entries_have_required_cron_and_summary_queue():
    schedule = dict(celery_app.conf.beat_schedule)
    reconcile = schedule["reconcile-model-usage"]
    cleanup = schedule["cleanup-model-usage"]

    assert isinstance(reconcile["schedule"], crontab)
    assert reconcile["schedule"].minute == {15}
    assert reconcile["options"] == {"queue": "summary"}
    assert cleanup["schedule"].minute == {40}
    assert cleanup["schedule"].hour == {3}
    assert cleanup["options"] == {"queue": "summary"}


def test_named_schedule_dictionaries_merge_to_same_union_in_both_orders():
    from app.workers.celery_app import CELERY_BEAT_SCHEDULE
    from app.workers.cleanup_tasks import CLEANUP_BEAT_SCHEDULE

    first = {}
    first.update(CELERY_BEAT_SCHEDULE)
    first.update(CLEANUP_BEAT_SCHEDULE)
    second = {}
    second.update(CLEANUP_BEAT_SCHEDULE)
    second.update(CELERY_BEAT_SCHEDULE)

    assert first == second
    assert set(first) == {
        "reconcile-conversation-summaries",
        "backfill-conversation-summaries",
        "reconcile-model-usage",
        "cleanup-model-usage",
        "cleanup-temp-files",
        "cleanup-stuck-documents",
        "cleanup-abandoned-interrupts",
    }


def test_retry_short_circuits_when_tracking_disabled(monkeypatch):
    from app.core.config import settings
    from app.workers import model_usage as worker

    monkeypatch.setattr(settings, "model_usage_tracking_enabled", False)
    monkeypatch.setattr(
        worker,
        "deserialize_record_command",
        lambda _payload: (_ for _ in ()).throw(AssertionError("payload must not be read")),
    )

    assert worker.retry_model_usage_write_task.run({"secret": "ignored"}) == {
        "tracking_enabled": False
    }


def test_retry_duplicate_emits_duplicate_persistence_metric(monkeypatch):
    from types import SimpleNamespace

    from app.core.config import settings
    from app.observability.model_usage import ModelUsageMetrics
    from app.workers import model_usage as worker

    metrics = ModelUsageMetrics()
    monkeypatch.setattr(settings, "model_usage_tracking_enabled", True)
    monkeypatch.setattr(worker, "model_usage_metrics", metrics)
    monkeypatch.setattr(
        worker,
        "deserialize_record_command",
        lambda _payload: SimpleNamespace(event_key="stable-key"),
    )
    monkeypatch.setattr(
        worker,
        "_build_repository",
        lambda: SimpleNamespace(record_event=lambda _command: SimpleNamespace(inserted=False)),
    )

    result = worker.retry_model_usage_write_task.run({})

    assert result == {"event_key": "stable-key", "inserted": False}
    assert metrics.persistence.labels(outcome="duplicate", failure_class="none")._value.get() == 1


def test_failed_retry_emits_bounded_retry_persistence_metric(monkeypatch):
    from types import SimpleNamespace

    import pytest

    from app.core.config import settings
    from app.observability.model_usage import ModelUsageMetrics
    from app.workers import model_usage as worker

    metrics = ModelUsageMetrics()
    monkeypatch.setattr(settings, "model_usage_tracking_enabled", True)
    monkeypatch.setattr(worker, "model_usage_metrics", metrics)
    monkeypatch.setattr(
        worker,
        "deserialize_record_command",
        lambda _payload: SimpleNamespace(event_key="stable-key"),
    )

    def fail(_command):
        raise ConnectionError("tenant content must not escape")

    monkeypatch.setattr(
        worker,
        "_build_repository",
        lambda: SimpleNamespace(record_event=fail),
    )

    with pytest.raises(ConnectionError):
        worker.retry_model_usage_write_task.run({})

    assert (
        metrics.persistence.labels(
            outcome="retry_enqueued", failure_class="connection"
        )._value.get()
        == 1
    )
