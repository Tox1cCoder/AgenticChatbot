"""Celery tasks for the model-usage ledger: failed-write retry and maintenance.

Three tasks, all routed to the ``summary`` queue:

* ``retry_model_usage_write_task`` -- durable delivery of a single ledger write
  that failed inline in the recorder. Its payload is the content-free,
  JSON-safe :class:`RecordEventCommand` (produced by
  ``app.usage.recorder.serialize_record_command``); it carries no prompts,
  responses, tool arguments, or raw provider payloads. Replaying it is
  idempotent because ``record_event`` upserts on ``event_key``, so the same
  ``operation_id:attempt`` never double-counts. Delivery uses exponential
  backoff governed by ``model_usage_retry_*`` settings (these govern the
  *ledger write*, never a provider call).
* ``reconcile_model_usage_task`` -- rebuild the trailing minute rollups exactly
  from raw events over ``model_usage_reconcile_minutes``.
* ``cleanup_model_usage_task`` -- batched retention deletes of raw events and
  rollups older than the configured retention windows.

Repositories resolve lazily from the application container so every worker
uses the same configured database/session provider as API processes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.observability.model_usage import model_usage_metrics
from app.repositories.model_usage import ModelUsageRepository
from app.usage.recorder import deserialize_record_command
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _build_repository() -> ModelUsageRepository:
    """Resolve a repository over the application's shared session provider.

    Deferred until task execution so worker-module import stays cheap and free
    of a database dependency.
    """
    from app.core.container import get_container

    return get_container().model_usage_repository()


def _retry_delay(retries: int) -> int:
    """Exponential ledger-write backoff: ``base * 2**retries`` seconds."""
    return int(settings.model_usage_retry_base_seconds * (2 ** max(0, retries)))


@celery_app.task(
    bind=True,
    name="app.workers.model_usage.retry_model_usage_write_task",
    ignore_result=True,
    max_retries=settings.model_usage_retry_max_attempts,
)
def retry_model_usage_write_task(self, payload: dict) -> dict:
    """Replay one previously-failed ledger write; idempotent on ``event_key``."""
    if not settings.model_usage_tracking_enabled:
        return {"tracking_enabled": False}
    command = deserialize_record_command(payload)
    repository = _build_repository()
    try:
        result = repository.record_event(command)
    except Exception as exc:
        logger.warning(
            "model_usage failed-write retry attempt=%s event_key=%s failure=%s",
            self.request.retries,
            command.event_key,
            type(exc).__name__,
        )
        outcome = (
            "dropped"
            if self.request.retries >= settings.model_usage_retry_max_attempts
            else "retry_enqueued"
        )
        model_usage_metrics.record_persistence(
            outcome,
            failure_class=type(exc).__name__,
        )
        raise self.retry(exc=exc, countdown=_retry_delay(self.request.retries)) from exc
    model_usage_metrics.record_persistence("stored" if result.inserted else "duplicate")
    return {"event_key": command.event_key, "inserted": bool(result.inserted)}


def reconcile_model_usage(*, now: datetime | None = None) -> int:
    """Rebuild exactly the trailing configured COMPLETE UTC minutes."""
    if not settings.model_usage_tracking_enabled:
        return 0
    complete_minute = (
        (now or datetime.now(timezone.utc))
        .astimezone(timezone.utc)
        .replace(second=0, microsecond=0)
    )
    start = complete_minute - timedelta(minutes=settings.model_usage_reconcile_minutes)
    return _build_repository().reconcile_minute_range(
        start_inclusive=start,
        end_exclusive=complete_minute,
        chunk_minutes=settings.model_usage_reconcile_chunk_minutes,
    )


@celery_app.task(
    name="app.workers.model_usage.reconcile_model_usage_task",
    ignore_result=True,
)
def reconcile_model_usage_task() -> int:
    """Rebuild the trailing minute rollups exactly from raw events."""
    return reconcile_model_usage()


def cleanup_model_usage(*, now: datetime | None = None) -> dict:
    """Delete configured retention windows using repository-bounded batches."""
    if not settings.model_usage_tracking_enabled:
        return {
            "raw_events_deleted": 0,
            "rollups_deleted": 0,
            "tracking_enabled": False,
        }
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    raw_cutoff = current - timedelta(days=settings.model_usage_raw_retention_days)
    rollup_cutoff = current - timedelta(days=settings.model_usage_rollup_retention_days)
    repository = _build_repository()
    raw_deleted = repository.delete_raw_events_older_than(
        raw_cutoff, batch_size=settings.model_usage_cleanup_batch_size
    )
    rollup_deleted = repository.delete_rollups_older_than(
        rollup_cutoff, batch_size=settings.model_usage_cleanup_batch_size
    )
    return {
        "raw_events_deleted": raw_deleted,
        "rollups_deleted": rollup_deleted,
    }


@celery_app.task(
    name="app.workers.model_usage.cleanup_model_usage_task",
    ignore_result=True,
)
def cleanup_model_usage_task() -> dict:
    """Delete raw events and rollups older than their retention windows."""
    return cleanup_model_usage()
