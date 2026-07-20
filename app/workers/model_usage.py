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

Task 5 builds the repository directly here (its DI wiring lands in Task 6), so
the tasks are functional and routable today.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.repositories.model_usage import ModelUsageRepository
from app.usage.recorder import deserialize_record_command
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _build_repository() -> ModelUsageRepository:
    """Construct a repository over the shared session factory.

    Deferred until task execution so worker-module import stays cheap and free
    of a database dependency.
    """
    from app.database.database import Database

    return ModelUsageRepository(session_factory=Database().session)


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
        raise self.retry(exc=exc, countdown=_retry_delay(self.request.retries)) from exc
    return {"event_key": command.event_key, "inserted": bool(result.inserted)}


@celery_app.task(
    name="app.workers.model_usage.reconcile_model_usage_task",
    ignore_result=True,
)
def reconcile_model_usage_task() -> int:
    """Rebuild the trailing minute rollups exactly from raw events."""
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(minutes=settings.model_usage_reconcile_minutes)
    repository = _build_repository()
    return repository.reconcile_minute_range(start_inclusive=start, end_exclusive=end)


@celery_app.task(
    name="app.workers.model_usage.cleanup_model_usage_task",
    ignore_result=True,
)
def cleanup_model_usage_task() -> dict:
    """Delete raw events and rollups older than their retention windows."""
    now = datetime.now(timezone.utc)
    raw_cutoff = now - timedelta(days=settings.model_usage_raw_retention_days)
    rollup_cutoff = now - timedelta(days=settings.model_usage_rollup_retention_days)
    repository = _build_repository()
    raw_deleted = repository.delete_raw_events_older_than(
        raw_cutoff, batch_size=settings.model_usage_cleanup_batch_size
    )
    rollup_deleted = repository.delete_rollups_older_than(
        rollup_cutoff, batch_size=settings.model_usage_cleanup_batch_size
    )
    return {"raw_events_deleted": raw_deleted, "rollups_deleted": rollup_deleted}
