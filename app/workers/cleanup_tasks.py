import asyncio
import logging
import sys
from datetime import datetime, timezone

import redis
from celery.schedules import crontab

from app.ai.agents.rag_agent import RAGAgent
from app.core.config import settings
from app.core.container import get_container
from app.services.checkpoint_retention_service import CheckpointRetentionService
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


celery_app.conf.beat_schedule = {
    "cleanup-temp-files": {
        "task": "app.workers.cleanup_tasks.cleanup_temp_files_task",
        "schedule": crontab(minute=0, hour="*/6"),
    },
    "cleanup-stuck-documents": {
        "task": "app.workers.document_processor.cleanup_failed_documents",
        "schedule": crontab(minute=30, hour="*/1"),
    },
    "cleanup-abandoned-interrupts": {
        "task": "app.workers.cleanup_tasks.cleanup_abandoned_interrupts",
        "schedule": crontab(minute="*/10"),  # Run every 10 minutes
    },
}


@celery_app.task(name="app.workers.cleanup_tasks.cleanup_temp_files_task")
def cleanup_temp_files_task(older_than_hours: int = 24):
    try:
        container = get_container()
        processing_service = container.document_processing_service()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                processing_service.cleanup_temp_files(older_than_hours)
            )
            logger.info(f"Temp file cleanup completed: {result}")
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Temp file cleanup task failed: {str(e)}")
        return {
            "files_removed": 0,
            "error": str(e),
            "message": "Temp file cleanup task failed",
        }


@celery_app.task(name="app.workers.cleanup_tasks.health_check_task")
def health_check_task():
    try:
        container = get_container()
        settings = container.config()
        qdrant_client = container.qdrant_client()
        embedding_service = container.rag_embedding_service()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            rag_agent = RAGAgent(
                settings=settings,
                qdrant_client=qdrant_client,
                embedding_service=embedding_service,
                collection_name=settings.qdrant_collection_name,
            )
            loop.run_until_complete(rag_agent.initialize())
            status = rag_agent.get_status()
            loop.run_until_complete(rag_agent.cleanup())

            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "rag_agent_status": status,
                "message": "Health check passed",
            }
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"success": False, "error": str(e), "message": "Health check failed"}


@celery_app.task(name="app.workers.cleanup_tasks.cleanup_abandoned_interrupts")
def cleanup_abandoned_interrupts():
    """
    Background task to clean up abandoned HITL interrupts.

    This task:
    1. Delegates DB-backed expiry (authoritative) and checkpoint-thread
       cleanup for expired interrupts + soft-deleted conversations to
       CheckpointRetentionService.
    2. Scans Redis for expired interrupt keys and deletes them (supplementary),
       then cleans up their checkpoint threads too.
    """
    try:
        now = datetime.now(timezone.utc)

        # ── Redis cleanup (supplementary) ────────────────────────────────────────
        redis_expired_threads, redis_expired_count, active_count = (
            _scan_and_expire_redis_interrupts(now)
        )

        # ── DB-backed expiry + checkpoint cleanup (authoritative) ──────────────
        retention_counts = {
            "pending_interrupts_inspected": 0,
            "hitl_interrupts_expired": 0,
            "hitl_checkpoint_threads_deleted": 0,
            "soft_deleted_conversations_inspected": 0,
            "conversation_checkpoint_threads_deleted": 0,
        }
        redis_checkpoints_cleaned = 0
        try:
            container = get_container()
            hitl_repo = container.hitl_interrupt_repository()
            conversation_repo = container.conversation_repository()

            # psycopg3 async requires a SelectorEventLoop; the default loop on
            # Windows is a ProactorEventLoop the checkpoint pool cannot use
            # (mirrors app.main's policy). Elsewhere the default loop is fine.
            if sys.platform == "win32":
                loop = asyncio.WindowsSelectorEventLoopPolicy().new_event_loop()
            else:
                loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                retention_counts, redis_checkpoints_cleaned = loop.run_until_complete(
                    _run_checkpoint_retention_cleanup(
                        now=now,
                        hitl_repo=hitl_repo,
                        conversation_repo=conversation_repo,
                        redis_expired_thread_ids=redis_expired_threads,
                    )
                )
            finally:
                loop.close()
        except Exception as retention_exc:
            logger.warning("Checkpoint retention cleanup failed: %s", retention_exc, exc_info=True)

        db_expired_count = retention_counts["hitl_interrupts_expired"]
        checkpoint_cleaned_count = (
            retention_counts["hitl_checkpoint_threads_deleted"]
            + retention_counts["conversation_checkpoint_threads_deleted"]
            + redis_checkpoints_cleaned
        )

        total_expired = db_expired_count + redis_expired_count
        if total_expired > 0:
            logger.info(
                "HITL cleanup: %d DB records expired, %d Redis keys expired, "
                "%d checkpoint states cleaned, %d soft-deleted conversations swept",
                db_expired_count,
                redis_expired_count,
                checkpoint_cleaned_count,
                retention_counts["soft_deleted_conversations_inspected"],
            )

        return {
            "success": True,
            "timestamp": now.isoformat(),
            "db_expired_interrupts": db_expired_count,
            "redis_expired_interrupts": redis_expired_count,
            "checkpoints_cleaned": checkpoint_cleaned_count,
            "active_interrupts": active_count,
            "soft_deleted_conversations_inspected": (
                retention_counts["soft_deleted_conversations_inspected"]
            ),
            "message": (
                f"Cleanup completed: {db_expired_count} DB + {redis_expired_count} Redis "
                f"expired, {checkpoint_cleaned_count} checkpoints cleaned, "
                f"{active_count} active"
            ),
        }

    except Exception as e:
        logger.error("Cleanup abandoned interrupts task failed: %s", str(e))
        return {
            "success": False,
            "error": str(e),
            "message": "Cleanup abandoned interrupts task failed",
        }


def _scan_and_expire_redis_interrupts(now: datetime) -> tuple[list[str], int, int]:
    """
    Scan Redis for interrupt keys past the approval timeout and delete them.

    Redis timestamps are supplementary bookkeeping only — the DB HITLInterrupt
    table remains the source of truth for interrupt lifecycle state, so a
    failure here must not block the DB-backed retention pass.

    Returns:
        (expired_thread_ids, expired_count, active_count)
    """
    redis_url = getattr(settings, "redis_url", "") or ""
    if not redis_url.strip():
        return [], 0, 0

    expired_thread_ids: list[str] = []
    expired_count = 0
    active_count = 0

    try:
        redis_client = redis.from_url(redis_url)
        interrupt_pattern = "interrupt:*"

        for key in redis_client.scan_iter(match=interrupt_pattern):
            try:
                stored_timestamp = redis_client.get(key)
                if not stored_timestamp:
                    continue

                stored_time = datetime.fromisoformat(stored_timestamp.decode("utf-8"))
                # Ensure comparison is between two tz-aware datetimes
                if stored_time.tzinfo is None:
                    stored_time = stored_time.replace(tzinfo=timezone.utc)

                elapsed_minutes = (now - stored_time).total_seconds() / 60

                if elapsed_minutes > settings.hitl_approval_timeout_minutes:
                    expired_count += 1

                    # Key format: "interrupt:{conversation_id}:{interrupt_id}"
                    key_parts = key.decode("utf-8").split(":")
                    conversation_id = key_parts[1] if len(key_parts) > 1 else None

                    if conversation_id:
                        expired_thread_ids.append(conversation_id)

                    redis_client.delete(key)
                else:
                    active_count += 1

            except Exception:
                continue
    except Exception as redis_exc:
        logger.warning("Redis interrupt expiry scan failed: %s", redis_exc, exc_info=True)
        return [], 0, 0

    return expired_thread_ids, expired_count, active_count


async def _run_checkpoint_retention_cleanup(
    now: datetime,
    hitl_repo,
    conversation_repo,
    redis_expired_thread_ids: list[str],
) -> tuple[dict[str, int], int]:
    """
    Run CheckpointRetentionService plus ad-hoc cleanup for Redis-sourced threads.

    A dedicated CheckpointManager is created per invocation (rather than reusing
    the container's singleton) because this coroutine runs inside a fresh event
    loop each time the Celery task fires, and the underlying psycopg connection
    pool is bound to the loop that opened it.

    Returns:
        (retention_counts, redis_checkpoints_cleaned)
    """
    # Imported lazily to avoid loading langgraph/psycopg_pool at worker startup
    # for a task that may never run in a given process.
    from app.ai.checkpoint import CheckpointManager

    checkpoint_manager = CheckpointManager(db_url=settings.database_url, settings=settings)
    await checkpoint_manager.setup()

    try:
        retention_service = CheckpointRetentionService(
            checkpoint_manager=checkpoint_manager,
            hitl_interrupt_repository=hitl_repo,
            conversation_repository=conversation_repo,
        )
        retention_counts = await retention_service.cleanup_expired_and_deleted_threads(now=now)

        redis_checkpoints_cleaned = 0
        for thread_id in dict.fromkeys(redis_expired_thread_ids):
            try:
                if await checkpoint_manager.delete_thread(thread_id):
                    redis_checkpoints_cleaned += 1
            except Exception:
                logger.warning(
                    "Failed to delete checkpoint thread %s (Redis-sourced expiry)",
                    thread_id,
                    exc_info=True,
                )
                continue

        return retention_counts, redis_checkpoints_cleaned
    finally:
        await checkpoint_manager.cleanup()
