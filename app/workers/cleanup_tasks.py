import asyncio
import logging
from datetime import datetime, timezone

import redis
from celery.schedules import crontab

from app.ai.agents.rag_agent import RAGAgent
from app.core.config import settings
from app.core.container import get_container
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
    1. Queries the DB for PENDING interrupts whose expiry time has passed and
       marks them EXPIRED (authoritative source of truth).
    2. Scans Redis for expired interrupt keys and deletes them (supplementary).
    3. Cleans up LangGraph checkpoint state for expired threads.
    """
    try:
        now = datetime.now(timezone.utc)
        expired_threads: list[str] = []

        # ── DB-backed expiry (authoritative) ────────────────────────────────────
        db_expired_count = 0
        try:
            container = get_container()
            hitl_repo = container.hitl_interrupt_repository()
            expired_records = hitl_repo.get_expired_pending(now)
            expired_threads.extend(
                record.thread_id for record in expired_records if getattr(record, "thread_id", None)
            )
            for record in expired_records:
                try:
                    hitl_repo.mark_expired(record.id)
                    db_expired_count += 1
                except Exception:
                    pass
        except Exception as db_exc:
            logger.warning("DB interrupt expiry check failed: %s", db_exc, exc_info=True)

        # ── Redis cleanup (supplementary) ────────────────────────────────────────
        redis_url = getattr(settings, "redis_url", "") or ""
        expired_count = 0
        checkpoint_cleaned_count = 0
        active_count = 0

        if redis_url.strip():
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
                            expired_threads.append(conversation_id)

                        redis_client.delete(key)
                    else:
                        active_count += 1

                except Exception:
                    continue

        # ── Checkpoint cleanup ───────────────────────────────────────────────────
        if expired_threads:
            expired_threads = list(dict.fromkeys(expired_threads))
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                checkpoint_cleaned_count = loop.run_until_complete(
                    _cleanup_checkpoint_states(expired_threads)
                )
            finally:
                loop.close()

        total_expired = db_expired_count + expired_count
        if total_expired > 0:
            logger.info(
                "HITL cleanup: %d DB records expired, %d Redis keys expired, "
                "%d checkpoint states cleaned",
                db_expired_count,
                expired_count,
                checkpoint_cleaned_count,
            )

        return {
            "success": True,
            "timestamp": now.isoformat(),
            "db_expired_interrupts": db_expired_count,
            "redis_expired_interrupts": expired_count,
            "checkpoints_cleaned": checkpoint_cleaned_count,
            "active_interrupts": active_count,
            "message": (
                f"Cleanup completed: {db_expired_count} DB + {expired_count} Redis "
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


async def _cleanup_checkpoint_states(thread_ids: list) -> int:
    """
    Clean up LangGraph checkpoint states for the given thread IDs.

    Args:
        thread_ids: List of thread IDs (conversation IDs) to clean up

    Returns:
        Number of successfully cleaned checkpoint states
    """
    cleaned_count = 0

    try:
        from app.ai.checkpoint import CheckpointManager

        checkpoint_manager = CheckpointManager(
            db_url=settings.database_url,
            settings=settings,
        )
        await checkpoint_manager.setup()

        for thread_id in thread_ids:
            try:
                if await checkpoint_manager.delete_thread(thread_id):
                    cleaned_count += 1
            except Exception:
                continue

        await checkpoint_manager.cleanup()

    except Exception:
        pass

    return cleaned_count
