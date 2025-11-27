import logging
import asyncio
from datetime import datetime, timedelta
import redis

from celery.schedules import crontab

from app.core.container import get_container
from app.core.config import settings
from app.workers.celery_app import celery_app
from app.ai.agents.rag_agent import RAGAgent

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
        embedding_model = container.embedding_model()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            rag_agent = RAGAgent(
                settings=settings,
                qdrant_client=qdrant_client,
                embedding_model=embedding_model,
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
    """
    try:
        redis_url = getattr(settings, "redis_url", "") or ""
        if not redis_url.strip():
            message = "Redis URL not configured; skipping interrupt cleanup task."
            logger.debug(message)
            return {
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "expired_interrupts_cleaned": 0,
                "active_interrupts": 0,
                "message": message,
            }

        # Connect to Redis
        redis_client = redis.from_url(redis_url)

        # Scan for all interrupt keys
        interrupt_pattern = "interrupt:*"
        expired_count = 0
        active_count = 0

        for key in redis_client.scan_iter(match=interrupt_pattern):
            try:
                # Get the stored timestamp
                stored_timestamp = redis_client.get(key)
                if not stored_timestamp:
                    continue

                # Parse timestamp
                stored_time = datetime.fromisoformat(stored_timestamp.decode("utf-8"))
                elapsed_minutes = (datetime.utcnow() - stored_time).total_seconds() / 60

                if elapsed_minutes > settings.hitl_approval_timeout_minutes:
                    # Interrupt has expired
                    expired_count += 1

                    # Extract conversation_id and interrupt_id from key
                    # Key format: "interrupt:{conversation_id}:{interrupt_id}"
                    key_parts = key.decode("utf-8").split(":")
                    conversation_id = key_parts[1] if len(key_parts) > 1 else "unknown"
                    interrupt_id = key_parts[2] if len(key_parts) > 2 else "unknown"

                    # Delete the expired key
                    redis_client.delete(key)
                else:
                    active_count += 1

            except Exception as e:
                logger.error(f"Error processing interrupt key {key}: {e}")
                continue

        result = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "expired_interrupts_cleaned": expired_count,
            "active_interrupts": active_count,
            "message": f"Cleanup completed: {expired_count} expired, {active_count} active",
        }

        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired interrupts")

        return result

    except Exception as e:
        logger.error(f"Cleanup abandoned interrupts task failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Cleanup abandoned interrupts task failed",
        }
