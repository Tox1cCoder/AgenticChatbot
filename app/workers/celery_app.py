from celery import Celery

from app.core.config import get_settings

settings = get_settings()


celery_app = Celery("chatbot_tasks")


celery_app.conf.broker_url = settings.celery_broker_url
celery_app.conf.result_backend = settings.celery_result_backend


celery_app.conf.task_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.result_serializer = "json"
celery_app.conf.timezone = "UTC"
celery_app.conf.enable_utc = True


celery_app.conf.worker_prefetch_multiplier = settings.celery_worker_prefetch_multiplier
celery_app.conf.task_acks_late = True
celery_app.conf.task_reject_on_worker_lost = True
celery_app.conf.worker_cancel_long_running_tasks_on_connection_loss = (
    settings.celery_worker_cancel_long_running_tasks_on_connection_loss
)

celery_app.conf.broker_connection_retry = True
celery_app.conf.broker_connection_retry_on_startup = True
celery_app.conf.broker_transport_options = {
    "health_check_interval": settings.celery_broker_health_check_interval,
    "socket_keepalive": settings.celery_broker_socket_keepalive,
    "retry_on_timeout": settings.celery_broker_retry_on_timeout,
    "visibility_timeout": settings.celery_broker_visibility_timeout,
}
celery_app.conf.result_backend_transport_options = {
    "visibility_timeout": settings.celery_broker_visibility_timeout,
}
celery_app.conf.visibility_timeout = settings.celery_broker_visibility_timeout

celery_app.conf.task_time_limit = settings.celery_worker_time_limit
celery_app.conf.task_soft_time_limit = settings.celery_worker_soft_time_limit

celery_app.conf.imports = (
    "app.workers.document_processor",
    "app.workers.cleanup_tasks",
    "app.workers.conversation_compaction",
    "app.workers.model_usage",
)

celery_app.conf.task_routes = {
    "app.workers.document_processor.parse_document_task": {"queue": "parse"},
    "app.workers.document_processor.index_document_task": {"queue": "index"},
    "app.workers.conversation_compaction.compact_conversation_task": {"queue": "summary"},
    "app.workers.conversation_compaction.compact_backfill_conversation_task": {"queue": "summary"},
    "app.workers.conversation_compaction.reconcile_conversation_summaries_task": {
        "queue": "summary"
    },
    "app.workers.conversation_compaction.backfill_conversation_summaries_task": {
        "queue": "summary"
    },
    "app.workers.model_usage.retry_model_usage_write_task": {"queue": "summary"},
    "app.workers.model_usage.reconcile_model_usage_task": {"queue": "summary"},
    "app.workers.model_usage.cleanup_model_usage_task": {"queue": "summary"},
}

celery_app.conf.beat_schedule = {
    "reconcile-conversation-summaries": {
        "task": "app.workers.conversation_compaction.reconcile_conversation_summaries_task",
        "schedule": settings.conversation_summary_reconcile_seconds,
        "options": {"queue": "summary"},
    },
    "backfill-conversation-summaries": {
        "task": "app.workers.conversation_compaction.backfill_conversation_summaries_task",
        "schedule": 3600,
        "options": {"queue": "summary"},
    },
}
