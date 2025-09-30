import os

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


celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.task_acks_late = True
celery_app.conf.task_reject_on_worker_lost = True

celery_app.conf.imports = (
    "app.workers.document_processor",
    "app.workers.cleanup_tasks",
)
