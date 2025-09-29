"""
Celery Application Configuration

This module sets up the Celery application for background task processing.
"""

from celery import Celery
import os
from app.core.config import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery("chatbot_tasks")

# Configuration
celery_app.conf.broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_app.conf.result_backend = os.getenv(
    "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
)

# Task settings
celery_app.conf.task_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.result_serializer = "json"
celery_app.conf.timezone = "UTC"
celery_app.conf.enable_utc = True

# Worker settings
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.task_acks_late = True
celery_app.conf.task_reject_on_worker_lost = True

celery_app.autodiscover_tasks(["app.workers"])
