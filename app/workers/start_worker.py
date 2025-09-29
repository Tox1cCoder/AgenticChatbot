"""
Celery Worker Startup Script

This script is used to start the Celery worker for background document processing.

Usage:
    python -m app.workers.start_worker

Or using Celery directly:
    celery -A app.workers.celery_app worker --loglevel=info --concurrency=2

Requirements:
    - Redis server must be running on localhost:6379
    - Database must be accessible
    - Qdrant server should be running for vector operations
"""

import os
import sys
import subprocess
from pathlib import Path


def start_worker():
    """Start the Celery worker with appropriate settings."""

    # Ensure we're in the correct directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    # Set environment variables if not already set
    if not os.getenv("CELERY_BROKER_URL"):
        os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379/0"

    if not os.getenv("CELERY_RESULT_BACKEND"):
        os.environ["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"

    # Start the worker
    cmd = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "app.workers.celery_app",
        "worker",
        "--loglevel=info",
        "--concurrency=2",
        "--max-tasks-per-child=10",
        "--time-limit=300",
        "--soft-time-limit=240",
    ]

    print("Starting Celery worker...")
    print("Command:", " ".join(cmd))

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down worker...")
    except Exception as e:
        print(f"Error starting worker: {e}")


if __name__ == "__main__":
    start_worker()
