import os
import platform
import subprocess
import sys
from pathlib import Path


def start_worker():
    """Start the Celery worker"""

    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

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

    # Detect platform and add appropriate pool configuration
    system = platform.system()
    if system == "Windows":
        cmd.append("--pool=solo")
        print(f"Detected platform: {system} - Using 'solo' pool")
    else:
        print(f"Detected platform: {system} - Using default 'prefork' pool")

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
