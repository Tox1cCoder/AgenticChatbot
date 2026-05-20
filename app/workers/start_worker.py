import os
import platform
import subprocess
import sys
from pathlib import Path


def _resolve_worker_pool(configured_pool: str, system: str) -> str:
    """Resolve the configured pool name, expanding 'auto' per platform.

    Windows defaults to ``threads`` because the legacy ``solo`` pool runs
    one task at a time, which makes batch uploads serialize even when
    concurrency is set. Linux defaults to ``prefork`` for stronger
    isolation in production.
    """
    pool = (configured_pool or "auto").strip().lower()
    if pool != "auto":
        return pool
    return "threads" if system == "Windows" else "prefork"


def start_worker():
    """Start the Celery worker using config-driven flags."""
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    # Defer settings import until invocation so test monkeypatches of env
    # vars are picked up via ``get_settings.cache_clear()``.
    from app.core.config import get_settings

    settings = get_settings()
    system = platform.system()
    pool = _resolve_worker_pool(settings.celery_worker_pool, system)
    concurrency = settings.celery_worker_concurrency
    if pool == "solo" and concurrency != 1:
        # The solo pool only ever runs one task at a time. Clamp the
        # concurrency value so the printed banner does not mislead.
        concurrency = 1

    cmd = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "app.workers.celery_app",
        "worker",
        "--loglevel=info",
        f"--pool={pool}",
        f"--concurrency={concurrency}",
        f"--max-tasks-per-child={settings.celery_worker_max_tasks_per_child}",
        f"--time-limit={settings.celery_worker_time_limit}",
        f"--soft-time-limit={settings.celery_worker_soft_time_limit}",
    ]

    print(
        "Starting Celery worker: "
        f"pool={pool} concurrency={concurrency} "
        f"prefetch={settings.celery_worker_prefetch_multiplier} "
        f"(platform={system})"
    )
    print("Command:", " ".join(cmd))

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down worker...")
    except Exception as e:
        print(f"Error starting worker: {e}")


if __name__ == "__main__":
    start_worker()
