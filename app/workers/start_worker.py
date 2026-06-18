import logging
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def _check_mineru_service(settings) -> None:
    """Log a prominent warning if mineru_api_url is configured but unreachable."""
    url = str(getattr(settings, "mineru_api_url", "") or "").strip()
    if not url:
        return  # Not configured — cold-start mode, no check needed

    # Probe a health/docs endpoint
    probe_url = url.rstrip("/") + "/docs"  # mineru-api serves FastAPI /docs
    try:
        with urllib.request.urlopen(probe_url, timeout=5) as resp:
            if resp.status < 400:
                logger.info("MinerU service at %s is reachable.", url)
                return
    except Exception:
        pass

    logger.warning(
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║  WARNING: MinerU API service is NOT reachable            ║\n"
        "║  MINERU_API_URL = %s\n"
        "║  Document parsing will fail until the service is running.║\n"
        "║  Start it: scripts/start_mineru_service.ps1              ║\n"
        "╚══════════════════════════════════════════════════════════╝",
        url,
    )


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


def _build_worker_cmd(
    queues: str,
    concurrency: int,
    pool: str,
    time_limit: int,
    soft_time_limit: int,
    label: str,
    settings,
) -> list[str]:
    """Return the argv list for a single Celery worker process."""
    return [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "app.workers.celery_app",
        "worker",
        "--loglevel=info",
        f"--queues={queues}",
        f"--pool={pool}",
        f"--concurrency={concurrency}",
        f"--hostname=worker-{label}@%h",
        f"--max-tasks-per-child={settings.celery_worker_max_tasks_per_child}",
        f"--time-limit={time_limit}",
        f"--soft-time-limit={soft_time_limit}",
    ]


def _spawn_worker(
    queues: str,
    concurrency: int,
    pool: str,
    time_limit: int,
    soft_time_limit: int,
    label: str,
    settings,
) -> subprocess.Popen:
    """Start a Celery worker subprocess for *queues* and return its Popen handle."""
    cmd = _build_worker_cmd(
        queues=queues,
        concurrency=concurrency,
        pool=pool,
        time_limit=time_limit,
        soft_time_limit=soft_time_limit,
        label=label,
        settings=settings,
    )
    print(f"Starting {label} worker: queues={queues} pool={pool} concurrency={concurrency}")
    print(f"Command [{label}]:", " ".join(cmd))
    return subprocess.Popen(cmd)


def start_worker():
    """Start parse and index Celery workers using config-driven flags.

    Spawns two parallel worker processes:
    - parse worker  : consumes the ``parse`` queue, concurrency from celery_parse_concurrency
    - index worker  : consumes the ``index`` queue, concurrency from celery_index_concurrency

    Dev-mode tip: for a minimal single-process setup start Celery manually with
    ``celery -A app.workers.celery_app worker -Q parse,index`` to handle both queues.
    """
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    # Defer settings import until invocation so test monkeypatches of env
    # vars are picked up via ``get_settings.cache_clear()``.
    from app.core.config import get_settings

    settings = get_settings()
    _check_mineru_service(settings)

    system = platform.system()
    pool = _resolve_worker_pool(settings.celery_worker_pool, system)

    parse_concurrency = settings.celery_parse_concurrency
    index_concurrency = settings.celery_index_concurrency
    if pool == "solo":
        # The solo pool only ever runs one task at a time. Clamp both
        # concurrency values so the printed banner does not mislead.
        parse_concurrency = 1
        index_concurrency = 1

    p_parse = _spawn_worker(
        queues="parse",
        concurrency=parse_concurrency,
        pool=pool,
        time_limit=settings.mineru_timeout + 60,
        soft_time_limit=settings.mineru_timeout + 30,
        label="parse",
        settings=settings,
    )
    p_index = _spawn_worker(
        queues="index",
        concurrency=index_concurrency,
        pool=pool,
        time_limit=settings.celery_index_time_limit,
        soft_time_limit=settings.celery_index_time_limit - 30,
        label="index",
        settings=settings,
    )

    print(f"Both workers started (PIDs: parse={p_parse.pid}, index={p_index.pid})")

    try:
        p_parse.wait()
        p_index.wait()
    except KeyboardInterrupt:
        print("\nShutting down workers...")
        if p_parse.poll() is None:
            p_parse.terminate()
        if p_index.poll() is None:
            p_index.terminate()
        p_parse.wait()
        p_index.wait()
    except Exception as e:
        print(f"Error waiting for workers: {e}")
        if p_parse.poll() is None:
            p_parse.terminate()
        if p_index.poll() is None:
            p_index.terminate()
        raise


if __name__ == "__main__":
    start_worker()
