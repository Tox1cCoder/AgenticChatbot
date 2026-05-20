"""Celery worker startup is config-driven and uses parallel-capable pools.

Replaces the prior assertion that Windows always appends ``--pool=solo``.
"""

from __future__ import annotations


def _captured_cmd(monkeypatch, *, system: str, env: dict[str, str] | None = None) -> list[str]:
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr("app.workers.start_worker.os.chdir", lambda _path: None)
    monkeypatch.setattr("app.workers.start_worker.platform.system", lambda: system)
    monkeypatch.setattr("app.workers.start_worker.sys.executable", "python")

    for key in (
        "CELERY_WORKER_POOL",
        "CELERY_WORKER_CONCURRENCY",
        "CELERY_WORKER_PREFETCH_MULTIPLIER",
        "CELERY_WORKER_MAX_TASKS_PER_CHILD",
        "CELERY_WORKER_TIME_LIMIT",
        "CELERY_WORKER_SOFT_TIME_LIMIT",
    ):
        monkeypatch.delenv(key, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)

    # Force the settings cache to rebuild so env values take effect.
    from app.core import config as _cfg

    _cfg.get_settings.cache_clear()  # type: ignore[attr-defined]

    def fake_run(cmd):
        captured["cmd"] = cmd

    monkeypatch.setattr("app.workers.start_worker.subprocess.run", fake_run)

    from app.workers.start_worker import start_worker

    start_worker()
    return captured["cmd"]


def test_windows_auto_pool_resolves_to_threads(monkeypatch):
    cmd = _captured_cmd(monkeypatch, system="Windows", env={"CELERY_WORKER_POOL": "auto"})
    joined = " ".join(cmd)
    assert "--pool=threads" in joined
    assert "--pool=solo" not in joined


def test_linux_auto_pool_resolves_to_prefork(monkeypatch):
    cmd = _captured_cmd(monkeypatch, system="Linux", env={"CELERY_WORKER_POOL": "auto"})
    joined = " ".join(cmd)
    assert "--pool=prefork" in joined


def test_concurrency_is_configurable(monkeypatch):
    cmd = _captured_cmd(
        monkeypatch,
        system="Windows",
        env={"CELERY_WORKER_POOL": "threads", "CELERY_WORKER_CONCURRENCY": "4"},
    )
    joined = " ".join(cmd)
    assert "--concurrency=4" in joined
    assert "--pool=threads" in joined


def test_explicit_solo_remains_available(monkeypatch):
    cmd = _captured_cmd(monkeypatch, system="Windows", env={"CELERY_WORKER_POOL": "solo"})
    joined = " ".join(cmd)
    assert "--pool=solo" in joined


def test_time_limits_passed_through(monkeypatch):
    cmd = _captured_cmd(
        monkeypatch,
        system="Linux",
        env={
            "CELERY_WORKER_POOL": "prefork",
            "CELERY_WORKER_TIME_LIMIT": "120",
            "CELERY_WORKER_SOFT_TIME_LIMIT": "60",
            "CELERY_WORKER_MAX_TASKS_PER_CHILD": "5",
        },
    )
    joined = " ".join(cmd)
    assert "--time-limit=120" in joined
    assert "--soft-time-limit=60" in joined
    assert "--max-tasks-per-child=5" in joined


def test_document_processing_service_uses_factory_provider():
    """Threaded workers MUST get a fresh DocumentProcessingService per task.

    The service caches per-run mutable state (``_extracted_images``,
    ``_mineru_output_path``). If the container ever switches the provider
    to ``providers.Singleton`` the service would leak state across
    concurrent tasks. Keep this pinned to ``providers.Factory``.
    """
    from dependency_injector import providers

    from app.core.container import Container

    provider = Container.document_processing_service
    assert isinstance(provider, providers.Factory), (
        "document_processing_service must remain a providers.Factory so each "
        "Celery task gets a fresh instance — singleton/shared state is unsafe "
        "under the threaded worker pool."
    )
