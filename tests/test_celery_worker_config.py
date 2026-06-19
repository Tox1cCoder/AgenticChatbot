"""Celery worker startup is config-driven and uses parallel-capable pools.

Replaces the prior assertion that Windows always appends ``--pool=solo``.
"""

from __future__ import annotations

from .conftest import FakePopen


def _captured_cmds(
    monkeypatch, *, system: str, env: dict[str, str] | None = None
) -> list[list[str]]:
    """Return [parse_cmd, index_cmd] spawned by start_worker()."""
    spawned: list[list[str]] = []

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

    def fake_popen(cmd, **kwargs):
        spawned.append(cmd)
        return FakePopen(cmd)

    monkeypatch.setattr("app.workers.start_worker.subprocess.Popen", fake_popen)

    from app.workers.start_worker import start_worker

    start_worker()
    return spawned


def test_windows_auto_pool_resolves_to_threads(monkeypatch):
    cmds = _captured_cmds(monkeypatch, system="Windows", env={"CELERY_WORKER_POOL": "auto"})
    assert len(cmds) == 2, "Expected two worker processes (parse + index)"
    for cmd in cmds:
        joined = " ".join(cmd)
        assert "--pool=threads" in joined
        assert "--pool=solo" not in joined


def test_linux_auto_pool_resolves_to_prefork(monkeypatch):
    cmds = _captured_cmds(monkeypatch, system="Linux", env={"CELERY_WORKER_POOL": "auto"})
    assert len(cmds) == 2
    for cmd in cmds:
        joined = " ".join(cmd)
        assert "--pool=prefork" in joined


def test_concurrency_is_configurable(monkeypatch):
    cmds = _captured_cmds(
        monkeypatch,
        system="Windows",
        env={
            "CELERY_WORKER_POOL": "threads",
            "CELERY_PARSE_CONCURRENCY": "4",
            "CELERY_INDEX_CONCURRENCY": "2",
        },
    )
    assert len(cmds) == 2
    # parse worker
    parse_joined = " ".join(cmds[0])
    assert "--concurrency=4" in parse_joined
    assert "--pool=threads" in parse_joined
    assert "--queues=parse" in parse_joined
    # index worker
    index_joined = " ".join(cmds[1])
    assert "--concurrency=2" in index_joined
    assert "--pool=threads" in index_joined
    assert "--queues=index" in index_joined


def test_explicit_solo_remains_available(monkeypatch):
    cmds = _captured_cmds(monkeypatch, system="Windows", env={"CELERY_WORKER_POOL": "solo"})
    assert len(cmds) == 2
    for cmd in cmds:
        joined = " ".join(cmd)
        assert "--pool=solo" in joined


def test_time_limits_passed_through(monkeypatch):
    cmds = _captured_cmds(
        monkeypatch,
        system="Linux",
        env={
            "CELERY_WORKER_POOL": "prefork",
            "CELERY_WORKER_TIME_LIMIT": "120",
            "CELERY_WORKER_SOFT_TIME_LIMIT": "60",
            "CELERY_WORKER_MAX_TASKS_PER_CHILD": "5",
            "CELERY_INDEX_TIME_LIMIT": "120",
        },
    )
    assert len(cmds) == 2
    for cmd in cmds:
        joined = " ".join(cmd)
        assert "--max-tasks-per-child=5" in joined
    # parse worker time limit should be mineru_timeout + 60
    # (mineru_timeout defaults to 300 in settings, so expected value is 360)
    parse_joined = " ".join(cmds[0])
    assert "--time-limit=360" in parse_joined
    # index worker inherits time limit from celery_index_time_limit
    index_joined = " ".join(cmds[1])
    assert "--time-limit=120" in index_joined


def test_hostname_disambiguation(monkeypatch):
    """Each worker must have a distinct hostname for Celery monitoring."""
    cmds = _captured_cmds(monkeypatch, system="Linux", env={"CELERY_WORKER_POOL": "prefork"})
    assert len(cmds) == 2
    parse_joined = " ".join(cmds[0])
    index_joined = " ".join(cmds[1])
    assert "--hostname=worker-parse@%h" in parse_joined
    assert "--hostname=worker-index@%h" in index_joined


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


def test_celery_redis_broker_resilience_is_explicit():
    """Broker resets should reconnect cleanly and avoid duplicate late-ack work."""
    from app.workers.celery_app import celery_app

    assert celery_app.conf.broker_connection_retry is True
    assert celery_app.conf.broker_connection_retry_on_startup is True
    assert celery_app.conf.worker_cancel_long_running_tasks_on_connection_loss is True

    broker_options = dict(celery_app.conf.broker_transport_options or {})
    assert broker_options["health_check_interval"] == 30
    assert broker_options["socket_keepalive"] is True
    assert broker_options["retry_on_timeout"] is True
    assert broker_options["visibility_timeout"] == 3600

    result_options = dict(celery_app.conf.result_backend_transport_options or {})
    assert result_options["visibility_timeout"] == 3600
