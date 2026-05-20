import os


def test_start_worker_does_not_seed_default_celery_env_vars(monkeypatch):
    """Worker startup must not mutate broker/result env vars."""
    captured = {}

    monkeypatch.delenv("CELERY_BROKER_URL", raising=False)
    monkeypatch.delenv("CELERY_RESULT_BACKEND", raising=False)
    # Force settings to a known parallel-capable pool so the test does not
    # depend on ambient env vars.
    monkeypatch.setenv("CELERY_WORKER_POOL", "threads")
    monkeypatch.setattr("app.workers.start_worker.os.chdir", lambda _path: None)
    monkeypatch.setattr("app.workers.start_worker.platform.system", lambda: "Windows")
    monkeypatch.setattr("app.workers.start_worker.sys.executable", "python")

    from app.core import config as _cfg

    _cfg.get_settings.cache_clear()  # type: ignore[attr-defined]

    def fake_run(cmd):
        captured["cmd"] = cmd

    monkeypatch.setattr("app.workers.start_worker.subprocess.run", fake_run)

    from app.workers.start_worker import start_worker

    start_worker()

    assert "CELERY_BROKER_URL" not in os.environ
    assert "CELERY_RESULT_BACKEND" not in os.environ
    joined = " ".join(captured["cmd"])
    # Windows default should not be solo any more — the worker must run
    # tasks concurrently for batch uploads to actually overlap.
    assert "--pool=solo" not in joined
    assert "--pool=threads" in joined
