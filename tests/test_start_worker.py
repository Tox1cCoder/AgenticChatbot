import os


class FakePopen:
    """Minimal subprocess.Popen stub that records the command and supports wait()."""

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.pid = 99999

    def wait(self):
        return 0

    def terminate(self):
        pass


def test_start_worker_does_not_seed_default_celery_env_vars(monkeypatch):
    """Worker startup must not mutate broker/result env vars."""
    spawned = []

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

    def fake_popen(cmd, **kwargs):
        spawned.append(cmd)
        return FakePopen(cmd)

    monkeypatch.setattr("app.workers.start_worker.subprocess.Popen", fake_popen)

    from app.workers.start_worker import start_worker

    start_worker()

    assert "CELERY_BROKER_URL" not in os.environ
    assert "CELERY_RESULT_BACKEND" not in os.environ

    assert len(spawned) == 2, "Expected two worker processes (parse + index)"

    for cmd in spawned:
        joined = " ".join(cmd)
        # Windows default should not be solo any more — the worker must run
        # tasks concurrently for batch uploads to actually overlap.
        assert "--pool=solo" not in joined
        assert "--pool=threads" in joined
