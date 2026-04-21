import os

from app.workers.start_worker import start_worker


def test_start_worker_does_not_seed_default_celery_env_vars(monkeypatch):
    captured = {}

    monkeypatch.delenv("CELERY_BROKER_URL", raising=False)
    monkeypatch.delenv("CELERY_RESULT_BACKEND", raising=False)
    monkeypatch.setattr("app.workers.start_worker.os.chdir", lambda _path: None)
    monkeypatch.setattr("app.workers.start_worker.platform.system", lambda: "Windows")
    monkeypatch.setattr("app.workers.start_worker.sys.executable", "python")

    def fake_run(cmd):
        captured["cmd"] = cmd

    monkeypatch.setattr("app.workers.start_worker.subprocess.run", fake_run)

    start_worker()

    assert "CELERY_BROKER_URL" not in os.environ
    assert "CELERY_RESULT_BACKEND" not in os.environ
    assert captured["cmd"][-1] == "--pool=solo"
