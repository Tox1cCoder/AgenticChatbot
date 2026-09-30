"""Every task the document worker module registers can actually run.

A task with no route lands on Celery's default "celery" queue, which none of
the workers ``start_worker`` spawns consume, so anything enqueued there waits
forever. ``process_document_task`` was such a shim: registered, unrouted, and
called by nothing.
"""

from __future__ import annotations

from .test_celery_worker_config import _captured_cmds


def test_every_document_processor_task_lands_on_a_consumed_queue(monkeypatch):
    import app.workers.document_processor  # noqa: F401 - registers its tasks
    from app.workers.celery_app import celery_app

    consumed = {
        arg.removeprefix("--queues=")
        for cmd in _captured_cmds(monkeypatch, system="Linux")
        for arg in cmd
        if arg.startswith("--queues=")
    }
    routes = dict(celery_app.conf.task_routes)

    stranded = {}
    for task_name, task in celery_app.tasks.items():
        if not task_name.startswith("app.workers.document_processor."):
            continue
        queue = (
            routes.get(task_name, {}).get("queue")
            or getattr(task, "queue", None)
            or celery_app.conf.task_default_queue
        )
        if queue not in consumed:
            stranded[task_name] = queue

    assert not stranded, f"document tasks routed to unconsumed queues: {stranded}"
