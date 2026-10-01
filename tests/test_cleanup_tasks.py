from datetime import UTC, datetime, timedelta

from dependency_injector import providers

from app.core.config import settings
from app.workers import cleanup_tasks


def test_health_check_task_builds_the_agent_from_application_settings(monkeypatch):
    """The container's ``config`` provider is never loaded, so reading settings
    from it handed the agent an empty value and the task always reported failure."""
    import app.ai.agents.rag_agent as rag_agent_module
    from app.core.container import get_container

    captured = {}

    class FakeRAGAgent:
        def __init__(self, *, settings, qdrant_client, embedding_service, collection_name):
            captured["settings"] = settings
            captured["collection_name"] = collection_name

        async def initialize(self):
            return True

        def get_status(self):
            return {"collection_exists": True}

        async def cleanup(self):
            return None

    monkeypatch.setattr(rag_agent_module, "RAGAgent", FakeRAGAgent)
    container = get_container()
    container.qdrant_client.override(providers.Object(object()))
    container.rag_embedding_service.override(providers.Object(object()))
    try:
        result = cleanup_tasks.health_check_task.run()
    finally:
        container.qdrant_client.reset_override()
        container.rag_embedding_service.reset_override()

    assert result["success"] is True, result
    assert captured["settings"] is settings
    assert captured["collection_name"] == settings.qdrant_collection_name


def test_redis_interrupt_scan_closes_its_client(monkeypatch):
    now = datetime.now(UTC)
    stale = (now - timedelta(minutes=settings.hitl_approval_timeout_minutes + 5)).isoformat()

    class FakeRedis:
        closed = False
        deleted: list[bytes] = []

        def scan_iter(self, match):
            yield b"interrupt:conv-1:int-1"
            yield b"interrupt:conv-2:int-2"

        def get(self, key):
            if key == b"interrupt:conv-2:int-2":
                return b"not-a-timestamp"
            return stale.encode()

        def delete(self, key):
            self.deleted.append(key)

        def close(self):
            FakeRedis.closed = True

    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:6379/0")
    monkeypatch.setattr(cleanup_tasks.redis, "from_url", lambda _url: FakeRedis())

    expired_threads, expired, active = cleanup_tasks._scan_and_expire_redis_interrupts(now)

    assert expired_threads == ["conv-1"]
    assert (expired, active) == (1, 0)
    assert FakeRedis.closed is True
