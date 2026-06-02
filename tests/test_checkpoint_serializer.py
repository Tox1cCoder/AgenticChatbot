from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.ai.checkpoint as checkpoint_module
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole

_CHECKPOINT_TYPES = (
    AgentType,
    MessageRole,
    AgentResponse,
    AgentMessage,
)


def _expected_json_allowlist() -> list[tuple[str, ...]]:
    return [(*symbol.__module__.split("."), symbol.__name__) for symbol in _CHECKPOINT_TYPES]


def _expected_msgpack_allowlist() -> list[tuple[str, str]]:
    return [(symbol.__module__, symbol.__name__) for symbol in _CHECKPOINT_TYPES]


def test_build_checkpoint_serializer_uses_json_allowlist_for_current_langgraph(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSerializer:
        def __init__(self, *, allowed_json_modules=None):
            captured["allowed_json_modules"] = allowed_json_modules

    monkeypatch.setattr(checkpoint_module, "JsonPlusSerializer", FakeSerializer)

    serializer = checkpoint_module._build_checkpoint_serializer()

    assert isinstance(serializer, FakeSerializer)
    assert list(captured["allowed_json_modules"]) == _expected_json_allowlist()


def test_build_checkpoint_serializer_adds_msgpack_allowlist_when_supported(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSerializer:
        def __init__(self, *, allowed_json_modules=None, allowed_msgpack_modules=None):
            captured["allowed_json_modules"] = allowed_json_modules
            captured["allowed_msgpack_modules"] = allowed_msgpack_modules

    monkeypatch.setattr(checkpoint_module, "JsonPlusSerializer", FakeSerializer)

    serializer = checkpoint_module._build_checkpoint_serializer()

    assert isinstance(serializer, FakeSerializer)
    assert list(captured["allowed_json_modules"]) == _expected_json_allowlist()
    assert list(captured["allowed_msgpack_modules"]) == _expected_msgpack_allowlist()


def test_build_checkpoint_serializer_uses_msgpack_allowlist_method_when_constructor_lacks_kwarg(
    monkeypatch,
):
    captured: dict[str, object] = {}

    class FakeSerializer:
        def __init__(self, *, allowed_json_modules=None):
            captured["allowed_json_modules"] = allowed_json_modules

        def with_msgpack_allowlist(self, allowlist):
            captured["allowed_msgpack_modules"] = allowlist
            return self

    monkeypatch.setattr(checkpoint_module, "JsonPlusSerializer", FakeSerializer)

    serializer = checkpoint_module._build_checkpoint_serializer()

    assert isinstance(serializer, FakeSerializer)
    assert list(captured["allowed_json_modules"]) == _expected_json_allowlist()
    assert list(captured["allowed_msgpack_modules"]) == _expected_msgpack_allowlist()


@pytest.mark.asyncio
async def test_checkpoint_manager_delete_thread_delegates_to_async_saver():
    manager = checkpoint_module.CheckpointManager(
        db_url="postgresql://user:pass@localhost/db",
        settings=SimpleNamespace(checkpoint_schema="public"),
    )
    manager._initialized = True
    calls: list[str] = []

    class FakeCheckpointer:
        async def adelete_thread(self, thread_id: str) -> None:
            calls.append(thread_id)

    manager.checkpointer = FakeCheckpointer()

    deleted = await manager.delete_thread("thread-1")

    assert deleted is True
    assert calls == ["thread-1"]


@pytest.mark.asyncio
async def test_checkpoint_manager_delete_thread_falls_back_to_pool_sql():
    manager = checkpoint_module.CheckpointManager(
        db_url="postgresql://user:pass@localhost/db",
        settings=SimpleNamespace(checkpoint_schema="public"),
    )
    manager._initialized = True
    manager.checkpointer = SimpleNamespace()
    executed: list[tuple[str, tuple[str]]] = []

    class FakeConnection:
        async def execute(self, statement: str, params: tuple[str]) -> None:
            executed.append((statement, params))

    class FakePoolConnection:
        async def __aenter__(self):
            return FakeConnection()

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class FakePool:
        def connection(self):
            return FakePoolConnection()

    manager._pool = FakePool()

    deleted = await manager.delete_thread("thread-1")

    assert deleted is True
    assert executed == [
        ("DELETE FROM checkpoints WHERE thread_id = %s", ("thread-1",)),
        ("DELETE FROM checkpoint_blobs WHERE thread_id = %s", ("thread-1",)),
        ("DELETE FROM checkpoint_writes WHERE thread_id = %s", ("thread-1",)),
    ]
