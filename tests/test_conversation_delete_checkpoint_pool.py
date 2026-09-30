"""Deleting a conversation opens no checkpoint pool when checkpoints are disabled.

With ``enable_langgraph_checkpoints=False`` startup never builds the checkpoint
manager, so shutdown never closes it. A delete that reached the manager anyway
ran ``delete_thread()`` -> ``setup()`` and opened a psycopg pool nobody closed.
The retention sweep owns the threads of soft-deleted conversations.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from dependency_injector import providers

import app.ai.checkpoint as checkpoint_module
import app.services.conversation_service as conversation_module
from app.core.config import settings
from app.core.container import get_container


class _PoolSpy:
    opened: list[_PoolSpy] = []

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    async def open(self) -> None:
        _PoolSpy.opened.append(self)
        raise ConnectionError("no database in this test")

    async def close(self) -> None:
        return None


@pytest.fixture
def wired_service(monkeypatch):
    """The container's own conversation_service wiring, collaborators stubbed."""
    _PoolSpy.opened = []
    monkeypatch.setattr(checkpoint_module, "AsyncConnectionPool", _PoolSpy)
    manager = checkpoint_module.CheckpointManager(
        db_url="postgresql://user:pass@localhost/db",
        settings=SimpleNamespace(checkpoint_schema="public"),
    )
    container = get_container()
    overrides = {
        "checkpoint_manager": providers.Object(manager),
        "ai_service": providers.Object(SimpleNamespace(invalidate_history_cache=lambda _id: None)),
        "conversation_repository": providers.Object(SimpleNamespace(delete=lambda _id: True)),
        "conversation_validation_utils": providers.Object(
            SimpleNamespace(validate_conversation_access=lambda *_args: None)
        ),
        "user_validation_utils": providers.Object(SimpleNamespace()),
        "project_service": providers.Object(SimpleNamespace()),
    }

    def _build(*, checkpoints: bool):
        monkeypatch.setattr(settings, "enable_langgraph_checkpoints", checkpoints)
        return container.conversation_service()

    with container.override_providers(**overrides):
        yield _build


async def _delete_and_settle(service) -> None:
    assert service.delete_conversation(uuid4(), uuid4()) is True
    await asyncio.gather(*conversation_module._BACKGROUND_TASKS, return_exceptions=True)


async def test_delete_with_checkpoints_disabled_opens_no_pool(wired_service):
    await _delete_and_settle(wired_service(checkpoints=False))

    assert _PoolSpy.opened == []


async def test_delete_with_checkpoints_enabled_still_cleans_up_the_thread(wired_service):
    await _delete_and_settle(wired_service(checkpoints=True))

    assert len(_PoolSpy.opened) == 1
