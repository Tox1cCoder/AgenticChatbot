from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.checkpoint_retention_service import CheckpointRetentionService


def _make_hitl_record(interrupt_id: str, thread_id: str | None) -> SimpleNamespace:
    return SimpleNamespace(id=interrupt_id, thread_id=thread_id)


def _make_conversation(conversation_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=conversation_id)


@pytest.mark.asyncio
async def test_expired_pending_interrupts_are_marked_expired():
    now = datetime(2026, 7, 8, tzinfo=timezone.utc)
    expired = [
        _make_hitl_record("interrupt-1", "thread-1"),
        _make_hitl_record("interrupt-2", "thread-2"),
    ]

    hitl_repo = Mock()
    hitl_repo.get_expired_pending = Mock(return_value=expired)
    hitl_repo.mark_expired = Mock()

    conversation_repo = Mock()
    conversation_repo.get_soft_deleted = Mock(return_value=[])

    checkpoint_manager = Mock()
    checkpoint_manager.delete_thread = AsyncMock(return_value=True)

    service = CheckpointRetentionService(checkpoint_manager, hitl_repo, conversation_repo)
    counts = await service.cleanup_expired_and_deleted_threads(now=now)

    assert hitl_repo.mark_expired.call_count == 2
    hitl_repo.mark_expired.assert_any_call("interrupt-1")
    hitl_repo.mark_expired.assert_any_call("interrupt-2")
    assert counts["pending_interrupts_inspected"] == 2
    assert counts["hitl_interrupts_expired"] == 2


@pytest.mark.asyncio
async def test_expired_interrupt_thread_ids_are_passed_to_delete_thread():
    now = datetime(2026, 7, 8, tzinfo=timezone.utc)
    expired = [
        _make_hitl_record("interrupt-1", "thread-1"),
        _make_hitl_record("interrupt-2", "thread-2"),
    ]

    hitl_repo = Mock()
    hitl_repo.get_expired_pending = Mock(return_value=expired)
    hitl_repo.mark_expired = Mock()

    conversation_repo = Mock()
    conversation_repo.get_soft_deleted = Mock(return_value=[])

    checkpoint_manager = Mock()
    checkpoint_manager.delete_thread = AsyncMock(return_value=True)

    service = CheckpointRetentionService(checkpoint_manager, hitl_repo, conversation_repo)
    counts = await service.cleanup_expired_and_deleted_threads(now=now)

    checkpoint_manager.delete_thread.assert_any_call("thread-1")
    checkpoint_manager.delete_thread.assert_any_call("thread-2")
    assert checkpoint_manager.delete_thread.call_count == 2
    assert counts["hitl_checkpoint_threads_deleted"] == 2


@pytest.mark.asyncio
async def test_unexpired_pending_interrupts_are_not_deleted():
    """get_expired_pending is the sole source of expired records; unexpired
    (not-yet-due) interrupts are never returned by the repository query, so
    the service must not touch or delete anything for them."""
    now = datetime(2026, 7, 8, tzinfo=timezone.utc)

    hitl_repo = Mock()
    hitl_repo.get_expired_pending = Mock(return_value=[])
    hitl_repo.mark_expired = Mock()

    conversation_repo = Mock()
    conversation_repo.get_soft_deleted = Mock(return_value=[])

    checkpoint_manager = Mock()
    checkpoint_manager.delete_thread = AsyncMock(return_value=True)

    service = CheckpointRetentionService(checkpoint_manager, hitl_repo, conversation_repo)
    counts = await service.cleanup_expired_and_deleted_threads(now=now)

    hitl_repo.mark_expired.assert_not_called()
    checkpoint_manager.delete_thread.assert_not_called()
    assert counts["pending_interrupts_inspected"] == 0
    assert counts["hitl_interrupts_expired"] == 0
    assert counts["hitl_checkpoint_threads_deleted"] == 0


@pytest.mark.asyncio
async def test_cleanup_swallows_per_thread_checkpoint_error_and_continues():
    now = datetime(2026, 7, 8, tzinfo=timezone.utc)
    expired = [
        _make_hitl_record("interrupt-1", "thread-bad"),
        _make_hitl_record("interrupt-2", "thread-good"),
    ]

    hitl_repo = Mock()
    hitl_repo.get_expired_pending = Mock(return_value=expired)
    hitl_repo.mark_expired = Mock()

    conversation_repo = Mock()
    conversation_repo.get_soft_deleted = Mock(return_value=[])

    checkpoint_manager = Mock()

    async def _delete_thread(thread_id: str) -> bool:
        if thread_id == "thread-bad":
            raise RuntimeError("boom")
        return True

    checkpoint_manager.delete_thread = AsyncMock(side_effect=_delete_thread)

    service = CheckpointRetentionService(checkpoint_manager, hitl_repo, conversation_repo)
    counts = await service.cleanup_expired_and_deleted_threads(now=now)

    # Both HITL records are still marked expired (that step doesn't depend on
    # checkpoint deletion succeeding).
    assert counts["hitl_interrupts_expired"] == 2
    # Both threads were attempted despite the first one raising.
    checkpoint_manager.delete_thread.assert_any_call("thread-bad")
    checkpoint_manager.delete_thread.assert_any_call("thread-good")
    assert checkpoint_manager.delete_thread.call_count == 2
    # Only the successful deletion is counted.
    assert counts["hitl_checkpoint_threads_deleted"] == 1


@pytest.mark.asyncio
async def test_soft_deleted_conversation_threads_are_deleted():
    now = datetime(2026, 7, 8, tzinfo=timezone.utc)

    hitl_repo = Mock()
    hitl_repo.get_expired_pending = Mock(return_value=[])
    hitl_repo.mark_expired = Mock()

    soft_deleted = [_make_conversation("conv-1"), _make_conversation("conv-2")]
    conversation_repo = Mock()
    conversation_repo.get_soft_deleted = Mock(return_value=soft_deleted)

    checkpoint_manager = Mock()
    checkpoint_manager.delete_thread = AsyncMock(return_value=True)

    service = CheckpointRetentionService(checkpoint_manager, hitl_repo, conversation_repo)
    counts = await service.cleanup_expired_and_deleted_threads(now=now)

    checkpoint_manager.delete_thread.assert_any_call("conv-1")
    checkpoint_manager.delete_thread.assert_any_call("conv-2")
    assert counts["soft_deleted_conversations_inspected"] == 2
    assert counts["conversation_checkpoint_threads_deleted"] == 2


@pytest.mark.asyncio
async def test_schema_qualified_fallback_sql_used_when_adelete_thread_unavailable():
    """Exercise CheckpointManager.delete_thread directly (as the service calls
    it) to verify the SQL fallback is schema-qualified and deletes in
    dependency-safe order: writes/blobs (children) before checkpoints (parent).
    """
    import app.ai.checkpoint as checkpoint_module

    manager = checkpoint_module.CheckpointManager(
        db_url="postgresql://user:pass@localhost/db",
        settings=SimpleNamespace(checkpoint_schema="custom_schema"),
    )
    manager._initialized = True
    manager.checkpointer = SimpleNamespace()  # no adelete_thread attribute
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

    deleted = await manager.delete_thread("thread-99")

    assert deleted is True
    assert executed == [
        ('DELETE FROM "custom_schema"."checkpoint_writes" WHERE thread_id = %s', ("thread-99",)),
        ('DELETE FROM "custom_schema"."checkpoint_blobs" WHERE thread_id = %s', ("thread-99",)),
        ('DELETE FROM "custom_schema"."checkpoints" WHERE thread_id = %s', ("thread-99",)),
    ]


@pytest.mark.asyncio
async def test_cleanup_never_issues_ddl_against_checkpoint_tables():
    """Guard against regressions that would let retention cleanup drop or
    truncate the LangGraph checkpoint tables themselves."""
    now = datetime(2026, 7, 8, tzinfo=timezone.utc)

    hitl_repo = Mock()
    hitl_repo.get_expired_pending = Mock(
        return_value=[_make_hitl_record("interrupt-1", "thread-1")]
    )
    hitl_repo.mark_expired = Mock()

    conversation_repo = Mock()
    conversation_repo.get_soft_deleted = Mock(return_value=[_make_conversation("conv-1")])

    # spec=["delete_thread"] means any attempt to call something like
    # drop_tables()/truncate_tables() on the manager raises AttributeError,
    # rather than silently succeeding as a fresh Mock attribute would.
    checkpoint_manager = Mock(spec=["delete_thread"])
    checkpoint_manager.delete_thread = AsyncMock(return_value=True)

    service = CheckpointRetentionService(checkpoint_manager, hitl_repo, conversation_repo)
    await service.cleanup_expired_and_deleted_threads(now=now)

    for call in checkpoint_manager.delete_thread.await_args_list:
        thread_id = call.args[0] if call.args else call.kwargs.get("thread_id")
        assert isinstance(thread_id, str)
    assert not hasattr(checkpoint_manager, "drop_tables")
    assert not hasattr(checkpoint_manager, "truncate_tables")
