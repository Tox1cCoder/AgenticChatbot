"""Conversation deletion must reach the per-turn threads it owns.

Before routing-v2 a conversation had exactly one checkpoint thread named after
it, so deleting by conversation ID was complete. Now each turn has its own
thread, and deleting by conversation ID alone would leave every turn's
checkpoint behind — an unbounded leak of state for a conversation the user
asked to remove.

Active interrupts are the counterweight: a turn paused on a human decision must
survive ordinary expiry-driven cleanup.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

from app.services.checkpoint_retention_service import CheckpointRetentionService


class FakeCheckpointManager:
    def __init__(self, *, failing: set[str] | None = None):
        self.deleted: list[str] = []
        self._failing = failing or set()

    async def delete_thread(self, thread_id: str) -> bool:
        if thread_id in self._failing:
            raise RuntimeError(f"transient failure deleting {thread_id}")
        self.deleted.append(thread_id)
        return True


class FakeHitlRepository:
    def __init__(self, expired=None):
        self._expired = expired or []
        self.marked: list[object] = []

    def get_expired_pending(self, now):
        return self._expired

    def mark_expired(self, record_id):
        self.marked.append(record_id)


class FakeConversationRepository:
    def __init__(self, soft_deleted=None):
        self._soft_deleted = soft_deleted or []

    def get_soft_deleted(self):
        return self._soft_deleted


class FakeMessageRepository:
    def __init__(self, turn_ids_by_conversation=None):
        self._turn_ids = turn_ids_by_conversation or {}
        self.queried: list[str] = []

    def get_user_message_ids(self, conversation_id):
        self.queried.append(str(conversation_id))
        return self._turn_ids.get(str(conversation_id), [])


def _service(
    *,
    checkpoint_manager=None,
    hitl=None,
    conversations=None,
    messages=None,
) -> CheckpointRetentionService:
    return CheckpointRetentionService(
        checkpoint_manager or FakeCheckpointManager(),
        hitl or FakeHitlRepository(),
        conversations or FakeConversationRepository(),
        message_repository=messages,
    )


def _interrupt(thread_id: str):
    return SimpleNamespace(id=uuid4(), thread_id=thread_id)


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


# ----------------------------------------------------------------------
# conversation deletion
# ----------------------------------------------------------------------


async def test_deleting_a_conversation_removes_every_turn_thread_it_owns():
    conversation_id = "conversation-1"
    manager = FakeCheckpointManager()
    service = _service(
        checkpoint_manager=manager,
        conversations=FakeConversationRepository([SimpleNamespace(id=conversation_id)]),
        messages=FakeMessageRepository({conversation_id: ["message-1", "message-2"]}),
    )

    await service.cleanup_expired_and_deleted_threads(now=NOW)

    assert "routing-v2:conversation-1:message-1" in manager.deleted
    assert "routing-v2:conversation-1:message-2" in manager.deleted


async def test_deleting_a_conversation_also_removes_its_pre_v2_thread():
    """Old conversations still have one thread named after the conversation."""
    conversation_id = "conversation-1"
    manager = FakeCheckpointManager()
    service = _service(
        checkpoint_manager=manager,
        conversations=FakeConversationRepository([SimpleNamespace(id=conversation_id)]),
        messages=FakeMessageRepository({conversation_id: ["message-1"]}),
    )

    await service.cleanup_expired_and_deleted_threads(now=NOW)

    assert conversation_id in manager.deleted


async def test_cleanup_never_deletes_by_an_unbounded_prefix():
    conversation_id = "conversation-1"
    manager = FakeCheckpointManager()
    service = _service(
        checkpoint_manager=manager,
        conversations=FakeConversationRepository([SimpleNamespace(id=conversation_id)]),
        messages=FakeMessageRepository({conversation_id: ["message-1"]}),
    )

    await service.cleanup_expired_and_deleted_threads(now=NOW)

    for thread_id in manager.deleted:
        assert "%" not in thread_id
        assert "*" not in thread_id


async def test_cleanup_works_without_a_message_repository():
    """Deployments without one still get the pre-v2 behavior, not a crash."""
    conversation_id = "conversation-1"
    manager = FakeCheckpointManager()
    service = _service(
        checkpoint_manager=manager,
        conversations=FakeConversationRepository([SimpleNamespace(id=conversation_id)]),
        messages=None,
    )

    counts = await service.cleanup_expired_and_deleted_threads(now=NOW)

    assert manager.deleted == [conversation_id]
    assert counts["conversation_checkpoint_threads_deleted"] == 1


async def test_a_conversation_with_no_turns_deletes_only_its_own_thread():
    manager = FakeCheckpointManager()
    service = _service(
        checkpoint_manager=manager,
        conversations=FakeConversationRepository([SimpleNamespace(id="conversation-1")]),
        messages=FakeMessageRepository({"conversation-1": []}),
    )

    await service.cleanup_expired_and_deleted_threads(now=NOW)
    assert manager.deleted == ["conversation-1"]


# ----------------------------------------------------------------------
# expiry
# ----------------------------------------------------------------------


async def test_expired_interrupts_delete_their_exact_stored_thread():
    manager = FakeCheckpointManager()
    service = _service(
        checkpoint_manager=manager,
        hitl=FakeHitlRepository([_interrupt("routing-v2:conversation-1:message-9")]),
    )

    counts = await service.cleanup_expired_and_deleted_threads(now=NOW)

    assert manager.deleted == ["routing-v2:conversation-1:message-9"]
    assert counts["hitl_interrupts_expired"] == 1


async def test_an_interrupt_thread_is_deleted_verbatim_not_reconstructed():
    """Resume uses the stored ID, so cleanup must act on that same value."""
    stored = "routing-v2:conversation-1:message-9"
    manager = FakeCheckpointManager()
    service = _service(checkpoint_manager=manager, hitl=FakeHitlRepository([_interrupt(stored)]))

    await service.cleanup_expired_and_deleted_threads(now=NOW)
    assert manager.deleted == [stored]


async def test_a_still_pending_interrupt_is_not_touched():
    """Only interrupts the repository reports as expired are reaped."""
    manager = FakeCheckpointManager()
    service = _service(checkpoint_manager=manager, hitl=FakeHitlRepository([]))

    counts = await service.cleanup_expired_and_deleted_threads(now=NOW - timedelta(days=1))

    assert manager.deleted == []
    assert counts["hitl_interrupts_expired"] == 0


# ----------------------------------------------------------------------
# resilience
# ----------------------------------------------------------------------


async def test_one_failed_delete_does_not_abort_the_rest():
    conversation_id = "conversation-1"
    manager = FakeCheckpointManager(failing={"routing-v2:conversation-1:message-1"})
    service = _service(
        checkpoint_manager=manager,
        conversations=FakeConversationRepository([SimpleNamespace(id=conversation_id)]),
        messages=FakeMessageRepository({conversation_id: ["message-1", "message-2"]}),
    )

    await service.cleanup_expired_and_deleted_threads(now=NOW)

    assert "routing-v2:conversation-1:message-2" in manager.deleted


async def test_cleanup_is_idempotent_across_retries():
    conversation_id = "conversation-1"
    manager = FakeCheckpointManager()
    service = _service(
        checkpoint_manager=manager,
        conversations=FakeConversationRepository([SimpleNamespace(id=conversation_id)]),
        messages=FakeMessageRepository({conversation_id: ["message-1"]}),
    )

    first = await service.cleanup_expired_and_deleted_threads(now=NOW)
    second = await service.cleanup_expired_and_deleted_threads(now=NOW)

    assert first == second


async def test_a_message_repository_failure_falls_back_to_the_conversation_thread():
    class ExplodingMessageRepository:
        def get_user_message_ids(self, conversation_id):
            raise RuntimeError("database unavailable")

    manager = FakeCheckpointManager()
    service = _service(
        checkpoint_manager=manager,
        conversations=FakeConversationRepository([SimpleNamespace(id="conversation-1")]),
        messages=ExplodingMessageRepository(),
    )

    await service.cleanup_expired_and_deleted_threads(now=NOW)
    assert manager.deleted == ["conversation-1"]
