"""Retention policy for LangGraph checkpoint threads.

Consolidates the DB-driven cleanup logic that previously lived inline in the
`cleanup_abandoned_interrupts` Celery task so it can run standalone and be
unit tested without a live database or broker. This service never touches
the checkpoint tables' schema (no DROP/TRUNCATE) — it only deletes rows for
specific thread IDs via `CheckpointManager.delete_thread`.
"""

import logging
from datetime import datetime
from typing import Protocol

logger = logging.getLogger(__name__)


class _SupportsDeleteThread(Protocol):
    async def delete_thread(self, thread_id: str) -> bool: ...


class CheckpointRetentionService:
    """Expires stale HITL interrupts and reaps their checkpoint threads.

    Cleanup runs in a fixed order so the authoritative DB state (HITL
    interrupt status) is always updated before any checkpoint rows are
    removed:

    1. Expire PENDING HITL interrupts whose `expires_at` has passed.
    2. Delete checkpoint threads for those newly-expired interrupts.
    3. Delete checkpoint threads for soft-deleted conversations (a safety
       net for conversations whose best-effort delete-time cleanup never
       ran or failed).
    """

    def __init__(
        self,
        checkpoint_manager: _SupportsDeleteThread,
        hitl_interrupt_repository,
        conversation_repository,
    ):
        self.checkpoint_manager = checkpoint_manager
        self.hitl_interrupt_repository = hitl_interrupt_repository
        self.conversation_repository = conversation_repository

    async def cleanup_expired_and_deleted_threads(self, *, now: datetime) -> dict[str, int]:
        """Expire stale HITL interrupts and delete their checkpoint threads.

        Returns a dict of counts:
        - pending_interrupts_inspected: expired-but-still-PENDING HITL rows found
        - hitl_interrupts_expired: how many of those were successfully marked EXPIRED
        - hitl_checkpoint_threads_deleted: checkpoint threads removed for expired HITL rows
        - soft_deleted_conversations_inspected: soft-deleted conversations found
        - conversation_checkpoint_threads_deleted: checkpoint threads removed for them
        """
        counts = {
            "pending_interrupts_inspected": 0,
            "hitl_interrupts_expired": 0,
            "hitl_checkpoint_threads_deleted": 0,
            "soft_deleted_conversations_inspected": 0,
            "conversation_checkpoint_threads_deleted": 0,
        }

        expired_records = self.hitl_interrupt_repository.get_expired_pending(now)
        counts["pending_interrupts_inspected"] = len(expired_records)

        expired_thread_ids: list[str] = []
        for record in expired_records:
            try:
                self.hitl_interrupt_repository.mark_expired(record.id)
                counts["hitl_interrupts_expired"] += 1
            except Exception:
                logger.warning("Failed to mark HITL interrupt %s expired", record.id, exc_info=True)
                continue

            thread_id = getattr(record, "thread_id", None)
            if thread_id:
                expired_thread_ids.append(thread_id)

        # De-dup while preserving order (a thread can back multiple interrupts).
        counts["hitl_checkpoint_threads_deleted"] = await self._delete_checkpoint_threads(
            list(dict.fromkeys(expired_thread_ids))
        )

        soft_deleted_conversations = self.conversation_repository.get_soft_deleted()
        counts["soft_deleted_conversations_inspected"] = len(soft_deleted_conversations)
        conversation_thread_ids = [
            str(conversation.id) for conversation in soft_deleted_conversations
        ]
        counts["conversation_checkpoint_threads_deleted"] = await self._delete_checkpoint_threads(
            conversation_thread_ids
        )

        return counts

    async def _delete_checkpoint_threads(self, thread_ids: list[str]) -> int:
        """Delete checkpoint rows for each thread, swallowing per-thread errors.

        One bad thread (e.g. a transient connection blip) must not abort
        cleanup for the rest — checkpoint tables themselves are never dropped.
        """
        deleted_count = 0
        for thread_id in thread_ids:
            try:
                if await self.checkpoint_manager.delete_thread(thread_id):
                    deleted_count += 1
            except Exception:
                logger.warning(
                    "Failed to delete checkpoint thread %s during retention cleanup",
                    thread_id,
                    exc_info=True,
                )
                continue
        return deleted_count
