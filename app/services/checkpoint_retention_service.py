"""Retention policy for LangGraph checkpoint threads.

Consolidates the DB-driven cleanup logic that previously lived inline in the
`cleanup_abandoned_interrupts` Celery task so it can run standalone and be
unit tested without a live database or broker. This service never touches
the checkpoint tables' schema (no DROP/TRUNCATE) — it only deletes rows for
specific thread IDs via `CheckpointManager.delete_thread`.
"""

import logging
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Protocol

from app.ai.workflow.state import build_checkpoint_thread_id, parse_checkpoint_thread_id

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointRetentionService",
    "is_v2_checkpoint_thread_id",
    "owned_v2_thread_ids",
]


def is_v2_checkpoint_thread_id(thread_id: Any) -> bool:
    """Whether ``thread_id`` is a well-formed routing-v2 per-turn thread."""
    try:
        parse_checkpoint_thread_id(str(thread_id))
    except (ValueError, TypeError):
        return False
    return True


def owned_v2_thread_ids(conversation_id: Any, turn_ids: Iterable[Any]) -> list[str]:
    """Exact v2 thread IDs a conversation owns, built from persisted turn IDs.

    Enumerating exact IDs is the whole point: a prefix delete would reach
    threads this conversation does not own, including turns still paused on a
    human decision. A turn ID that cannot produce a valid thread is skipped
    rather than guessed at.
    """
    conversation = str(conversation_id or "").strip()
    if not conversation:
        return []

    thread_ids: list[str] = []
    seen: set[str] = set()
    for turn_id in turn_ids or []:
        try:
            thread_id = build_checkpoint_thread_id(conversation, str(turn_id or ""))
        except (ValueError, TypeError):
            continue
        if thread_id not in seen:
            seen.add(thread_id)
            thread_ids.append(thread_id)
    return thread_ids


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
        message_repository=None,
    ):
        self.checkpoint_manager = checkpoint_manager
        self.hitl_interrupt_repository = hitl_interrupt_repository
        self.conversation_repository = conversation_repository
        # Routing-v2 gives each turn its own checkpoint thread, so deleting a
        # conversation needs its turn IDs. Optional: without it, cleanup still
        # removes the pre-v2 conversation-named thread.
        self.message_repository = message_repository

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

        conversation_thread_ids: list[str] = []
        for conversation in soft_deleted_conversations:
            conversation_id = str(conversation.id)
            # The pre-v2 thread was named after the conversation; routing-v2
            # threads are per turn. A deleted conversation owns both.
            conversation_thread_ids.append(conversation_id)
            conversation_thread_ids.extend(
                owned_v2_thread_ids(conversation_id, self._turn_ids_for(conversation_id))
            )

        counts["conversation_checkpoint_threads_deleted"] = await self._delete_checkpoint_threads(
            list(dict.fromkeys(conversation_thread_ids))
        )

        return counts

    def _turn_ids_for(self, conversation_id: str) -> list[str]:
        """Persisted user-message IDs, which are this conversation's turn IDs.

        A lookup failure degrades to the pre-v2 thread rather than aborting
        cleanup: deleting less than intended is recoverable on the next run,
        while raising here would strand every later conversation in the batch.
        """
        if self.message_repository is None:
            return []
        lookup = getattr(self.message_repository, "get_user_message_ids", None)
        if not callable(lookup):
            return []
        try:
            return [str(turn_id) for turn_id in lookup(conversation_id) or []]
        except Exception:
            logger.warning(
                "Could not enumerate turn ids for conversation %s; "
                "falling back to the pre-v2 thread only",
                conversation_id,
                exc_info=True,
            )
            return []

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
