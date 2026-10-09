"""Bounded first-turn memory preparation for a forked conversation.

A fork copies a transcript prefix but never the source's memory summary, which
can cover rows past the cutoff. The child's own summary job is committed with
the fork and runs later on a worker. Until it has, ``ConversationHistoryProvider``
caps the copied prefix at the history budget before routing runs, and
``RequestBudgetService.preflight`` only compacts a request that is already near
the provider's hard limit -- so an early fact in a long prefix would be missing
from the first child turn.

``ForkMemoryPreparer`` closes that gap on the next-generation path, before the
workflow request is built, and only for the child's *first* turn: once any row
follows the copied prefix (besides the turn's own user message), later turns rely
on background compaction and never wait or claim here. That bound matters
because every persisted assistant reply re-queues the job with its attempts
reset, so a failing provider would otherwise cost a wait on every turn. When the
child has no valid memory and its copied prefix exceeds the tightest history
budget, it runs the child's existing durable
job through the ordinary worker (``run_conversation_compaction_now``): the same
lease, CAS write and retry accounting a Celery worker uses, never a second
summary path. The work runs in a thread because the worker's repository is
synchronous, and it is bounded by
``conversation_fork_memory_prepare_timeout_seconds``.

Every way preparation cannot finish -- compaction disabled, a lease held by a
worker, a dead job, a retry still in backoff, a timeout, a summary failure --
proceeds with the ordinary bounded history and logs one structured warning.
Preparation never fails a turn. Even when it succeeds the child gets the
application's normal bounded memory (a summary plus recent rows), not its full
transcript.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from app.ai.history import db_message_to_agent_message
from app.ai.token_instrumentation import HistoryBudgetConfig, trim_history_to_budget
from app.models.conversation_summary_job import SummaryJobStatus

logger = logging.getLogger(__name__)

#: Every agent key that reads prompt history. Which specialist answers is not
#: known until routing, and the router itself reads history first, so the
#: tightest budget among them decides whether copied context would be cut.
HISTORY_AGENT_KEYS = ("router", "chat", "rag", "search", "planning")

PreparationStatus = Literal["skipped", "prepared", "fallback"]


@dataclass(frozen=True)
class ForkMemoryPreparation:
    """What preparation did. ``fallback`` means bounded history without a summary."""

    status: PreparationStatus
    reason: str
    detail: str | None = None


def applicable_history_budget(settings: Any) -> tuple[int, int]:
    """The smallest positive ``(max_messages, max_tokens)`` any agent applies; 0 = uncapped."""
    messages: list[int] = []
    tokens: list[int] = []
    for agent_key in HISTORY_AGENT_KEYS:
        config = HistoryBudgetConfig.for_agent(agent_key, settings)
        if config.max_messages > 0:
            messages.append(int(config.max_messages))
        if config.max_tokens > 0:
            tokens.append(int(config.max_tokens))
    return min(messages, default=0), min(tokens, default=0)


def _in_future(moment: datetime | None) -> bool:
    return moment is not None and moment > datetime.now(UTC)


class ForkMemoryPreparer:
    """Build a forked child's first summary before its first prompt, or say why not."""

    def __init__(
        self,
        *,
        repository: Any,
        runner: Callable[..., Awaitable[Any]],
        settings: Any,
    ) -> None:
        self.repository = repository
        self.runner = runner
        self.settings = settings

    async def aprepare(
        self,
        *,
        conversation_id: UUID,
        owner_id: UUID,
        forked_through_sequence: int | None,
        current_message_id: UUID | None = None,
    ) -> ForkMemoryPreparation:
        """Prepare the child's memory within the timeout; never raises for a failed summary."""
        if forked_through_sequence is None:
            return ForkMemoryPreparation("skipped", "not_a_fork")
        timeout = float(self.settings.conversation_fork_memory_prepare_timeout_seconds)
        try:
            # On timeout the thread is not interrupted: it holds the job's lease
            # and finishes inside the worker's own provider timeout, so the
            # summary is ready for a later turn instead of leaving a stale lease.
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._prepare_blocking,
                    conversation_id,
                    owner_id,
                    forked_through_sequence,
                    current_message_id,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            result = ForkMemoryPreparation("fallback", "timeout", f"{timeout:g}s")
        except Exception as exc:
            result = ForkMemoryPreparation("fallback", "preparation_error", type(exc).__name__)
            self._warn(conversation_id, result, exc_info=exc)
            return result
        if result.status == "fallback":
            self._warn(conversation_id, result)
        return result

    def _prepare_blocking(
        self,
        conversation_id: UUID,
        owner_id: UUID,
        through_sequence: int,
        current_message_id: UUID | None,
    ) -> ForkMemoryPreparation:
        if self.repository.has_messages_after(
            conversation_id, through_sequence, excluding_message_id=current_message_id
        ):
            # Not the first turn: background compaction owns this child now,
            # including after a later edit invalidates its memory.
            logger.debug("Fork memory preparation skipped after the first turn %s", conversation_id)
            return ForkMemoryPreparation("skipped", "not_first_turn")
        if self.repository.get_owned_valid_memory(conversation_id, owner_id) is not None:
            return ForkMemoryPreparation("skipped", "memory_present")
        if not self._prefix_exceeds_budget(conversation_id, through_sequence):
            return ForkMemoryPreparation("skipped", "within_history_budget")
        if not self.settings.conversation_summary_enabled:
            return ForkMemoryPreparation("fallback", "compaction_disabled")
        refusal = self._job_refusal(conversation_id)
        if refusal is not None:
            return refusal
        # The worker is async but its repository is synchronous; this thread
        # gets its own loop exactly like the Celery task does.
        outcome = asyncio.run(self.runner(conversation_id, force=True))
        return self._interpret(outcome, conversation_id, owner_id)

    def _prefix_exceeds_budget(self, conversation_id: UUID, through_sequence: int) -> bool:
        """Whether history assembly would drop part of the copied prefix."""
        max_messages, max_tokens = applicable_history_budget(self.settings)
        if max_messages <= 0 and max_tokens <= 0:
            return False
        limit = max_messages + 1 if max_messages > 0 else None
        rows = self.repository.load_fork_prefix_tail(conversation_id, through_sequence, limit=limit)
        if max_messages > 0 and len(rows) > max_messages:
            return True
        messages = [
            message
            for message in (db_message_to_agent_message(row) for row in rows)
            if message is not None
        ]
        kept = trim_history_to_budget(messages, max_messages=max_messages, max_tokens=max_tokens)
        return len(kept) < len(messages)

    def _job_refusal(self, conversation_id: UUID) -> ForkMemoryPreparation | None:
        """Refuse what the durable lifecycle says must not run now; reopen an idle job."""
        state = self.repository.get_job_state(conversation_id)
        if state is None:
            return ForkMemoryPreparation("fallback", "summary_job_missing")
        status = state.status
        if status == SummaryJobStatus.PROCESSING.value:
            return ForkMemoryPreparation("fallback", "lease_busy")
        if status == SummaryJobStatus.DEAD.value:
            return ForkMemoryPreparation("fallback", "summary_job_dead")
        if status == SummaryJobStatus.RETRY.value and _in_future(state.available_at):
            # Claiming by id ignores ``available_at``; honour the backoff here.
            return ForkMemoryPreparation("fallback", "summary_job_backoff")
        if status == SummaryJobStatus.IDLE.value:
            # A background run found the prefix below its own trigger and went
            # idle without memory; ask again through the ordinary request path.
            self.repository.request_backfill(conversation_id)
        return None

    def _interpret(
        self, outcome: Any, conversation_id: UUID, owner_id: UUID
    ) -> ForkMemoryPreparation:
        status = str(getattr(outcome, "status", "") or "")
        code = str(getattr(outcome, "code", "") or "") or None
        if status == "noop" and code == "not_claimed":
            return ForkMemoryPreparation("fallback", "lease_busy")
        if status == "disabled":
            return ForkMemoryPreparation("fallback", "compaction_disabled")
        memory = self.repository.get_owned_valid_memory(conversation_id, owner_id)
        if status == "completed" and memory is not None:
            return ForkMemoryPreparation("prepared", "summary_ready")
        return ForkMemoryPreparation("fallback", "summary_failed", code)

    @staticmethod
    def _warn(
        conversation_id: UUID,
        result: ForkMemoryPreparation,
        *,
        exc_info: BaseException | None = None,
    ) -> None:
        fields = {
            "conversation_id": str(conversation_id),
            "reason": result.reason,
            "detail": result.detail,
        }
        logger.warning(
            "Fork memory preparation fell back to bounded history "
            "conversation_id=%s reason=%s detail=%s: copied messages beyond the "
            "history budget are not in this turn's context until the child's summary exists",
            fields["conversation_id"],
            fields["reason"],
            fields["detail"],
            extra={"fork_memory": fields},
            exc_info=exc_info,
        )


__all__ = [
    "HISTORY_AGENT_KEYS",
    "ForkMemoryPreparation",
    "ForkMemoryPreparer",
    "applicable_history_budget",
]
