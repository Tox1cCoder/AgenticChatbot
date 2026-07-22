"""Canonical prompt-history assembly for all agents.

The provider is the single place that builds prompt memory. Every agent
node receives both:

* a dedicated lower-priority memory message containing validated canonical
  JSON, when owned valid memory exists; and
* recent unsummarized DB rows after its sequence cursor.

This replaces multiple prompt-memory paths with a single deterministic
source of truth keyed on database message sequences.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from cachetools import TTLCache

from app.ai.conversation_memory import ConversationMemory
from app.ai.canvas_state import CanvasArtifactSnapshot, canvas_snapshot_from_message
from app.ai.schemas import AgentMessage, MessageRole
from app.ai.token_instrumentation import HistoryBudgetConfig, trim_history_to_budget
from app.models.enums import MessageRole as DBMessageRole
from app.models.message import Message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoryBudget:
    """Per-agent cap on prompt memory."""

    agent_key: str
    max_messages: int
    max_tokens: int


@dataclass(frozen=True)
class ConversationHistoryContext:
    """Composed memory context for a single agent invocation."""

    conversation_id: str
    user_id: str
    memory: AgentMessage | None
    memory_sequence: int | None
    messages: list[AgentMessage]
    budget: HistoryBudget
    summary_version: int = 0


def db_message_to_agent_message(message: Message) -> AgentMessage | None:
    """Translate a DB row into an ``AgentMessage`` suitable for prompt history.

    Returns ``None`` when the row should not appear in prompt history at all
    (soft-deleted, empty paused/interrupt placeholder). The strategy layer
    already filters most of these but this is a defensive last pass for any
    callers that bypass it.
    """
    if message.deleted_at is not None:
        return None

    raw_sequence = getattr(message, "sequence", None)
    metadata = {
        "message_id": str(message.id),
        "sequence": int(raw_sequence) if raw_sequence is not None else None,
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }

    raw_metadata = message.message_metadata if isinstance(message.message_metadata, dict) else {}
    raw_attachments = raw_metadata.get("attachments")
    attachments = raw_attachments if isinstance(raw_attachments, list) else None

    sender = message.sender
    if sender == DBMessageRole.user.value:
        return AgentMessage(
            role=MessageRole.USER,
            content=message.content or "",
            metadata=metadata,
            attachments=attachments,
        )
    if sender == DBMessageRole.assistant.value:
        if not (message.content or "").strip():
            return None
        return AgentMessage(
            role=MessageRole.ASSISTANT,
            content=message.content,
            metadata=metadata,
        )
    return None


@dataclass
class _CacheKey:
    conversation_id: str
    user_id: str
    current_message_id: str | None
    agent_key: str
    memory_sequence: int | None
    summary_version: int

    def as_tuple(self) -> tuple:
        return (
            self.conversation_id,
            self.user_id,
            self.current_message_id,
            self.agent_key,
            self.memory_sequence,
            self.summary_version,
        )


class ConversationHistoryProvider:
    """Single history-context builder shared by every agent node."""

    def __init__(
        self,
        message_repository: Any,
        summary_repository: Any,
        settings: Any,
    ):
        self.message_repository = message_repository
        self.summary_repository = summary_repository
        self.settings = settings

        max_conversations = int(getattr(settings, "memory_cache_max_conversations", 256) or 256)
        ttl_seconds = int(getattr(settings, "memory_cache_ttl_seconds", 60) or 60)
        self._cache: TTLCache = TTLCache(maxsize=max_conversations, ttl=ttl_seconds)
        self._locks: dict[str, asyncio.Lock] = {}

    async def build_context(
        self,
        *,
        conversation_id: UUID | str,
        user_id: UUID | str,
        current_message_id: UUID | str | None,
        agent_key: str,
    ) -> ConversationHistoryContext:
        conversation_uuid = self._coerce_uuid(conversation_id)
        user_uuid = self._coerce_uuid(user_id)
        current_uuid = self._coerce_uuid(current_message_id) if current_message_id else None
        budget = self._budget_for(agent_key)

        summary = self._safe_get_summary(conversation_uuid, user_uuid)
        memory = self._memory_message(summary)
        memory_sequence = (
            int(summary.last_summarized_sequence)
            if summary is not None and summary.last_summarized_sequence is not None
            else None
        )
        summary_version = int(getattr(summary, "summary_version", 0) or 0) if summary else 0

        cache_key = _CacheKey(
            conversation_id=str(conversation_uuid),
            user_id=str(user_uuid),
            current_message_id=str(current_uuid) if current_uuid else None,
            agent_key=agent_key,
            memory_sequence=memory_sequence,
            summary_version=summary_version,
        ).as_tuple()

        lock = self._locks.setdefault(str(conversation_uuid), asyncio.Lock())
        async with lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

            # ``max_messages == 0`` is documented as "no cap"; pass ``None`` so
            # the repository skips ``.limit()`` entirely. The token-based
            # ``trim_history_to_budget`` step still bounds the prompt size.
            history_limit: int | None = (
                budget.max_messages if budget.max_messages and budget.max_messages > 0 else None
            )

            try:
                rows = self.message_repository.get_prompt_history(
                    conversation_uuid,
                    before_message_id=current_uuid,
                    after_sequence=memory_sequence,
                    limit=history_limit,
                )
            except Exception as exc:
                logger.warning(
                    "History provider failed to load prompt history for %s: %s",
                    conversation_uuid,
                    exc,
                )
                rows = []

            agent_messages = self._normalize_rows(rows)
            trimmed = trim_history_to_budget(
                agent_messages,
                max_messages=budget.max_messages,
                max_tokens=budget.max_tokens,
            )

            context = ConversationHistoryContext(
                conversation_id=str(conversation_uuid),
                user_id=str(user_uuid),
                memory=memory,
                memory_sequence=memory_sequence,
                messages=([memory] if memory is not None else []) + trimmed,
                budget=budget,
                summary_version=summary_version,
            )
            self._cache[cache_key] = context
            return context

    def invalidate(self, conversation_id: UUID | str) -> None:
        """Drop every cached context for the conversation."""
        prefix = str(conversation_id)
        keys_to_drop = [key for key in list(self._cache.keys()) if key[0] == prefix]
        for key in keys_to_drop:
            self._cache.pop(key, None)
        # Drop the per-conversation lock too — a fresh one will be created on
        # the next call. Holding a stale lock would only slow callers down.
        self._locks.pop(prefix, None)

    async def get_latest_canvas_artifact(
        self,
        *,
        conversation_id: UUID | str,
        user_id: UUID | str,
    ) -> CanvasArtifactSnapshot | None:
        """Load the latest valid canvas independently of prompt-history bounds."""
        conversation_uuid = self._coerce_uuid(conversation_id)
        # Preserve the provider's identifier validation at this trusted boundary.
        self._coerce_uuid(user_id)

        try:
            candidates = self.message_repository.get_canvas_artifact_candidates(
                conversation_uuid
            )
            latest_assistant = self.message_repository.get_latest_assistant_by_conversation(
                conversation_uuid
            )
        except Exception as exc:
            logger.warning(
                "History provider could not load canvas state for %s: %s",
                conversation_uuid,
                exc,
            )
            return None

        latest_assistant_id = str(latest_assistant.id) if latest_assistant is not None else None
        for candidate in candidates:
            snapshot = canvas_snapshot_from_message(candidate)
            if snapshot is None:
                logger.warning(
                    "Skipping invalid canvas artifact metadata message_id=%s conversation_id=%s",
                    getattr(candidate, "id", None),
                    conversation_uuid,
                )
                continue
            return snapshot.with_latest_assistant(snapshot.message_id == latest_assistant_id)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_get_summary(self, conversation_uuid: UUID, user_uuid: UUID):
        if self.summary_repository is None:
            return None
        try:
            summary = self.summary_repository.get_owned_valid_memory(conversation_uuid, user_uuid)
            if summary is None or not bool(getattr(summary, "is_valid", False)):
                return None
            if getattr(summary, "last_summarized_sequence", None) is None:
                return None
            ConversationMemory.model_validate(summary.summary_payload)
            return summary
        except Exception as exc:
            logger.warning(
                "History provider could not load summary for %s: %s",
                conversation_uuid,
                exc,
            )
            return None

    @staticmethod
    def _memory_message(summary: Any | None) -> AgentMessage | None:
        if summary is None:
            return None
        memory = ConversationMemory.model_validate(summary.summary_payload)
        return AgentMessage(
            role=MessageRole.MEMORY,
            content=memory.to_untrusted_reference(),
            metadata={
                "memory_sequence": int(summary.last_summarized_sequence),
                "summary_version": int(summary.summary_version),
                "untrusted_reference": True,
            },
        )

    def _budget_for(self, agent_key: str) -> HistoryBudget:
        config = HistoryBudgetConfig.for_agent(agent_key, self.settings)
        return HistoryBudget(
            agent_key=agent_key,
            max_messages=int(config.max_messages or 0),
            max_tokens=int(config.max_tokens or 0),
        )

    def _normalize_rows(self, rows: Iterable[Message]) -> list[AgentMessage]:
        normalized: list[AgentMessage] = []
        for row in rows:
            agent_message = db_message_to_agent_message(row)
            if agent_message is not None:
                normalized.append(agent_message)
        return normalized

    @staticmethod
    def _coerce_uuid(value: UUID | str) -> UUID:
        if isinstance(value, UUID):
            return value
        return UUID(str(value))


__all__ = [
    "ConversationHistoryContext",
    "ConversationHistoryProvider",
    "HistoryBudget",
    "db_message_to_agent_message",
]
