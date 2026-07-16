"""Request-path bridge to durable and forced conversation compaction."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.ai.conversation_memory import ConversationMemory


@dataclass(frozen=True)
class CompactedMemoryReference:
    content: str
    cursor: int


class RequestCompactionCoordinator:
    """Connect request preflight to durable PostgreSQL-backed compaction."""

    def __init__(self, *, repository: Any, publisher: Any, runner: Any, enabled: bool) -> None:
        self.repository = repository
        self.publisher = publisher
        self.runner = runner
        self.enabled = bool(enabled)

    def request_durable(self, conversation_id: UUID | str) -> bool:
        if not self.enabled:
            return False
        parsed_id = self._uuid(conversation_id)
        requested = bool(self.repository.request_backfill(parsed_id))
        if requested:
            self.publisher(parsed_id)
        return requested

    async def compact_now(
        self,
        conversation_id: UUID | str,
        user_id: UUID | str,
    ) -> CompactedMemoryReference | None:
        if not self.enabled:
            return None
        parsed_conversation_id = self._uuid(conversation_id)
        parsed_user_id = self._uuid(user_id)
        self.repository.request_backfill(parsed_conversation_id)
        outcome = self.runner(parsed_conversation_id, force=True)
        if inspect.isawaitable(outcome):
            await outcome
        memory_row = self.repository.get_owned_valid_memory(
            parsed_conversation_id,
            parsed_user_id,
        )
        if memory_row is None or memory_row.last_summarized_sequence is None:
            return None
        try:
            memory = ConversationMemory.model_validate(memory_row.summary_payload)
        except Exception:
            return None
        return CompactedMemoryReference(
            content=memory.to_untrusted_reference(),
            cursor=int(memory_row.last_summarized_sequence),
        )

    @staticmethod
    def _uuid(value: UUID | str) -> UUID:
        return value if isinstance(value, UUID) else UUID(str(value))


__all__ = ["CompactedMemoryReference", "RequestCompactionCoordinator"]
