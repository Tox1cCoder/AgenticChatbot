"""Commands and snapshots for the generation lifecycle.

:class:`GenerationSnapshot` is the one shape every transport returns — internal
SSE, AI SDK v6, the client-backend proxy and the JSON status endpoint all
project this and nothing else. What it deliberately omits is as much of the
contract as what it carries: a checkpoint thread id is a resume handle, and a
budget or research-accounting blob is server bookkeeping. Neither belongs in a
response a client can read.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.generation import GenerationCommandAction, GenerationStatus

__all__ = [
    "CommandClaim",
    "ContinueGenerationCommand",
    "CreateGeneration",
    "GenerationSnapshot",
    "StopGenerationCommand",
]


def _require_text(value: str, field: str) -> str:
    cleaned = " ".join(str(value or "").split())
    if not cleaned:
        raise ValueError(f"{field} must not be blank")
    return cleaned


class GenerationSnapshot(BaseModel):
    """The public state of one generation. Immutable by construction."""

    model_config = ConfigDict(frozen=True)

    generation_id: UUID
    logical_turn_id: str
    conversation_id: UUID
    status: GenerationStatus
    version: int
    execution_epoch: int
    continuation_id: UUID | None = None
    continuation_available: bool = False
    continuation_block_reason: str | None = None
    assistant_message_id: UUID | None = None
    terminal_reason: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> GenerationSnapshot:
        """Project a ``Generation`` row, naming every field that crosses over.

        Written out rather than derived from the model so that adding a column
        does not silently start publishing it.
        """
        return cls(
            generation_id=row.id,
            logical_turn_id=row.logical_turn_id,
            conversation_id=row.conversation_id,
            status=row.status,
            version=row.version,
            execution_epoch=row.execution_epoch,
            continuation_id=row.continuation_id,
            continuation_available=bool(row.continuation_available),
            continuation_block_reason=row.continuation_block_reason,
            assistant_message_id=row.assistant_message_id,
            terminal_reason=row.terminal_reason,
        )


class CreateGeneration(BaseModel):
    """Everything needed before the first model call of a new turn."""

    conversation_id: UUID
    user_id: UUID
    logical_turn_id: str = Field(min_length=1, max_length=160)
    checkpoint_thread_id: str = Field(min_length=1, max_length=320)
    active_agent_id: str | None = Field(default=None, max_length=160)

    @field_validator("logical_turn_id", "checkpoint_thread_id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        # ``min_length`` counts whitespace, and the logical turn is the table's
        # unique key: a row of spaces would claim the turn and never match again.
        return _require_text(value, "identity")


class StopGenerationCommand(BaseModel):
    """Stop one generation, fenced against the version the client last saw."""

    generation_id: UUID
    conversation_id: UUID
    user_id: UUID
    idempotency_key: str = Field(min_length=8, max_length=160)
    # R5: without a fence, a delayed replay of a Stop issued against epoch 0
    # cancels whatever epoch is running when it arrives.
    expected_version: int = Field(ge=1)


class ContinueGenerationCommand(BaseModel):
    """Continue a paused or stopped generation from its exact checkpoint."""

    generation_id: UUID
    continuation_id: UUID
    conversation_id: UUID
    user_id: UUID
    idempotency_key: str = Field(min_length=8, max_length=160)
    expected_version: int = Field(ge=1)
    inline_rich_response_v1: bool = False


class CommandClaim(BaseModel):
    """Whether this caller owns a command, or is replaying someone else's.

    ``claimed`` is the whole point. A caller that claimed the command must
    execute it and then record its result; a caller that did not must return
    ``result`` unchanged and do nothing else — that is what makes a retried
    Stop return "stopped" rather than stopping something new.
    """

    model_config = ConfigDict(frozen=True)

    claimed: bool
    action: GenerationCommandAction
    fence: int
    result: dict[str, Any] | None = None
