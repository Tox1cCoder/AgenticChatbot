"""Durable record of one mutating tool execution.

A LangGraph checkpoint is written after a node returns. A mutation that
reached the provider and then lost the process therefore leaves no trace, and
the replay calls the provider again — a second charge, a second message sent,
a second row created. This table closes that window: a row is reserved before
the call and completed alongside its effect.

``outcome_unknown`` is a first-class state, not an error bucket. A reserved row
with no completion and no provider-side deduplication means nobody can say
whether the effect happened; retrying risks a duplicate and reporting failure
risks denying a real effect, so the honest answer is that it is unknown and a
human reconciles it.
"""

import enum
import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, String, func
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.models.base import Base

__all__ = ["ReceiptStatus", "ToolExecutionReceipt"]


class ReceiptStatus(str, enum.Enum):
    """Lifecycle of one mutation receipt."""

    RESERVED = "reserved"
    COMPLETED = "completed"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"


def _enum_member_values(enum_class: type[enum.Enum]) -> list[str]:
    """Persist stable enum values instead of Python member names."""
    return [str(member.value) for member in enum_class]


class ToolExecutionReceipt(Base):
    """One mutating tool call, keyed by its execution identity.

    ``execution_key`` is a SHA-256 digest of ``(thread_id, dispatch_id,
    task_id, tool_call_id)`` and is unique. The uniqueness constraint is the
    mechanism, not a safety net: two concurrent replays race to insert, one
    wins, and the loser reads the winner's row instead of calling the provider.
    """

    __tablename__ = "tool_execution_receipts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # --- execution identity ---------------------------------------------
    execution_key = Column(String(64), nullable=False, unique=True)
    status = Column(
        SQLEnum(
            ReceiptStatus,
            name="tool_execution_receipt_status",
            create_type=True,
            values_callable=_enum_member_values,
        ),
        nullable=False,
    )

    # --- owners ----------------------------------------------------------
    # Every read is filtered by owner. A receipt is not a global cache: one
    # user's completed effect must never answer another user's call.
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True
    )
    turn_id = Column(String(160), nullable=False)
    thread_id = Column(String(320), nullable=False, index=True)

    # --- call provenance -------------------------------------------------
    dispatch_id = Column(String(64), nullable=False)
    task_id = Column(String(160), nullable=False)
    tool_call_id = Column(String(255), nullable=False)
    qualified_tool_id = Column(String(512), nullable=False, index=True)
    provider_idempotency = Column(Boolean, nullable=False, default=False)

    # --- outcome ---------------------------------------------------------
    # Bounded by construction: a large result is offloaded and only its
    # reference is stored, so replay never depends on a row growing without
    # limit.
    result_json = Column(JSONB, nullable=True)
    artifact_ref = Column(String(512), nullable=True)
    provider_receipt_id = Column(String(255), nullable=True)
    error_code = Column(String(128), nullable=True)

    __table_args__ = (
        Index("ix_tool_execution_receipts_owner_status", "user_id", "status"),
        Index("ix_tool_execution_receipts_conversation_turn", "conversation_id", "turn_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<ToolExecutionReceipt(execution_key='{self.execution_key[:12]}...', "
            f"status={self.status.value}, tool='{self.qualified_tool_id}')>"
        )
