"""Durable coalescing conversation-compaction job."""

from __future__ import annotations

import enum

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class SummaryJobStatus(str, enum.Enum):
    IDLE = "idle"
    PENDING = "pending"
    PROCESSING = "processing"
    RETRY = "retry"
    DEAD = "dead"


class ConversationSummaryJob(Base):
    """Latest durable compaction target and worker lease per conversation."""

    __tablename__ = "conversation_summary_jobs"
    __table_args__ = (
        PrimaryKeyConstraint(
            "conversation_id",
            name="pk_conversation_summary_jobs",
        ),
        ForeignKeyConstraint(
            ["conversation_id", "requested_through_sequence"],
            ["messages.conversation_id", "messages.sequence"],
            name="fk_summary_job_conversation_sequence",
        ),
        CheckConstraint(
            "status IN ('idle', 'pending', 'processing', 'retry', 'dead')",
            name="ck_summary_jobs_status",
        ),
        CheckConstraint(
            "attempt_count >= 0",
            name="ck_summary_jobs_attempt_count_nonnegative",
        ),
        Index("ix_summary_jobs_due", "status", "available_at"),
    )

    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "conversations.id",
            name="fk_summary_job_conversation",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    requested_through_sequence = Column(BigInteger, nullable=False)
    status = Column(
        String(16),
        nullable=False,
        default=SummaryJobStatus.PENDING.value,
        server_default=text("'pending'"),
    )
    attempt_count = Column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    available_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        server_default=func.now(),
    )
    lease_token = Column(UUID(as_uuid=True), nullable=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True)
    last_error_code = Column(String(64), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        onupdate=func.now(),
        server_default=func.now(),
    )

    conversation = relationship("Conversation", back_populates="summary_job")

    def __repr__(self) -> str:
        return (
            f"<ConversationSummaryJob(conversation_id={self.conversation_id}, "
            f"target={self.requested_through_sequence}, status={self.status!r})>"
        )
