"""Model for tracking task plans in conversations with planning mode."""

import uuid

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.models.base import Base
from app.models.enums import TaskStatus, TaskStatusType


class TaskPlan(Base):
    """
    Track task plans for conversations with planning mode enabled.

    This model stores individual tasks with their status and metadata
    for structured task management within conversations.
    """

    __tablename__ = "task_plans"
    __table_args__ = (
        # Non-unique fallback index kept for query performance (pre-constraint rows).
        # The unique constraint on (conversation_id, task_order) is the authoritative
        # database-level invariant enforced by the migration that resequences dupes first.
        UniqueConstraint(
            "conversation_id",
            "task_order",
            name="uq_task_plan_conversation_order",
            deferrable=True,
            initially="DEFERRED",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True
    )
    task_order = Column(Integer, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(TaskStatusType, nullable=False, default=TaskStatus.pending)

    task_metadata = Column(JSONB, nullable=True, default=None, server_default="'{}'::jsonb")
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="task_plans")

    def __repr__(self) -> str:
        return (
            f"<TaskPlan(id={self.id}, task_order={self.task_order}, "
            f"status={self.status}, conversation_id={self.conversation_id})>"
        )
