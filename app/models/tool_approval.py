"""Model for tracking human-in-the-loop tool approval decisions."""

import uuid
from sqlalchemy import Column, String, ForeignKey, DateTime, Enum as SQLEnum, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import enum

from app.models.base import Base


class DecisionType(str, enum.Enum):
    """Enum for tool approval decision types."""

    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"


class ToolApproval(Base):
    """
    Track all human approval decisions for tool calls in HITL workflows.

    This model provides a complete audit trail for compliance, debugging,
    and analytics purposes. Each record represents a single tool approval decision.
    """

    __tablename__ = "tool_approvals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # Foreign keys
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )

    # Interrupt tracking
    interrupt_id = Column(String(255), nullable=False, index=True)

    # Tool call details
    tool_name = Column(String(255), nullable=False)
    tool_call_id = Column(String(255), nullable=False)
    original_args = Column(JSONB, nullable=False)
    modified_args = Column(JSONB, nullable=True)

    # Decision information
    decision = Column(
        SQLEnum(DecisionType, name="decision_type", create_type=True), nullable=False
    )
    decided_at = Column(
        DateTime(timezone=True), default=func.now(), nullable=False, index=True
    )

    # Relationships
    conversation = relationship("Conversation", backref="tool_approvals")
    user = relationship("User", backref="tool_approvals")

    def __repr__(self) -> str:
        return (
            f"<ToolApproval(id={self.id}, tool_name='{self.tool_name}', "
            f"decision={self.decision.value}, conversation_id={self.conversation_id})>"
        )
