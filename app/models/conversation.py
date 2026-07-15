import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base
from app.models.enums import PlanLifecycleType


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        CheckConstraint(
            "next_message_sequence > 0",
            name="ck_conversations_next_message_sequence_positive",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    persona_prompt = Column(Text, nullable=True)
    planning_mode_enabled = Column(Boolean, default=False, nullable=False)
    next_message_sequence = Column(
        BigInteger,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    # Explicit plan lifecycle state; NULL means no plan has been created.
    plan_lifecycle = Column(PlanLifecycleType, nullable=True, default=None)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")
    documents = relationship("Document", back_populates="conversation")
    task_plans = relationship("TaskPlan", back_populates="conversation")
    memory_summary = relationship(
        "ConversationMemorySummary",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    summary_job = relationship(
        "ConversationSummaryJob",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title='{self.title}', owner_id={self.owner_id})>"
