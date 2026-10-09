import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
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
        CheckConstraint(
            "forked_through_sequence IS NULL OR forked_through_sequence > 0",
            name="ck_conversations_forked_through_sequence_positive",
        ),
        CheckConstraint(
            "(fork_request_id IS NULL) = (fork_request_fingerprint IS NULL)",
            name="ck_conversations_fork_request_fields_together",
        ),
        # Includes soft-deleted rows on purpose: a retry must find the deleted
        # child (and answer 410) instead of creating a second one.
        UniqueConstraint(
            "owner_id", "fork_request_id", name="uq_conversations_owner_fork_request"
        ),
        Index(
            "ix_conversations_project_updated",
            "project_id",
            "updated_at",
            postgresql_where=text("deleted_at IS NULL"),
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
    # Nullable: a conversation may belong to at most one project, or none.
    project_id = Column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True, index=True
    )
    # Lineage. SET NULL: hard-deleting the source must not delete or block the
    # fork; ``forked_through_sequence`` keeps the cutoff once the references go.
    # ``use_alter``: conversations and messages now reference each other.
    forked_from_conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "conversations.id",
            ondelete="SET NULL",
            name="fk_conversations_forked_from_conversation_id",
        ),
        nullable=True,
        index=True,
    )
    forked_from_message_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "messages.id",
            ondelete="SET NULL",
            name="fk_conversations_forked_from_message_id",
            use_alter=True,
        ),
        nullable=True,
        index=True,
    )
    forked_through_sequence = Column(BigInteger, nullable=True)
    # Idempotency key and canonical request hash; internal, never serialized.
    fork_request_id = Column(UUID(as_uuid=True), nullable=True)
    fork_request_fingerprint = Column(String(64), nullable=True)

    # Relationships
    user = relationship("User", back_populates="conversations")
    # ``foreign_keys``: ``forked_from_message_id`` is a second path to messages.
    messages = relationship(
        "Message",
        back_populates="conversation",
        foreign_keys="Message.conversation_id",
    )
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
