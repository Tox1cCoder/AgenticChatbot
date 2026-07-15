"""Structured, sequence-scoped conversation memory."""

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    PrimaryKeyConstraint,
    SmallInteger,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class ConversationMemorySummary(Base):
    """Validated derived memory owned one-to-one by a conversation."""

    __tablename__ = "conversation_memory_summaries"
    __table_args__ = (
        PrimaryKeyConstraint(
            "conversation_id",
            name="pk_conversation_memory_summaries",
        ),
        ForeignKeyConstraint(
            ["conversation_id", "last_summarized_sequence"],
            ["messages.conversation_id", "messages.sequence"],
            name="fk_memory_summary_conversation_sequence",
        ),
        CheckConstraint(
            "summary_schema_version > 0",
            name="ck_memory_summary_schema_version_positive",
        ),
        CheckConstraint(
            "summary_version > 0",
            name="ck_memory_summary_version_positive",
        ),
        CheckConstraint(
            "source_message_count >= 0",
            name="ck_memory_summary_source_message_count_nonnegative",
        ),
        CheckConstraint(
            "source_token_count >= 0",
            name="ck_memory_summary_source_token_count_nonnegative",
        ),
        CheckConstraint(
            "summary_token_count >= 0",
            name="ck_memory_summary_token_count_nonnegative",
        ),
    )

    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "conversations.id",
            name="fk_memory_summary_conversation",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    summary_payload = Column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    summary_schema_version = Column(
        SmallInteger,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    last_summarized_sequence = Column(BigInteger, nullable=True)
    summary_version = Column(
        BigInteger,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    source_message_count = Column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    source_token_count = Column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    summary_token_count = Column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    provider = Column(String(64), nullable=False, default="", server_default=text("''"))
    model = Column(String(255), nullable=False, default="", server_default=text("''"))
    tokenizer = Column(String(128), nullable=False, default="", server_default=text("''"))
    prompt_version = Column(String(64), nullable=False, default="", server_default=text("''"))
    is_valid = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
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

    conversation = relationship("Conversation", back_populates="memory_summary")

    def __repr__(self) -> str:
        return (
            f"<ConversationMemorySummary(conversation_id={self.conversation_id}, "
            f"version={self.summary_version}, cursor={self.last_summarized_sequence}, "
            f"valid={self.is_valid})>"
        )
