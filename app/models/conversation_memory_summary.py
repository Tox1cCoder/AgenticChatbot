"""Durable conversation memory summary model.

Stores a single rolling summary per conversation alongside a database
``messages.id`` cursor. The cursor lets the prompt-history pipeline replay
recent unsummarized messages without overlapping the summary, replacing
the previous LangGraph-message-ID approach.
"""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class ConversationMemorySummary(Base):
    """Per-conversation rolling summary with a DB-message cursor."""

    __tablename__ = "conversation_memory_summaries"
    __table_args__ = (
        Index(
            "ux_conversation_memory_summaries_conversation_id",
            "conversation_id",
            unique=True,
        ),
        Index(
            "ix_conversation_memory_summaries_last_message",
            "last_summarized_message_id",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id"),
        nullable=False,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )
    summary_text = Column(Text, nullable=False, default="")
    # Cursor to the newest DB message folded into the summary. NULL means the
    # summary spans no committed range yet.
    last_summarized_message_id = Column(
        UUID(as_uuid=True),
        ForeignKey("messages.id"),
        nullable=True,
    )
    source_message_count = Column(Integer, nullable=False, default=0)
    estimated_tokens = Column(Integer, nullable=False, default=0)
    # Monotonically incremented on every upsert; useful as a cache-busting key.
    summary_version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    conversation = relationship("Conversation", backref="memory_summary")
    user = relationship("User")
    last_summarized_message = relationship("Message", foreign_keys=[last_summarized_message_id])

    def __repr__(self) -> str:
        return (
            f"<ConversationMemorySummary(conversation_id={self.conversation_id}, "
            f"version={self.summary_version}, "
            f"cursor={self.last_summarized_message_id})>"
        )
