from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import Column, ForeignKey, Text, Index, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base

from app.models.enums import MessageRoleType

# Create independent base for this model
Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"

    # Independent attribute declarations - no inheritance
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True
    )
    sender = Column(MessageRoleType, nullable=False)
    content = Column(Text, nullable=False)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    feedback = relationship("Feedback", back_populates="message")

    # Index for efficient querying by conversation and timestamp
    __table_args__ = (
        Index("idx_message_conversation_created", "conversation_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, conversation_id={self.conversation_id}, sender={self.sender})>"
