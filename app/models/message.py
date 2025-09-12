from sqlalchemy import Column, ForeignKey, Text, Index, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import BaseModel
from app.models.enums import MessageRoleType


class Message(BaseModel):
    __tablename__ = "messages"

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
