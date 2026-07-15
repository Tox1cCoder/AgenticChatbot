import uuid

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.models.base import Base
from app.models.enums import MessageRoleType


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
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
    message_metadata = Column(JSONB, nullable=True, default=dict)
    sequence = Column(BigInteger, nullable=False)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    feedback = relationship("Feedback", back_populates="message", uselist=False, lazy="joined")

    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "sequence",
            name="uq_messages_conversation_sequence",
        ),
        CheckConstraint("sequence > 0", name="ck_messages_sequence_positive"),
        Index("idx_message_conversation_created", "conversation_id", "created_at"),
        Index(
            "ix_messages_prompt_history",
            "conversation_id",
            "sequence",
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Message(id={self.id}, conversation_id={self.conversation_id}, sender={self.sender})>"
        )
