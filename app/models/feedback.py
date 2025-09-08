from sqlalchemy import Column, ForeignKey, Text, SmallInteger, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import BaseModel


class Feedback(BaseModel):
    __tablename__ = "feedback"

    message_id = Column(
        UUID(as_uuid=True), ForeignKey("messages.id"), nullable=False, index=True
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    rating = Column(SmallInteger, nullable=False)  # Rating from 1 to 5
    comment = Column(Text, nullable=True)

    # Relationships
    message = relationship("Message", back_populates="feedback")
    user = relationship("User", back_populates="feedback")

    # Unique constraint to ensure one feedback per user per message
    __table_args__ = (
        UniqueConstraint("message_id", "user_id", name="uq_feedback_message_user"),
        Index("idx_feedback_message_user", "message_id", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<Feedback(id={self.id}, message_id={self.message_id}, user_id={self.user_id}, rating={self.rating})>"
