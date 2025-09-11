from sqlalchemy import Column, ForeignKey, Text, SmallInteger, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import BaseModel


class Feedback(BaseModel):
    __tablename__ = "feedback"

    message_id = Column(
        UUID(as_uuid=True),
        ForeignKey("message.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("user.id"), nullable=False, index=True
    )
    rating = Column(SmallInteger, nullable=False)  # Rating from 1 to 5
    comment = Column(Text, nullable=True)

    # Relationships
    message = relationship("Message", back_populates="feedback")
    user = relationship("User", back_populates="feedback")

    # Index for efficient querying
    __table_args__ = (Index("idx_feedbacks_message_user", "message_id", "user_id"),)

    def __repr__(self) -> str:
        return f"<Feedback(id={self.id}, message_id={self.message_id}, user_id={self.user_id}, rating={self.rating})>"
