import uuid

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    SmallInteger,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # UNIQUE on its own, which is also why there is no (message_id, user_id)
    # index: any lookup by message finds at most one row already.
    message_id = Column(
        UUID(as_uuid=True),
        ForeignKey("messages.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    rating = Column(SmallInteger, nullable=False)  # Rating from 1 to 5
    comment = Column(Text, nullable=True)

    # Relationships
    message = relationship("Message", back_populates="feedback")
    user = relationship("User", back_populates="feedback")

    # The API schemas already bound it to 1..5; this holds for raw SQL too.
    __table_args__ = (
        CheckConstraint("rating BETWEEN 1 AND 5", name="ck_feedbacks_rating_range"),
    )

    def __repr__(self) -> str:
        return (
            f"<Feedback(id={self.id}, message_id={self.message_id}, "
            f"user_id={self.user_id}, rating={self.rating})>"
        )
