from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.feedback import Feedback
from app.repositories.base import BaseRepository
from app.schemas.feedback import FeedbackCreate, FeedbackUpdate


class FeedbackRepository(BaseRepository[Feedback, FeedbackCreate, FeedbackUpdate]):
    """Repository for Feedback model with custom methods"""

    def __init__(self, db: Session):
        super().__init__(Feedback, db)

    def get_by_message_id(
        self, message_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Feedback]:
        """Get feedback by message ID"""
        stmt = (
            select(Feedback)
            .where(Feedback.message_id == message_id)
            .order_by(Feedback.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def get_by_user_id(
        self, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Feedback]:
        """Get feedback by user ID"""
        stmt = (
            select(Feedback)
            .where(Feedback.user_id == user_id)
            .order_by(Feedback.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def get_by_message_and_user(
        self, message_id: UUID, user_id: UUID
    ) -> Optional[Feedback]:
        """Get feedback by message and user (should be unique)"""
        stmt = select(Feedback).where(
            Feedback.message_id == message_id, Feedback.user_id == user_id
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def get_average_rating_for_message(self, message_id: UUID) -> Optional[float]:
        """Get average rating for a message"""
        from sqlalchemy import func

        stmt = select(func.avg(Feedback.rating)).where(
            Feedback.message_id == message_id
        )
        result = self.db.execute(stmt).scalar()
        return float(result) if result is not None else None