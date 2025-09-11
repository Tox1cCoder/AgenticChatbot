from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.feedback import Feedback
from app.repositories.strategy import Repository, DefaultCRUDStrategy
from app.schemas.feedback import FeedbackCreate, FeedbackUpdate


class FeedbackCRUDStrategy(
    DefaultCRUDStrategy[Feedback, FeedbackCreate, FeedbackUpdate]
):
    """Custom CRUD strategy for Feedback operations"""

    def get_by_message_id(
        self, db: Session, message_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Feedback]:
        """Get feedback by message ID"""
        stmt = (
            select(Feedback)
            .where(Feedback.message_id == message_id)
            .order_by(Feedback.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(db.execute(stmt).scalars().all())

    def get_by_user_id(
        self, db: Session, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Feedback]:
        """Get feedback by user ID"""
        stmt = (
            select(Feedback)
            .where(Feedback.user_id == user_id)
            .order_by(Feedback.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(db.execute(stmt).scalars().all())

    def get_by_message_and_user(
        self, db: Session, message_id: UUID, user_id: UUID
    ) -> Optional[Feedback]:
        """Get feedback by message and user (should be unique per ERD)"""
        stmt = select(Feedback).where(
            Feedback.message_id == message_id, Feedback.user_id == user_id
        )
        return db.execute(stmt).scalar_one_or_none()

    def get_average_rating_for_message(
        self, db: Session, message_id: UUID
    ) -> Optional[float]:
        """Get average rating for a message"""
        from sqlalchemy import func

        stmt = select(func.avg(Feedback.rating)).where(
            Feedback.message_id == message_id
        )
        result = db.execute(stmt).scalar()
        return float(result) if result is not None else None

    def get_comment_for_message(self, db: Session, message_id: UUID) -> Optional[str]:
        """Get comment for a message (unique per ERD constraint)"""
        stmt = select(Feedback.comment).where(Feedback.message_id == message_id)
        result = db.execute(stmt).scalar()
        return result if result is not None else None


class FeedbackRepository(Repository[Feedback, FeedbackCreate, FeedbackUpdate]):
    """Repository for Feedback model"""

    def __init__(self, db: Session):
        strategy = FeedbackCRUDStrategy(Feedback)
        super().__init__(db, strategy)

    def get_by_message_id(
        self, message_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Feedback]:
        """Get feedback by message ID"""
        return self._crud_strategy.get_by_message_id(self.db, message_id, skip, limit)

    def get_by_user_id(
        self, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Feedback]:
        """Get feedback by user ID"""
        return self._crud_strategy.get_by_user_id(self.db, user_id, skip, limit)

    def get_by_message_and_user(
        self, message_id: UUID, user_id: UUID
    ) -> Optional[Feedback]:
        """Get feedback by message and user (should be unique per ERD)"""
        return self._crud_strategy.get_by_message_and_user(self.db, message_id, user_id)

    def get_average_rating_for_message(self, message_id: UUID) -> Optional[float]:
        """Get average rating for a message"""
        return self._crud_strategy.get_average_rating_for_message(self.db, message_id)

    def get_comment_for_message(self, message_id: UUID) -> Optional[str]:
        """Get comment for a message (unique per ERD constraint)"""
        return self._crud_strategy.get_comment_for_message(self.db, message_id)
