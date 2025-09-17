from typing import List, Optional
from uuid import UUID
from contextlib import AbstractContextManager
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

    def get_rating_for_message(self, db: Session, message_id: UUID) -> Optional[float]:
        """Get rating for a message"""
        from sqlalchemy import func

        stmt = select(Feedback.rating).where(Feedback.message_id == message_id)
        result = db.execute(stmt).scalar()
        return result if result is not None else None

    def get_comment_for_message(self, db: Session, message_id: UUID) -> Optional[str]:
        """Get comment for a message (unique per ERD constraint)"""
        stmt = select(Feedback.comment).where(Feedback.message_id == message_id)
        result = db.execute(stmt).scalar()
        return result if result is not None else None


class FeedbackRepository:
    """Repository for Feedback model using session factory pattern"""

    def __init__(self, session_factory: callable):
        """Initialize repository with session factory for dependency injection."""
        self.session_factory = session_factory
        self._crud_strategy = FeedbackCRUDStrategy(Feedback)

    def get_by_message_id(
        self, message_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Feedback]:
        """Get feedback by message ID"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_message_id(
                session, message_id, skip, limit
            )

    def get_by_user_id(
        self, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Feedback]:
        """Get feedback by user ID"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_user_id(session, user_id, skip, limit)

    def get_by_message_and_user(
        self, message_id: UUID, user_id: UUID
    ) -> Optional[Feedback]:
        """Get feedback by message and user (should be unique per ERD)"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_message_and_user(
                session, message_id, user_id
            )

    def get_rating_for_message(self, message_id: UUID) -> Optional[float]:
        """Get rating for a message"""
        with self.session_factory() as session:
            return self._crud_strategy.get_rating_for_message(session, message_id)

    def get_comment_for_message(self, message_id: UUID) -> Optional[str]:
        """Get comment for a message (unique per ERD constraint)"""
        with self.session_factory() as session:
            return self._crud_strategy.get_comment_for_message(session, message_id)

    def create(self, input_schema: FeedbackCreate) -> Feedback:
        """Create a new feedback"""
        with self.session_factory() as session:
            return self._crud_strategy.create(session, input_schema)

    def get_by_id(self, id: UUID) -> Optional[Feedback]:
        """Get feedback by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_id(session, id)

    def get_all(self, skip: int = 0, limit: int = 100) -> list[Feedback]:
        """Get all feedback with pagination"""
        with self.session_factory() as session:
            return self._crud_strategy.get_all(session, skip, limit)

    def update(self, id: UUID, input_schema: FeedbackUpdate) -> Optional[Feedback]:
        """Update feedback by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.update(session, id, input_schema)

    def delete(self, id: UUID) -> bool:
        """Delete feedback by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.delete(session, id)

    def exists(self, id: UUID) -> bool:
        """Check if feedback exists"""
        with self.session_factory() as session:
            return self._crud_strategy.exists(session, id)
