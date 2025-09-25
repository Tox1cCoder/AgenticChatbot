from typing import List, Optional
from uuid import UUID
from contextlib import AbstractContextManager
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.feedback import Feedback
from app.repositories.command_strategy import DefaultCommandStrategy
from app.repositories.query_strategy import DefaultQueryStrategy
from app.schemas.feedback import FeedbackCreate, FeedbackUpdate


class FeedbackCRUDStrategy(
    DefaultCommandStrategy[Feedback, FeedbackCreate, FeedbackUpdate],
    DefaultQueryStrategy[Feedback],
):
    """Custom CRUD strategy for Feedback operations"""

    def __init__(self, model: type[Feedback]):
        DefaultCommandStrategy.__init__(self, model)
        DefaultQueryStrategy.__init__(self, model)

    def get_by_message_id(self, db: Session, message_id: UUID) -> Optional[Feedback]:
        """Get feedback by message ID"""
        statement = select(Feedback).where(Feedback.message_id == message_id)
        return db.execute(statement).scalar_one_or_none()

    def get_by_user_id(self, db: Session, user_id: UUID) -> List[Feedback]:
        """Get all feedback by user ID (no pagination needed for user's own feedback)"""
        statement = (
            select(Feedback)
            .where(Feedback.user_id == user_id)
            .order_by(Feedback.created_at.desc())
        )
        return list(db.execute(statement).scalars().all())

    def get_rating_for_message(self, db: Session, message_id: UUID) -> Optional[float]:
        """Get rating for a message"""
        statement = select((Feedback.rating)).where(Feedback.message_id == message_id)
        result = db.execute(statement).scalar_one_or_none()
        return result if result is not None else None


class FeedbackRepository:
    """Repository for Feedback model using session factory pattern"""

    def __init__(self, session_factory: callable):
        """Initialize repository with session factory for dependency injection."""
        self.session_factory = session_factory
        self._crud_strategy = FeedbackCRUDStrategy(Feedback)

    def get_by_message_id(self, message_id: UUID) -> Optional[Feedback]:
        """Get feedback by message ID (1-1 relationship per ERD)"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_message_id(session, message_id)

    def get_by_user_id(self, user_id: UUID) -> List[Feedback]:
        """Get all feedback by user ID (no pagination needed for user's own feedback)"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_user_id(session, user_id)

    def get_rating_for_message(self, message_id: UUID) -> Optional[float]:
        """Get rating for a message"""
        with self.session_factory() as session:
            return self._crud_strategy.get_rating_for_message(session, message_id)

    def create(self, input_schema: FeedbackCreate) -> Feedback:
        """Create a new feedback"""
        with self.session_factory() as session:
            return self._crud_strategy.create(session, input_schema)

    def get_by_id(self, id: UUID) -> Optional[Feedback]:
        """Get feedback by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.get_by_id(session, id)

    def get_all(self) -> List[Feedback]:
        """Get all feedback (removed pagination as not needed for admin operations)"""
        with self.session_factory() as session:
            statement = select(Feedback).order_by(Feedback.created_at.desc())
            return list(session.execute(statement).scalars().all())

    def update(self, id: UUID, input_schema: FeedbackUpdate) -> Optional[Feedback]:
        """Update feedback by ID"""
        with self.session_factory() as session:
            db_obj = self._crud_strategy.get_by_id(session, id)
            if db_obj is None:
                return None
            return self._crud_strategy.update(session, db_obj, input_schema)

    def delete(self, id: UUID) -> bool:
        """Delete feedback by ID"""
        with self.session_factory() as session:
            return self._crud_strategy.delete(session, id)

    def exists(self, id: UUID) -> bool:
        """Check if feedback exists"""
        with self.session_factory() as session:
            return self._crud_strategy.exists(session, id)

    def user_owns_feedback(self, user_id: UUID, feedback_id: UUID) -> bool:
        """Check if a user owns a specific feedback entry"""
        with self.session_factory() as session:
            feedback = self._crud_strategy.get_by_id(session, feedback_id)
            return feedback is not None and feedback.user_id == user_id
