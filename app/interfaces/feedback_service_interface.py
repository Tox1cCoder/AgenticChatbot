"""
Feedback service interface definition
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from app.schemas.feedback import FeedbackCreate, FeedbackUpdate, FeedbackRead


class IFeedbackService(ABC):
    """Interface for Feedback service operations"""

    @abstractmethod
    def create_feedback(
        self, feedback_create_data: FeedbackCreate, user_id: UUID
    ) -> FeedbackRead:
        """Create new feedback with validation"""
        pass

    @abstractmethod
    def get_by_id(self, feedback_id: UUID) -> FeedbackRead:
        """Get feedback by ID"""
        pass

    @abstractmethod
    def get_by_message(
        self, message_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[FeedbackRead]:
        """Get all feedback for a message"""
        pass

    @abstractmethod
    def get_by_user(
        self, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[FeedbackRead]:
        """Get all feedback by a user"""
        pass

    @abstractmethod
    def get_user_feedback_for_message(
        self, message_id: UUID, user_id: UUID
    ) -> Optional[FeedbackRead]:
        """Get specific user's feedback for a message"""
        pass

    @abstractmethod
    def get_message_rating_stats(self, message_id: UUID) -> dict:
        """Get rating statistics for a message"""
        pass

    @abstractmethod
    def update_feedback(
        self, feedback_id: UUID, user_id: UUID, feedback_update_data: FeedbackUpdate
    ) -> FeedbackRead:
        """Update feedback with ownership validation"""
        pass

    @abstractmethod
    def delete_feedback(self, feedback_id: UUID, user_id: UUID) -> bool:
        """Delete feedback with ownership validation"""
        pass
