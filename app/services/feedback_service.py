from __future__ import annotations
from typing import List, Optional
from uuid import UUID
from fastapi import HTTPException, status

from app.repositories.feedback import FeedbackRepository
from app.repositories.message import MessageRepository
from app.repositories.user import UserRepository
from app.schemas.feedback import FeedbackCreate, FeedbackUpdate, FeedbackRead
from app.factories.feedback_factory import FeedbackFactory


class FeedbackService:
    """Service layer for Feedback operations"""

    def __init__(
        self,
        feedback_repository: FeedbackRepository,
        message_repository: MessageRepository,
        user_repository: UserRepository,
    ):
        """
        Initialize FeedbackService with injected dependencies.

        Args:
            feedback_repository: Injected feedback repository
            message_repository: Injected message repository
            user_repository: Injected user repository
        """
        self.repository = feedback_repository
        self.message_repository = message_repository
        self.user_repository = user_repository
        self.repository = feedback_repository
        self.message_repository = message_repository
        self.user_repository = user_repository

    def create_feedback(
        self, feedback_create_data: FeedbackCreate, user_id: UUID
    ) -> FeedbackRead:
        """Create new feedback with validation"""
        # Validate user exists
        if not self.user_repository.exists(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        # Validate message exists
        if not self.message_repository.exists(feedback_create_data.message_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
            )

        # Check if user already provided feedback for this message
        existing_feedback_entity = self.repository.get_by_message_and_user(
            feedback_create_data.message_id, user_id
        )

        if existing_feedback_entity:
            # Update existing feedback
            from app.schemas.feedback import FeedbackUpdate

            update_data = FeedbackUpdate(
                rating=feedback_create_data.rating, comment=feedback_create_data.comment
            )
            updated_feedback = self.repository.update(
                existing_feedback_entity, update_data
            )
            return FeedbackRead.model_validate(updated_feedback)

        # Create new feedback entity using factory
        feedback_entity = FeedbackFactory.create_from_schema(
            feedback_create_data, user_id
        )
        created_feedback = self.repository.create(feedback_entity)
        return FeedbackRead.model_validate(created_feedback)

    def get_feedback_by_id(self, feedback_id: UUID) -> FeedbackRead:
        """Get feedback by ID"""
        feedback_entity = self.repository.get_by_id(feedback_id)
        if not feedback_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found"
            )
        return FeedbackRead.model_validate(feedback_entity)

    def get_feedback_by_message(
        self, message_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[FeedbackRead]:
        """Get all feedback for a message"""
        # Validate message exists
        if not self.message_repository.exists(message_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
            )

        feedback_entities = self.repository.get_by_message_id(
            message_id, skip=skip, limit=limit
        )
        return [FeedbackRead.model_validate(feedback) for feedback in feedback_entities]

    def get_feedback_by_user(
        self, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[FeedbackRead]:
        """Get all feedback by a user"""
        # Validate user exists
        if not self.user_repository.exists(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        feedback_entities = self.repository.get_by_user_id(
            user_id, skip=skip, limit=limit
        )
        return [FeedbackRead.model_validate(feedback) for feedback in feedback_entities]

    def get_user_feedback_for_message(
        self, message_id: UUID, user_id: UUID
    ) -> Optional[FeedbackRead]:
        """Get specific user's feedback for a message"""
        # Validate message exists
        if not self.message_repository.exists(message_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
            )

        # Validate user exists
        if not self.user_repository.exists(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        feedback_entity = self.repository.get_by_message_and_user(message_id, user_id)
        return FeedbackRead.model_validate(feedback_entity) if feedback_entity else None

    def get_message_rating_stats(self, message_id: UUID) -> dict:
        """Get rating statistics for a message"""
        # Validate message exists
        if not self.message_repository.exists(message_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
            )

        avg_rating = self.repository.get_rating_for_message(message_id)

        return {
            "message_id": message_id,
            "rating": avg_rating,
            "comment": self.repository.get_comment_for_message(message_id),
        }

    def update_feedback(
        self, feedback_id: UUID, user_id: UUID, feedback_update_data: FeedbackUpdate
    ) -> FeedbackRead:
        """Update feedback with ownership validation"""
        feedback_entity = self.repository.get_by_id(feedback_id)
        if not feedback_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found"
            )

        # Validate user owns the feedback
        if feedback_entity.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this feedback",
            )

        updated_feedback = self.repository.update(feedback_entity, feedback_update_data)
        return FeedbackRead.model_validate(updated_feedback)

    def delete_feedback(self, feedback_id: UUID, user_id: UUID) -> bool:
        """Delete feedback with ownership validation"""
        feedback_entity = self.repository.get_by_id(feedback_id)
        if not feedback_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found"
            )

        # Validate user owns the feedback
        if feedback_entity.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this feedback",
            )

        return self.repository.delete(feedback_id)
