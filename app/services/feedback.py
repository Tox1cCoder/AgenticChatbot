from typing import List, Optional
from uuid import UUID
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.repositories.feedback import FeedbackRepository
from app.repositories.message import MessageRepository
from app.repositories.user import UserRepository
from app.schemas.feedback import FeedbackCreate, FeedbackUpdate, FeedbackRead


class FeedbackService:
    """Service layer for Feedback operations"""

    def __init__(self, db: Session):
        self.repository = FeedbackRepository(db)
        self.message_repository = MessageRepository(db)
        self.user_repository = UserRepository(db)

    def create_feedback(
        self, feedback_data: FeedbackCreate, user_id: UUID
    ) -> FeedbackRead:
        """Create new feedback with validation"""
        # Validate user exists
        if not self.user_repository.exists(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        # Validate message exists
        if not self.message_repository.exists(feedback_data.message_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
            )

        # Check if user already provided feedback for this message
        existing_feedback = self.repository.get_by_message_and_user(
            feedback_data.message_id, user_id
        )
        if existing_feedback:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Feedback already exists for this message",
            )

        # Create feedback data with user_id
        feedback_dict = feedback_data.model_dump()
        feedback_dict["user_id"] = user_id

        # Create a proper schema class with user_id for repository
        from pydantic import BaseModel

        class FeedbackCreateWithUserId(BaseModel):
            message_id: UUID
            user_id: UUID
            rating: int
            comment: Optional[str] = None

            def model_dump(self):
                return {
                    "message_id": self.message_id,
                    "user_id": self.user_id,
                    "rating": self.rating,
                    "comment": self.comment,
                }

        feedback_create_db = FeedbackCreateWithUserId(**feedback_dict)
        feedback = self.repository.create(feedback_create_db)
        return FeedbackRead.model_validate(feedback)

    def get_feedback_by_id(self, feedback_id: UUID) -> FeedbackRead:
        """Get feedback by ID"""
        feedback = self.repository.get_by_id(feedback_id)
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found"
            )
        return FeedbackRead.model_validate(feedback)

    def get_feedback_by_message(
        self, message_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[FeedbackRead]:
        """Get all feedback for a message"""
        # Validate message exists
        if not self.message_repository.exists(message_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
            )

        feedback_list = self.repository.get_by_message_id(
            message_id, skip=skip, limit=limit
        )
        return [FeedbackRead.model_validate(feedback) for feedback in feedback_list]

    def get_feedback_by_user(
        self, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[FeedbackRead]:
        """Get all feedback by a user"""
        # Validate user exists
        if not self.user_repository.exists(user_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        feedback_list = self.repository.get_by_user_id(user_id, skip=skip, limit=limit)
        return [FeedbackRead.model_validate(feedback) for feedback in feedback_list]

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

        feedback = self.repository.get_by_message_and_user(message_id, user_id)
        return FeedbackRead.model_validate(feedback) if feedback else None

    def get_message_rating_stats(self, message_id: UUID) -> dict:
        """Get rating statistics for a message"""
        # Validate message exists
        if not self.message_repository.exists(message_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
            )

        avg_rating = self.repository.get_average_rating_for_message(message_id)
        feedback_count = self.repository.get_feedback_count_for_message(message_id)

        return {
            "message_id": message_id,
            "average_rating": avg_rating,
            "feedback_count": feedback_count,
        }

    def update_feedback(
        self, feedback_id: UUID, user_id: UUID, feedback_data: FeedbackUpdate
    ) -> FeedbackRead:
        """Update feedback with ownership validation"""
        feedback = self.repository.get_by_id(feedback_id)
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found"
            )

        # Validate user owns the feedback
        if feedback.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this feedback",
            )

        updated_feedback = self.repository.update(feedback, feedback_data)
        return FeedbackRead.model_validate(updated_feedback)

    def delete_feedback(self, feedback_id: UUID, user_id: UUID) -> bool:
        """Delete feedback with ownership validation"""
        feedback = self.repository.get_by_id(feedback_id)
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found"
            )

        # Validate user owns the feedback
        if feedback.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this feedback",
            )

        return self.repository.delete(feedback_id)
