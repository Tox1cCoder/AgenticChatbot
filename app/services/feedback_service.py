from __future__ import annotations
from typing import List, Optional
from uuid import UUID

from app.core.exceptions import (
    ValidationException,
    ResourceNotFoundException,
    AuthorizationException,
)
from app.repositories.feedback import FeedbackRepository
from app.repositories.message import MessageRepository
from app.repositories.user import UserRepository
from app.schemas.feedback import FeedbackCreate, FeedbackUpdate, FeedbackRead
from app.factories.feedback_factory import FeedbackFactory
from app.utils.validation.user_validation import UserValidationUtils
from app.utils.validation.message_validation import MessageValidationUtils
from app.utils.validation.feedback_validation import FeedbackValidationUtils
from app.interfaces.feedback_service_interface import IFeedbackService


class FeedbackService(IFeedbackService):
    """Service layer for Feedback operations"""

    def __init__(
        self,
        feedback_repository: FeedbackRepository,
        message_repository: MessageRepository,
        user_repository: UserRepository,
        user_validation_utils: UserValidationUtils,
        message_validation_utils: MessageValidationUtils,
        feedback_validation_utils: FeedbackValidationUtils,
    ):
        """
        Initialize FeedbackService with injected dependencies.

        Args:
            feedback_repository: Injected feedback repository
            message_repository: Injected message repository
            user_repository: Injected user repository
            user_validation_utils: Injected user validation utils
            message_validation_utils: Injected message validation utils
            feedback_validation_utils: Injected feedback validation utils
        """
        self.repository = feedback_repository
        self.message_repository = message_repository
        self.user_repository = user_repository
        self.user_validation_utils = user_validation_utils
        self.message_validation_utils = message_validation_utils
        self.feedback_validation_utils = feedback_validation_utils

    def create_feedback(
        self, feedback_create_data: FeedbackCreate, user_id: UUID
    ) -> FeedbackRead:
        """Create new feedback with validation"""
        self.user_validation_utils.validate_user_exists(user_id)

        self.message_validation_utils.validate_message_exists(
            feedback_create_data.message_id
        )

        existing_feedback_entity = self.repository.get_by_message_and_user(
            feedback_create_data.message_id, user_id
        )

        if existing_feedback_entity:
            update_data = FeedbackUpdate(
                rating=feedback_create_data.rating, comment=feedback_create_data.comment
            )
            updated_feedback = self.repository.update(
                existing_feedback_entity.id, update_data
            )
            return FeedbackRead.model_validate(updated_feedback)

        feedback_entity = FeedbackFactory.create_from_schema(
            feedback_create_data, user_id
        )
        created_feedback = self.repository.create(feedback_entity)
        return FeedbackRead.model_validate(created_feedback)

    def get_by_id(self, feedback_id: UUID) -> FeedbackRead:
        """Get feedback by ID"""
        feedback_entity = self.repository.get_by_id(feedback_id)
        if not feedback_entity:
            raise ResourceNotFoundException(
                detail="Feedback not found", error_code="FEEDBACK_NOT_FOUND"
            )
        return FeedbackRead.model_validate(feedback_entity)

    def get_by_message(
        self, message_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[FeedbackRead]:
        """Get all feedback for a message"""
        # Validate message exists
        self.message_validation_utils.validate_message_exists(message_id)

        # Convert skip to page for repository call
        page = (skip // limit) + 1 if limit > 0 else 1
        paginated_result = self.repository.get_by_message_id(
            message_id, page=page, limit=limit
        )
        feedback_entities = paginated_result.items
        return [FeedbackRead.model_validate(feedback) for feedback in feedback_entities]

    def get_feedbacks_by_message_id(
        self, message_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[FeedbackRead]:
        """Get all feedbacks for a message - alias for get_by_message"""
        return self.get_by_message(message_id, skip, limit)

    def get_by_user(
        self, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[FeedbackRead]:
        """Get all feedback by a user"""
        # Validate user exists
        self.user_validation_utils.validate_user_exists(user_id)

        # Convert skip to page for repository call
        page = (skip // limit) + 1 if limit > 0 else 1
        paginated_result = self.repository.get_by_user_id(
            user_id, page=page, limit=limit
        )
        feedback_entities = paginated_result.items
        return [FeedbackRead.model_validate(feedback) for feedback in feedback_entities]

    def get_user_feedback_for_message(
        self, message_id: UUID, user_id: UUID
    ) -> Optional[FeedbackRead]:
        """Get specific user's feedback for a message"""
        # Validate message exists
        self.message_validation_utils.validate_message_exists(message_id)

        # Validate user exists
        self.user_validation_utils.validate_user_exists(user_id)

        feedback_entity = self.repository.get_by_message_and_user(message_id, user_id)
        return FeedbackRead.model_validate(feedback_entity) if feedback_entity else None

    def get_message_rating_stats(self, message_id: UUID) -> dict:
        """Get rating statistics for a message"""
        # Validate message exists
        self.message_validation_utils.validate_message_exists(message_id)

        return {
            "messageId": message_id,
            "rating": self.repository.get_rating_for_message(message_id),
        }

    def update_feedback(
        self, feedback_id: UUID, user_id: UUID, feedback_update_data: FeedbackUpdate
    ) -> FeedbackRead:
        """Update feedback with ownership validation"""
        if not self.feedback_validation_utils.validate_feedback_exists(feedback_id):
            raise ResourceNotFoundException(
                detail="Feedback not found", error_code="FEEDBACK_NOT_FOUND"
            )

        # Validate user owns the feedback
        if not self.feedback_validation_utils.validate_user_owns_feedback(
            user_id, feedback_id
        ):
            raise AuthorizationException(
                detail="Access denied to this feedback",
                error_code="FEEDBACK_ACCESS_DENIED",
            )

        updated_feedback = self.repository.update(feedback_id, feedback_update_data)
        return FeedbackRead.model_validate(updated_feedback)

    def delete_feedback(self, feedback_id: UUID, user_id: UUID) -> bool:
        """Delete feedback with ownership validation"""
        if not self.feedback_validation_utils.validate_feedback_exists(feedback_id):
            raise ResourceNotFoundException(
                detail="Feedback not found", error_code="FEEDBACK_NOT_FOUND"
            )

        if not self.feedback_validation_utils.validate_user_owns_feedback(
            user_id, feedback_id
        ):
            raise AuthorizationException(
                detail="Access denied to this feedback",
                error_code="FEEDBACK_ACCESS_DENIED",
            )

        return self.repository.delete(feedback_id)
