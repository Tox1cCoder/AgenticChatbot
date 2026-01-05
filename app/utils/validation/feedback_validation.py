"""
Feedback validation utilities
"""

from uuid import UUID

from app.repositories.feedback import FeedbackRepository
from app.utils.validation.base_validation import BaseValidationUtils


class FeedbackValidationUtils(BaseValidationUtils):
    """Utilities for feedback-related validations"""

    def _init_repositories(self):
        """Initialize feedback repository"""
        self.feedback_repository = FeedbackRepository(self.session_factory)

    def validate_feedback_exists(self, feedback_id: UUID) -> bool:
        """
        Validate that a feedback entry exists.

        Returns:
            bool: True if feedback exists, False otherwise
        """
        return self.feedback_repository.exists(feedback_id)

    def validate_user_owns_feedback(self, user_id: UUID, feedback_id: UUID) -> bool:
        """
        Validate that a user owns a specific feedback entry.

        Returns:
            bool: True if user owns the feedback, False otherwise
        """
        return self.feedback_repository.user_owns_feedback(user_id, feedback_id)
