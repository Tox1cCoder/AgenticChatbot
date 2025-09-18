"""
Feedback validation utilities
"""

from uuid import UUID

from app.repositories.feedback import FeedbackRepository


class FeedbackValidationUtils:
    """Utilities for feedback-related validations"""

    def __init__(self, session_factory: callable):
        """Initialize validation utils with session factory for dependency injection."""
        self.session_factory = session_factory
        self.feedback_repository = FeedbackRepository(session_factory)

    def validate_feedback_exists(self, feedback_id: UUID) -> bool:
        """Validate that a feedback entry exists"""
        return self.feedback_repository.exists(feedback_id)

    def validate_user_owns_feedback(
        self, user_id: UUID, feedback_id: UUID
    ) -> bool:
        """Validate that a user owns a specific feedback entry"""
        return self.feedback_repository.user_owns_feedback(
            user_id, feedback_id
        )
