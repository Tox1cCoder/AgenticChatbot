"""
User validation utilities
"""

from typing import Optional
from uuid import UUID

from app.repositories.user import UserRepository
from app.core.exceptions import ValidationException, ResourceNotFoundException
from app.utils.validation.base_validation import BaseValidationUtils


class UserValidationUtils(BaseValidationUtils):
    """Utilities for user-related validations"""

    def _init_repositories(self):
        """Initialize user repository"""
        self.user_repository = UserRepository(self.session_factory)

    def validate_email_availability(
        self, email: str, exclude_user_id: Optional[UUID] = None
    ):
        """
        Validate that an email is available.

        Raises:
            ValidationException: If email is already registered
        """
        if self.user_repository.email_exists(email, exclude_id=exclude_user_id):
            raise ValidationException(
                detail="Email is already registered", error_code="EMAIL_NOT_AVAILABLE"
            )

    def validate_username_availability(
        self, username: str, exclude_user_id: Optional[UUID] = None
    ):
        """
        Validate that a username is available.

        Raises:
            ValidationException: If username is already taken
        """
        if self.user_repository.username_exists(username, exclude_id=exclude_user_id):
            raise ValidationException(
                detail="Username is already taken", error_code="USERNAME_NOT_AVAILABLE"
            )

    def validate_user_exists(self, user_id: UUID):
        """
        Validate that a user exists.

        Raises:
            ResourceNotFoundException: If user is not found
        """
        if not self.user_repository.exists(user_id):
            raise ResourceNotFoundException(
                detail="User not found", error_code="USER_NOT_FOUND"
            )

    def validate_email_and_username_availability(
        self, email: str, username: str, exclude_user_id: Optional[UUID] = None
    ):
        """
        Validate both email and username availability.

        Raises:
            ValidationException: If email or username is not available
        """
        self.validate_email_availability(email, exclude_user_id)
        self.validate_username_availability(username, exclude_user_id)
