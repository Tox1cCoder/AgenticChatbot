"""
User validation utilities
"""

from typing import Optional
from uuid import UUID
from contextlib import AbstractContextManager
from sqlalchemy.orm import Session

from app.repositories.user import UserRepository


class UserValidationUtils:
    """Utilities for user-related validations"""

    def __init__(self, session_factory: callable):
        """Initialize validation utils with session factory for dependency injection."""
        self.session_factory = session_factory
        self.user_repository = UserRepository(session_factory)

    def is_email_available(
        self, email: str, exclude_user_id: Optional[UUID] = None
    ) -> bool:
        """Check if email is available for use"""
        return not self.user_repository.email_exists(email, exclude_id=exclude_user_id)

    def is_username_available(
        self, username: str, exclude_user_id: Optional[UUID] = None
    ) -> bool:
        """Check if username is available for use"""
        return not self.user_repository.username_exists(
            username, exclude_id=exclude_user_id
        )

    def validate_user_exists(self, user_id: UUID) -> bool:
        """Validate that a user exists"""
        return self.user_repository.exists(user_id)

    def validate_email_and_username_availability(
        self, email: str, username: str, exclude_user_id: Optional[UUID] = None
    ) -> tuple[bool, list[str]]:
        """
        Validate both email and username availability
        Returns (is_valid, errors_list)
        """
        errors = []

        if not self.is_email_available(email, exclude_user_id):
            errors.append("Email is already registered")

        if not self.is_username_available(username, exclude_user_id):
            errors.append("Username is already taken")

        return len(errors) == 0, errors
