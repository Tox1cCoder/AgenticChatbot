"""
Validation utilities for service layer
"""

from typing import Optional
from uuid import UUID
from contextlib import AbstractContextManager
from sqlalchemy.orm import Session

from app.repositories.user import UserRepository
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository


class UserValidationService:
    """Service for user-related validations"""

    def __init__(self, session_factory: callable):
        """Initialize validation service with session factory for dependency injection."""
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
            errors.append("Email already registered")

        if not self.is_username_available(username, exclude_user_id):
            errors.append("Username already taken")

        return len(errors) == 0, errors


class ConversationValidationService:
    """Service for conversation-related validations"""

    def __init__(self, session_factory: callable):
        """Initialize validation service with session factory for dependency injection."""
        self.session_factory = session_factory
        self.conversation_repository = ConversationRepository(session_factory)
        self.user_repository = UserRepository(session_factory)

    def validate_conversation_exists(self, conversation_id: UUID) -> bool:
        """Validate that a conversation exists"""
        return self.conversation_repository.exists(conversation_id)

    def validate_user_owns_conversation(
        self, user_id: UUID, conversation_id: UUID
    ) -> bool:
        """Validate that a user owns a specific conversation"""
        return self.conversation_repository.user_owns_conversation(
            user_id, conversation_id
        )

    def validate_conversation_access(
        self, user_id: UUID, conversation_id: UUID
    ) -> tuple[bool, list[str]]:
        """
        Validate conversation exists and user has access
        Returns (is_valid, errors_list)
        """
        errors = []

        if not self.validate_conversation_exists(conversation_id):
            errors.append("Conversation not found")
            return False, errors

        if not self.validate_user_owns_conversation(user_id, conversation_id):
            errors.append("Access denied to this conversation")

        return len(errors) == 0, errors


class MessageValidationService:
    """Service for message-related validations"""

    def __init__(self, session_factory: callable):
        """Initialize validation service with session factory for dependency injection."""
        self.session_factory = session_factory
        self.message_repository = MessageRepository(session_factory)
        self.conversation_repository = ConversationRepository(session_factory)

    def validate_message_exists(self, message_id: UUID) -> bool:
        """Validate that a message exists"""
        return self.message_repository.exists(message_id)

    def validate_message_access(
        self, user_id: UUID, message_id: UUID
    ) -> tuple[bool, list[str]]:
        """
        Validate message exists and user has access through conversation ownership
        Returns (is_valid, errors_list)
        """
        errors = []

        if not self.validate_message_exists(message_id):
            errors.append("Message not found")
            return False, errors

        # Get message to check conversation ownership
        message = self.message_repository.get_by_id(message_id)
        if not message:
            errors.append("Message not found")
            return False, errors

        if not self.conversation_repository.user_owns_conversation(
            user_id, message.conversation_id
        ):
            errors.append("Access denied to this conversation")

        return len(errors) == 0, errors
