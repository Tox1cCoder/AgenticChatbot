"""
Message validation utilities
"""

from typing import Optional
from uuid import UUID
from contextlib import AbstractContextManager
from sqlalchemy.orm import Session

from app.repositories.message import MessageRepository
from app.repositories.conversation import ConversationRepository


class MessageValidationUtils:
    """Utilities for message-related validations"""

    def __init__(self, session_factory: callable):
        """Initialize validation utils with session factory for dependency injection."""
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
