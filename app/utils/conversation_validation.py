"""
Conversation validation utilities
"""

from typing import Optional
from uuid import UUID
from contextlib import AbstractContextManager
from sqlalchemy.orm import Session

from app.repositories.conversation import ConversationRepository
from app.repositories.user import UserRepository


class ConversationValidationUtils:
    """Utilities for conversation-related validations"""

    def __init__(self, session_factory: callable):
        """Initialize validation utils with session factory for dependency injection."""
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
