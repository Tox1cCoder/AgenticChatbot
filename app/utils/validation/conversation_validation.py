"""
Conversation validation utilities
"""

from uuid import UUID

from app.repositories.conversation import ConversationRepository
from app.core.exceptions import ResourceNotFoundException, AuthorizationException


class ConversationValidationUtils:
    """Utilities for conversation-related validations"""

    def __init__(self, session_factory: callable):
        """Initialize validation utils with session factory for dependency injection."""
        self.session_factory = session_factory
        self.conversation_repository = ConversationRepository(session_factory)

    def validate_conversation_exists(self, conversation_id: UUID):
        """Validate that a conversation exists"""
        if not self.conversation_repository.exists(conversation_id):
            raise ResourceNotFoundException(
                detail="Conversation not found", error_code="CONVERSATION_NOT_FOUND"
            )

    def validate_user_owns_conversation(self, user_id: UUID, conversation_id: UUID):
        """Validate that a user owns a specific conversation"""
        if not self.conversation_repository.user_owns_conversation(
            user_id, conversation_id
        ):
            raise AuthorizationException(
                detail="Access denied to this conversation",
                error_code="CONVERSATION_ACCESS_DENIED",
            )

    def validate_conversation_access(self, user_id: UUID, conversation_id: UUID):
        """
        Validate conversation exists and user has access
        """
        self.validate_conversation_exists(conversation_id)
        self.validate_user_owns_conversation(user_id, conversation_id)