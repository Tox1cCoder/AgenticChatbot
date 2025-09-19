"""
Message validation utilities
"""

from uuid import UUID

from app.repositories.message import MessageRepository
from app.repositories.conversation import ConversationRepository
from app.core.exceptions import ResourceNotFoundException, AuthorizationException
from app.utils.validation.base_validation import BaseValidationUtils


class MessageValidationUtils(BaseValidationUtils):
    """Utilities for message-related validations"""

    def _init_repositories(self):
        """Initialize message and conversation repositories"""
        self.message_repository = MessageRepository(self.session_factory)
        self.conversation_repository = ConversationRepository(self.session_factory)

    def validate_message_exists(self, message_id: UUID):
        """
        Validate that a message exists.

        Raises:
            ResourceNotFoundException: If message is not found
        """
        if not self.message_repository.exists(message_id):
            raise ResourceNotFoundException(
                detail="Message not found", error_code="MESSAGE_NOT_FOUND"
            )

    def validate_message_access(self, user_id: UUID, message_id: UUID):
        """
        Validate message exists and user has access through conversation ownership.

        Raises:
            ResourceNotFoundException: If message is not found
            AuthorizationException: If user does not have access to the conversation
        """
        self.validate_message_exists(message_id)
        message = self.message_repository.get_by_id(message_id)
        if not self.conversation_repository.user_owns_conversation(
            user_id, message.conversation_id
        ):
            raise AuthorizationException(
                detail="Access denied to this conversation",
                error_code="CONVERSATION_ACCESS_DENIED",
            )
