from __future__ import annotations
from typing import List, TYPE_CHECKING
from uuid import UUID

from app.core.exceptions import (
    ValidationException,
    ResourceNotFoundException,
    AuthorizationException,
)
from app.repositories.conversation import ConversationRepository
from app.repositories.user import UserRepository
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationRead,
)
from app.factories.conversation_factory import ConversationFactory
from app.utils.user_validation import UserValidationUtils
from app.utils.conversation_validation import ConversationValidationUtils
from app.interfaces.conversation_service_interface import IConversationService


class ConversationService(IConversationService):
    """Service layer for Conversation operations"""

    def __init__(
        self,
        conversation_repository: ConversationRepository,
        user_repository: UserRepository,
        user_validation_utils: UserValidationUtils,
        conversation_validation_utils: ConversationValidationUtils,
    ):
        """
        Initialize ConversationService with injected dependencies.

        Args:
            conversation_repository: Injected conversation repository
            user_repository: Injected user repository
            user_validation_utils: Injected user validation utils
            conversation_validation_utils: Injected conversation validation utils
        """
        self.repository = conversation_repository
        self.user_repository = user_repository
        self.user_validation_utils = user_validation_utils
        self.conversation_validation_utils = conversation_validation_utils

    def create_conversation(
        self, conversation_create_data: ConversationCreate, owner_id: UUID
    ) -> ConversationRead:
        """Create a new conversation with validation"""
        if not self.user_validation_utils.validate_user_exists(owner_id):
            raise ResourceNotFoundException(
                detail="User not found", error_code="USER_NOT_FOUND"
            )

        conversation_entity = ConversationFactory.create_from_schema(
            conversation_create_data, owner_id
        )
        created_conversation = self.repository.create(conversation_entity)
        return ConversationRead.model_validate(created_conversation)

    def get_by_id(self, conversation_id: UUID) -> ConversationRead:
        """Get conversation by ID"""
        if not self.conversation_validation_utils.validate_conversation_exists(
            conversation_id
        ):
            raise ResourceNotFoundException(
                detail="Conversation not found", error_code="CONVERSATION_NOT_FOUND"
            )

        conversation_entity = self.repository.get_by_id(conversation_id)
        return ConversationRead.model_validate(conversation_entity)

    def get_user_conversations(
        self, owner_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[ConversationRead]:
        """Get all conversations for a user with validation"""
        if not self.user_validation_utils.validate_user_exists(owner_id):
            raise ResourceNotFoundException(
                detail="User not found", error_code="USER_NOT_FOUND"
            )

        conversation_entities = self.repository.get_by_owner_id(
            owner_id, skip=skip, limit=limit
        )
        return [
            ConversationRead.model_validate(conversation_entity)
            for conversation_entity in conversation_entities
        ]

    def get_conversation_with_messages(
        self, conversation_id: UUID, owner_id: UUID
    ) -> ConversationRead:
        """Get conversation with messages, ensuring user owns it"""
        is_valid, validation_errors = (
            self.conversation_validation_utils.validate_conversation_access(
                owner_id, conversation_id
            )
        )

        if not is_valid:
            if "Conversation not found" in validation_errors:
                raise ResourceNotFoundException(
                    detail="Conversation not found", error_code="CONVERSATION_NOT_FOUND"
                )
            else:
                raise AuthorizationException(
                    detail="Access denied to this conversation",
                    error_code="CONVERSATION_ACCESS_DENIED",
                )

        conversation_entity = self.repository.get_with_messages(conversation_id)
        return ConversationRead.model_validate(conversation_entity)

    def update_conversation(
        self,
        conversation_id: UUID,
        owner_id: UUID,
        conversation_update_data: ConversationUpdate,
    ) -> ConversationRead:
        """Update conversation with ownership validation"""
        is_valid, validation_errors = (
            self.conversation_validation_utils.validate_conversation_access(
                owner_id, conversation_id
            )
        )

        if not is_valid:
            if "Conversation not found" in validation_errors:
                raise ResourceNotFoundException(
                    detail="Conversation not found", error_code="CONVERSATION_NOT_FOUND"
                )
            else:
                raise AuthorizationException(
                    detail="Access denied to this conversation",
                    error_code="CONVERSATION_ACCESS_DENIED",
                )

        conversation_entity = self.repository.get_by_id(conversation_id)
        updated_conversation = self.repository.update(
            conversation_entity.id, conversation_update_data
        )
        return ConversationRead.model_validate(updated_conversation)

    def delete_conversation(self, conversation_id: UUID, owner_id: UUID) -> bool:
        """Delete conversation with ownership validation"""
        is_valid, validation_errors = (
            self.conversation_validation_utils.validate_conversation_access(
                owner_id, conversation_id
            )
        )

        if not is_valid:
            if "Conversation not found" in validation_errors:
                raise ResourceNotFoundException(
                    detail="Conversation not found", error_code="CONVERSATION_NOT_FOUND"
                )
            else:
                raise AuthorizationException(
                    detail="Access denied to this conversation",
                    error_code="CONVERSATION_ACCESS_DENIED",
                )

        return self.repository.delete(conversation_id)
