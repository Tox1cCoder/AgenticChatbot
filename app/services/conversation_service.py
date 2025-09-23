from __future__ import annotations
from typing import List, Optional
from uuid import UUID

from app.repositories.conversation import ConversationRepository
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationRead,
)
from app.factories.conversation_factory import ConversationFactory
from app.utils.validation.user_validation import UserValidationUtils
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.interfaces.conversation_service_interface import IConversationService


class ConversationService(IConversationService):
    """Service layer for Conversation operations"""

    def __init__(
        self,
        conversation_repository: ConversationRepository,
        user_validation_utils: UserValidationUtils,
        conversation_validation_utils: ConversationValidationUtils,
    ):
        self.repository = conversation_repository
        self.user_validation_utils = user_validation_utils
        self.conversation_validation_utils = conversation_validation_utils

    def create_conversation(
        self, conversation_create_data: ConversationCreate, owner_id: UUID
    ) -> ConversationRead:
        self.user_validation_utils.validate_user_exists(owner_id)
        conversation_entity = ConversationFactory.create_from_schema(
            conversation_create_data, owner_id
        )
        created_conversation = self.repository.create(conversation_entity)
        return ConversationRead.model_validate(created_conversation)

    def get_by_id(self, conversation_id: UUID) -> ConversationRead:
        self.conversation_validation_utils.validate_conversation_exists(conversation_id)
        conversation_entity = self.repository.get_by_id(conversation_id)
        return ConversationRead.model_validate(conversation_entity)

    def get_user_conversations(
        self,
        owner_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: Optional[str] = None,
        order_direction: str = "desc",
    ) -> List[ConversationRead]:
        self.user_validation_utils.validate_user_exists(owner_id)
        conversation_entities = self.repository.get_by_owner_id(
            owner_id,
            page=page,
            limit=limit,
            order_by=order_by,
            order_direction=order_direction,
        )
        return [
            ConversationRead.model_validate(conversation_entity)
            for conversation_entity in conversation_entities
        ]

    def get_conversation_with_messages(
        self, conversation_id: UUID, owner_id: UUID
    ) -> ConversationRead:
        self.conversation_validation_utils.validate_conversation_access(
            owner_id, conversation_id
        )
        conversation_entity = self.repository.get_with_messages(conversation_id)
        return ConversationRead.model_validate(conversation_entity)

    def update_conversation(
        self,
        conversation_id: UUID,
        owner_id: UUID,
        conversation_update_data: ConversationUpdate,
    ) -> ConversationRead:
        self.conversation_validation_utils.validate_conversation_access(
            owner_id, conversation_id
        )
        conversation_entity = self.repository.get_by_id(conversation_id)
        updated_conversation = self.repository.update(
            conversation_entity.id, conversation_update_data
        )
        return ConversationRead.model_validate(updated_conversation)

    def delete_conversation(self, conversation_id: UUID, owner_id: UUID) -> bool:
        self.conversation_validation_utils.validate_conversation_access(
            owner_id, conversation_id
        )
        return self.repository.delete(conversation_id)
