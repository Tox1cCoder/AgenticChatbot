from __future__ import annotations
from typing import List, TYPE_CHECKING
from uuid import UUID
from fastapi import HTTPException, status

from app.repositories.conversation import ConversationRepository
from app.repositories.user import UserRepository
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationRead,
)
from app.factories.conversation_factory import ConversationFactory

if TYPE_CHECKING:
    from app.core.container import DIContainer


class ConversationService:
    """Service layer for Conversation operations"""

    def __init__(
        self,
        container: DIContainer,
        conversation_repository: ConversationRepository,
        user_repository: UserRepository,
    ):
        """
        Initialize ConversationService with injected dependencies.

        Args:
            container: DI container for additional dependency resolution
            conversation_repository: Injected conversation repository
            user_repository: Injected user repository
        """
        self.container = container
        self.repository = conversation_repository
        self.user_repository = user_repository

    def create_conversation(
        self, conversation_create_data: ConversationCreate, owner_id: UUID
    ) -> ConversationRead:
        """Create a new conversation (authentication handled at API layer)"""
        # API layer authentication ensures user exists - no duplicate validation needed

        # Create conversation entity using factory
        conversation_entity = ConversationFactory.create_from_schema(
            conversation_create_data, owner_id
        )

        # Save to repository
        created_conversation = self.repository.create(conversation_entity)
        return ConversationRead.model_validate(created_conversation)

    def get_conversation_by_id(self, conversation_id: UUID) -> ConversationRead:
        """Get conversation by ID"""
        conversation_entity = self.repository.get_by_id(conversation_id)
        if not conversation_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )
        return ConversationRead.model_validate(conversation_entity)

    def get_user_conversations(
        self, owner_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[ConversationRead]:
        """Get all conversations for a user (authentication handled at API layer)"""
        # API layer authentication ensures user exists - no duplicate validation needed

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
        if not self.repository.user_owns_conversation(owner_id, conversation_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation",
            )

        conversation_entity = self.repository.get_with_messages(conversation_id)
        if not conversation_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )
        return ConversationRead.model_validate(conversation_entity)

    def update_conversation(
        self,
        conversation_id: UUID,
        owner_id: UUID,
        conversation_update_data: ConversationUpdate,
    ) -> ConversationRead:
        """Update conversation with ownership validation"""
        if not self.repository.user_owns_conversation(owner_id, conversation_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation",
            )

        conversation_entity = self.repository.get_by_id(conversation_id)
        if not conversation_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )

        updated_conversation = self.repository.update(
            conversation_entity, conversation_update_data
        )
        return ConversationRead.model_validate(updated_conversation)

    def delete_conversation(self, conversation_id: UUID, owner_id: UUID) -> bool:
        """Delete conversation with ownership validation"""
        if not self.repository.user_owns_conversation(owner_id, conversation_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation",
            )

        return self.repository.delete(conversation_id)
