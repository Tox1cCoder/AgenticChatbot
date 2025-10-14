"""
Conversation service interface definition
"""

from abc import ABC, abstractmethod
from typing import List
from uuid import UUID

from app.repositories.utils.pagination import Paginator
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationRead,
)


class IConversationService(ABC):
    """Interface for Conversation service operations"""

    @abstractmethod
    def create_conversation(
        self, conversation_create_data: ConversationCreate, owner_id: UUID
    ) -> ConversationRead:
        """Create a new conversation with validation"""
        pass

    @abstractmethod
    def get_by_id(self, conversation_id: UUID) -> ConversationRead:
        """Get conversation by ID"""
        pass

    @abstractmethod
    def get_by_user_id(
        self,
        owner_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: str = "updated_at",
        order_direction: str = "desc",
        include: List[str] = None,
        latest_messages: int = 3,
    ) -> Paginator[ConversationRead]:
        """Get all conversations for a user with optional includes"""
        pass

    @abstractmethod
    def get_conversation_with_messages(
        self, conversation_id: UUID, owner_id: UUID
    ) -> ConversationRead:
        """Get conversation with messages, ensuring user owns it"""
        pass

    @abstractmethod
    def update_conversation(
        self,
        conversation_id: UUID,
        owner_id: UUID,
        conversation_update_data: ConversationUpdate,
    ) -> ConversationRead:
        """Update conversation with ownership validation"""
        pass

    @abstractmethod
    def delete_conversation(self, conversation_id: UUID, owner_id: UUID) -> bool:
        """Delete conversation with ownership validation"""
        pass
