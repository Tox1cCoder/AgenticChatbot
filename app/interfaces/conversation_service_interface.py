"""
Conversation service interface definition
"""

from abc import ABC, abstractmethod
from typing import List
from uuid import UUID

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
    def get_user_conversations(
        self, owner_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[ConversationRead]:
        """Get all conversations for a user with validation"""
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
