"""
Message service interface definition
"""

from abc import ABC, abstractmethod
from typing import List
from uuid import UUID

from app.schemas.message import MessageCreate, MessageUpdate, MessageRead


class IMessageService(ABC):
    """Interface for Message service operations"""

    @abstractmethod
    def create_message(
        self, message_create_data: MessageCreate, conversation_id: UUID, user_id: UUID
    ) -> MessageRead:
        """Create a new message with validation"""
        pass

    @abstractmethod
    def get_by_id(self, message_id: UUID, user_id: UUID) -> MessageRead:
        """Get message by ID with access validation"""
        pass

    @abstractmethod
    def get_conversation_messages(
        self, conversation_id: UUID, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[MessageRead]:
        """Get all messages in a conversation with access validation"""
        pass

    @abstractmethod
    def get_user_messages(
        self, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[MessageRead]:
        """Get all messages by a user"""
        pass

    @abstractmethod
    def update_message(
        self, message_id: UUID, user_id: UUID, message_update_data: MessageUpdate
    ) -> MessageRead:
        """Update message with ownership validation"""
        pass

    @abstractmethod
    def delete_message(self, message_id: UUID, user_id: UUID) -> bool:
        """Delete message with ownership validation"""
        pass
