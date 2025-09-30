"""
Message service interface definition
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from app.repositories.utils.pagination import Paginator
from app.schemas.message import MessageCreate, MessageUpdate, MessageRead


class IMessageService(ABC):
    """Interface for Message service operations"""

    @abstractmethod
    async def create_message(self, message_create_data: MessageCreate) -> MessageRead:
        """Create a new message with validation"""
        pass

    @abstractmethod
    def get_by_id(self, message_id: UUID, user_id: UUID) -> MessageRead:
        """Get message by ID with access validation"""
        pass

    @abstractmethod
    def get_conversation_messages(
        self,
        conversation_id: UUID,
        user_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: Optional[str] = None,
        order_direction: str = "asc",
    ) -> Paginator[MessageRead]:
        """Get all messages in a conversation with pagination"""
        pass

    @abstractmethod
    def get_user_messages(
        self,
        user_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: Optional[str] = None,
        order_direction: str = "desc",
    ) -> Paginator[MessageRead]:
        """Get all messages by a user with pagination"""
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
