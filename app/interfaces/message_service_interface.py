"""
Message service interface definition
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import UUID

from app.ai.schemas import InterruptDecision
from app.repositories.utils.pagination import Paginator
from app.schemas.message import MessageCreate, MessageUpdate, MessageRead


class IMessageService(ABC):
    """Interface for Message service operations"""

    @abstractmethod
    async def create_message(
        self, message_create_data: MessageCreate, user_id: UUID
    ) -> MessageRead:
        """Create a new message with validation"""
        pass

    @abstractmethod
    async def create_message_stream(
        self,
        message_create_data: MessageCreate,
        user_id: UUID,
        bot_message_id: Optional[UUID] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Create a new message and stream the assistant response"""
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
        include_feedback: bool = False,
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
        include_feedback: bool = False,
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

    @abstractmethod
    async def resume_workflow(
        self, conversation_id: UUID, user_id: UUID, user_input: Optional[str] = None
    ) -> MessageRead:
        """Resume a paused workflow and return the bot's response message"""
        pass

    @abstractmethod
    async def resume_message_creation_stream(
        self,
        thread_id: str,
        conversation_id: UUID,
        user_id: UUID,
        decisions: List[InterruptDecision],
        interrupt_id: Optional[str] = None,
        bot_message_id: Optional[UUID] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Resume an interrupted workflow and stream the assistant response"""
        pass

    @abstractmethod
    async def stop_message_generation(
        self,
        conversation_id: UUID,
        user_id: UUID,
        user_message_id: UUID,
    ) -> dict:
        """
        Request cancellation of an in-flight streaming generation.

        Returns a dict with:
          - ``status``: ``"cancelled"`` | ``"not_inflight"``
          - ``message``: optional ``MessageRead`` (the persisted partial/final message)
        """
        pass
