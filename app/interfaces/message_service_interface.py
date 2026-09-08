"""
Message service interface definition
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from uuid import UUID

from app.repositories.utils.pagination import Paginator
from app.schemas.message import MessageCreate, MessageRead, MessageUpdate
from app.schemas.workflow import InterruptDecision
from app.services.event_streaming.events import V3StreamEvent


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
        bot_message_id: UUID | None = None,
    ) -> AsyncGenerator[V3StreamEvent, None]:
        """Create a new message and stream the assistant response as canonical v3 events"""
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
        order_by: str | None = None,
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
        order_by: str | None = None,
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
        self, conversation_id: UUID, user_id: UUID, user_input: str | None = None
    ) -> MessageRead:
        """Resume a paused workflow and return the bot's response message"""
        pass

    @abstractmethod
    async def resume_message_creation_stream(
        self,
        thread_id: str,
        conversation_id: UUID,
        user_id: UUID,
        decisions: list[InterruptDecision],
        interrupt_id: str | None = None,
        device_id: UUID | None = None,
        bot_message_id: UUID | None = None,
        inline_rich_response_v1: bool = False,
    ) -> AsyncGenerator[V3StreamEvent, None]:
        """Resume an interrupted workflow and stream canonical v3 assistant events"""
        pass

    @abstractmethod
    async def stop_message_generation(
        self,
        conversation_id: UUID,
        user_id: UUID,
        user_message_id: UUID,
        wait_seconds: float = 5.0,
    ) -> dict:
        """
        Request cancellation of an in-flight streaming generation.

        Returns a dict with:
          - ``status``: ``"cancelled"`` when the producer confirmed,
            ``"stop_requested"`` when it has not answered within
            ``wait_seconds``, ``"not_inflight"`` when no such generation is held
          - ``message``: optional ``MessageRead`` (the persisted partial/final message)

        ``stop_requested`` is a pending state, not a failure: the producer may
        be mid-provider-call, and reporting ``cancelled`` before it confirms
        would claim something no one has verified.
        """
        pass
