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

          - ``generation``: the durable lifecycle snapshot, when the deployment
            has the generation control service wired. This is what a client
            should read; ``status`` above is kept for existing ones.

        ``stop_requested`` is a pending state, not a failure: the producer may
        be mid-provider-call, and reporting ``cancelled`` before it confirms
        would claim something no one has verified.
        """
        pass

    @abstractmethod
    async def stop_generation(
        self,
        *,
        generation_id: UUID,
        conversation_id: UUID,
        user_id: UUID,
        idempotency_key: str,
        expected_version: int,
    ):
        """Stop one generation durably, returning its ``GenerationSnapshot``.

        The canonical Stop. It works when the command lands on a different
        worker than the stream, which is the whole reason the lifecycle row
        exists; :meth:`stop_message_generation` is the turn-scoped entry point
        that resolves a user message id to a generation and delegates here.

        ``expected_version`` fences the command (R5): a delayed replay issued
        against an earlier state is refused rather than executed against
        whatever epoch happens to be running when it arrives.
        """
        pass

    @abstractmethod
    async def continue_message_generation_stream(
        self,
        *,
        generation_id: UUID,
        continuation_id: UUID,
        conversation_id: UUID,
        user_id: UUID,
        idempotency_key: str,
        expected_version: int,
        bot_message_id: UUID | None = None,
        inline_rich_response_v1: bool = False,
    ) -> AsyncGenerator[V3StreamEvent, None]:
        """Resume a paused turn from its exact checkpoint.

        Not a new turn: no user message is appended, the router is not
        consulted, and the specialist the turn already chose is the one that
        resumes. ``continuation_id`` is single-use, so a replayed Continue
        cannot open a second epoch on the same answer.
        """
        pass
