"""
Conversation validation utilities
"""

from collections.abc import Callable
from typing import Any
from uuid import UUID

from app.core.exceptions import AuthorizationException, ResourceNotFoundException
from app.repositories.conversation import ConversationRepository


class ConversationValidationUtils:
    """Utilities for conversation-related validations"""

    def __init__(
        self,
        session_factory: callable,
        async_session_factory: Callable[[], Any] | None = None,
    ):
        """Initialize validation utils with session factory for dependency injection."""
        self.session_factory = session_factory
        self.conversation_repository = ConversationRepository(
            session_factory,
            async_session_factory=async_session_factory,
        )

    def validate_conversation_exists(self, conversation_id: UUID):
        """Validate that a conversation exists"""
        if not self.conversation_repository.exists(conversation_id):
            raise ResourceNotFoundException(
                detail="Conversation not found", error_code="CONVERSATION_NOT_FOUND"
            )

    def validate_user_owns_conversation(self, user_id: UUID, conversation_id: UUID):
        """Validate that a user owns a specific conversation"""
        if not self.conversation_repository.user_owns_conversation(user_id, conversation_id):
            raise AuthorizationException(
                detail="Access denied to this conversation",
                error_code="CONVERSATION_ACCESS_DENIED",
            )

    def validate_conversation_access(self, user_id: UUID, conversation_id: UUID):
        """
        Validate conversation exists and user has access
        """
        self.validate_conversation_exists(conversation_id)
        self.validate_user_owns_conversation(user_id, conversation_id)

    async def avalidate_conversation_exists(self, conversation_id: UUID):
        """Async twin of :meth:`validate_conversation_exists`."""
        if not await self.conversation_repository.aexists(conversation_id):
            raise ResourceNotFoundException(
                detail="Conversation not found", error_code="CONVERSATION_NOT_FOUND"
            )

    async def avalidate_user_owns_conversation(self, user_id: UUID, conversation_id: UUID):
        """Async twin of :meth:`validate_user_owns_conversation`."""
        if not await self.conversation_repository.auser_owns_conversation(user_id, conversation_id):
            raise AuthorizationException(
                detail="Access denied to this conversation",
                error_code="CONVERSATION_ACCESS_DENIED",
            )

    async def avalidate_conversation_access(self, user_id: UUID, conversation_id: UUID):
        """Async twin of :meth:`validate_conversation_access`.

        The two checks stay separate and ordered exactly as in the sync path so
        a missing conversation still raises ``ResourceNotFoundException`` rather
        than ``AuthorizationException``.
        """
        await self.avalidate_conversation_exists(conversation_id)
        await self.avalidate_user_owns_conversation(user_id, conversation_id)
