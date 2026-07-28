from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any
from uuid import UUID

from app.factories.conversation_factory import ConversationFactory
from app.interfaces.conversation_service_interface import IConversationService
from app.repositories.conversation import ConversationRepository
from app.repositories.utils.pagination import Paginator
from app.schemas.conversation import (
    ConversationCreate,
    ConversationRead,
    ConversationUpdate,
)
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.utils.validation.pagination_validation import validate_pagination_params
from app.utils.validation.user_validation import UserValidationUtils

logger = logging.getLogger(__name__)


class ConversationService(IConversationService):
    """Service layer for Conversation operations"""

    def __init__(
        self,
        conversation_repository: ConversationRepository,
        user_validation_utils: UserValidationUtils,
        conversation_validation_utils: ConversationValidationUtils,
        ai_service: Any | None = None,
        checkpoint_manager: Any | None = None,
    ):
        self.repository = conversation_repository
        self.user_validation_utils = user_validation_utils
        self.conversation_validation_utils = conversation_validation_utils
        # Optional dependencies (Memory Refactor 2026-04-29) — used to clear
        # in-process memory caches and LangGraph checkpoint state when a
        # conversation is deleted. Wired through the DI container.
        self.ai_service = ai_service
        self.checkpoint_manager = checkpoint_manager

    def _convert_to_read_schema(
        self, conversation_entity, include: list[str] = None
    ) -> ConversationRead:
        if include is None:
            include = []

        conv_dict = {
            "id": conversation_entity.id,
            "created_at": conversation_entity.created_at,
            "updated_at": conversation_entity.updated_at,
            "deleted_at": conversation_entity.deleted_at,
            "owner_id": conversation_entity.owner_id,
            "title": conversation_entity.title,
            "persona_prompt": conversation_entity.persona_prompt,
            "planning_mode_enabled": getattr(conversation_entity, "planning_mode_enabled", False),
            "plan_lifecycle": getattr(conversation_entity, "plan_lifecycle", None),
        }

        if hasattr(conversation_entity, "message_count"):
            conv_dict["message_count"] = conversation_entity.message_count
        else:
            conv_dict["message_count"] = None

        if "messages" in include:
            try:
                messages = conversation_entity.__dict__.get("messages")

                if messages is not None:
                    from app.schemas.message import MessageRead

                    conv_dict["messages"] = [
                        MessageRead.model_validate(message) for message in list(messages)
                    ]
                else:
                    conv_dict["messages"] = None
            except Exception:
                conv_dict["messages"] = None
        else:
            conv_dict["messages"] = None

        return ConversationRead.model_validate(conv_dict)

    def create_conversation(
        self, conversation_create_data: ConversationCreate, owner_id: UUID
    ) -> ConversationRead:
        self.user_validation_utils.validate_user_exists(owner_id)
        conversation_entity = ConversationFactory.create_from_schema(
            conversation_create_data, owner_id
        )
        created_conversation = self.repository.create(conversation_entity)
        return self._convert_to_read_schema(created_conversation, include=[])

    def get_by_id(self, conversation_id: UUID) -> ConversationRead:
        self.conversation_validation_utils.validate_conversation_exists(conversation_id)
        conversation_entity = self.repository.get_by_id(conversation_id)
        return self._convert_to_read_schema(conversation_entity, include=[])

    def get_by_id_for_user(self, conversation_id: UUID, owner_id: UUID) -> ConversationRead:
        self.conversation_validation_utils.validate_conversation_access(owner_id, conversation_id)
        conversation_entity = self.repository.get_by_id(conversation_id)
        return self._convert_to_read_schema(conversation_entity, include=[])

    def get_by_user_id(
        self,
        owner_id: UUID,
        page: int = 1,
        limit: int = 10,
        order_by: str = "updated_at",
        order_direction: str = "desc",
        include: list[str] = None,
        latest_messages: int = 3,
        search: str | None = None,
    ) -> Paginator[ConversationRead]:
        """Get user conversations with optional includes"""
        if include is None:
            include = []

        # Validate pagination parameters
        validate_pagination_params(page, limit)

        # Validate order_by field
        valid_order_fields = ["created_at", "updated_at"]
        if order_by not in valid_order_fields:
            raise ValueError(f"Invalid order_by field. Must be one of: {valid_order_fields}")

        # Validate order_direction
        valid_directions = ["asc", "desc"]
        if order_direction.lower() not in valid_directions:
            raise ValueError(f"Invalid order_direction. Must be one of: {valid_directions}")

        normalized_search = search.strip().lower() if isinstance(search, str) else ""

        self.user_validation_utils.validate_user_exists(owner_id)
        paginated_conversations = self.repository.get_by_owner_id(
            owner_id,
            page=page,
            limit=limit,
            order_by=order_by,
            order_direction=order_direction,
            include=include,
            latest_messages=latest_messages,
            search=normalized_search or None,
        )
        # Convert items to ConversationRead schemas
        conversation_reads = [
            self._convert_to_read_schema(conversation_entity, include=include)
            for conversation_entity in paginated_conversations.items
        ]

        # Return new Paginator with converted items
        return Paginator.create(conversation_reads, paginated_conversations.meta.total, page, limit)

    def get_conversation_with_messages(
        self, conversation_id: UUID, owner_id: UUID
    ) -> ConversationRead:
        self.conversation_validation_utils.validate_conversation_access(owner_id, conversation_id)
        conversation_entity = self.repository.get_with_messages(conversation_id)
        return self._convert_to_read_schema(conversation_entity, include=["messages"])

    def update_conversation(
        self,
        conversation_id: UUID,
        owner_id: UUID,
        conversation_update_data: ConversationUpdate,
    ) -> ConversationRead:
        self.conversation_validation_utils.validate_conversation_access(owner_id, conversation_id)
        conversation_entity = self.repository.get_by_id(conversation_id)
        updated_conversation = self.repository.update(
            conversation_entity.id, conversation_update_data
        )
        return self._convert_to_read_schema(updated_conversation, include=[])

    def delete_conversation(self, conversation_id: UUID, owner_id: UUID) -> bool:
        self.conversation_validation_utils.validate_conversation_access(owner_id, conversation_id)
        deleted = self.repository.delete(conversation_id)
        if deleted:
            # Drop cached prompt history so a future re-creation under the
            # same UUID does not see stale memory.
            if self.ai_service is not None:
                with contextlib.suppress(Exception):
                    self.ai_service.invalidate_history_cache(str(conversation_id))
            # Best-effort checkpoint thread cleanup. Errors are non-fatal —
            # the conversation row is already soft-deleted.
            if self.checkpoint_manager is not None:
                with contextlib.suppress(Exception):
                    asyncio.create_task(self._delete_checkpoint_thread_async(str(conversation_id)))
        return deleted

    async def _delete_checkpoint_thread_async(self, thread_id: str) -> None:
        try:
            await self.checkpoint_manager.delete_thread(thread_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Checkpoint cleanup on conversation delete failed (thread=%s): %s",
                thread_id,
                exc,
            )
