"""
Message factory for creating Message entities
"""

from __future__ import annotations

from typing import Dict, Any
from uuid import uuid4, UUID

from app.schemas.message import MessageCreate
from app.models.enums import MessageRole
from app.utils.timestamp_utils import TimestampUtils
from app.core.response_constants import normalize_message_content


class MessageFactory:
    """Factory for creating Message entities"""

    @staticmethod
    def create_from_schema(message_data: MessageCreate) -> Dict[str, Any]:
        """Create Message data dictionary from MessageCreate schema"""
        return {
            "id": uuid4(),
            "conversation_id": message_data.conversation_id,
            "sender": MessageRole.user.value,  # Default role assignment
            "content": message_data.content,
            "created_at": TimestampUtils.now(),
        }

    @staticmethod
    def create_from_schema_with_role(
        message_data: MessageCreate,
        role: MessageRole,
        message_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Create Message data dictionary from MessageCreate schema with specified role"""
        metadata = dict(message_metadata) if message_metadata else {}

        # Store attachments in metadata if present
        if hasattr(message_data, "attachments") and message_data.attachments:
            metadata["attachments"] = message_data.attachments
        if (
            hasattr(message_data, "model_config_field")
            and message_data.model_config_field
        ):
            metadata["model_request"] = message_data.model_config_field

        return {
            "id": uuid4(),
            "conversation_id": message_data.conversation_id,
            "sender": role.value,
            "content": message_data.content,
            "message_metadata": metadata,
            "created_at": TimestampUtils.now(),
        }

    @staticmethod
    def create_from_dict(message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create Message data dictionary from dictionary"""
        return {
            "id": message_data.get("id", uuid4()),
            "conversation_id": message_data["conversation_id"],
            "sender": message_data["sender"],
            "content": message_data["content"],
            "created_at": message_data.get("created_at", TimestampUtils.now()),
        }

    @staticmethod
    def _normalize_content(
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        """Wrapper for backward compatibility."""
        return normalize_message_content(content, metadata)

    @staticmethod
    def create_bot_response(
        conversation_id: UUID,
        content: str,
        message_metadata: Dict[str, Any] | None = None,
        id: UUID | None = None,
    ) -> Dict[str, Any]:
        """Create bot response message data dictionary"""
        metadata = dict(message_metadata) if message_metadata else {}
        return {
            "id": id or uuid4(),
            "conversation_id": conversation_id,
            "sender": MessageRole.assistant.value,
            "content": MessageFactory._normalize_content(content, metadata),
            "message_metadata": metadata,
            "created_at": TimestampUtils.now(),
        }
