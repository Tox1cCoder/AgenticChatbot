"""
Message factory for creating Message entities
"""

from typing import Dict, Any, Optional
from uuid import uuid4, UUID

from app.models.message import Message
from app.schemas.message import MessageCreate
from app.models.enums import MessageRole
from app.utils.timestamp_utils import TimestampUtils


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
        message_data: MessageCreate, role: MessageRole
    ) -> Dict[str, Any]:
        """Create Message data dictionary from MessageCreate schema with specified role"""
        return {
            "id": uuid4(),
            "conversation_id": message_data.conversation_id,
            "sender": role.value,
            "content": message_data.content,
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
    def create_bot_response(conversation_id: UUID, content: str) -> Dict[str, Any]:
        """Create bot response message data dictionary"""
        return {
            "id": uuid4(),
            "conversation_id": conversation_id,
            "sender": MessageRole.assistant.value,
            "content": content,
            "created_at": TimestampUtils.now(),
        }
