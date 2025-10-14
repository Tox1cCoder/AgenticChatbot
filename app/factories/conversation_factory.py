"""
Conversation factory for creating Conversation entities
"""

from typing import Dict, Any
from uuid import uuid4, UUID

from app.schemas.conversation import ConversationCreate
from app.utils.timestamp_utils import TimestampUtils


class ConversationFactory:
    """Factory for creating Conversation entities"""

    @staticmethod
    def create_from_schema(
        conversation_data: ConversationCreate, owner_id: UUID
    ) -> Dict[str, Any]:
        """Create Conversation data dictionary from ConversationCreate schema"""
        timestamps = TimestampUtils.get_timestamp_dict()
        return {
            "id": uuid4(),
            "owner_id": owner_id,
            "title": conversation_data.title,
            **timestamps,
        }

    @staticmethod
    def create_from_dict(conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create Conversation data dictionary from dictionary"""
        timestamps = TimestampUtils.get_timestamp_dict(
            created_at=conversation_data.get("created_at"),
            updated_at=conversation_data.get("updated_at"),
        )

        return {
            "id": conversation_data.get("id", uuid4()),
            "owner_id": conversation_data["owner_id"],
            "title": conversation_data["title"],
            **timestamps,
        }
