"""
Conversation factory for creating Conversation entities
"""

from typing import Dict, Any
from uuid import uuid4, UUID
from datetime import datetime, timezone

from app.models.conversation import Conversation
from app.schemas.conversation import ConversationCreate


class ConversationFactory:
    """Factory for creating Conversation entities"""

    @staticmethod
    def create_from_schema(
        conversation_data: ConversationCreate, owner_id: UUID
    ) -> Dict[str, Any]:
        """Create Conversation data dictionary from ConversationCreate schema"""
        return {
            "id": uuid4(),
            "owner_id": owner_id,
            "title": conversation_data.title,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

    @staticmethod
    def create_from_dict(conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create Conversation data dictionary from dictionary"""
        now = datetime.now(timezone.utc)

        return {
            "id": conversation_data.get("id", uuid4()),
            "owner_id": conversation_data["owner_id"],
            "title": conversation_data["title"],
            "created_at": conversation_data.get("created_at", now),
            "updated_at": conversation_data.get("updated_at", now),
        }
