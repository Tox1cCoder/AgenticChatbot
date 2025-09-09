"""
Feedback factory for creating Feedback entities
"""

from typing import Dict, Any
from uuid import uuid4, UUID
from datetime import datetime, timezone

from app.models.feedback import Feedback
from app.schemas.feedback import FeedbackCreate


class FeedbackFactory:
    """Factory for creating Feedback entities"""

    @staticmethod
    def create_from_schema(
        feedback_data: FeedbackCreate, submitter_id: UUID
    ) -> Feedback:
        """Create Feedback entity from FeedbackCreate schema"""
        return Feedback(
            id=uuid4(),
            message_id=feedback_data.message_id,
            user_id=submitter_id,
            rating=feedback_data.rating,
            comment=feedback_data.comment,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def create_from_dict(feedback_data: Dict[str, Any]) -> Feedback:
        """Create Feedback entity from dictionary"""
        now = datetime.now(timezone.utc)

        return Feedback(
            id=feedback_data.get("id", uuid4()),
            message_id=feedback_data["message_id"],
            user_id=feedback_data["user_id"],
            rating=feedback_data["rating"],
            comment=feedback_data.get("comment"),
            created_at=feedback_data.get("created_at", now),
            updated_at=feedback_data.get("updated_at", now),
        )
