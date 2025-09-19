"""
Feedback factory for creating Feedback entities
"""

from typing import Dict, Any
from uuid import uuid4, UUID

from app.models.feedback import Feedback
from app.schemas.feedback import FeedbackCreate
from app.utils.timestamp_utils import TimestampUtils


class FeedbackFactory:
    """Factory for creating Feedback entities"""

    @staticmethod
    def create_from_schema(
        feedback_data: FeedbackCreate, submitter_id: UUID
    ) -> Dict[str, Any]:
        """Create Feedback data dictionary from FeedbackCreate schema"""
        timestamps = TimestampUtils.get_timestamp_dict()
        return {
            "id": uuid4(),
            "message_id": feedback_data.message_id,
            "user_id": submitter_id,
            "rating": feedback_data.rating,
            "comment": feedback_data.comment,
            **timestamps,
        }

    @staticmethod
    def create_from_dict(feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create Feedback data dictionary from dictionary"""
        timestamps = TimestampUtils.get_timestamp_dict(
            created_at=feedback_data.get("created_at"),
            updated_at=feedback_data.get("updated_at"),
        )

        return {
            "id": feedback_data.get("id", uuid4()),
            "message_id": feedback_data["message_id"],
            "user_id": feedback_data["user_id"],
            "rating": feedback_data["rating"],
            "comment": feedback_data.get("comment"),
            **timestamps,
        }
