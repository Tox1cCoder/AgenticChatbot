"""
Feedback factory for creating Feedback entities
"""

from typing import Any
from uuid import UUID, uuid4

from app.schemas.feedback import FeedbackCreate
from app.utils.timestamp_utils import TimestampUtils


class FeedbackFactory:
    """Factory for creating Feedback entities"""

    @staticmethod
    def create_from_schema(feedback_data: FeedbackCreate, submitter_id: UUID) -> dict[str, Any]:
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
