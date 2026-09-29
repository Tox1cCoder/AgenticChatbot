"""
User factory for creating User entities
"""

from typing import Any
from uuid import uuid4

from app.core.security import hash_password
from app.schemas.user import UserCreate
from app.utils.timestamp_utils import TimestampUtils


class UserFactory:
    """Factory for creating User entities"""

    @staticmethod
    def create_from_schema(user_data: UserCreate) -> dict[str, Any]:
        """Create User data dictionary from UserCreate schema"""
        timestamps = TimestampUtils.get_timestamp_dict()
        return {
            "id": uuid4(),
            "username": user_data.username,
            "email": user_data.email,
            "password_hash": hash_password(user_data.password),
            "avatar_url": user_data.avatar_url,
            **timestamps,
        }
