"""
User factory for creating User entities
"""

from typing import Dict, Any
from uuid import uuid4

from app.schemas.user import UserCreate
from app.core.security import hash_password
from app.utils.timestamp_utils import TimestampUtils


class UserFactory:
    """Factory for creating User entities"""

    @staticmethod
    def create_from_schema(user_data: UserCreate) -> Dict[str, Any]:
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

    @staticmethod
    def create_from_dict(user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create User data dictionary from dictionary"""
        timestamps = TimestampUtils.get_timestamp_dict(
            created_at=user_data.get("created_at"),
            updated_at=user_data.get("updated_at"),
        )

        return {
            "id": user_data.get("id", uuid4()),
            "username": user_data["username"],
            "email": user_data["email"],
            "password_hash": user_data["password_hash"],
            "avatar_url": user_data.get("avatar_url"),
            **timestamps,
        }
