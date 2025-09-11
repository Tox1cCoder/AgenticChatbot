"""
User factory for creating User entities
"""

from typing import Dict, Any
from uuid import uuid4
from datetime import datetime, timezone

from app.models.user import User
from app.schemas.user import UserCreate
from app.core.security import hash_password


class UserFactory:
    """Factory for creating User entities"""

    @staticmethod
    def create_from_schema(user_data: UserCreate) -> Dict[str, Any]:
        """Create User data dictionary from UserCreate schema"""
        return {
            "id": uuid4(),
            "username": user_data.username,
            "email": user_data.email,
            "password_hash": hash_password(user_data.password),
            "avatar_url": user_data.avatar_url,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

    @staticmethod
    def create_from_dict(user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create User data dictionary from dictionary"""
        now = datetime.now(timezone.utc)

        return {
            "id": user_data.get("id", uuid4()),
            "username": user_data["username"],
            "email": user_data["email"],
            "password_hash": user_data["password_hash"],
            "avatar_url": user_data.get("avatar_url"),
            "created_at": user_data.get("created_at", now),
            "updated_at": user_data.get("updated_at", now),
        }
