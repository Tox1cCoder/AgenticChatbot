"""
User service interface definition
"""

from abc import ABC, abstractmethod
from uuid import UUID

from app.schemas.user import UserCreate, UserInDB, UserRead, UserUpdate


class IUserService(ABC):
    """Interface for User service operations"""

    @abstractmethod
    def create_user(self, user_create_data: UserCreate) -> UserRead:
        """Create a new user with validation"""
        pass

    @abstractmethod
    def get_by_id(self, user_id: UUID) -> UserRead:
        """Get user by ID"""
        pass

    @abstractmethod
    def get_by_email(self, email: str) -> UserRead | None:
        """Get user by email"""
        pass

    @abstractmethod
    def get_by_email_with_password(self, email: str) -> UserInDB | None:
        """Get user by email with password hash for authentication"""
        pass

    @abstractmethod
    def get_by_username(self, username: str) -> UserRead | None:
        """Get user by username"""
        pass

    @abstractmethod
    def get_all(self, skip: int = 0, limit: int = 100) -> list[UserRead]:
        """Get all users with pagination"""
        pass

    @abstractmethod
    def update_user(self, user_id: UUID, user_update_data: UserUpdate) -> UserRead:
        """Update user with validation"""
        pass
