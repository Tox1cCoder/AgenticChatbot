"""
User service interface definition
"""

from abc import ABC, abstractmethod
from uuid import UUID

from app.core.security.token_version import TokenState
from app.schemas.user import UserCreate, UserInDB, UserRead


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
    def get_by_email_with_password(self, email: str) -> UserInDB | None:
        """Get user by email with password hash for authentication"""
        pass

    @abstractmethod
    def get_token_state(self, user_id: UUID) -> TokenState | None:
        """The user's current token version and soft-delete state, uncached"""
        pass
