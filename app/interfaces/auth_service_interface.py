"""
Authentication service interface definition
"""

from abc import ABC, abstractmethod
from uuid import UUID

from app.schemas.responses.token_response import LoginRequest


class IAuthService(ABC):
    @abstractmethod
    def authenticate_user(self, login_data: LoginRequest) -> dict:
        pass

    @abstractmethod
    def refresh_access_token(self, user_id: UUID) -> str:
        pass
