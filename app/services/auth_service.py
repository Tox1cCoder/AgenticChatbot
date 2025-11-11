"""
Authentication service for handling login operations
"""

from fastapi import HTTPException, status

from app.core.security import verify_password, create_access_token, create_refresh_token
from app.schemas.responses.token_response import LoginRequest


from app.interfaces.auth_service_interface import IAuthService
from app.interfaces.user_service_interface import IUserService
from app.services.jwt_service import JwtService


class AuthService(IAuthService):
    """Service for authentication operations"""

    def __init__(self, user_service: IUserService, jwt_service: JwtService) -> None:
        self.user_service = user_service
        self.jwt_service = jwt_service

    def authenticate_user(self, login_data: LoginRequest) -> dict:
        """Authenticate user and return token data"""
        # Get user by email with password hash for authentication
        user = self.user_service.get_by_email_with_password(login_data.email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify password
        if not verify_password(login_data.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Create tokens
        token_data = {"sub": str(user.id)}
        access_token = create_access_token(token_data, self.jwt_service)
        refresh_token = create_refresh_token(token_data, self.jwt_service)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user_id": str(user.id),
        }
