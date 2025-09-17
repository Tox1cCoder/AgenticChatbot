"""Authentication API endpoints for user login, signup, and token management"""

from datetime import timedelta
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from uuid import UUID

from dependency_injector.wiring import Provide, inject

from app.core.container import Container
from app.services.user_service import UserService
from app.core.config import settings
from app.core.security import (
    verify_password,
    create_access_token,
    create_refresh_token,
)
from app.core.auth import get_refresh_token_user_id, security
from app.schemas.user import UserCreate, UserRead
from pydantic import BaseModel, EmailStr

router = APIRouter(prefix="/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = settings.access_token_expire_minutes * 60
    user_id: str  # Include user_id in token response


class RefreshTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = settings.access_token_expire_minutes * 60


@router.post("/signup", response_model=UserRead, status_code=status.HTTP_201_CREATED)
@inject
async def signup(
    user_data: UserCreate,
    user_service: Annotated[UserService, Depends(Provide[Container.user_service])],
) -> UserRead:
    """Register a new user"""
    return user_service.create_user(user_data)


@router.post("/login", response_model=TokenResponse)
@inject
async def login(
    login_data: LoginRequest,
    user_service: Annotated[UserService, Depends(Provide[Container.user_service])],
) -> TokenResponse:
    """Authenticate user and return JWT tokens"""
    try:
        # Get user by email with password hash for authentication
        user = user_service.get_user_by_email_with_password(login_data.email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Verify password
        if not verify_password(login_data.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Create tokens
        token_data = {"sub": str(user.id)}
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user_id=str(user.id),  # Include user_id in response
        )

    except HTTPException:
        # Re-raise HTTP exceptions (authentication failures)
        raise
    except Exception as e:
        # Log unexpected errors and return generic authentication failure
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
        )


@router.post("/refresh", response_model=RefreshTokenResponse)
@inject
async def refresh_token(
    user_service: Annotated[UserService, Depends(Provide[Container.user_service])],
    user_id: UUID = Depends(get_refresh_token_user_id),
) -> RefreshTokenResponse:
    """Get new access token using refresh token"""
    # Verify user still exists
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    # Create new access token
    token_data = {"sub": str(user.id)}
    access_token = create_access_token(token_data)

    return RefreshTokenResponse(access_token=access_token)


@router.post("/logout")
async def logout():
    """Logout endpoint (client should discard tokens)"""
    # Placeholder: Blacklist the token
    return {"message": "Successfully logged out. Please discard your tokens."}
