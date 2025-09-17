"""Authentication API endpoints for user login, signup, and token management"""

import logging
from datetime import timedelta
from typing import Annotated
from fastapi import APIRouter, Depends, status
from fastapi.security import HTTPAuthorizationCredentials
from uuid import UUID

from dependency_injector.wiring import Provide, inject

from app.core.container import Container
from app.core.exceptions import AuthenticationException, ResourceNotFoundException
from app.interfaces.user_service_interface import IUserService
from app.services.auth_service import AuthService
from app.core.config import settings
from app.core.security import (
    verify_password,
    create_access_token,
    create_refresh_token,
)
from app.core.auth import get_refresh_token_user_id, security
from app.schemas.user import UserCreate, UserRead
from pydantic import BaseModel, EmailStr, Field

router = APIRouter(prefix="/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    accessToken: str = Field(alias="access_token")
    refreshToken: str = Field(alias="refresh_token")
    tokenType: str = Field(default="bearer", alias="token_type")
    expiresIn: int = Field(
        default=settings.access_token_expire_minutes * 60, alias="expires_in"
    )
    userId: str = Field(alias="user_id")

    class Config:
        validate_by_name = True


class RefreshTokenResponse(BaseModel):
    accessToken: str = Field(alias="access_token")
    tokenType: str = Field(default="bearer", alias="token_type")
    expiresIn: int = Field(
        default=settings.access_token_expire_minutes * 60, alias="expires_in"
    )

    class Config:
        validate_by_name = True


@router.post("/signup", response_model=UserRead, status_code=status.HTTP_201_CREATED)
@inject
async def signup(
    user_data: UserCreate,
    user_service: Annotated[IUserService, Depends(Provide[Container.user_service])],
) -> UserRead:
    """Register a new user"""
    return user_service.create_user(user_data)


@router.post("/login", response_model=TokenResponse)
@inject
async def login(
    login_data: LoginRequest,
    user_service: Annotated[IUserService, Depends(Provide[Container.user_service])],
) -> TokenResponse:
    """Authenticate user and return JWT tokens"""
    # Create auth service instance with user service
    auth_service = AuthService(user_service)
    # Authenticate user using auth service
    auth_response = auth_service.authenticate_user(login_data)
    return TokenResponse(**auth_response)


@router.post("/refresh", response_model=RefreshTokenResponse)
@inject
async def refresh_token(
    user_service: Annotated[IUserService, Depends(Provide[Container.user_service])],
    user_id: UUID = Depends(get_refresh_token_user_id),
) -> RefreshTokenResponse:
    """Get new access token using refresh token"""
    # Verify user still exists
    user = user_service.get_by_id(user_id)
    if not user:
        raise AuthenticationException(
            detail="User not found", error_code="USER_NOT_FOUND"
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
