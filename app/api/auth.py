"""Authentication API endpoints for user login, signup, and token management"""

import logging
from typing import Annotated
from fastapi import APIRouter, Depends, status
from uuid import UUID

from dependency_injector.wiring import Provide, inject

from app.core.container import Container
from app.interfaces.user_service_interface import IUserService
from app.interfaces.auth_service_interface import IAuthService
from app.core.security import create_access_token
from app.core.auth import get_refresh_token_user_id
from app.schemas.user import UserCreate, UserRead
from app.schemas.responses.api_response import ApiResponse
from app.schemas.responses.token_response import (
    TokenResponse,
    LoginRequest,
    RefreshTokenResponse,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/signup", response_model=UserRead, status_code=status.HTTP_201_CREATED)
@inject
async def signup(
    user_data: UserCreate,
    user_service: Annotated[IUserService, Depends(Provide[Container.user_service])],
) -> UserRead:
    """Register a new user"""
    return user_service.create_user(user_data)


@router.post("/login", response_model=ApiResponse[TokenResponse])
@inject
async def login(
    login_data: LoginRequest,
    auth_service: Annotated[IAuthService, Depends(Provide[Container.auth_service])],
) -> ApiResponse[TokenResponse]:
    """Authenticate user and return JWT tokens"""
    auth_response = auth_service.authenticate_user(login_data)
    return ApiResponse(
        data=TokenResponse(**auth_response), message="Login successful"
    )


@router.post("/refresh", response_model=RefreshTokenResponse)
@inject
async def refresh_token(
    user_service: Annotated[IUserService, Depends(Provide[Container.user_service])],
    user_id: UUID = Depends(get_refresh_token_user_id),
) -> RefreshTokenResponse:
    """Get new access token using refresh token"""
    user = user_service.get_by_id(user_id)
    token_data = {"sub": str(user.id)}
    access_token = create_access_token(token_data)
    return RefreshTokenResponse(access_token=access_token)


@router.post("/logout")
async def logout():
    """Logout endpoint (client should discard tokens)"""
    # Placeholder: Blacklist the token
    return {"message": "Successfully logged out. Please discard your tokens."}