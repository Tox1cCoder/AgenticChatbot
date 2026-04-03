import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, status

from app.core.dependency_injection import AppAutoInjector
from app.core.security import create_access_token
from app.interfaces.auth_service_interface import IAuthService
from app.interfaces.user_service_interface import IUserService
from app.schemas.responses.api_response import ApiResponse
from app.schemas.responses.token_response import (
    LoginRequest,
    RefreshTokenResponse,
    TokenResponse,
)
from app.schemas.user import UserCreate, UserRead
from app.services.jwt_service import JwtService

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/signup", response_model=ApiResponse[UserRead], status_code=status.HTTP_201_CREATED)
@AppAutoInjector.auto_inject()
async def signup(
    user_data: UserCreate,
    user_service: IUserService,
) -> ApiResponse[UserRead]:
    """Register a new user"""
    created_user = user_service.create_user(user_data)
    return ApiResponse(success=True, message="User created successfully", data=created_user)


@router.post("/login", response_model=ApiResponse[TokenResponse])
@AppAutoInjector.auto_inject()
async def login(
    login_data: LoginRequest,
    auth_service: IAuthService,
) -> ApiResponse[TokenResponse]:
    """Authenticate user and return JWT tokens"""
    auth_response = auth_service.authenticate_user(login_data)
    return ApiResponse(
        success=True, message="Login successful", data=TokenResponse(**auth_response)
    )


@router.post("/refresh", response_model=ApiResponse[RefreshTokenResponse])
@AppAutoInjector.auto_inject()
async def refresh_token(
    user_service: IUserService,
    jwt_service: JwtService,
    refresh_user_id: UUID,
) -> ApiResponse[RefreshTokenResponse]:
    """Get new access token using refresh token"""
    user = user_service.get_by_id(refresh_user_id)
    token_data = {"sub": str(user.id)}
    access_token = create_access_token(token_data, jwt_service)
    return ApiResponse(
        success=True,
        message="Token refreshed successfully",
        data=RefreshTokenResponse(access_token=access_token),
    )


@router.post("/logout", response_model=ApiResponse[Any])
@AppAutoInjector.auto_inject()
async def logout(
    current_user_id: UUID,
) -> ApiResponse[Any]:
    """Logout endpoint with token invalidation"""

    logging.info(f"User {current_user_id} logged out successfully")

    return ApiResponse(
        success=True,
        message="Successfully logged out. Please discard your tokens.",
    )
