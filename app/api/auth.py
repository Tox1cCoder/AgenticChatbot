import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, status

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.auth_service_interface import IAuthService
from app.interfaces.user_service_interface import IUserService
from app.schemas.responses.api_response import ApiResponse
from app.schemas.responses.token_response import (
    LoginRequest,
    RefreshTokenResponse,
    TokenResponse,
)
from app.schemas.user import UserCreate, UserRead

logger = logging.getLogger(__name__)
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
    auth_service: IAuthService,
    refresh_user_id: UUID,
) -> ApiResponse[RefreshTokenResponse]:
    """Get new access token using refresh token"""
    access_token = auth_service.refresh_access_token(refresh_user_id)
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
    """Acknowledge a logout. Tokens are stateless and stay valid until they expire."""

    logger.info("User %s logged out", current_user_id)

    return ApiResponse(
        success=True,
        message="Successfully logged out. Please discard your tokens.",
    )
