"""Authentication API endpoints for user login, signup, and token management"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from uuid import UUID

from app.database.session import get_db
from app.core.container import get_container
from app.services.user_service import UserService
from app.core.security import (
    verify_password,
    create_access_token,
    create_refresh_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
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
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


class RefreshTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


def get_user_service(db: Session = Depends(get_db)) -> UserService:
    """Dependency to get UserService instance"""
    container = get_container()
    container.set_session(db)
    return container.get("user_service")


@router.post("/signup", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def signup(
    user_data: UserCreate, user_service: UserService = Depends(get_user_service)
) -> UserRead:
    """Register a new user"""
    return user_service.create_user(user_data)


@router.post("/login", response_model=TokenResponse)
async def login(
    login_data: LoginRequest, user_service: UserService = Depends(get_user_service)
) -> TokenResponse:
    """Authenticate user and return JWT tokens"""
    # Get user by email
    user = user_service.get_user_by_email(login_data.email)
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

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_token(
    user_id: UUID = Depends(get_refresh_token_user_id),
    user_service: UserService = Depends(get_user_service),
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
    # Blacklist the token placeholder
    return {"message": "Successfully logged out. Please discard your tokens."}
