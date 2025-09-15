"""Authentication dependencies and middleware for FastAPI"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from uuid import UUID
from typing import Optional

from app.core.security import get_user_id_from_token, verify_token, verify_refresh_token

# Security scheme for JWT Bearer token
security = HTTPBearer()


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UUID:
    """
    Dependency to get current authenticated user ID from JWT token
    Prevents ID injection by extracting user_id from validated token
    Uses PyJWT for enhanced security and FastAPI compatibility
    """
    token = credentials.credentials
    user_id_str = get_user_id_from_token(token)

    try:
        return UUID(user_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID format in token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> Optional[UUID]:
    """
    Optional authentication dependency for endpoints that can work with or without auth
    Compatible with PyJWT error handling
    """
    if not credentials:
        return None

    try:
        token = credentials.credentials
        user_id_str = get_user_id_from_token(token)
        return UUID(user_id_str)
    except (HTTPException, ValueError):
        return None


async def get_refresh_token_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UUID:
    """
    Dependency to get user ID from refresh token
    Validates refresh token type and extracts user_id
    """
    token = credentials.credentials
    payload = verify_refresh_token(token)
    user_id_str = payload.get("sub")

    if user_id_str is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token: missing user ID",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return UUID(user_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID format in refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_user_ownership(resource_user_id: UUID, authenticated_user_id: UUID) -> None:
    """
    Utility function to ensure authenticated user owns the resource
    Prevents ID injection attacks by validating ownership
    Enhanced for PyJWT implementation
    """
    if resource_user_id != authenticated_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: insufficient permissions",
        )
