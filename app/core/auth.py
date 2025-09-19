"""Authentication dependencies and middleware for FastAPI"""

from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from uuid import UUID
import jwt

from app.core.security import get_user_id_from_token, verify_refresh_token
from app.core.exceptions import TokenExpiredException, AuthenticationException, AuthorizationException

security = HTTPBearer()


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UUID:
    """
    Dependency to get current authenticated user ID from JWT token
    """
    token = credentials.credentials
    try:
        user_id_str = get_user_id_from_token(token)
        return UUID(user_id_str)
    except jwt.ExpiredSignatureError:
        raise TokenExpiredException()
    except ValueError:
        raise AuthenticationException(
            detail="Invalid user ID format in token",
            error_code="INVALID_USER_ID_FORMAT",
        )

async def get_refresh_token_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UUID:
    """
    Dependency to get user ID from refresh token
    Validates refresh token type and extracts user_id
    """
    token = credentials.credentials
    try:
        payload = verify_refresh_token(token)
        user_id_str = payload.get("sub")

        if user_id_str is None:
            raise AuthenticationException(
                detail="Invalid refresh token: missing user ID",
                error_code="INVALID_REFRESH_TOKEN",
            )

        return UUID(user_id_str)
    except jwt.ExpiredSignatureError:
        raise TokenExpiredException()
    except ValueError:
        raise AuthenticationException(
            detail="Invalid user ID format in refresh token",
            error_code="INVALID_USER_ID_FORMAT",
        )


def require_user_ownership(resource_user_id: UUID, authenticated_user_id: UUID) -> None:
    """
    Utility function to ensure authenticated user owns the resource
    """
    if resource_user_id != authenticated_user_id:
        raise AuthorizationException(
            detail="Access denied: insufficient permissions",
            error_code="INSUFFICIENT_PERMISSIONS",
        )