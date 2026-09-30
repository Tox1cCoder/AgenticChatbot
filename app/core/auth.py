"""Authentication dependencies and middleware for FastAPI"""

from uuid import UUID

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from app.core.exceptions import (
    AuthenticationException,
    AuthorizationException,
    ResourceNotFoundException,
    TokenExpiredException,
)
from app.core.security import decode_access_token, token_version, verify_refresh_token
from app.core.security.token_version import TokenState, token_version_claim
from app.models.user import User
from app.services.jwt_service import JwtService

security = HTTPBearer()


def get_jwt_service() -> JwtService:
    from app.core.container import container

    return container.jwt_service()


def get_user_service():
    from app.core.container import container

    return container.user_service()


def _require_current_token(payload: dict, state: TokenState | None) -> None:
    """Refuse a token whose user is gone or whose version has been revoked.

    A token with no ``ver`` claim counts as version 0, so tokens issued before
    versions existed keep working until they expire.
    """
    if state is None or state.deleted:
        raise AuthenticationException(
            detail="Authenticated user no longer exists",
            error_code="AUTHENTICATED_USER_NOT_FOUND",
        )
    version = token_version_claim(payload)
    if version is None:
        raise AuthenticationException(
            detail="Invalid authentication credentials",
            error_code="INVALID_CREDENTIALS",
        )
    if version < state.version:
        raise AuthenticationException(detail="Token has been revoked", error_code="TOKEN_REVOKED")


async def _arequire_current_token(payload: dict, user_id: UUID) -> None:
    """:func:`_require_current_token` for an async dependency.

    A cache hit is answered on the event loop; only a miss, which reads the
    users table, goes to the threadpool.
    """
    cache = token_version.token_state_cache
    hit, state = cache.peek(user_id)
    if not hit:
        state = await run_in_threadpool(cache.get, user_id)
    _require_current_token(payload, state)


def _subject_user_id(payload: dict, detail: str) -> UUID:
    try:
        return UUID(payload["sub"])
    except (TypeError, ValueError) as e:
        raise AuthenticationException(detail=detail, error_code="INVALID_USER_ID_FORMAT") from e


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),  # noqa: B008
    jwt_service: JwtService = Depends(get_jwt_service),  # noqa: B008
) -> UUID:
    """
    Dependency to get current authenticated user ID from JWT token
    """
    try:
        payload = decode_access_token(credentials.credentials, jwt_service)
    except jwt.ExpiredSignatureError as e:
        raise TokenExpiredException() from e
    user_id = _subject_user_id(payload, "Invalid user ID format in token")
    await _arequire_current_token(payload, user_id)
    return user_id


async def get_refresh_token_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),  # noqa: B008
    jwt_service: JwtService = Depends(get_jwt_service),  # noqa: B008
) -> UUID:
    """
    Dependency to get user ID from refresh token
    Validates refresh token type and extracts user_id
    """
    try:
        payload = verify_refresh_token(credentials.credentials, jwt_service)
    except jwt.ExpiredSignatureError as e:
        raise TokenExpiredException() from e
    if payload.get("sub") is None:
        raise AuthenticationException(
            detail="Invalid refresh token: missing user ID",
            error_code="INVALID_REFRESH_TOKEN",
        )
    user_id = _subject_user_id(payload, "Invalid user ID format in refresh token")
    await _arequire_current_token(payload, user_id)
    return user_id


def require_user_ownership(resource_user_id: UUID, authenticated_user_id: UUID) -> None:
    """
    Utility function to ensure authenticated user owns the resource
    """
    if resource_user_id != authenticated_user_id:
        raise AuthorizationException(
            detail="Access denied: insufficient permissions",
            error_code="INSUFFICIENT_PERMISSIONS",
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),  # noqa: B008
    jwt_service: JwtService = Depends(get_jwt_service),  # noqa: B008
) -> User:
    """Resolve the bearer token to a (password-less) User.

    Deliberately sync: the user lookup is a blocking database read, and FastAPI
    runs a sync dependency in its threadpool. As ``async def`` it ran on the
    event loop and stalled every in-flight stream for each authenticated request.
    """
    try:
        payload = decode_access_token(credentials.credentials, jwt_service)
    except jwt.ExpiredSignatureError as e:
        raise TokenExpiredException() from e
    user_id = _subject_user_id(payload, "Invalid user ID format in token")
    _require_current_token(payload, token_version.token_state_cache.get(user_id))

    try:
        user_read = get_user_service().get_by_id(user_id)
    except ResourceNotFoundException as e:
        raise AuthenticationException(
            detail="Authenticated user no longer exists",
            error_code="AUTHENTICATED_USER_NOT_FOUND",
        ) from e

    # A minimal User for auth purposes, without the password hash.
    return User(
        id=user_read.id,
        username=user_read.username,
        email=user_read.email,
        created_at=user_read.created_at,
        updated_at=user_read.updated_at,
        deleted_at=user_read.deleted_at,
        avatar_url=user_read.avatar_url,
        password_hash="",
    )
