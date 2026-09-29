"""
JWT token utilities
"""

from datetime import timedelta

from app.core.exceptions import AuthenticationException
from app.services.jwt_service import JwtService


def create_access_token(
    data: dict, jwt_service: JwtService, expires_delta: timedelta | None = None
) -> str:
    return jwt_service.create_access_token(data, expires_delta)


def get_user_id_from_token(token: str, jwt_service: JwtService) -> str:
    """Return the ``sub`` of an access token, refusing every other token kind.

    Refresh tokens and widget WebSocket tokens are signed with the same key and
    carry the same ``sub``. Without the ``type`` check a refresh token (valid
    for days) or a widget token (handed to the browser inside a ``ws_url``)
    authenticated every route as a full access token. Access tokens are the
    only ones minted without a ``type`` claim.
    """
    payload = jwt_service.decode_token(token)
    if payload.get("type") is not None:
        raise AuthenticationException(
            detail="Invalid token type", error_code="INVALID_TOKEN_TYPE"
        )
    user_id = payload.get("sub")
    if user_id is None:
        raise AuthenticationException(
            detail="Invalid authentication credentials",
            error_code="INVALID_CREDENTIALS",
        )
    return user_id


def create_refresh_token(data: dict, jwt_service: JwtService) -> str:
    return jwt_service.create_refresh_token(data)


def verify_refresh_token(token: str, jwt_service: JwtService) -> dict:
    return jwt_service.verify_refresh_token(token)
