"""
JWT token utilities
"""

from datetime import timedelta

from app.services.jwt_service import JwtService


def create_access_token(
    data: dict, jwt_service: JwtService, expires_delta: timedelta | None = None
) -> str:
    """Create JWT access token using JwtService"""
    return jwt_service.create_access_token(data, expires_delta)


def verify_token(token: str, jwt_service: JwtService) -> dict:
    """Verify and decode JWT token"""
    return jwt_service.decode_token(token)


def get_user_id_from_token(token: str, jwt_service: JwtService) -> str:
    """Extract user ID from JWT token"""
    return jwt_service.get_user_id_from_token(token)


def create_refresh_token(data: dict, jwt_service: JwtService) -> str:
    """Create refresh token"""
    return jwt_service.create_refresh_token(data)


def verify_refresh_token(token: str, jwt_service: JwtService) -> dict:
    """Verify refresh token"""
    return jwt_service.verify_refresh_token(token)
