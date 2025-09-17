"""Security utilities including password hashing and JWT authentication"""

import bcrypt
from typing import Optional
from datetime import timedelta

from app.services.jwt_service import JwtService


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    password_bytes = password.encode("utf-8")
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hashed_bytes)


# JWT service instance for token operations
jwt_service = JwtService()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token using JwtService"""
    return jwt_service.create_access_token(data, expires_delta)


def verify_token(token: str) -> dict:
    """Verify and decode JWT token using JwtService"""
    return jwt_service.decode_token(token)


def get_user_id_from_token(token: str) -> str:
    """Extract user ID from JWT token using JwtService"""
    return jwt_service.get_user_id_from_token(token)


def create_refresh_token(data: dict) -> str:
    """Create refresh token using JwtService"""
    return jwt_service.create_refresh_token(data)


def verify_refresh_token(token: str) -> dict:
    """Verify refresh token using JwtService"""
    return jwt_service.verify_refresh_token(token)
