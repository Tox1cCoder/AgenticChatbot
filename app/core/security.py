"""Security utilities including password hashing and JWT authentication"""

import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, status

from app.core.config import settings


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


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token using PyJWT (FastAPI recommended)"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify and decode JWT token using PyJWT (FastAPI recommended)"""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_user_id_from_token(token: str) -> str:
    """Extract user ID from JWT token"""
    payload = verify_token(token)
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


def create_refresh_token(data: dict) -> str:
    """Create refresh token with longer expiration"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(
        days=settings.refresh_token_expire_days
    )
    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "refresh"})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)


def verify_refresh_token(token: str) -> dict:
    """Verify refresh token and ensure it's the correct type"""
    payload = verify_token(token)
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload