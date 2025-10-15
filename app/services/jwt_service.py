"""JWT Service for token management and authentication utilities"""

from datetime import datetime, timezone, timedelta
from typing import Optional

import jwt

from app.core.config import settings
from app.core.exceptions import TokenExpiredException, AuthenticationException


class JwtService:
    """JWT token service for authentication operations"""

    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes
        self.refresh_token_expire_days = settings.refresh_token_expire_days

    def _calculate_expiration_time(self, delta: Optional[timedelta] = None) -> datetime:
        """Calculate token expiration time"""
        if delta:
            return datetime.now(timezone.utc) + delta
        return datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)

    def create_access_token(
        self, data: dict, expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = self._calculate_expiration_time(expires_delta)
        to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, data: dict) -> str:
        """Create refresh token with longer expiration"""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc), "type": "refresh"})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def decode_token(self, token: str) -> dict:
        """Decode and verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise TokenExpiredException()
        except jwt.InvalidTokenError:
            raise AuthenticationException(
                detail="Invalid authentication credentials",
                error_code="INVALID_CREDENTIALS",
            )

    def verify_refresh_token(self, token: str) -> dict:
        """Verify refresh token and ensure correct type"""
        payload = self.decode_token(token)
        if payload.get("type") != "refresh":
            raise AuthenticationException(
                detail="Invalid token type", error_code="INVALID_TOKEN_TYPE"
            )
        return payload

    def get_user_id_from_token(self, token: str) -> str:
        """Extract user ID from JWT token"""
        payload = self.decode_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise AuthenticationException(
                detail="Invalid authentication credentials",
                error_code="INVALID_CREDENTIALS",
            )
        return user_id