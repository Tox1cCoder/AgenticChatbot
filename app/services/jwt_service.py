"""JWT Service for token management and authentication utilities"""

from datetime import datetime, timedelta, timezone

import jwt

from app.core.config import settings
from app.core.exceptions import AuthenticationException, TokenExpiredException
from app.core.server_secrets import get_signing_key


class JwtService:
    """Signs and verifies tokens with the server signing key.

    The key is read per call, not captured here: the container builds one of
    these per request, and ``get_signing_key`` answers from the process cache
    once startup has resolved it.
    """

    def __init__(self):
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes
        self.refresh_token_expire_days = settings.refresh_token_expire_days

    def _calculate_expiration_time(
        self, delta: timedelta | None = None, now: datetime | None = None
    ) -> datetime:
        if now is None:
            now = datetime.now(timezone.utc)
        if delta:
            return now + delta
        return now + timedelta(minutes=self.access_token_expire_minutes)

    def create_access_token(self, data: dict, expires_delta: timedelta | None = None) -> str:
        to_encode = data.copy()
        now = datetime.now(timezone.utc)
        expire = self._calculate_expiration_time(expires_delta, now)
        to_encode.update({"exp": int(expire.timestamp()), "iat": int(now.timestamp())})
        return jwt.encode(to_encode, get_signing_key(), algorithm=self.algorithm)

    def create_refresh_token(self, data: dict) -> str:
        to_encode = data.copy()
        now = datetime.now(timezone.utc)
        expire = now + timedelta(days=self.refresh_token_expire_days)
        to_encode.update(
            {
                "exp": int(expire.timestamp()),
                "iat": int(now.timestamp()),
                "type": "refresh",
            }
        )
        return jwt.encode(to_encode, get_signing_key(), algorithm=self.algorithm)

    def decode_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, get_signing_key(), algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError as e:
            raise TokenExpiredException() from e
        except jwt.InvalidTokenError as e:
            raise AuthenticationException(
                detail="Invalid authentication credentials",
                error_code="INVALID_CREDENTIALS",
            ) from e

    def verify_refresh_token(self, token: str) -> dict:
        payload = self.decode_token(token)
        if payload.get("type") != "refresh":
            raise AuthenticationException(
                detail="Invalid token type", error_code="INVALID_TOKEN_TYPE"
            )
        return payload
