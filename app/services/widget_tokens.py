"""Short-lived signed tokens that admit one browser to one widget's WebSocket.

Server-only, and apart from ``widget_runtime`` for that reason: the client sidecar
bundle ships ``widget_runtime`` for its store, never mints or verifies these tokens,
and must not carry the server's key resolution.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from app.core.server_secrets import get_signing_key

WIDGET_TOKEN_TTL_SECONDS = 300  # 5 minutes


class WidgetTokenService:
    """Mint and verify short-lived signed tokens scoped to a single widget.

    Without an explicit ``secret`` the server signing key is used, the same key
    as access tokens; ``decode_access_token`` refuses these by their ``type``.
    """

    def __init__(self, secret: str | None = None, algorithm: str = "HS256") -> None:
        self._secret = secret
        self._algorithm = algorithm

    def _key(self) -> str:
        return self._secret or get_signing_key()

    def mint(
        self,
        *,
        widget_id: str,
        session_id: str,
        user_id: str,
        ttl_seconds: int = WIDGET_TOKEN_TTL_SECONDS,
    ) -> tuple[str, datetime]:
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=ttl_seconds)
        payload = {
            "sub": user_id,
            "wid": widget_id,
            "sid": session_id,
            "type": "widget",
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
        }
        token = jwt.encode(payload, self._key(), algorithm=self._algorithm)
        return token, expires_at

    def verify(self, token: str) -> dict[str, Any]:
        """Verify token and return claims. Raises jwt.InvalidTokenError on failure."""
        payload = jwt.decode(token, self._key(), algorithms=[self._algorithm])
        if payload.get("type") != "widget":
            raise jwt.InvalidTokenError("Not a widget token")
        return payload


_widget_token_service: WidgetTokenService | None = None


def get_widget_token_service() -> WidgetTokenService:
    global _widget_token_service
    if _widget_token_service is None:
        _widget_token_service = WidgetTokenService()
    return _widget_token_service
