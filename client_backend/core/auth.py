"""
Authentication helpers for the local client backend.
"""

import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from client_backend.core.security import LocalSessionPayload, verify_local_session_token
from client_backend.services.runtime_bridge import get_runtime_bridge
from client_backend.services.upstream_auth import get_upstream_auth_service

_bearer_scheme = HTTPBearer(auto_error=False)


def _build_compat_session_payload(user_id: str) -> LocalSessionPayload:
    now = datetime.now(timezone.utc)
    bridge = get_runtime_bridge()
    return LocalSessionPayload(
        user_id=str(user_id),
        server_user_id=str(user_id),
        device_id=bridge.get_registered_device_id(),
        device_identifier=bridge.get_device_identifier(),
        iat=now,
        exp=now + timedelta(hours=1),
    )


def _extract_user_id_from_bearer_token(token: str) -> str | None:
    try:
        payload = jwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False},
            algorithms=["HS256", "RS256"],
        )
    except Exception:
        return None

    user_id = payload.get("sub")
    return str(user_id) if user_id else None


async def require_local_session(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> LocalSessionPayload:
    """
    Require a valid local session token or the active upstream access token.

    The desktop UI currently stores the upstream `accessToken` returned by
    `/auth/login`, so the local backend must continue accepting it to preserve
    the existing endpoint contract.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing local session bearer token",
        )

    auth_service = get_upstream_auth_service()
    raw_token = credentials.credentials

    try:
        payload = verify_local_session_token(raw_token)
    except Exception:
        payload = None
    else:
        current_user_id = auth_service.get_current_user_id()
        if not auth_service.is_authenticated() or not current_user_id:
            restored = await auth_service.restore_session(str(payload.user_id))
            if not restored:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="No active upstream session. Login or restore the session first.",
                )
            current_user_id = auth_service.get_current_user_id()

        if str(current_user_id) != str(payload.user_id):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Local session token does not match the active upstream user.",
            )
        return payload

    current_user_id = auth_service.get_current_user_id()
    if not auth_service.is_authenticated() or not current_user_id:
        # The `sub` claim is read from an UNVERIFIED token, so it is only ever a
        # hint about which stored session to rehydrate. Authorization below still
        # requires the presented token to equal that session's access token;
        # never treat a successful restore as proof of the bearer's identity.
        hinted_user_id = _extract_user_id_from_bearer_token(raw_token)
        if hinted_user_id:
            await auth_service.restore_session(hinted_user_id)
            current_user_id = auth_service.get_current_user_id()

    if not auth_service.is_authenticated() or not current_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No active upstream session. Login or restore the session first.",
        )

    current_access_token = auth_service.get_current_access_token()
    if current_access_token and secrets.compare_digest(raw_token, current_access_token):
        return _build_compat_session_payload(str(current_user_id))

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Bearer token is not a valid local session or the active upstream access token.",
    )
