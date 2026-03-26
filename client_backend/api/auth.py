"""
Authentication API endpoints for the client backend.

These endpoints preserve the current server contract for the desktop UI while
also maintaining the local runtime bridge and local-session support.
"""

from typing import Any

from fastapi import APIRouter, Header, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict

from client_backend.api.common import raise_server_error
from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.security import (
    LocalSessionError,
    create_local_session_token,
    verify_local_session_token,
)
from client_backend.services.runtime_bridge import get_runtime_bridge
from client_backend.services.server_api import get_server_client
from client_backend.services.upstream_auth import get_upstream_auth_service

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    """Login request body compatible with current clients."""

    email: str | None = None
    username: str | None = None
    password: str

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    def resolved_email(self) -> str:
        email = (self.email or self.username or "").strip()
        if not email:
            raise ValueError("`email` is required")
        return email


def _build_local_session_token(user_id: str) -> str:
    bridge = get_runtime_bridge()
    return create_local_session_token(
        user_id=user_id,
        server_user_id=user_id,
        device_identifier=bridge.get_device_identifier(),
        device_id=bridge.get_registered_device_id(),
    )


def _augment_auth_response(
    payload: dict[str, Any], *, include_local_session: bool
) -> dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return payload

    bridge = get_runtime_bridge()
    resolved_user_id = (
        data.get("userId")
        or data.get("user_id")
        or get_upstream_auth_service().get_current_user_id()
    )
    augmented = dict(payload)
    augmented_data = dict(data)
    if resolved_user_id and "userId" not in augmented_data:
        augmented_data["userId"] = resolved_user_id
    augmented_data["serverUrl"] = client_settings.server_api_base_url
    augmented_data["deviceId"] = bridge.get_registered_device_id()
    augmented_data["deviceIdentifier"] = bridge.get_device_identifier()
    if include_local_session and resolved_user_id:
        augmented_data["localSessionToken"] = _build_local_session_token(str(resolved_user_id))
    augmented["data"] = augmented_data
    return augmented


async def _start_runtime_bridge_after_auth() -> None:
    """Start the runtime bridge and wait briefly for initial registration."""
    bridge = get_runtime_bridge()
    runtime_started = await bridge.start(wait_for_connection=False)
    if not runtime_started:
        logger.warning("Runtime bridge did not start immediately after authentication")
        return

    connected = await bridge.start(
        wait_for_connection=True,
        timeout_seconds=min(5, client_settings.server_api_timeout_seconds),
    )
    if not connected:
        logger.warning("Runtime bridge did not establish a session within the startup window")


@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(payload: dict[str, Any], response: Response) -> dict[str, Any]:
    """Proxy signup directly to the canonical server."""
    try:
        response.status_code = status.HTTP_201_CREATED
        return await get_server_client().post(
            "/auth/signup",
            json=payload,
            include_auth_headers=False,
        )
    except Exception as exc:
        raise_server_error(exc)


@router.post("/login")
async def login(request: LoginRequest) -> dict[str, Any]:
    """
    Login to the upstream server.

    Returns the same wrapped auth payload shape as the server, with local
    runtime metadata added as extra fields under `data`.
    """
    auth_service = get_upstream_auth_service()

    try:
        result = await auth_service.login(request.resolved_email(), request.password)
        await _start_runtime_bridge_after_auth()
        return _augment_auth_response(result, include_local_session=True)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    except Exception as exc:
        raise_server_error(exc)


@router.post("/refresh")
async def refresh(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """
    Refresh the current upstream access token.

    The local backend continues to own upstream tokens, so the current active
    session is refreshed and the server's wrapped response is returned.
    """
    auth_service = get_upstream_auth_service()
    if not auth_service.is_authenticated():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No active upstream session to refresh",
        )

    current_refresh_token = auth_service.get_current_refresh_token()
    if authorization and current_refresh_token:
        expected = f"Bearer {current_refresh_token}"
        if authorization.strip() != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token does not match the active local session",
            )

    try:
        result = await auth_service.refresh()
        return _augment_auth_response(result, include_local_session=True)
    except Exception as exc:
        raise_server_error(exc)


@router.post("/logout")
async def logout() -> dict[str, Any]:
    """
    Logout from the upstream server and stop the local runtime bridge.
    """
    auth_service = get_upstream_auth_service()

    try:
        await get_runtime_bridge().stop()
        upstream_response = await auth_service.logout()
        if upstream_response is None:
            upstream_response = {
                "success": True,
                "message": "Successfully logged out. Please discard your tokens.",
                "data": None,
                "error": None,
            }
        return upstream_response
    except Exception as exc:
        raise_server_error(exc)


@router.post("/restore")
async def restore_session(user_id: str) -> dict[str, Any]:
    """
    Restore a session from locally stored credentials.
    """
    auth_service = get_upstream_auth_service()

    try:
        success = await auth_service.restore_session(user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to restore session - credentials may be expired",
            )

        await _start_runtime_bridge_after_auth()
        current_access_token = auth_service.get_current_access_token()
        current_refresh_token = auth_service.get_current_refresh_token()
        if not current_access_token or not current_refresh_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Stored session was restored without usable tokens",
            )

        payload = {
            "success": True,
            "message": "Session restored successfully",
            "data": {
                "accessToken": current_access_token,
                "refreshToken": current_refresh_token,
                "tokenType": "bearer",
                "expiresIn": None,
                "userId": user_id,
            },
            "error": None,
        }
        return _augment_auth_response(payload, include_local_session=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise_server_error(exc)


@router.get("/session")
async def get_session_info() -> dict[str, Any]:
    """
    Get current local session/runtime information.
    """
    auth_service = get_upstream_auth_service()
    bridge = get_runtime_bridge()

    return {
        "userId": auth_service.get_current_user_id() or "",
        "authenticated": auth_service.is_authenticated(),
        "serverUrl": auth_service._client.base_url,
        "deviceId": bridge.get_registered_device_id(),
        "deviceIdentifier": bridge.get_device_identifier(),
    }


@router.get("/users")
async def list_stored_users() -> dict[str, Any]:
    """
    List users with locally stored sessions that can be restored.
    """
    auth_service = get_upstream_auth_service()
    return {"users": auth_service.list_stored_users()}


@router.post("/verify-local-token")
async def verify_local_token(token: str) -> dict[str, Any]:
    """
    Verify a locally-issued session token.
    """
    try:
        payload = verify_local_session_token(token)
        return {
            "valid": True,
            "userId": payload.user_id,
            "deviceId": payload.device_id,
            "deviceIdentifier": payload.device_identifier,
            "expiresAt": payload.exp.isoformat(),
        }
    except LocalSessionError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e)) from e
