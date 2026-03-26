import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from client_backend.api import auth as auth_api
from client_backend.core import auth as auth_module
from client_backend.core.security import create_local_session_token


class _AuthStub:
    def __init__(
        self,
        user_id: str,
        authenticated: bool = True,
        access_token: str | None = None,
    ):
        self._user_id = user_id
        self._authenticated = authenticated
        self._access_token = access_token

    def get_current_user_id(self) -> str:
        return self._user_id

    def is_authenticated(self) -> bool:
        return self._authenticated

    def get_current_access_token(self) -> str | None:
        return self._access_token


@pytest.mark.asyncio
async def test_require_local_session_accepts_matching_user(monkeypatch):
    user_id = "user-123"
    token = create_local_session_token(
        user_id=user_id,
        server_user_id=user_id,
        device_identifier="device-abc",
    )
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    monkeypatch.setattr(auth_module, "get_upstream_auth_service", lambda: _AuthStub(user_id))

    payload = await auth_module.require_local_session(credentials)

    assert payload.user_id == user_id
    assert payload.device_identifier == "device-abc"


@pytest.mark.asyncio
async def test_require_local_session_rejects_mismatched_user(monkeypatch):
    token = create_local_session_token(
        user_id="user-123",
        server_user_id="user-123",
        device_identifier="device-abc",
    )
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    monkeypatch.setattr(auth_module, "get_upstream_auth_service", lambda: _AuthStub("user-456"))

    try:
        await auth_module.require_local_session(credentials)
    except HTTPException as exc:
        assert exc.status_code == 401
        assert "does not match" in str(exc.detail)
    else:
        raise AssertionError("Expected HTTPException for mismatched user")


@pytest.mark.asyncio
async def test_require_local_session_accepts_active_upstream_access_token(monkeypatch):
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="upstream-access-token",
    )

    class _BridgeStub:
        def get_registered_device_id(self) -> str:
            return "device-id-123"

        def get_device_identifier(self) -> str:
            return "device-identifier-123"

    monkeypatch.setattr(
        auth_module,
        "get_upstream_auth_service",
        lambda: _AuthStub("user-123", access_token="upstream-access-token"),
    )
    monkeypatch.setattr(auth_module, "get_runtime_bridge", lambda: _BridgeStub())

    payload = await auth_module.require_local_session(credentials)

    assert payload.user_id == "user-123"
    assert payload.device_id == "device-id-123"
    assert payload.device_identifier == "device-identifier-123"


@pytest.mark.asyncio
async def test_login_returns_server_shape_with_local_metadata(monkeypatch):
    class _AuthServiceStub:
        async def login(self, email: str, password: str) -> dict:
            assert email == "user@example.com"
            assert password == "secret"
            return {
                "success": True,
                "message": "Login successful",
                "data": {
                    "accessToken": "access-token",
                    "refreshToken": "refresh-token",
                    "tokenType": "bearer",
                    "expiresIn": 3600,
                    "userId": "user-123",
                },
                "error": None,
            }

    class _BridgeStub:
        def get_registered_device_id(self) -> str:
            return "device-id-123"

        def get_device_identifier(self) -> str:
            return "device-identifier-123"

    async def _noop_start_runtime():
        return None

    monkeypatch.setattr(auth_api, "get_upstream_auth_service", lambda: _AuthServiceStub())
    monkeypatch.setattr(auth_api, "get_runtime_bridge", lambda: _BridgeStub())
    monkeypatch.setattr(auth_api, "_start_runtime_bridge_after_auth", _noop_start_runtime)

    result = await auth_api.login(
        auth_api.LoginRequest(email="user@example.com", password="secret")
    )

    assert result["success"] is True
    assert result["data"]["accessToken"] == "access-token"
    assert result["data"]["userId"] == "user-123"
    assert result["data"]["deviceId"] == "device-id-123"
    assert result["data"]["deviceIdentifier"] == "device-identifier-123"
    assert result["data"]["localSessionToken"]


def test_augment_auth_response_restores_user_context_for_refresh(monkeypatch):
    class _AuthServiceStub:
        def get_current_user_id(self) -> str:
            return "user-123"

    class _BridgeStub:
        def get_registered_device_id(self) -> str:
            return "device-id-123"

        def get_device_identifier(self) -> str:
            return "device-identifier-123"

    monkeypatch.setattr(auth_api, "get_upstream_auth_service", lambda: _AuthServiceStub())
    monkeypatch.setattr(auth_api, "get_runtime_bridge", lambda: _BridgeStub())

    result = auth_api._augment_auth_response(
        {
            "success": True,
            "message": "Token refreshed successfully",
            "data": {
                "access_token": "new-access-token",
                "token_type": "bearer",
                "expires_in": 3600,
            },
            "error": None,
        },
        include_local_session=True,
    )

    assert result["data"]["userId"] == "user-123"
    assert result["data"]["serverUrl"] == auth_api.client_settings.server_api_base_url
    assert result["data"]["deviceId"] == "device-id-123"
    assert result["data"]["deviceIdentifier"] == "device-identifier-123"
    assert result["data"]["localSessionToken"]
