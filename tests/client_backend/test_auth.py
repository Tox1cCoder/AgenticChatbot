from datetime import UTC

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from client_backend.api import auth as auth_api
from client_backend.core import auth as auth_module
from client_backend.core.security import create_local_session_token
from client_backend.services.server_api import TokenPair


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

    def get_current_username(self) -> str | None:
        return None


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
        def is_authenticated(self) -> bool:
            return False

        def get_current_username(self) -> str | None:
            return None

        async def login(self, email: str, password: str) -> TokenPair:
            assert email == "user@example.com"
            assert password == "secret"
            return TokenPair(
                access_token="access-token",
                refresh_token="refresh-token",
                token_type="bearer",
                expires_in=3600,
                user_id="user-123",
            )

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


@pytest.mark.asyncio
async def test_login_rejects_switching_active_user_without_logout(monkeypatch):
    login_calls: list[tuple[str, str]] = []

    class _AuthServiceStub:
        def is_authenticated(self) -> bool:
            return True

        def get_current_user_id(self) -> str:
            return "user-123"

        def get_current_username(self) -> str | None:
            return "user-one@example.com"

        async def login(self, email: str, password: str) -> TokenPair:
            login_calls.append((email, password))
            raise AssertionError("login() should not be called when another user is active")

    monkeypatch.setattr(auth_api, "get_upstream_auth_service", lambda: _AuthServiceStub())

    with pytest.raises(HTTPException) as exc_info:
        await auth_api.login(auth_api.LoginRequest(email="user-two@example.com", password="secret"))

    assert exc_info.value.status_code == 409
    assert "log out" in str(exc_info.value.detail).lower()
    assert login_calls == []


@pytest.mark.asyncio
async def test_restore_rejects_switching_active_user_without_logout(monkeypatch):
    restore_calls: list[str] = []

    class _AuthServiceStub:
        def is_authenticated(self) -> bool:
            return True

        def get_current_user_id(self) -> str:
            return "user-123"

        async def restore_session(self, user_id: str) -> bool:
            restore_calls.append(user_id)
            raise AssertionError(
                "restore_session() should not be called when another user is active"
            )

    monkeypatch.setattr(auth_api, "get_upstream_auth_service", lambda: _AuthServiceStub())

    with pytest.raises(HTTPException) as exc_info:
        await auth_api.restore_session("user-456")

    assert exc_info.value.status_code == 409
    assert "log out" in str(exc_info.value.detail).lower()
    assert restore_calls == []


class _RefreshAuthStub:
    def __init__(self) -> None:
        self.refresh_calls = 0

    def is_authenticated(self) -> bool:
        return True

    def get_current_refresh_token(self) -> str:
        return "active-refresh-token"

    def get_current_user_id(self) -> str:
        return "user-123"

    async def refresh(self) -> TokenPair:
        self.refresh_calls += 1
        return TokenPair(access_token="new-access", refresh_token="active-refresh-token")


@pytest.mark.asyncio
@pytest.mark.parametrize("authorization", [None, "", "Bearer wrong", "Bearer ümlaut"])
async def test_refresh_requires_the_active_refresh_token(monkeypatch, authorization):
    """Without the header, any local process could mint fresh tokens here."""
    stub = _RefreshAuthStub()
    monkeypatch.setattr(auth_api, "get_upstream_auth_service", lambda: stub)

    with pytest.raises(HTTPException) as exc_info:
        await auth_api.refresh(authorization=authorization)

    assert exc_info.value.status_code == 401
    assert stub.refresh_calls == 0


@pytest.mark.asyncio
async def test_refresh_accepts_the_active_refresh_token(monkeypatch):
    stub = _RefreshAuthStub()

    class _BridgeStub:
        def get_registered_device_id(self) -> str:
            return "device-id-123"

        def get_device_identifier(self) -> str:
            return "device-identifier-123"

    monkeypatch.setattr(auth_api, "get_upstream_auth_service", lambda: stub)
    monkeypatch.setattr(auth_api, "get_runtime_bridge", lambda: _BridgeStub())

    result = await auth_api.refresh(authorization="Bearer active-refresh-token")

    assert result["data"]["accessToken"] == "new-access"
    assert stub.refresh_calls == 1


@pytest.mark.asyncio
async def test_require_local_session_rejects_non_ascii_bearer_with_401(monkeypatch):
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="tökén")
    monkeypatch.setattr(
        auth_module,
        "get_upstream_auth_service",
        lambda: _AuthStub("user-123", access_token="upstream-access-token"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await auth_module.require_local_session(credentials)

    assert exc_info.value.status_code == 401


@pytest.mark.parametrize("hostile", ["..", "../../outside", "..\\..\\outside", "a/b", " "])
def test_profile_paths_refuse_a_user_id_that_is_not_one_component(hostile):
    """/auth/restore and unverified JWT claims both reach this with raw input."""
    from client_backend.core.paths import get_profile_subdir, profile_subdir_path

    with pytest.raises(ValueError):
        profile_subdir_path(hostile, "session")
    with pytest.raises(ValueError):
        get_profile_subdir(hostile, "session")


@pytest.mark.asyncio
async def test_restore_of_a_traversing_user_id_touches_nothing(tmp_path, monkeypatch):
    from client_backend.core.config import client_settings
    from client_backend.services.upstream_auth import UpstreamAuthService

    class _ClientStub:
        base_url = "http://server.test"

        def set_tokens(self, tokens):
            raise AssertionError("no tokens may be loaded for a malformed id")

    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path / "profile"))
    outside = tmp_path / "outside" / "session"

    restored = await UpstreamAuthService(server_client=_ClientStub()).restore_session(
        "../../outside"
    )

    assert restored is False
    assert not outside.exists()


@pytest.mark.asyncio
async def test_rejected_restore_does_not_leave_dead_tokens_on_the_client(tmp_path, monkeypatch):
    from datetime import datetime

    from client_backend.core.config import client_settings
    from client_backend.services.server_api import AuthenticationError
    from client_backend.services.upstream_auth import StoredCredentials, UpstreamAuthService

    class _ClientStub:
        base_url = "http://server.test"

        def __init__(self) -> None:
            self.tokens = None

        def set_tokens(self, tokens):
            self.tokens = tokens

        def get_tokens(self):
            return self.tokens

        async def refresh_token(self):
            raise AuthenticationError("refresh rejected", status_code=401)

    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path / "profile"))
    client = _ClientStub()
    service = UpstreamAuthService(server_client=client)
    service._save_credentials(
        StoredCredentials(
            user_id="user-123",
            username="user@example.com",
            tokens=TokenPair(access_token="dead-access", refresh_token="dead-refresh"),
            stored_at=datetime.now(UTC),
            server_url=client.base_url,
        )
    )

    assert await service.restore_session("user-123") is False
    assert client.get_tokens() is None


def test_unreadable_credentials_log_the_error_type_not_its_text(tmp_path, monkeypatch, caplog):
    """A pydantic error echoes its input, so logging it would write the tokens to the log."""
    import json
    import logging

    from client_backend.core.config import client_settings
    from client_backend.core.paths import profile_subdir_path
    from client_backend.services.upstream_auth import UpstreamAuthService

    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path / "profile"))
    path = profile_subdir_path("user-123", "session") / "credentials.json"
    path.parent.mkdir(parents=True)
    # A legacy plaintext file missing a required field: the ValidationError
    # text would repeat the whole input, tokens included.
    path.write_text(
        json.dumps(
            {
                "user_id": "user-123",
                "username": "user@example.com",
                "tokens": {"access_token": "sk-leaky-access", "refresh_token": "sk-leaky-refresh"},
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        loaded = UpstreamAuthService(server_client=object())._load_credentials("user-123")

    assert loaded is None
    assert "ValidationError" in caplog.text
    assert "sk-leaky" not in caplog.text


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


# ── Logout needs the session it ends ────────────────────────────────────────


class _StoppableBridge:
    def __init__(self) -> None:
        self.stop_calls = 0

    async def stop(self) -> None:
        self.stop_calls += 1

    def get_registered_device_id(self) -> None:
        return None

    def get_device_identifier(self) -> str:
        return "device-identifier-123"


@pytest.fixture
def signed_in_sidecar(tmp_path, monkeypatch):
    """The real auth service, signed in with credentials stored on disk."""
    from datetime import datetime

    import httpx
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from client_backend.core.config import client_settings
    from client_backend.services.server_api import ServerAPIClient
    from client_backend.services.upstream_auth import StoredCredentials, UpstreamAuthService

    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path / "profile"))
    upstream_calls: list[str] = []

    def _upstream(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request.url.path)
        return httpx.Response(200, json={"success": True, "message": "Logged out"})

    server_client = ServerAPIClient(base_url="http://server.test")
    server_client._client = httpx.AsyncClient(  # noqa: SLF001 - exercises the HTTP boundary
        base_url=server_client.base_url,
        transport=httpx.MockTransport(_upstream),
    )
    tokens = TokenPair(access_token="upstream-access", refresh_token="upstream-refresh")
    server_client.set_tokens(tokens)
    service = UpstreamAuthService(server_client=server_client)
    service._credentials = StoredCredentials(
        user_id="user-123",
        username="user@example.com",
        tokens=tokens,
        stored_at=datetime.now(UTC),
        server_url=server_client.base_url,
    )
    service._current_user_id = "user-123"
    service._save_credentials(service._credentials)
    credentials_path = service._get_credentials_path("user-123")

    bridge = _StoppableBridge()
    monkeypatch.setattr(auth_api, "get_upstream_auth_service", lambda: service)
    monkeypatch.setattr(auth_module, "get_upstream_auth_service", lambda: service)
    monkeypatch.setattr(auth_api, "get_runtime_bridge", lambda: bridge)
    monkeypatch.setattr(auth_module, "get_runtime_bridge", lambda: bridge)

    app = FastAPI()
    app.include_router(auth_api.router)
    return TestClient(app), service, bridge, credentials_path, upstream_calls


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer not-the-session"}, {"Authorization": "Basic dXNlcg=="}],
)
def test_logout_without_the_session_is_refused_and_the_session_survives(
    signed_in_sidecar, headers
):
    """Any local process can reach the port; only the session holder may end it."""
    client, service, bridge, credentials_path, upstream_calls = signed_in_sidecar

    response = client.post("/auth/logout", headers=headers)

    assert response.status_code == 401
    assert credentials_path.exists()
    assert service.is_authenticated()
    assert bridge.stop_calls == 0
    assert upstream_calls == []


def test_logout_with_the_local_session_ends_it(signed_in_sidecar):
    client, service, bridge, credentials_path, upstream_calls = signed_in_sidecar
    token = create_local_session_token(
        user_id="user-123",
        server_user_id="user-123",
        device_identifier="device-identifier-123",
    )

    response = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert not credentials_path.exists()
    assert not service.is_authenticated()
    assert bridge.stop_calls == 1
    assert upstream_calls == ["/auth/logout"]
