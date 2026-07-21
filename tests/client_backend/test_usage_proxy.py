"""Contract tests for model-usage routes exposed by the local sidecar."""

from __future__ import annotations

import json
from collections.abc import Callable
from uuid import uuid4

import httpx
import pytest


class _ActiveAuthService:
    def __init__(self, *, user_id: str, access_token: str) -> None:
        self._user_id = user_id
        self._access_token = access_token

    def get_current_user_id(self) -> str:
        return self._user_id

    def is_authenticated(self) -> bool:
        return True

    def get_current_access_token(self) -> str:
        return self._access_token


def _upstream_response(
    payload: object,
    *,
    status_code: int = 200,
) -> Callable[[httpx.Request], httpx.Response]:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            content=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
            request=request,
        )

    return _handler


def _build_clients(monkeypatch, handler: Callable[[httpx.Request], httpx.Response]):
    from client_backend.core.security import create_local_session_token
    from client_backend.main import create_app
    from client_backend.services.server_api import ServerAPIClient, TokenPair

    user_id = str(uuid4())
    access_token = "canonical-upstream-access-token"
    server_client = ServerAPIClient(base_url="https://canonical.example.test")
    server_client.set_tokens(
        TokenPair(
            access_token=access_token,
            refresh_token="refresh-token",
            user_id=user_id,
        )
    )
    server_client._client = httpx.AsyncClient(  # noqa: SLF001 - exercises the HTTP boundary
        base_url=server_client.base_url,
        transport=httpx.MockTransport(handler),
    )

    auth_service = _ActiveAuthService(user_id=user_id, access_token=access_token)
    monkeypatch.setattr(
        "client_backend.core.auth.get_upstream_auth_service",
        lambda: auth_service,
    )
    monkeypatch.setattr(
        "client_backend.api.common.get_server_client",
        lambda: server_client,
    )

    local_token = create_local_session_token(
        user_id=user_id,
        server_user_id=user_id,
        device_identifier="local-installation",
    )
    sidecar_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()),
        base_url="http://sidecar.test",
        headers={"Authorization": f"Bearer {local_token}"},
    )
    return sidecar_client, server_client, access_token


@pytest.mark.asyncio
@pytest.mark.parametrize("mount", ["", "/api"])
async def test_dashboard_proxy_preserves_query_identity_and_upstream_bearer(
    monkeypatch,
    mount,
):
    captured: list[httpx.Request] = []
    payload = {"success": True, "data": {"totals": {"totalTokens": 12}}}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _upstream_response(payload)(request)

    sidecar, server, access_token = _build_clients(monkeypatch, _handler)
    try:
        response = await sidecar.get(
            f"{mount}/usage/dashboard"
            "?timezone=America%2FNew_York&provider=gemini&provider=openai"
            "&userId=untrusted-camel&user_id=untrusted-snake&deviceId=foreign-device"
        )
    finally:
        await sidecar.aclose()
        await server.close()

    assert response.status_code == 200
    assert response.json() == payload
    assert len(captured) == 1
    upstream = captured[0]
    assert upstream.method == "GET"
    assert upstream.url.path == "/usage/dashboard"
    assert upstream.url.query == (
        b"timezone=America%2FNew_York&provider=gemini&provider=openai"
        b"&userId=untrusted-camel&user_id=untrusted-snake&deviceId=foreign-device"
    )
    assert upstream.headers["authorization"] == f"Bearer {access_token}"


@pytest.mark.asyncio
@pytest.mark.parametrize("mount", ["", "/api"])
@pytest.mark.parametrize(
    ("status_code", "payload"),
    [
        (200, {"success": True, "data": {"series": []}}),
        (404, {"success": False, "message": "Conversation not found", "data": None}),
    ],
)
async def test_conversation_proxy_preserves_canonical_status_and_body(
    monkeypatch,
    mount,
    status_code,
    payload,
):
    conversation_id = uuid4()
    captured: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _upstream_response(payload, status_code=status_code)(request)

    sidecar, server, _ = _build_clients(monkeypatch, _handler)
    try:
        response = await sidecar.get(
            f"{mount}/usage/conversations/{conversation_id}"
            "?bucket=day&timezone=Asia%2FBangkok&dimension=model&dimension=operation"
        )
    finally:
        await sidecar.aclose()
        await server.close()

    assert response.status_code == status_code
    assert response.json() == payload
    assert len(captured) == 1
    upstream = captured[0]
    assert upstream.url.path == f"/usage/conversations/{conversation_id}"
    assert upstream.url.query == (
        b"bucket=day&timezone=Asia%2FBangkok&dimension=model&dimension=operation"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mount", ["", "/api"])
async def test_usage_routes_require_a_local_session(monkeypatch, mount):
    requests: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _upstream_response({"success": True})(request)

    sidecar, server, _ = _build_clients(monkeypatch, _handler)
    sidecar.headers.pop("Authorization")
    try:
        response = await sidecar.get(f"{mount}/usage/dashboard")
    finally:
        await sidecar.aclose()
        await server.close()

    assert response.status_code == 401
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mount", ["", "/api"])
async def test_conversation_usage_route_rejects_non_uuid_without_proxying(
    monkeypatch,
    mount,
):
    requests: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _upstream_response({"success": True})(request)

    sidecar, server, _ = _build_clients(monkeypatch, _handler)
    try:
        response = await sidecar.get(f"{mount}/usage/conversations/not-a-uuid")
    finally:
        await sidecar.aclose()
        await server.close()

    assert response.status_code == 422
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/ai/usage/dashboard", "/api/ai/usage/dashboard"])
async def test_usage_proxy_has_no_ai_alias(monkeypatch, path):
    sidecar, server, _ = _build_clients(
        monkeypatch,
        _upstream_response({"success": True}),
    )
    try:
        response = await sidecar.get(path)
    finally:
        await sidecar.aclose()
        await server.close()

    assert response.status_code == 404
