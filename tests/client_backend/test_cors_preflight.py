"""The sidecar must answer browser CORS preflights itself, before auth/routing.

The FE (e.g. http://localhost:3000) talks to the sidecar cross-origin with an
Authorization header, so every call is preflighted. CORSMiddleware intercepts
the OPTIONS request before ``require_local_session`` runs; if this contract
regresses (middleware removed/reordered, or the configured FE origin dropped),
every FE call fails with "preflight ... does not have HTTP ok status".

Origins are an explicit allowlist rather than ``*``: this process executes local
shell commands and skill runtimes, so an arbitrary page must not be able to
drive it. Both directions are covered here — a configured origin is echoed, an
unconfigured one gets no allow-origin header.
"""

from __future__ import annotations

import httpx
import pytest

from client_backend.main import create_app


@pytest.fixture()
def cors_client() -> httpx.AsyncClient:
    # ASGITransport without lifespan: middleware behavior only, no MCP/skills
    # startup side effects.
    transport = httpx.ASGITransport(app=create_app())
    return httpx.AsyncClient(transport=transport, base_url="http://sidecar.local")


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_method", ["GET", "PATCH", "DELETE", "POST"])
async def test_preflight_succeeds_without_local_session(cors_client, requested_method):
    async with cors_client as client:
        response = await client.options(
            "/ai/conversations/827faf55-1041-4357-9e1a-0f7d031fa546",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": requested_method,
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert requested_method in response.headers["access-control-allow-methods"]


@pytest.mark.asyncio
async def test_preflight_from_unconfigured_origin_is_not_allowed(cors_client):
    async with cors_client as client:
        response = await client.options(
            "/messages/stop",
            headers={
                "Origin": "http://evil.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )

    assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
async def test_preflight_allows_authorization_header(cors_client):
    async with cors_client as client:
        response = await client.options(
            "/messages/stop",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )

    assert response.status_code == 200
    allowed = response.headers.get("access-control-allow-headers", "").lower()
    assert "authorization" in allowed
