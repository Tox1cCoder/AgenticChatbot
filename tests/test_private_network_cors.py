from __future__ import annotations

import httpx
import pytest


async def _private_network_preflight(app, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.local") as client:
        return await client.options(
            path,
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "PATCH",
                "Access-Control-Request-Headers": "authorization,content-type",
                "Access-Control-Request-Private-Network": "true",
            },
        )


@pytest.mark.asyncio
async def test_server_allows_private_network_preflight_for_lan_frontend():
    from app.main import create_app

    response = await _private_network_preflight(
        create_app(),
        "/ai/conversations/5ce46cff-af1f-4723-9608-83ebbcaf01ba",
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-private-network"] == "true"
    assert "PATCH" in response.headers["access-control-allow-methods"]


@pytest.mark.asyncio
async def test_client_backend_allows_private_network_preflight_for_lan_frontend():
    from client_backend.main import create_app

    response = await _private_network_preflight(
        create_app(),
        "/ai/conversations/5ce46cff-af1f-4723-9608-83ebbcaf01ba",
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-private-network"] == "true"
    assert "PATCH" in response.headers["access-control-allow-methods"]
