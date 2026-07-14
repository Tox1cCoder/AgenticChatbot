"""The sidecar proxies HITL resources without device stamping."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from client_backend.api import proxy as proxy_module
from client_backend.api.proxy import router
from client_backend.core.auth import require_local_session


@pytest.fixture
def client(monkeypatch):
    captured = {}

    async def _fake_proxy(request, *, upstream_path, **kwargs):
        from fastapi.responses import JSONResponse

        captured["path"] = upstream_path
        captured["method"] = request.method
        captured["params_override"] = kwargs.get("params_override")
        return JSONResponse(status_code=200, content={"success": True, "data": {}})

    monkeypatch.setattr(proxy_module, "proxy_server_request", _fake_proxy)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_local_session] = lambda: object()
    return TestClient(app), captured


def test_get_hitl_settings_proxies_without_device_param(client):
    test_client, captured = client
    resp = test_client.get("/hitl/settings")
    assert resp.status_code == 200
    assert captured["path"] == "/hitl/settings"
    assert captured["method"] == "GET"
    assert captured["params_override"] is None  # per-user, no device stamping


def test_post_and_delete_hitl_settings_proxy(client):
    test_client, captured = client
    assert test_client.post("/hitl/settings", json={"items": []}).status_code == 200
    assert captured["path"] == "/hitl/settings"
    assert (
        test_client.request(
            "DELETE", "/hitl/settings", params={"scope_type": "server", "scope_value": "excel"}
        ).status_code
        == 200
    )
    assert captured["method"] == "DELETE"


def test_get_hitl_interrupt_proxies_without_device_context(client):
    test_client, captured = client

    response = test_client.get("/hitl/interrupts/int-1")

    assert response.status_code == 200
    assert captured["path"] == "/hitl/interrupts/int-1"
    assert captured["method"] == "GET"
    assert captured["params_override"] is None
