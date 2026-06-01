"""Tests for the local backend custom-agent proxy routes."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _fake_server(payload, status_code=200):
    class FakeResponse:
        def __init__(self):
            self.status_code = status_code
            self.headers = {"content-type": "application/json"}
            self.content = json.dumps(payload).encode("utf-8")

        def json(self):
            return payload

    class FakeServerClient:
        def __init__(self):
            self.calls: list[tuple[str, str, dict]] = []

        async def request_response(self, method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            return FakeResponse()

    return FakeServerClient()


def _client(monkeypatch, server_client):
    from client_backend.api import proxy as proxy_api

    monkeypatch.setattr("client_backend.api.common.get_server_client", lambda: server_client)
    app = FastAPI()
    app.include_router(proxy_api.router)
    app.dependency_overrides[proxy_api.require_local_session] = lambda: object()
    return TestClient(app), proxy_api


def _runtime_bridge(device_id: str | None):
    class FakeBridge:
        def get_registered_device_id(self):
            return device_id

    return FakeBridge()


@pytest.mark.asyncio
async def test_proxy_list_and_create_custom_agents(monkeypatch):
    server = _fake_server([{"id": "a1"}])
    client, _ = _client(monkeypatch, server)
    with client:
        assert client.get("/custom-agents").status_code == 200
        assert client.post("/custom-agents", json={"name": "x"}).status_code == 200
    methods = [(m, p) for m, p, _ in server.calls]
    assert ("GET", "/custom-agents") in methods
    assert ("POST", "/custom-agents") in methods


@pytest.mark.asyncio
async def test_proxy_options_and_single_agent(monkeypatch):
    server = _fake_server({"ok": True})
    client, _ = _client(monkeypatch, server)
    with client:
        assert client.get("/custom-agents/options").status_code == 200
        assert client.get("/custom-agents/abc").status_code == 200
        assert client.delete("/custom-agents/abc").status_code == 200
    paths = [(m, p) for m, p, _ in server.calls]
    assert ("GET", "/custom-agents/options") in paths
    assert ("GET", "/custom-agents/abc") in paths
    assert ("DELETE", "/custom-agents/abc") in paths


@pytest.mark.asyncio
async def test_proxy_forwards_active_device_id_for_options_and_mutations(monkeypatch):
    server = _fake_server({"ok": True})
    client, _ = _client(monkeypatch, server)
    monkeypatch.setattr(
        "client_backend.api.proxy.get_runtime_bridge",
        lambda: _runtime_bridge("device-123"),
    )

    with client:
        assert client.get("/custom-agents/options").status_code == 200
        assert client.post("/custom-agents", json={"name": "x"}).status_code == 200
        assert client.patch("/custom-agents/abc", json={"name": "y"}).status_code == 200

    params_by_call = {(method, path): kwargs["params"] for method, path, kwargs in server.calls}
    assert ("deviceId", "device-123") in params_by_call[("GET", "/custom-agents/options")]
    assert ("deviceId", "device-123") in params_by_call[("POST", "/custom-agents")]
    assert ("deviceId", "device-123") in params_by_call[("PATCH", "/custom-agents/abc")]


@pytest.mark.asyncio
async def test_proxy_conversation_custom_agents(monkeypatch):
    server = _fake_server([{"id": "a1"}])
    client, _ = _client(monkeypatch, server)
    with client:
        assert client.get("/conversations/c1/custom-agents").status_code == 200
        assert (
            client.put("/conversations/c1/custom-agents", json={"customAgentIds": []}).status_code
            == 200
        )
    paths = [(m, p) for m, p, _ in server.calls]
    assert ("GET", "/conversations/c1/custom-agents") in paths
    assert ("PUT", "/conversations/c1/custom-agents") in paths
