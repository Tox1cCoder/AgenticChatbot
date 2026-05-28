"""Tests for the local backend widget action proxy."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_proxy_widget_action_forwards_to_canonical_server(monkeypatch):
    from client_backend.api import proxy as proxy_api

    upstream_payload = {
        "widget_id": "w-1",
        "session_id": "conv-1",
        "action_key": "explain_current_state",
        "content": "Explain damping=0.4 in the context of the current answer.",
    }

    class FakeJsonResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = json.dumps(upstream_payload).encode("utf-8")

        def json(self):
            return upstream_payload

    class FakeServerClient:
        def __init__(self):
            self.calls: list[tuple[str, str, dict]] = []

        async def request_response(self, method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            return FakeJsonResponse()

    server_client = FakeServerClient()
    monkeypatch.setattr(
        "client_backend.api.common.get_server_client",
        lambda: server_client,
    )

    app = FastAPI()
    app.include_router(proxy_api.router)
    app.dependency_overrides[proxy_api.require_local_session] = lambda: object()

    with TestClient(app) as client:
        response = client.post(
            "/widgets/w-1/actions/explain_current_state",
            json={
                "state_patch": {"control_values": {"damping": 0.4}},
                "input_values": {"note": "demo"},
            },
        )

    assert response.status_code == 200
    assert response.json() == upstream_payload
    assert server_client.calls
    method, path, _ = server_client.calls[0]
    assert method == "POST"
    assert path == "/widgets/w-1/actions/explain_current_state"


@pytest.mark.asyncio
async def test_proxy_widget_action_passes_through_404(monkeypatch):
    from client_backend.api import proxy as proxy_api

    upstream_payload = {"error": "Widget action explain_current_state not found"}

    class FakeJsonResponse:
        status_code = 404
        headers = {"content-type": "application/json"}
        content = json.dumps(upstream_payload).encode("utf-8")

        def json(self):
            return upstream_payload

    class FakeServerClient:
        async def request_response(self, method, path, **kwargs):
            return FakeJsonResponse()

    monkeypatch.setattr(
        "client_backend.api.common.get_server_client",
        lambda: FakeServerClient(),
    )

    app = FastAPI()
    app.include_router(proxy_api.router)
    app.dependency_overrides[proxy_api.require_local_session] = lambda: object()

    with TestClient(app) as client:
        response = client.post(
            "/widgets/missing/actions/explain_current_state",
            json={},
        )

    assert response.status_code == 404
    assert "not found" in response.json()["error"]
