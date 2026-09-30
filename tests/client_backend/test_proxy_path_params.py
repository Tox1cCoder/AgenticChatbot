"""A proxied path parameter must stay one path segment on the upstream server.

``/ai/conversations/%2E%2E`` arrives with ``conversation_id == ".."``; formatted
into ``/ai/conversations/..`` it would be normalised by the HTTP client and
reach a different upstream route than the one this sidecar allowlists.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from client_backend.api import documents as documents_module
from client_backend.api import proxy as proxy_module
from client_backend.core.auth import require_local_session


@pytest.fixture
def client(monkeypatch):
    forwarded: list[str] = []

    async def _fake_proxy(request, *, upstream_path, **kwargs):
        forwarded.append(upstream_path)
        return JSONResponse(status_code=200, content={"success": True})

    monkeypatch.setattr(proxy_module, "proxy_server_request", _fake_proxy)
    monkeypatch.setattr(documents_module, "proxy_server_request", _fake_proxy)
    monkeypatch.setattr(
        proxy_module,
        "get_runtime_bridge",
        lambda: type("Bridge", (), {"get_registered_device_id": staticmethod(lambda: None)})(),
    )
    app = FastAPI()
    app.include_router(proxy_module.router)
    app.include_router(documents_module.router)
    app.dependency_overrides[require_local_session] = lambda: object()
    return TestClient(app), forwarded


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/ai/conversations/%2E%2E"),
        ("DELETE", "/ai/conversations/%2E%2E"),
        ("GET", "/ai/conversations/%2e"),
        ("GET", "/ai/conversations/%2E%2E/messages"),
        ("GET", "/custom-agents/%2E%2E"),
        ("POST", "/task-plans/%2E%2E/complete"),
        ("PUT", "/messages/m1/feedbacks/%2E%2E"),
        ("GET", "/providers/%2E%2E/models"),
        ("GET", "/documents/%2E%2E"),
        ("DELETE", "/documents/%2E%2E"),
        ("GET", "/documents/task/%2E%2E"),
        ("GET", "/documents/conversation/%2E%2E"),
        ("GET", "/ai/conversations/a%5C.."),
    ],
)
def test_a_dot_segment_path_parameter_is_refused_and_never_forwarded(client, method, path):
    test_client, forwarded = client

    response = test_client.request(method, path)

    assert response.status_code == 404
    assert forwarded == []


def test_an_ordinary_path_parameter_is_still_forwarded(client):
    test_client, forwarded = client

    response = test_client.get("/ai/conversations/3f2a.b-1/messages")

    assert response.status_code == 200
    assert forwarded == ["/ai/conversations/3f2a.b-1/messages"]
