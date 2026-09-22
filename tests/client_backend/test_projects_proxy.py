"""Tests for the local backend project proxy routes.

The primary regression covered here: pressing "Save" to create a project in
the Streamlit UI produced a 404, because the canonical server grew nine
``/projects`` routes but nobody added a matching proxy to the client_backend
sidecar (an explicit allowlist proxy with no catch-all passthrough). The
``test_create_project_reaches_sidecar`` case below is the exact bug the user
hit: it must fail with a 404 before the sidecar app registers ``projects_router``
and pass once it is wired into ``client_backend.main``.
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from client_backend.core.auth import require_local_session
from client_backend.main import create_app


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


def _sidecar_app_client(monkeypatch, server_client):
    """Build the real sidecar app exactly as ``client_backend.main`` does.

    Using ``create_app()`` exercises the actual ``compatibility_routers``
    registration, so this reproduces the user's bug (and its fix) faithfully
    rather than only testing the projects router in isolation: before
    ``projects_router`` is added to that list, ``POST /projects`` 404s here
    exactly as it did against the real sidecar process.
    """
    monkeypatch.setattr("client_backend.api.common.get_server_client", lambda: server_client)
    app = create_app()
    app.dependency_overrides[require_local_session] = lambda: object()
    return TestClient(app)


def _projects_router_client(monkeypatch, server_client):
    """Mount only the projects router, bypassing local-session auth."""
    from client_backend.api import projects as projects_api

    monkeypatch.setattr("client_backend.api.common.get_server_client", lambda: server_client)
    app = FastAPI()
    app.include_router(projects_api.router)
    app.dependency_overrides[projects_api.require_local_session] = lambda: object()
    return TestClient(app), projects_api


def test_create_project_reaches_sidecar(monkeypatch):
    """POST /projects must forward to the canonical server, not 404.

    This is the exact request the user's "Save" button sent
    (``POST /projects HTTP/1.1`` -> 404 Not Found before the fix). It is
    built against the real sidecar app (``client_backend.main.create_app()``)
    so it fails the same way the live process did: before
    ``projects_router`` is registered in ``compatibility_routers``, this app
    has no ``/projects`` route at all.
    """
    server = _fake_server({"id": "p1", "name": "New Project"}, status_code=201)
    client = _sidecar_app_client(monkeypatch, server)

    with client:
        response = client.post("/projects", json={"name": "New Project"})

    assert response.status_code == 201
    methods = [(m, p) for m, p, _ in server.calls]
    assert ("POST", "/projects") in methods


def test_list_projects_forwards(monkeypatch):
    server = _fake_server([{"id": "p1"}])
    client, _ = _projects_router_client(monkeypatch, server)

    with client:
        response = client.get("/projects")

    assert response.status_code == 200
    methods = [(m, p) for m, p, _ in server.calls]
    assert ("GET", "/projects") in methods


def test_get_update_delete_project_forward(monkeypatch):
    server = _fake_server({"id": "p1"})
    client, _ = _projects_router_client(monkeypatch, server)

    with client:
        assert client.get("/projects/p1").status_code == 200
        assert client.patch("/projects/p1", json={"name": "Renamed"}).status_code == 200
        assert client.delete("/projects/p1").status_code == 200

    paths = [(m, p) for m, p, _ in server.calls]
    assert ("GET", "/projects/p1") in paths
    assert ("PATCH", "/projects/p1") in paths
    assert ("DELETE", "/projects/p1") in paths


def test_project_custom_agents_forward(monkeypatch):
    server = _fake_server([{"id": "a1"}])
    client, _ = _projects_router_client(monkeypatch, server)

    with client:
        assert client.get("/projects/p1/custom-agents").status_code == 200
        assert (
            client.put(
                "/projects/p1/custom-agents", json={"customAgentIds": ["a1"]}
            ).status_code
            == 200
        )

    paths = [(m, p) for m, p, _ in server.calls]
    assert ("GET", "/projects/p1/custom-agents") in paths
    assert ("PUT", "/projects/p1/custom-agents") in paths


def test_project_conversation_membership_forwards(monkeypatch):
    server = _fake_server({"ok": True})
    client, _ = _projects_router_client(monkeypatch, server)

    with client:
        attach = client.put("/projects/p1/conversations/c1")
        detach = client.delete("/projects/p1/conversations/c1")

    assert attach.status_code == 200
    assert detach.status_code == 200
    paths = [(m, p) for m, p, _ in server.calls]
    assert ("PUT", "/projects/p1/conversations/c1") in paths
    assert ("DELETE", "/projects/p1/conversations/c1") in paths


def test_list_conversations_forwards_project_id(monkeypatch):
    """GET /conversations?projectId=... must forward projectId upstream.

    Before this fix, ``list_conversations`` built an explicit ``params``
    allowlist that silently dropped ``projectId``, so a project's
    conversation list would show every conversation instead of just its own.
    """
    from client_backend.api import conversations as conversations_api

    server = _fake_server({"items": []})
    monkeypatch.setattr("client_backend.api.common.get_server_client", lambda: server)
    app = FastAPI()
    app.include_router(conversations_api.router)
    app.dependency_overrides[conversations_api.require_local_session] = lambda: object()

    with TestClient(app) as client:
        response = client.get("/conversations/?projectId=proj-1")

    assert response.status_code == 200
    assert len(server.calls) == 1
    _method, _path, kwargs = server.calls[0]
    assert kwargs["params"]["projectId"] == "proj-1"
