import json

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from client_backend.api import conversations as conversations_api
from client_backend.core.auth import require_local_session
from client_backend.main import create_app


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(conversations_api.router)
    app.dependency_overrides[conversations_api.require_local_session] = lambda: object()
    return app


def test_conversation_routes_use_proxy_server_request(monkeypatch):
    calls: list[dict] = []

    async def _proxy(
        request: Request,
        *,
        upstream_path: str,
        params_override=None,
    ):
        calls.append(
            {
                "method": request.method,
                "path": upstream_path,
                "params": params_override,
            }
        )
        return JSONResponse(
            {"path": upstream_path}, status_code=201 if request.method == "POST" else 200
        )

    monkeypatch.setattr(conversations_api, "proxy_server_request", _proxy)

    with TestClient(_build_app()) as client:
        list_response = client.get("/conversations/")
        messages_response = client.get("/conversations/conv-1/messages")
        title_response = client.post("/conversations/generate-title", json={"message": "hello"})
        create_response = client.post("/conversations/", json={"title": "New title"})
        get_response = client.get("/conversations/conv-1")
        patch_response = client.patch("/conversations/conv-1", json={"title": "Renamed"})
        delete_response = client.delete("/conversations/conv-1")

    assert list_response.status_code == 200
    assert messages_response.status_code == 200
    assert title_response.status_code == 201
    assert create_response.status_code == 201
    assert get_response.status_code == 200
    assert patch_response.status_code == 200
    assert delete_response.status_code == 200
    assert calls == [
        {
            "method": "GET",
            "path": "/conversations/",
            "params": {"page": 1, "limit": 20, "include": [], "latestMessages": 3},
        },
        {
            "method": "GET",
            "path": "/conversations/conv-1/messages",
            "params": {"page": 1, "limit": 50, "include": []},
        },
        {
            "method": "POST",
            "path": "/conversations/generate-title",
            "params": None,
        },
        {
            "method": "POST",
            "path": "/conversations/",
            "params": None,
        },
        {
            "method": "GET",
            "path": "/conversations/conv-1",
            "params": None,
        },
        {
            "method": "PATCH",
            "path": "/conversations/conv-1",
            "params": None,
        },
        {
            "method": "DELETE",
            "path": "/conversations/conv-1",
            "params": None,
        },
    ]


def test_list_conversations_forwards_pagination_sorting(monkeypatch):
    calls: list[dict] = []

    async def _proxy(
        request: Request,
        *,
        upstream_path: str,
        params_override=None,
    ):
        calls.append(
            {
                "path": upstream_path,
                "params": params_override,
            }
        )
        return JSONResponse({"path": upstream_path})

    monkeypatch.setattr(conversations_api, "proxy_server_request", _proxy)

    with TestClient(_build_app()) as client:
        response = client.get(
            "/conversations/"
            "?page=2&limit=5&orderBy=createdAt&orderDirection=asc"
            "&latestMessages=7&include=messages&include=feedback"
            "&search=Roadmap%20Q3"
        )

    assert response.status_code == 200
    assert calls == [
        {
            "path": "/conversations/",
            "params": {
                "page": 2,
                "limit": 5,
                "orderBy": "createdAt",
                "orderDirection": "asc",
                "include": ["messages", "feedback"],
                "latestMessages": 7,
                "search": "Roadmap Q3",
            },
        }
    ]


class _FakeUpstreamResponse:
    def __init__(self, payload: dict, status_code: int) -> None:
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        self.content = json.dumps(payload).encode("utf-8")
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeServerClient:
    def __init__(self, payload: dict, status_code: int) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self._response = _FakeUpstreamResponse(payload, status_code)

    async def request_response(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        return self._response


def _sidecar(monkeypatch, payload: dict, status_code: int, *, signed_in: bool = True):
    server = _FakeServerClient(payload, status_code)
    monkeypatch.setattr("client_backend.api.common.get_server_client", lambda: server)
    app = create_app()
    if signed_in:
        app.dependency_overrides[require_local_session] = lambda: object()
    return TestClient(app), server


_FORK_BODY = {"messageId": "m-1", "requestId": "r-1", "title": "Alt"}


@pytest.mark.parametrize("prefix", ["", "/api"])
@pytest.mark.parametrize("status_code", [201, 200])
def test_fork_is_proxied_at_both_aliases_keeping_the_upstream_status(
    monkeypatch, prefix, status_code
):
    payload = {"success": True, "data": {"id": "child-1"}}
    client, server = _sidecar(monkeypatch, payload, status_code)

    with client:
        response = client.post(f"{prefix}/conversations/conv-1/fork", json=_FORK_BODY)

    assert response.status_code == status_code
    assert response.json() == payload
    method, path, kwargs = server.calls[0]
    assert (method, path) == ("POST", "/conversations/conv-1/fork")
    assert kwargs["json"] == _FORK_BODY


@pytest.mark.parametrize(
    ("status_code", "code"),
    [
        (409, "conversation_turn_conflict"),
        (409, "FORK_POINT_NOT_COMPLETED"),
        (410, "FORK_RESULT_DELETED"),
        (413, "FORK_HISTORY_TOO_LARGE"),
        (404, "FORK_MESSAGE_NOT_FOUND"),
    ],
)
def test_fork_upstream_error_envelopes_pass_through_unchanged(monkeypatch, status_code, code):
    envelope = {"success": False, "code": code, "message": "refused"}
    client, _ = _sidecar(monkeypatch, envelope, status_code)

    with client:
        response = client.post("/api/conversations/conv-1/fork", json=_FORK_BODY)

    assert response.status_code == status_code
    assert response.json() == envelope


@pytest.mark.parametrize("prefix", ["", "/api"])
def test_fork_without_a_local_session_is_rejected_and_never_forwarded(monkeypatch, prefix):
    client, server = _sidecar(monkeypatch, {"success": True}, 201, signed_in=False)

    with client:
        response = client.post(f"{prefix}/conversations/conv-1/fork", json=_FORK_BODY)

    assert response.status_code in (401, 403)
    assert server.calls == []
