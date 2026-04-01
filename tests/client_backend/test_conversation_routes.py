from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from client_backend.api import conversations as conversations_api


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
        inject_device_context: bool = False,
        params_override=None,
    ):
        calls.append(
            {
                "method": request.method,
                "path": upstream_path,
                "params": params_override,
                "inject_device_context": inject_device_context,
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
            "inject_device_context": False,
        },
        {
            "method": "GET",
            "path": "/conversations/conv-1/messages",
            "params": {"page": 1, "limit": 50, "include": []},
            "inject_device_context": False,
        },
        {
            "method": "POST",
            "path": "/conversations/generate-title",
            "params": None,
            "inject_device_context": False,
        },
        {
            "method": "POST",
            "path": "/conversations/",
            "params": None,
            "inject_device_context": False,
        },
        {
            "method": "GET",
            "path": "/conversations/conv-1",
            "params": None,
            "inject_device_context": False,
        },
        {
            "method": "PATCH",
            "path": "/conversations/conv-1",
            "params": None,
            "inject_device_context": False,
        },
        {
            "method": "DELETE",
            "path": "/conversations/conv-1",
            "params": None,
            "inject_device_context": False,
        },
    ]
