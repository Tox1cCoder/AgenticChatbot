"""A forwarded path parameter must stay one upstream path segment.

``/ai/conversations/%2E%2E`` arrives with ``conversation_id == ".."``; formatted
into ``/ai/conversations/..`` it would be normalised by the HTTP client and
reach a different upstream route than the one this sidecar allowlists. The
refusal lives in ``ServerAPIClient``, the one place every forwarded request
passes, so these tests drive the real routers against a recording transport
rather than stubbing the forwarding helper away.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI

from client_backend.api import common as common_module
from client_backend.api import conversations as conversations_module
from client_backend.api import documents as documents_module
from client_backend.api import messages as messages_module
from client_backend.api import projects as projects_module
from client_backend.api import proxy as proxy_module
from client_backend.core.auth import require_local_session
from client_backend.services import server_api
from client_backend.services.server_api import ServerAPIClient, UnforwardablePathError


class _DisconnectedBridge:
    def is_connected(self) -> bool:
        return False

    def get_registered_device_id(self) -> None:
        return None


class _SignedOutUpstream:
    def is_authenticated(self) -> bool:
        return False


def _recording_server_client(forwarded: list[httpx.Request]) -> ServerAPIClient:
    def _handler(request: httpx.Request) -> httpx.Response:
        forwarded.append(request)
        return httpx.Response(
            200,
            content=json.dumps({"success": True}).encode(),
            headers={"content-type": "application/json"},
        )

    client = ServerAPIClient(base_url="https://canonical.example.test")
    client._client = httpx.AsyncClient(  # noqa: SLF001 - exercises the HTTP boundary
        base_url=client.base_url,
        transport=httpx.MockTransport(_handler),
    )
    return client


@pytest.fixture
async def sidecar(monkeypatch):
    forwarded: list[httpx.Request] = []
    server_client = _recording_server_client(forwarded)
    monkeypatch.setattr(server_api, "_server_client", server_client)
    monkeypatch.setattr(common_module, "get_runtime_bridge", _DisconnectedBridge)
    monkeypatch.setattr(proxy_module, "get_runtime_bridge", _DisconnectedBridge)
    monkeypatch.setattr(messages_module, "get_upstream_auth_service", _SignedOutUpstream)

    app = FastAPI()
    for module in (
        proxy_module,
        documents_module,
        conversations_module,
        projects_module,
        messages_module,
    ):
        app.include_router(module.router)
    app.include_router(messages_module.ai_sdk_router)
    app.dependency_overrides[require_local_session] = lambda: object()

    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://sidecar.test",
    )
    try:
        yield client, forwarded
    finally:
        await client.aclose()
        await server_client.close()


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
        ("GET", "/conversations/%2E%2E"),
        ("GET", "/conversations/%2E%2E/messages"),
        ("PATCH", "/conversations/%2E%2E"),
        ("DELETE", "/conversations/%2E%2E"),
        ("GET", "/conversations/%252E%252E"),
        ("GET", "/conversations/a%252Fb"),
        ("GET", "/conversations/abc%3Fall=1"),
        ("GET", "/ai/conversations/abc%23/messages"),
        ("GET", "/projects/%2E%2E"),
        ("PATCH", "/projects/%2E%2E"),
        ("GET", "/projects/%2E%2E/custom-agents"),
        ("PUT", "/projects/p1/conversations/%2E%2E"),
        ("DELETE", "/projects/%2E%2E/conversations/c1"),
        ("GET", "/messages/%2E%2E"),
        ("GET", "/messages/generations/%2E%2E?conversation_id=c1"),
    ],
)
async def test_a_dot_segment_path_parameter_is_refused_and_never_forwarded(sidecar, method, path):
    client, forwarded = sidecar

    response = await client.request(method, path)

    assert response.status_code == 404
    assert forwarded == []


@pytest.mark.parametrize("path", ["/api/chat/%2E%2E", "/ai/chat/%2E%2E"])
async def test_a_streamed_chat_to_a_dot_segment_id_is_never_forwarded(sidecar, path):
    client, forwarded = sidecar

    response = await client.post(path, json={"messages": []})

    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    assert forwarded == []
    assert [event.get("statusCode") for event in events] == [404]


@pytest.mark.parametrize(
    ("path", "upstream"),
    [
        ("/ai/conversations/3f2a.b-1/messages", "/ai/conversations/3f2a.b-1/messages"),
        ("/conversations/a..b", "/conversations/a..b"),
        ("/projects/p1/conversations/c1", "/projects/p1/conversations/c1"),
    ],
)
async def test_an_ordinary_path_parameter_is_still_forwarded(sidecar, path, upstream):
    client, forwarded = sidecar

    response = await client.get(path) if "/projects/" not in path else await client.put(path)

    assert response.status_code == 200
    assert [request.url.path for request in forwarded] == [upstream]


@pytest.mark.parametrize("path", ["/chat-images/..", "/web-images/%2E", "/users/a%2F..%2Fb"])
async def test_every_sending_method_of_the_server_client_refuses_the_path(path):
    forwarded: list[httpx.Request] = []
    client = _recording_server_client(forwarded)
    try:
        with pytest.raises(UnforwardablePathError) as refused:
            await client.request_response("GET", path)
        with pytest.raises(UnforwardablePathError):
            async with client.stream_media(path):
                pass
        with pytest.raises(UnforwardablePathError):
            async for _event in client.stream_sse(path):
                pass
    finally:
        await client.close()

    assert refused.value.status_code == 404
    assert forwarded == []
