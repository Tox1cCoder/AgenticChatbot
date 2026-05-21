import json

import pytest

from client_backend.services.server_api import ServerAPIClient, ServerAPIError


class _ServerAPIClientStub(ServerAPIClient):
    def __init__(self):
        super().__init__(base_url="http://example.test", timeout=5)
        self.stream_calls: list[tuple[str, dict]] = []

    async def stream_sse(self, path: str, json: dict | None = None, **kwargs):
        self.stream_calls.append((path, json or {}))
        yield {"type": "complete"}


@pytest.mark.asyncio
async def test_resume_ai_sdk_interrupt_uses_canonical_server_route():
    client = _ServerAPIClientStub()

    events = [event async for event in client.resume_ai_sdk_interrupt({"threadId": "thread-1"})]

    assert events == [{"type": "complete"}]
    assert client.stream_calls == [
        (
            "/ai/resume-interrupt",
            {"threadId": "thread-1"},
        )
    ]


@pytest.mark.asyncio
async def test_stream_sse_reads_streaming_error_body_before_raising(monkeypatch):
    class FakeResponse:
        status_code = 422

        def __init__(self):
            self._read = False
            self._body = b'{"detail":"invalid payload"}'

        async def aread(self):
            self._read = True
            return self._body

        @property
        def content(self):
            if not self._read:
                raise RuntimeError("Attempted to access streaming response content without read()")
            return self._body

        @property
        def text(self):
            if not self._read:
                raise RuntimeError("Attempted to access streaming response text without read()")
            return self._body.decode("utf-8")

        def json(self):
            if not self._read:
                raise RuntimeError("Attempted to access streaming response json without read()")
            return {"detail": "invalid payload"}

        async def aiter_lines(self):
            if False:
                yield ""

    class FakeStream:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, *args):
            return False

    client = ServerAPIClient(base_url="http://example.test", timeout=5)
    client._tokens = type("T", (), {"access_token": "fake"})()

    real_client = await client._get_client()
    monkeypatch.setattr(real_client, "stream", lambda *args, **kwargs: FakeStream())

    with pytest.raises(ServerAPIError) as exc_info:
        [event async for event in client.stream_sse("/messages/stream")]

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == {"detail": "invalid payload"}


@pytest.mark.asyncio
async def test_stream_sse_passes_through_nested_context_window_metadata(monkeypatch):
    """A non-heartbeat ``complete`` event carrying nested
    ``message.metadata.context_window`` must be yielded unchanged so the
    desktop client can render the context-window indicator.
    """

    context_window_payload = {
        "provider": "openai",
        "model": "gpt-4o",
        "context_window_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "source": "registry",
        "known": True,
        "used_tokens": 12000,
        "used_token_source": "actual_input",
        "usage_ratio": 0.09375,
        "display_state": "ok",
    }
    complete_event = {
        "type": "complete",
        "message": {
            "id": "msg-1",
            "metadata": {"context_window": context_window_payload},
        },
    }

    sse_lines = [
        f"data: {json.dumps(complete_event)}",
        "",
        "data: [DONE]",
        "",
    ]

    class FakeResponse:
        status_code = 200

        async def aiter_lines(self):
            for line in sse_lines:
                yield line

        async def aread(self):
            return b""

        async def aclose(self):
            pass

    class FakeStream:
        def __init__(self, *args, **kwargs):
            self.response = FakeResponse()

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, *args):
            return False

    client = ServerAPIClient(base_url="http://example.test", timeout=5)
    client._tokens = type("T", (), {"access_token": "fake"})()

    real_client = await client._get_client()
    monkeypatch.setattr(real_client, "stream", lambda *args, **kwargs: FakeStream())

    events = [event async for event in client.stream_sse("/messages/stream")]

    assert events == [complete_event]
    # Deep equality above already covers it, but pin the critical leaf to
    # make a future regression (e.g. accidental key filtering) easy to spot.
    assert events[0]["message"]["metadata"]["context_window"] == context_window_payload


@pytest.mark.asyncio
async def test_proxy_server_request_preserves_arbitrary_response_fields(monkeypatch):
    """``proxy_server_request`` must forward the upstream JSON body verbatim,
    including newly-added fields like camelCase ``contextWindowTokens``
    inside ``/model-config/options``. Existing per-route tests stub the
    proxy helper itself, so the pass-through behaviour is not exercised
    end-to-end anywhere else.
    """

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from client_backend.api import proxy as proxy_api
    from client_backend.api.common import proxy_server_request

    upstream_payload = {
        "success": True,
        "message": "ok",
        "data": {
            "providers": [
                {
                    "providerType": "openai",
                    "models": [
                        {
                            "name": "gpt-4o",
                            "contextWindowTokens": 128000,
                            "maxOutputTokens": 16384,
                        }
                    ],
                }
            ],
            "agentConfig": {"provider": "openai", "model": "gpt-4o"},
        },
        "error": None,
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

        async def request_response(self, method: str, path: str, **kwargs):
            self.calls.append((method, path, kwargs))
            return FakeJsonResponse()

    server_client = FakeServerClient()
    # ``proxy_server_request`` resolves the client via
    # ``client_backend.api.common.get_server_client``; patch there.
    monkeypatch.setattr(
        "client_backend.api.common.get_server_client",
        lambda: server_client,
    )

    app = FastAPI()
    app.include_router(proxy_api.router)
    app.dependency_overrides[proxy_api.require_local_session] = lambda: object()

    # Sanity: the module re-exports the real proxy helper, not a stub.
    assert proxy_server_request is not None

    with TestClient(app) as client:
        response = client.get("/model-config/options")

    assert response.status_code == 200
    assert response.json() == upstream_payload
    # The nested per-model context window must pass through unchanged so
    # the frontend can surface the new context-window indicator.
    forwarded_model = response.json()["data"]["providers"][0]["models"][0]
    assert forwarded_model["contextWindowTokens"] == 128000
    assert forwarded_model["maxOutputTokens"] == 16384
    assert server_client.calls and server_client.calls[0][:2] == (
        "GET",
        "/model-config/options",
    )
