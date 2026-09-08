from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from client_backend.api import common as common_api
from client_backend.api import messages as messages_api
from client_backend.core.security import LocalSessionPayload
from client_backend.services.server_api import ServerAPIError


class _AuthServiceStub:
    def is_authenticated(self) -> bool:
        return True


class _BridgeStub:
    def __init__(self):
        self.device_id: str | None = None
        self.start_calls: list[tuple[bool, int | None]] = []

    def is_connected(self) -> bool:
        return self.device_id is not None

    async def start(
        self,
        *,
        wait_for_connection: bool,
        timeout_seconds: int | None = None,
    ) -> bool:
        self.start_calls.append((wait_for_connection, timeout_seconds))
        if wait_for_connection:
            self.device_id = "device-123"
        return True

    def get_registered_device_id(self) -> str | None:
        return self.device_id


class _ServerClientStub:
    def __init__(self):
        self.chat_calls: list[tuple[str, dict]] = []
        self.resume_calls: list[dict] = []
        self.internal_stream_calls: list[dict] = []
        self.internal_stream_events: list[dict] = [{"type": "start"}]
        self.continue_calls: list[dict] = []
        self.continue_ai_sdk_calls: list[dict] = []
        self.request_calls: list[dict] = []
        self.upstream_status: int = 200
        self.upstream_body: dict = {"success": True, "data": {"status": "cancelled"}}

    async def stream_ai_sdk_chat(self, conversation_id: str, payload: dict):
        self.chat_calls.append((conversation_id, payload))
        yield {"type": "start"}

    async def resume_ai_sdk_interrupt(self, payload: dict):
        self.resume_calls.append(payload)
        yield {"type": "data-interrupt", "data": {"threadId": "thread-1"}}

    async def stream_internal_message(self, payload: dict):
        self.internal_stream_calls.append(payload)
        for event in self.internal_stream_events:
            yield event

    async def continue_generation(self, payload: dict):
        self.continue_calls.append(payload)
        yield {"type": "generation_start", "generation_id": "gen-1"}

    async def continue_ai_sdk_generation(self, payload: dict):
        self.continue_ai_sdk_calls.append(payload)
        yield {"type": "data-generation", "data": {"phase": "start"}}

    async def request_response(self, method: str, path: str, **kwargs):
        """Mimic httpx enough to carry a status code alongside a JSON body."""
        self.request_calls.append({"method": method, "path": path, **kwargs})
        return _UpstreamResponse(self.upstream_status, self.upstream_body)


class _UpstreamResponse:
    """The subset of an httpx response the JSON proxy helper reads."""

    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = {"content-type": "application/json"}
        self.content = json.dumps(body).encode()

    def json(self) -> dict:
        return self._body


def _session() -> LocalSessionPayload:
    now = datetime.now(timezone.utc)
    return LocalSessionPayload(
        user_id="user-123",
        server_user_id="user-123",
        device_id=None,
        device_identifier="device-abc",
        iat=now,
        exp=now + timedelta(hours=1),
    )


async def _collect_streaming_body(response) -> str:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        chunks.append(chunk)
    return "".join(chunks)


@pytest.mark.asyncio
async def test_ai_sdk_chat_reconnects_runtime_bridge_and_injects_device_id(monkeypatch):
    bridge = _BridgeStub()
    server_client = _ServerClientStub()

    monkeypatch.setattr(messages_api, "get_runtime_bridge", lambda: bridge)
    monkeypatch.setattr(messages_api, "get_upstream_auth_service", lambda: _AuthServiceStub())
    monkeypatch.setattr(messages_api, "get_server_client", lambda: server_client)
    monkeypatch.setattr(common_api, "get_runtime_bridge", lambda: bridge)

    response = await messages_api.ai_sdk_chat(
        "conversation-1",
        {"messages": [{"role": "user", "content": "hello"}]},
        _session=_session(),
    )
    body = await _collect_streaming_body(response)

    # The bridge should have been started with wait_for_connection=True
    # so device_id is available before the message is forwarded.
    assert len(bridge.start_calls) == 1
    assert bridge.start_calls[0][0] is True  # wait_for_connection
    assert bridge.start_calls[0][1] == 15  # timeout_seconds
    assert server_client.chat_calls == [
        (
            "conversation-1",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "device_id": "device-123",
            },
        )
    ]
    assert 'data: {"type": "start"}' in body
    assert "data: [DONE]" in body


@pytest.mark.asyncio
async def test_ai_sdk_resume_interrupt_reconnects_runtime_bridge_before_forwarding(monkeypatch):
    bridge = _BridgeStub()
    server_client = _ServerClientStub()

    monkeypatch.setattr(messages_api, "get_runtime_bridge", lambda: bridge)
    monkeypatch.setattr(messages_api, "get_upstream_auth_service", lambda: _AuthServiceStub())
    monkeypatch.setattr(messages_api, "get_server_client", lambda: server_client)
    monkeypatch.setattr(common_api, "get_runtime_bridge", lambda: bridge)

    response = await messages_api.ai_sdk_resume_interrupt(
        {
            "threadId": "thread-1",
            "conversationId": "conversation-1",
            "interruptId": "interrupt-1",
            "decisions": [{"type": "approve", "taskId": "tool-1"}],
        },
        _session=_session(),
    )
    body = await _collect_streaming_body(response)

    assert len(bridge.start_calls) == 1
    assert bridge.start_calls[0][0] is True
    assert server_client.resume_calls == [
        {
            "threadId": "thread-1",
            "conversationId": "conversation-1",
            "interruptId": "interrupt-1",
            "decisions": [{"type": "approve", "taskId": "tool-1"}],
            "device_id": "device-123",
        }
    ]
    assert '"type": "data-interrupt"' in body
    assert "data: [DONE]" in body


@pytest.mark.asyncio
async def test_messages_stream_route_forwards_context_window_metadata(monkeypatch):
    """The ``/messages/stream`` sidecar route must forward
    ``complete.message.metadata.context_window`` to the desktop client
    without stripping or rewriting any nested keys.
    """

    bridge = _BridgeStub()
    server_client = _ServerClientStub()

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
    server_client.internal_stream_events = [complete_event]

    monkeypatch.setattr(messages_api, "get_runtime_bridge", lambda: bridge)
    monkeypatch.setattr(messages_api, "get_upstream_auth_service", lambda: _AuthServiceStub())
    monkeypatch.setattr(messages_api, "get_server_client", lambda: server_client)
    monkeypatch.setattr(common_api, "get_runtime_bridge", lambda: bridge)

    response = await messages_api.create_message_stream(
        {"messages": [{"role": "user", "content": "hello"}]},
        _session=_session(),
    )
    body = await _collect_streaming_body(response)

    # Internal stream was called with the device-id-augmented payload.
    assert server_client.internal_stream_calls == [
        {
            "messages": [{"role": "user", "content": "hello"}],
            "device_id": "device-123",
        }
    ]

    # Extract the single ``data:`` line corresponding to the complete event
    # and assert that its JSON is byte-for-byte equivalent to the upstream
    # event — no nested keys dropped.
    data_lines = [
        line[len("data: ") :]
        for line in body.splitlines()
        if line.startswith("data: ") and line[len("data: ") :] != "[DONE]"
    ]
    assert len(data_lines) == 1
    forwarded_event = json.loads(data_lines[0])
    assert forwarded_event == complete_event
    assert forwarded_event["message"]["metadata"]["context_window"] == context_window_payload


@pytest.mark.asyncio
async def test_internal_sidecar_stream_error_retains_upstream_identity():
    async def source():
        raise ServerAPIError(
            "Server error: 409",
            status_code=409,
            detail={
                "code": "INTERRUPT_CONFLICT",
                "message": "Interrupt was claimed by a concurrent request.",
            },
        )
        yield  # pragma: no cover

    response = messages_api._build_sse_response(source())
    body = await _collect_streaming_body(response)
    payloads = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]

    assert payloads == [
        {
            "type": "error",
            "error": "Interrupt was claimed by a concurrent request.",
            "status_code": 409,
            "error_code": "INTERRUPT_CONFLICT",
        }
    ]


@pytest.mark.asyncio
async def test_ai_sdk_sidecar_stream_error_retains_upstream_identity():
    async def source():
        raise ServerAPIError(
            "Server error: 409",
            status_code=409,
            detail={
                "code": "INTERRUPT_CONFLICT",
                "message": "Interrupt was claimed by a concurrent request.",
            },
        )
        yield  # pragma: no cover

    response = messages_api._build_sse_response(source(), ai_sdk=True)
    body = await _collect_streaming_body(response)
    payloads = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]

    assert payloads == [
        {
            "type": "error",
            "errorText": "Interrupt was claimed by a concurrent request.",
            "statusCode": 409,
            "errorCode": "INTERRUPT_CONFLICT",
        }
    ]
    assert body.endswith("data: [DONE]\n\n")


# ----------------------------------------------------------------------
# generation controls
# ----------------------------------------------------------------------


def _wire(monkeypatch, bridge, server_client) -> None:
    monkeypatch.setattr(messages_api, "get_runtime_bridge", lambda: bridge)
    monkeypatch.setattr(messages_api, "get_upstream_auth_service", lambda: _AuthServiceStub())
    monkeypatch.setattr(messages_api, "get_server_client", lambda: server_client)
    monkeypatch.setattr(common_api, "get_runtime_bridge", lambda: bridge)


@pytest.mark.asyncio
async def test_a_pending_stop_keeps_its_202_through_the_proxy():
    """The status is the signal, and flattening it is a lie.

    The proxy used to return the parsed body, so a ``202`` — accepted but not
    confirmed by any worker — arrived at the client as ``200``, which reads as
    "the turn is over".
    """
    bridge = _BridgeStub()
    server_client = _ServerClientStub()
    server_client.upstream_status = 202
    server_client.upstream_body = {"success": True, "data": {"status": "stop_requested"}}

    with pytest.MonkeyPatch.context() as monkeypatch:
        _wire(monkeypatch, bridge, server_client)
        response = await messages_api.stop_generation(
            {"conversationId": "conversation-1", "generationId": "gen-1", "expectedVersion": 2},
            _session=_session(),
        )

    assert response.status_code == 202
    assert json.loads(response.body)["data"]["status"] == "stop_requested"


@pytest.mark.asyncio
async def test_a_settled_stop_keeps_its_200_through_the_proxy():
    bridge = _BridgeStub()
    server_client = _ServerClientStub()
    server_client.upstream_status = 200

    with pytest.MonkeyPatch.context() as monkeypatch:
        _wire(monkeypatch, bridge, server_client)
        response = await messages_api.stop_generation(
            {"conversationId": "conversation-1", "generationId": "gen-1", "expectedVersion": 2},
            _session=_session(),
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_the_stop_proxy_forwards_the_fence_and_key_unchanged():
    """A fence the proxy rewrote would defeat the point of having one."""
    bridge = _BridgeStub()
    server_client = _ServerClientStub()

    with pytest.MonkeyPatch.context() as monkeypatch:
        _wire(monkeypatch, bridge, server_client)
        await messages_api.stop_generation(
            {
                "conversationId": "conversation-1",
                "generationId": "gen-1",
                "expectedVersion": 7,
                "idempotencyKey": "stop-key-abcdef",
            },
            _session=_session(),
        )

    forwarded = server_client.request_calls[0]
    assert forwarded["method"] == "POST"
    assert forwarded["path"] == "/messages/stop"
    assert forwarded["json"]["expectedVersion"] == 7
    assert forwarded["json"]["idempotencyKey"] == "stop-key-abcdef"


@pytest.mark.asyncio
async def test_the_generation_snapshot_proxy_forwards_the_conversation_scope():
    """Without the conversation the upstream read is not owner-scoped."""
    bridge = _BridgeStub()
    server_client = _ServerClientStub()
    server_client.upstream_body = {"success": True, "data": {"status": "continuable"}}

    with pytest.MonkeyPatch.context() as monkeypatch:
        _wire(monkeypatch, bridge, server_client)
        response = await messages_api.get_generation(
            "gen-1", "conversation-1", _session=_session()
        )

    forwarded = server_client.request_calls[0]
    assert forwarded["method"] == "GET"
    assert forwarded["path"] == "/messages/generations/gen-1"
    assert forwarded["params"] == {"conversation_id": "conversation-1"}
    assert json.loads(response.body)["data"]["status"] == "continuable"


@pytest.mark.asyncio
async def test_the_snapshot_route_is_not_shadowed_by_the_message_route():
    """FastAPI matches in declaration order.

    ``/messages/{message_id}`` is declared after this one on purpose: reversed,
    it captures "generations" as a message id and the snapshot endpoint becomes
    unreachable — a 404 that looks like an ownership refusal.
    """
    from client_backend.api.messages import router

    paths = [route.path for route in router.routes]
    assert "/messages/generations/{generation_id}" in paths
    assert paths.index("/messages/generations/{generation_id}") < paths.index(
        "/messages/{message_id}"
    )


@pytest.mark.asyncio
async def test_continue_reconnects_the_runtime_bridge_before_forwarding():
    """A continued epoch binds the same client tools as the original.

    Skipping the bridge would let a continuation lose them mid-answer, which
    fails closed as a tool error rather than anything the user can act on.
    """
    bridge = _BridgeStub()
    server_client = _ServerClientStub()

    with pytest.MonkeyPatch.context() as monkeypatch:
        _wire(monkeypatch, bridge, server_client)
        response = await messages_api.continue_generation(
            {
                "conversationId": "conversation-1",
                "generationId": "gen-1",
                "continuationId": "cont-1",
                "expectedVersion": 4,
                "idempotencyKey": "continue-key-0001",
            },
            _session=_session(),
        )
        body = await _collect_streaming_body(response)

    assert bridge.start_calls and bridge.start_calls[0][0] is True
    assert server_client.continue_calls == [
        {
            "conversationId": "conversation-1",
            "generationId": "gen-1",
            "continuationId": "cont-1",
            "expectedVersion": 4,
            "idempotencyKey": "continue-key-0001",
            "device_id": "device-123",
        }
    ]
    assert "generation_start" in body


@pytest.mark.asyncio
async def test_the_ai_sdk_continue_proxy_uses_the_ai_sdk_stream_shape():
    """Same command, different adapter — that is what parity means here."""
    bridge = _BridgeStub()
    server_client = _ServerClientStub()

    with pytest.MonkeyPatch.context() as monkeypatch:
        _wire(monkeypatch, bridge, server_client)
        response = await messages_api.ai_sdk_continue_generation(
            {
                "conversationId": "conversation-1",
                "generationId": "gen-1",
                "continuationId": "cont-1",
                "expectedVersion": 4,
                "idempotencyKey": "continue-key-0001",
            },
            _session=_session(),
        )
        body = await _collect_streaming_body(response)

    assert server_client.continue_ai_sdk_calls[0]["generationId"] == "gen-1"
    assert "data-generation" in body
    # The AI SDK transport terminates with its sentinel; the internal one does not.
    assert body.endswith("data: [DONE]\n\n")
