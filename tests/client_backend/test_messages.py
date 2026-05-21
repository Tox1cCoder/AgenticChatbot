from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from client_backend.api import common as common_api
from client_backend.api import messages as messages_api
from client_backend.core.security import LocalSessionPayload


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
    assert (
        forwarded_event["message"]["metadata"]["context_window"]
        == context_window_payload
    )
