from __future__ import annotations

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

    async def stream_ai_sdk_chat(self, conversation_id: str, payload: dict):
        self.chat_calls.append((conversation_id, payload))
        yield {"type": "start"}

    async def resume_ai_sdk_interrupt(self, payload: dict):
        self.resume_calls.append(payload)
        yield {"type": "data-interrupt", "data": {"threadId": "thread-1"}}


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

    assert bridge.start_calls[0][0] is False
    assert bridge.start_calls[1][0] is True
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

    assert bridge.start_calls[0][0] is False
    assert bridge.start_calls[1][0] is True
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
