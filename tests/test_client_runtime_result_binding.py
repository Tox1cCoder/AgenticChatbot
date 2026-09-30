"""A device can answer only the requests the server dispatched to it.

The gateway hands every ``tool_result`` a device sends to the shared store. The
store must check the request was dispatched to that same device, or any
connected device could resolve another device's (another user's) tool call with
a forged result. Both backends are exercised.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.api.device_runtime import DeviceRuntimeGateway
from app.schemas.runtime_protocol import ToolDispatchRequest, ToolDispatchResult
from app.services import client_runtime_store as runtime_store_module
from app.services.client_runtime_store import (
    DeviceSessionRecord,
    InMemoryClientRuntimeStore,
    RedisClientRuntimeStore,
)


def _redis_store() -> RedisClientRuntimeStore:
    url = os.getenv("TEST_REDIS_URL") or ""
    if not url.strip():
        from app.core.config import settings

        url = str(getattr(settings, "redis_url", "") or "")
    if not url.strip():
        pytest.skip("no Redis URL available")
    try:
        return RedisClientRuntimeStore(url)
    except Exception as exc:  # noqa: BLE001 - any connection failure means skip
        pytest.skip(f"Redis unavailable: {exc}")


@pytest.fixture(params=["memory", "redis"])
async def store(request, monkeypatch):
    backend = InMemoryClientRuntimeStore() if request.param == "memory" else _redis_store()
    monkeypatch.setattr(runtime_store_module, "_store", backend)
    owner = DeviceSessionRecord(device_id=uuid4(), session_id="owner", user_id=uuid4())
    await backend.put_session(owner)
    try:
        yield backend, owner
    finally:
        await backend.delete_session(owner.device_id)
        await backend.close()


class _RecordingSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, message: dict) -> None:
        self.sent.append(message)


def _gateway(device_id: UUID) -> tuple[DeviceRuntimeGateway, _RecordingSocket]:
    socket = _RecordingSocket()
    gateway = DeviceRuntimeGateway(
        websocket=socket,
        device_id=device_id,
        session=SimpleNamespace(user_id=uuid4(), device_id=device_id, session_id="s"),
        service=SimpleNamespace(),
    )
    return gateway, socket


def _request() -> ToolDispatchRequest:
    return ToolDispatchRequest(
        request_id=str(uuid4()),
        tool_name="read_file",
        qualified_tool_id="desktop-commander::read_file",
        arguments={"path": "notes.txt"},
        timeout_seconds=5,
    )


def _result_message(request_id: str, text: str) -> dict:
    result = ToolDispatchResult(
        request_id=request_id, success=True, result=text, execution_time_ms=1
    )
    return result.model_dump(mode="json")


async def test_a_result_from_another_device_does_not_resolve_the_request(store):
    backend, owner = store
    request = _request()
    waiting = asyncio.create_task(backend.dispatch_request(owner, request, 5))
    assert await backend.get_next_request(owner.device_id, timeout_seconds=2) == request

    intruder, intruder_socket = _gateway(uuid4())
    await intruder._handle_tool_result(_result_message(request.request_id, "forged"))

    owner_gateway, owner_socket = _gateway(owner.device_id)
    await owner_gateway._handle_tool_result(_result_message(request.request_id, "genuine"))

    outcome = await asyncio.wait_for(waiting, timeout=5)
    assert outcome["result"] == "genuine"
    assert intruder_socket.sent == []
    assert owner_socket.sent == [{"type": "ack", "message_id": request.request_id}]


async def test_a_result_for_a_request_nobody_dispatched_is_refused(store):
    backend, _ = store

    accepted = await backend.publish_result(
        ToolDispatchResult(request_id=str(uuid4()), success=True, execution_time_ms=1),
        device_id=uuid4(),
    )

    assert accepted is False
