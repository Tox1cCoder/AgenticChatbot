"""A request the server stops waiting for must not run on the device.

The server stops waiting when the device misses its deadline or the user
presses Stop. If the request is still queued it is withdrawn, so it never
reaches the device; if the device already has it, a cancel follows it. Both
backends are exercised: in-memory for development, Redis for production, where
the waiting worker and the worker holding the device's socket can differ.
"""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from app.schemas.runtime_protocol import RuntimeCancelMessage, ToolDispatchRequest
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
async def store(request):
    backend = InMemoryClientRuntimeStore() if request.param == "memory" else _redis_store()
    session = DeviceSessionRecord(device_id=uuid4(), session_id="session-1", user_id=uuid4())
    await backend.put_session(session)
    try:
        yield backend, session
    finally:
        await backend.delete_session(session.device_id)
        await backend.close()


def _request() -> ToolDispatchRequest:
    return ToolDispatchRequest(
        request_id=str(uuid4()),
        tool_name="start_process",
        qualified_tool_id="desktop-commander::start_process",
        arguments={"command": "npm test"},
        timeout_seconds=5,
    )


async def _delivered(backend, device_id) -> list:
    messages = []
    while (message := await backend.get_next_request(device_id, timeout_seconds=1)) is not None:
        messages.append(message)
    return messages


async def test_request_withdrawn_before_delivery_never_reaches_the_device(store):
    backend, session = store
    request = _request()

    with pytest.raises(TimeoutError):
        await backend.dispatch_request(session, request, 0.05)

    delivered = await _delivered(backend, session.device_id)
    assert not [message for message in delivered if isinstance(message, ToolDispatchRequest)]


async def test_request_already_delivered_is_followed_by_a_cancel(store):
    backend, session = store
    request = _request()
    waiting = asyncio.create_task(backend.dispatch_request(session, request, 30))

    forwarded = await backend.get_next_request(session.device_id, timeout_seconds=2)
    assert forwarded == request

    waiting.cancel()  # the user pressed Stop
    with pytest.raises(asyncio.CancelledError):
        await waiting

    follow_up = await backend.get_next_request(session.device_id, timeout_seconds=2)
    assert follow_up == RuntimeCancelMessage(request_id=request.request_id)


async def test_a_cancel_that_cannot_be_sent_does_not_stop_forwarding(monkeypatch):
    from types import SimpleNamespace

    from app.api.device_runtime import DeviceRuntimeGateway
    from app.services import client_runtime_store as runtime_store_module

    backend = InMemoryClientRuntimeStore()
    monkeypatch.setattr(runtime_store_module, "_store", backend)
    device_id = uuid4()
    later = _request()
    backend._withdraw(device_id, "abandoned-request")
    await backend._get_queue(device_id).put(later)
    forwarded: list[dict] = []

    class FlakySocket:
        async def send_json(self, message):
            if message["type"] == "cancel":
                raise ConnectionResetError("socket hiccup")
            forwarded.append(message)

    gateway = DeviceRuntimeGateway(
        websocket=FlakySocket(),
        device_id=device_id,
        session=SimpleNamespace(user_id=uuid4(), device_id=device_id, session_id="s"),
        service=SimpleNamespace(),
    )
    gateway._running = True
    loop_task = asyncio.create_task(gateway._dispatch_requests_loop())
    try:
        for _ in range(200):
            if forwarded:
                break
            await asyncio.sleep(0.01)
    finally:
        gateway._running = False
        loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)

    assert [message["request_id"] for message in forwarded] == [later.request_id]


async def test_request_that_lands_after_its_withdrawal_is_still_not_delivered():
    """A push cancelled mid-flight can still reach Redis after the withdrawal ran.

    Stop can interrupt the waiting worker while its queue push is on the wire:
    Redis applies the push anyway, possibly after the withdrawal's LREM found
    nothing. The request must still never be handed to the device.
    """
    backend = _redis_store()
    session = DeviceSessionRecord(device_id=uuid4(), session_id="session-1", user_id=uuid4())
    await backend.put_session(session)
    request = _request()
    try:
        await backend._withdraw(session.device_id, request.request_id, request.model_dump_json())
        await backend._async.rpush(
            backend._request_queue_key(session.device_id), request.model_dump_json()
        )

        delivered = await _delivered(backend, session.device_id)

        assert not [message for message in delivered if isinstance(message, ToolDispatchRequest)]
    finally:
        await backend.delete_session(session.device_id)
        await backend.close()
