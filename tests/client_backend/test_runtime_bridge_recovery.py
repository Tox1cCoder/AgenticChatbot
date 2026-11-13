"""Heartbeat failures must terminate the socket lifecycle so it can reconnect."""

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import httpx
import pytest
import websockets
from fastapi import WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.http11 import Response

from app.ai.client_runtime_tools import get_active_client_runtime_session
from app.api.device_runtime import DeviceRuntimeGateway
from app.core.config import settings
from app.schemas.runtime_protocol import RuntimeAckMessage
from app.services import client_runtime_store as store_module
from app.services.client_device_service import ClientDeviceService
from app.services.client_runtime_store import InMemoryClientRuntimeStore, RedisClientRuntimeStore
from client_backend.schemas.runtime import CatalogSyncResult, RuntimeStatus
from client_backend.services import runtime_bridge as bridge_module
from client_backend.services.runtime_bridge import RuntimeBridgeService
from client_backend.services.server_api import (
    AuthenticationError,
    ServerAPIError,
    ServerConnectionError,
)


class _Socket:
    def __init__(self, failure):
        self.failure = failure
        self.handshake_sent = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def recv(self):
        if not self.handshake_sent:
            self.handshake_sent = True
            return json.dumps({"type": "ack"})
        await asyncio.Future()

    async def send(self, message):
        if self.failure == "send":
            raise ConnectionResetError("heartbeat transport failed")


@pytest.mark.parametrize("failure", ["send", "no_ack", "catalog"])
async def test_failed_connection_clears_readiness_and_exits(monkeypatch, failure):
    bridge = RuntimeBridgeService(server_client=SimpleNamespace(base_url="http://server.test"))
    bridge._device_id = "device-a"
    bridge._session_id = "session-a"
    socket = _Socket(failure)
    monkeypatch.setattr(websockets, "connect", lambda *_args, **_kwargs: socket)
    monkeypatch.setattr(bridge_module.client_settings, "heartbeat_interval_seconds", 0.01)
    monkeypatch.setattr(bridge, "_build_websocket_url", lambda: "ws://server.test/runtime")
    bridge._websocket_base_url = "ws://server.test"

    async def refresh_catalogs():
        if failure == "catalog":
            raise RuntimeError("catalog sync failed")

    monkeypatch.setattr(bridge, "refresh_catalogs", refresh_catalogs)
    task = asyncio.create_task(bridge._connect_and_serve())
    try:
        done, _ = await asyncio.wait({task}, timeout=0.5)
        assert task in done, "A failed heartbeat left the receive loop running forever"
        expected_error = {
            "send": ConnectionResetError,
            "no_ack": TimeoutError,
            "catalog": RuntimeError,
        }[failure]
        with pytest.raises(expected_error):
            await task
        assert not bridge.is_connected()
        assert not bridge._connected_event.is_set()
        assert bridge._websocket is None
        assert bridge._heartbeat_task is None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_only_heartbeat_acknowledgments_update_heartbeat_time():
    bridge = RuntimeBridgeService(server_client=SimpleNamespace(base_url="http://server.test"))
    old_heartbeat = datetime.now(UTC) - timedelta(minutes=1)
    bridge._set_state(status=RuntimeStatus.CONNECTED, last_heartbeat=old_heartbeat)

    await bridge._handle_server_message(RuntimeAckMessage(message_id="tool-request-a"))
    assert bridge.get_runtime_state().last_heartbeat == old_heartbeat

    await bridge._handle_server_message(RuntimeAckMessage())
    assert bridge.get_runtime_state().last_heartbeat > old_heartbeat


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Invalid runtime configuration"),
        AuthenticationError("Refresh token rejected", status_code=401),
        ServerAPIError("Invalid registration", status_code=400),
        InvalidStatus(Response(401, "Unauthorized", websockets.Headers())),
        httpx.UnsupportedProtocol("Server URL is missing http:// or https://"),
        httpx.LocalProtocolError("Invalid client request headers"),
        FileNotFoundError("Local runtime configuration is missing"),
        PermissionError("Local runtime configuration is not readable"),
    ],
    ids=["configuration", "auth", "400", "upgrade-401", "url", "headers", "file", "permission"],
)
async def test_non_transient_failures_still_respect_retry_limit(monkeypatch, error):
    bridge = RuntimeBridgeService(server_client=SimpleNamespace(base_url="http://server.test"))
    monkeypatch.setattr(bridge_module.client_settings, "reconnect_delay_seconds", 0)
    monkeypatch.setattr(bridge_module.client_settings, "max_reconnect_attempts", 2)
    registrations = 0

    async def initialize_local_runtime():
        pass

    async def register_device():
        nonlocal registrations
        registrations += 1
        if registrations > 5:
            pytest.fail("A permanent failure bypassed the reconnect limit")
        raise error.with_traceback(None)

    monkeypatch.setattr(bridge, "_initialize_local_runtime", initialize_local_runtime)
    monkeypatch.setattr(bridge, "_register_device", register_device)

    await asyncio.wait_for(bridge._run_forever(), timeout=0.5)

    assert registrations == 2
    assert bridge.get_runtime_state().status == RuntimeStatus.ERROR
    assert not bridge.is_connected()


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("Opening handshake timed out"),
        ConnectionClosedError(None, None),
        InvalidStatus(Response(503, "Server recovering", websockets.Headers())),
    ],
    ids=["timeout", "closed-socket", "upgrade-503"],
)
async def test_transient_websocket_failures_keep_retrying_until_stopped(monkeypatch, error):
    bridge = RuntimeBridgeService(server_client=SimpleNamespace(base_url="http://server.test"))
    monkeypatch.setattr(bridge_module.client_settings, "reconnect_delay_seconds", 0)
    monkeypatch.setattr(bridge_module.client_settings, "max_reconnect_attempts", 2)
    retries_exceeded = asyncio.Event()
    attempts = 0

    async def initialize_local_runtime():
        pass

    async def register_device():
        return SimpleNamespace(device_id="device-a", session_id="session-a")

    async def connect_and_serve():
        nonlocal attempts
        attempts += 1
        if attempts > 5:
            retries_exceeded.set()
            await asyncio.Future()
        raise error

    monkeypatch.setattr(bridge, "_initialize_local_runtime", initialize_local_runtime)
    monkeypatch.setattr(bridge, "_register_device", register_device)
    monkeypatch.setattr(bridge, "_connect_and_serve", connect_and_serve)
    bridge._runtime_task = asyncio.create_task(bridge._run_forever())
    try:
        async with asyncio.timeout(1):
            await retries_exceeded.wait()
        assert not bridge._runtime_task.done()
    finally:
        # Skip the upstream disconnect check: no socket was ever established.
        bridge._device_id = None
        await bridge.stop()
    assert bridge._runtime_task is None
    assert bridge.get_runtime_state().status == RuntimeStatus.DISCONNECTED


async def test_websocket_handshake_eof_keeps_retrying(monkeypatch):
    """A resuming server/proxy may accept TCP but close before the HTTP upgrade."""
    bridge = RuntimeBridgeService(server_client=SimpleNamespace(base_url="http://server.test"))
    monkeypatch.setattr(bridge_module.client_settings, "reconnect_delay_seconds", 0)
    monkeypatch.setattr(bridge_module.client_settings, "max_reconnect_attempts", 2)
    retries_exceeded = asyncio.Event()
    connections = 0

    async def close_during_handshake(reader, writer):
        nonlocal connections
        try:
            await reader.read(4096)
            connections += 1
            if connections > 5:
                retries_exceeded.set()
        finally:
            writer.close()
            await writer.wait_closed()

    async def initialize_local_runtime():
        pass

    async def register_device():
        return SimpleNamespace(device_id="device-a", session_id="session-a")

    monkeypatch.setattr(bridge, "_initialize_local_runtime", initialize_local_runtime)
    monkeypatch.setattr(bridge, "_register_device", register_device)
    async with await asyncio.start_server(close_during_handshake, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        bridge._websocket_base_url = f"ws://127.0.0.1:{port}"
        bridge._runtime_task = asyncio.create_task(bridge._run_forever())
        try:
            async with asyncio.timeout(2):
                await retries_exceeded.wait()
            assert not bridge._runtime_task.done()
        finally:
            bridge._device_id = None
            await bridge.stop()


@pytest.fixture(params=["memory", "redis"])
async def runtime_store(request, monkeypatch):
    if request.param == "memory":
        store = InMemoryClientRuntimeStore()
    else:
        url = os.getenv("TEST_REDIS_URL") or store_module._runtime_redis_url()
        if not url:
            pytest.skip("No test Redis available")
        try:
            store = RedisClientRuntimeStore(url)
        except Exception as exc:
            pytest.skip(f"Redis unavailable: {exc}")
    monkeypatch.setattr(store_module, "_store", store)
    try:
        yield store
    finally:
        await store.close()


class _GatewaySocket:
    """Adapt a real local WebSocket to the gateway's ASGI socket interface."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def receive_text(self):
        try:
            return await self.websocket.recv()
        except websockets.ConnectionClosed as exc:
            raise WebSocketDisconnect() from exc

    async def send_json(self, message):
        await self.websocket.send(json.dumps(message))

    async def close(self, code=1000, reason=""):
        await self.websocket.close(code=code, reason=reason)


@pytest.mark.parametrize("session_loss", ["expired", "missing"])
@pytest.mark.parametrize(
    "outage_error",
    [
        None,
        ServerConnectionError("Network unavailable after wake"),
        ServerAPIError("Server recovering", status_code=503),
        ServerAPIError("Server busy", status_code=429),
        httpx.RemoteProtocolError("Server disconnected without sending a response"),
        ConnectionResetError("Connection reset after wake"),
    ],
    ids=["no-outage", "network", "503", "429", "http-transport", "socket"],
)
async def test_reconnect_restores_both_catalogs_after_repeated_session_loss(
    monkeypatch, runtime_store, session_loss, outage_error
):
    """Recover usable catalogs even when wake-up outlasts the old retry budget."""
    from websockets.asyncio.server import serve

    store = runtime_store
    monkeypatch.setattr(settings, "enable_client_runtime_bridge", True)
    monkeypatch.setattr(bridge_module, "is_process_elevated", lambda: False)
    monkeypatch.setattr(bridge_module.client_settings, "heartbeat_interval_seconds", 0.05)
    monkeypatch.setattr(bridge_module.client_settings, "reconnect_delay_seconds", 0)
    monkeypatch.setattr(bridge_module.client_settings, "max_reconnect_attempts", 2)

    user_id, device_id = uuid4(), uuid4()
    failed_registrations_remaining = 0
    service = ClientDeviceService.__new__(ClientDeviceService)
    service.repository = SimpleNamespace(
        update_status=lambda *_args: SimpleNamespace(user_id=user_id)
    )

    class _ServerClient:
        base_url = "http://server.test"

        def is_authenticated(self):
            return True

        async def register_device(self, **kwargs):
            nonlocal failed_registrations_remaining
            if failed_registrations_remaining:
                failed_registrations_remaining -= 1
                raise outage_error
            session_id = str(uuid4())
            await service.issue_runtime_session_id(device_id, session_id)
            return SimpleNamespace(device_id=str(device_id), session_id=session_id)

        async def update_device_tool_catalog(self, *, device_id, catalog):
            if not await service.update_tool_catalog(device_id, catalog):
                raise RuntimeError("No active session for tool catalog")
            return CatalogSyncResult(status="synced", tool_count=len(catalog["tools"]))

        async def update_device_skill_catalog(self, *, device_id, catalog):
            if not await service.update_skill_catalog(device_id, catalog):
                raise RuntimeError("No active session for skill catalog")
            return CatalogSyncResult(status="synced", skill_count=len(catalog["skills"]))

        async def list_connected_devices(self):
            return {
                "connected_devices": [
                    {"device_id": str(session.device_id)}
                    for session in service.get_active_sessions_for_user(user_id)
                ]
            }

    async def handle_socket(websocket):
        session_id = parse_qs(urlsplit(websocket.request.path).query)["session_id"][0]
        session = await service.start_session(device_id, session_id)
        gateway = DeviceRuntimeGateway(_GatewaySocket(websocket), device_id, session, service)
        await gateway.handle_connection()

    monkeypatch.setattr(
        bridge_module,
        "get_skills_registry",
        lambda: SimpleNamespace(
            get_skill_catalog=lambda **_kwargs: {
                "skills": [{"name": "demo", "description": "Device skill"}]
            }
        ),
    )

    async with serve(handle_socket, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        bridge = RuntimeBridgeService(
            server_client=_ServerClient(), websocket_base_url=f"ws://127.0.0.1:{port}"
        )

        async def initialize_local_runtime():
            pass

        async def build_tool_catalog():
            return {"tools": [{"name": "read_file", "qualified_id": "demo::read_file"}]}

        async def execute_tool_request(request):
            return {"content": [{"type": "text", "text": "Tool available after reconnect"}]}

        monkeypatch.setattr(bridge, "_initialize_local_runtime", initialize_local_runtime)
        monkeypatch.setattr(bridge, "_build_tool_catalog", build_tool_catalog)
        monkeypatch.setattr(bridge, "_execute_tool_request", execute_tool_request)
        try:
            assert await bridge.start(timeout_seconds=2)
            for _ in range(3):
                old_session_id = bridge.get_runtime_state().session_id
                if outage_error is not None:
                    failed_registrations_remaining = 5
                if session_loss == "expired":
                    session = store.get_session(device_id)
                    session.last_heartbeat = datetime.now(UTC) - timedelta(days=1)
                    await store.put_session(session)
                else:
                    await store.delete_session(device_id)

                async with asyncio.timeout(2):
                    while True:
                        if bridge._runtime_task.done():
                            pytest.fail(
                                f"Reconnect stopped: {bridge.get_runtime_state().error_message}"
                            )
                        session = get_active_client_runtime_session(
                            user_id=str(user_id), device_id=str(device_id)
                        )
                        if (
                            bridge.is_connected()
                            and bridge._connected_event.is_set()
                            and session is not None
                            and session.session_id != old_session_id
                            and session.tool_catalog
                            and session.skill_catalog
                        ):
                            break
                        await asyncio.sleep(0.01)

                assert session.tool_catalog["tools"][0]["qualified_id"] == "demo::read_file"
                assert session.skill_catalog["skills"][0]["name"] == "demo"
                assert session.tool_catalog_version == 1
                assert session.skill_catalog_version == 1
                assert failed_registrations_remaining == 0
                result = await service.dispatch_tool_call(
                    user_id=str(user_id),
                    device_id=str(device_id),
                    tool_name="read_file",
                    qualified_tool_id="demo::read_file",
                    arguments={},
                    execution_timeout_seconds=1,
                    response_timeout_seconds=2,
                    bound_session_id=session.session_id,
                    bound_catalog_version=session.tool_catalog_version,
                )
                assert result["success"] is True
                assert "Tool available after reconnect" in str(result["result"])
        finally:
            await bridge.stop()
            await store.delete_session(device_id)
