"""Heartbeat failures must terminate the socket lifecycle so it can reconnect."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest
import websockets
from fastapi import WebSocketDisconnect

from app.ai.client_runtime_tools import get_active_client_runtime_session
from app.api.device_runtime import DeviceRuntimeGateway
from app.core.config import settings
from app.schemas.runtime_protocol import RuntimeAckMessage
from app.services import client_runtime_store as store_module
from app.services.client_device_service import ClientDeviceService
from app.services.client_runtime_store import InMemoryClientRuntimeStore
from client_backend.schemas.runtime import CatalogSyncResult, RuntimeStatus
from client_backend.services import runtime_bridge as bridge_module
from client_backend.services.runtime_bridge import RuntimeBridgeService


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


async def test_consecutive_failed_connections_still_respect_retry_limit(monkeypatch):
    bridge = RuntimeBridgeService(server_client=SimpleNamespace(base_url="http://server.test"))
    monkeypatch.setattr(bridge_module.client_settings, "reconnect_delay_seconds", 0)
    monkeypatch.setattr(bridge_module.client_settings, "max_reconnect_attempts", 2)
    registrations = 0

    async def initialize_local_runtime():
        pass

    async def register_device():
        nonlocal registrations
        registrations += 1
        raise ConnectionError("Server unavailable")

    monkeypatch.setattr(bridge, "_initialize_local_runtime", initialize_local_runtime)
    monkeypatch.setattr(bridge, "_register_device", register_device)

    await asyncio.wait_for(bridge._run_forever(), timeout=0.5)

    assert registrations == 2
    assert bridge.get_runtime_state().status == RuntimeStatus.ERROR
    assert not bridge.is_connected()


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
async def test_reconnect_restores_both_catalogs_after_repeated_session_loss(
    monkeypatch, session_loss
):
    """Healthy connections must reset retries across separate idle/sleep cycles."""
    from websockets.asyncio.server import serve

    store = InMemoryClientRuntimeStore()
    monkeypatch.setattr(store_module, "_store", store)
    monkeypatch.setattr(settings, "enable_client_runtime_bridge", True)
    monkeypatch.setattr(bridge_module, "is_process_elevated", lambda: False)
    monkeypatch.setattr(bridge_module.client_settings, "heartbeat_interval_seconds", 0.05)
    monkeypatch.setattr(bridge_module.client_settings, "reconnect_delay_seconds", 0)
    monkeypatch.setattr(bridge_module.client_settings, "max_reconnect_attempts", 2)

    user_id, device_id = uuid4(), uuid4()
    service = ClientDeviceService.__new__(ClientDeviceService)
    service.repository = SimpleNamespace(
        update_status=lambda *_args: SimpleNamespace(user_id=user_id)
    )

    class _ServerClient:
        base_url = "http://server.test"

        def is_authenticated(self):
            return True

        async def register_device(self, **kwargs):
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

        monkeypatch.setattr(bridge, "_initialize_local_runtime", initialize_local_runtime)
        monkeypatch.setattr(bridge, "_build_tool_catalog", build_tool_catalog)
        try:
            assert await bridge.start(timeout_seconds=2)
            for _ in range(3):
                old_session_id = bridge.get_runtime_state().session_id
                if session_loss == "expired":
                    session = store.get_session(device_id)
                    session.last_heartbeat = datetime.now(UTC) - timedelta(days=1)
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
        finally:
            await bridge.stop()
