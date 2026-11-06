"""An open socket must not conceal an expired device runtime session."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.api.device_runtime import DeviceRuntimeGateway
from app.core.config import settings
from app.services import client_runtime_store as runtime_store_module
from app.services.client_device_service import ClientDeviceService
from app.services.client_runtime_store import DeviceSessionRecord, InMemoryClientRuntimeStore


class _Socket:
    def __init__(self):
        self.messages = []
        self.close_code = None

    async def send_json(self, message):
        self.messages.append(message)

    async def close(self, code=1000, reason=""):
        self.close_code = code


@pytest.mark.parametrize("session_state", ["active", "expired", "missing"])
async def test_heartbeat_only_acknowledges_a_live_session(monkeypatch, session_state):
    store = InMemoryClientRuntimeStore()
    monkeypatch.setattr(runtime_store_module, "_store", store)
    session = DeviceSessionRecord(device_id=uuid4(), session_id="session-a", user_id=uuid4())
    if session_state == "expired":
        session.last_heartbeat = datetime.now(UTC) - timedelta(
            seconds=settings.client_runtime_heartbeat_interval_seconds * 3
        )
    if session_state != "missing":
        await store.put_session(session)
    service = ClientDeviceService.__new__(ClientDeviceService)
    service.repository = SimpleNamespace(update_status=lambda *_args: None)
    socket = _Socket()
    gateway = DeviceRuntimeGateway(socket, session.device_id, session, service)
    gateway._running = True

    await gateway._handle_heartbeat({"type": "heartbeat"})

    if session_state == "active":
        assert socket.messages == [{"type": "ack", "message_id": None}]
        assert socket.close_code is None
        assert gateway._running
    else:
        assert socket.messages == []
        assert socket.close_code == 1012
        assert not gateway._running
