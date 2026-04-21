import asyncio
from types import SimpleNamespace

import pytest

from app.schemas.runtime_protocol import RuntimeErrorMessage
from client_backend.schemas.runtime import CatalogSyncResult, RuntimeStatus, ToolDispatchRequest
from client_backend.services import runtime_bridge as runtime_bridge_module
from client_backend.services.runtime_bridge import RuntimeBridgeService


class _ServerClientStub:
    def __init__(self):
        self.base_url = "http://server.test"
        self.tool_catalog_updates: list[tuple[str, dict]] = []
        self.skill_catalog_updates: list[tuple[str, dict]] = []
        self.register_calls: list[dict] = []

    async def update_device_tool_catalog(
        self, *, device_id: str, catalog: dict
    ) -> CatalogSyncResult:
        self.tool_catalog_updates.append((device_id, catalog))
        return CatalogSyncResult(status="synced", tool_count=len(catalog.get("tools", [])))

    async def update_device_skill_catalog(
        self, *, device_id: str, catalog: dict
    ) -> CatalogSyncResult:
        self.skill_catalog_updates.append((device_id, catalog))
        return CatalogSyncResult(
            status="synced",
            skill_count=len(catalog.get("skills", [])),
        )

    def is_authenticated(self) -> bool:
        return True

    async def register_device(self, **kwargs):
        self.register_calls.append(kwargs)
        return SimpleNamespace(device_id="device-123", session_id="session-123")


@pytest.mark.asyncio
async def test_refresh_catalogs_syncs_skill_summaries_without_content(monkeypatch):
    server_client = _ServerClientStub()
    bridge = RuntimeBridgeService(server_client=server_client)
    bridge._device_id = "device-123"

    include_content_calls: list[bool] = []

    class _RegistryStub:
        def get_skill_catalog(self, include_content: bool = True) -> dict:
            include_content_calls.append(include_content)
            if include_content:
                return {"skills": [{"name": "demo", "content": "secret"}]}
            return {"skills": [{"name": "demo", "description": "Device skill"}]}

    monkeypatch.setattr(
        runtime_bridge_module,
        "get_mcp_manager",
        lambda: SimpleNamespace(
            get_tool_catalog=lambda: {"tools": [], "server_count": 0, "active_servers": []}
        ),
    )
    monkeypatch.setattr(runtime_bridge_module, "get_skills_registry", lambda: _RegistryStub())

    await bridge.refresh_catalogs()

    assert include_content_calls == [False]
    assert server_client.tool_catalog_updates[0][0] == "device-123"
    assert server_client.skill_catalog_updates == [
        (
            "device-123",
            {"skills": [{"name": "demo", "description": "Device skill"}]},
        )
    ]


@pytest.mark.asyncio
async def test_runtime_bridge_executes_activate_skill_locally(monkeypatch):
    bridge = RuntimeBridgeService(server_client=_ServerClientStub())

    monkeypatch.setattr(
        runtime_bridge_module,
        "get_skills_registry",
        lambda: SimpleNamespace(
            get_skill=lambda name: (
                SimpleNamespace(name="demo", enabled=True, content="Follow the demo instructions.")
                if name == "demo"
                else None
            )
        ),
    )

    result = await bridge._execute_tool_request(
        ToolDispatchRequest(
            request_id="req-skill",
            tool_name="activate_skill",
            qualified_tool_id="client_skill::activate",
            arguments={"skill_name": "demo"},
            timeout_seconds=10,
        )
    )

    assert "Skill: demo" in result
    assert "Follow the demo instructions." in result


@pytest.mark.asyncio
async def test_handle_tool_request_sends_shared_typed_tool_result(monkeypatch):
    bridge = RuntimeBridgeService(server_client=_ServerClientStub())
    sent_payloads: list[dict] = []

    # Populate the catalog so _validate_tool_request passes.
    # The request uses tool_name="demo" / qualified_tool_id="demo::echo_text",
    # so the catalog entry must match both.  No tool_instance_id or session fields
    # are set on the request, so no additional validation is triggered.
    bridge._current_tool_catalog = {
        "demo::echo_text": {
            "qualified_id": "demo::echo_text",
            "name": "demo",
        }
    }

    async def _fake_execute_tool_request(request: ToolDispatchRequest) -> dict[str, bool]:
        assert isinstance(request, ToolDispatchRequest)
        return {"ok": True}

    async def _fake_send_runtime_message(payload) -> None:
        sent_payloads.append(payload)

    monkeypatch.setattr(bridge, "_execute_tool_request", _fake_execute_tool_request)
    monkeypatch.setattr(bridge, "_send_runtime_message", _fake_send_runtime_message)

    await bridge._handle_tool_request(
        ToolDispatchRequest(
            request_id="req-1",
            tool_name="demo",
            qualified_tool_id="demo::echo_text",
            arguments={"text": "notes"},
            timeout_seconds=5,
        )
    )

    assert len(sent_payloads) == 1
    payload = sent_payloads[0]
    assert payload.type == "tool_result"
    assert payload.request_id == "req-1"
    assert payload.success is True
    assert payload.result == {"ok": True}
    assert isinstance(payload.execution_time_ms, int)


@pytest.mark.asyncio
async def test_handle_server_message_uses_typed_runtime_error_context():
    bridge = RuntimeBridgeService(server_client=_ServerClientStub())

    await bridge._handle_server_message(
        RuntimeErrorMessage(message="remote failure", code="REMOTE_ERROR")
    )

    assert bridge.get_runtime_state().error_message == "remote failure"


@pytest.mark.asyncio
async def test_build_tool_catalog_returns_mcp_tools_only(monkeypatch):
    server_client = _ServerClientStub()
    bridge = RuntimeBridgeService(server_client=server_client)
    bridge._device_id = "device-123"
    bridge._session_id = "session-123"
    bridge._tool_catalog_version = 0

    monkeypatch.setattr(
        runtime_bridge_module,
        "get_mcp_manager",
        lambda: SimpleNamespace(
            get_tool_catalog=lambda: {
                "tools": [
                    {
                        "name": "echo_text",
                        "origin": "mcp",
                        "server_name": "demo",
                        "qualified_id": "demo::echo_text",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
                "server_count": 1,
                "active_servers": ["demo"],
            }
        ),
    )

    catalog = await bridge._build_tool_catalog()

    assert [tool["qualified_id"] for tool in catalog["tools"]] == ["demo::echo_text"]
    assert catalog["tools"][0]["tool_instance_id"]
    assert "native_tool_count" not in catalog


@pytest.mark.asyncio
async def test_register_device_capabilities_only_advertise_mcp_and_skills():
    server_client = _ServerClientStub()
    bridge = RuntimeBridgeService(server_client=server_client)

    await bridge._register_device()

    assert server_client.register_calls
    assert server_client.register_calls[0]["capabilities"] == {
        "local_mcp": True,
        "local_skills": True,
    }


@pytest.mark.asyncio
async def test_start_waits_when_connected_event_is_stale_during_reconnect():
    server_client = _ServerClientStub()
    bridge = RuntimeBridgeService(server_client=server_client)
    bridge._connected_event.set()
    bridge._set_state(status=RuntimeStatus.RECONNECTING)
    bridge._runtime_task = asyncio.create_task(asyncio.sleep(0.05))

    try:
        connected = await bridge.start(wait_for_connection=True, timeout_seconds=0.01)
    finally:
        bridge._runtime_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await bridge._runtime_task

    assert connected is False
    assert bridge._connected_event.is_set() is False
