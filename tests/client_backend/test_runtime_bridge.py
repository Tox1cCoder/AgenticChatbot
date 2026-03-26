from types import SimpleNamespace

import pytest

from client_backend.services import runtime_bridge as runtime_bridge_module
from client_backend.services.runtime_bridge import RuntimeBridgeService


class _ServerClientStub:
    def __init__(self):
        self.base_url = "http://server.test"
        self.tool_catalog_updates: list[tuple[str, dict]] = []
        self.skill_catalog_updates: list[tuple[str, dict]] = []

    async def update_device_tool_catalog(self, *, device_id: str, catalog: dict) -> None:
        self.tool_catalog_updates.append((device_id, catalog))

    async def update_device_skill_catalog(self, *, device_id: str, catalog: dict) -> None:
        self.skill_catalog_updates.append((device_id, catalog))

    def is_authenticated(self) -> bool:
        return True


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

    result = await bridge._execute_native_tool(
        qualified_tool_id="native::activate_skill",
        arguments={"skill_name": "demo"},
        timeout_seconds=10,
    )

    assert "Skill: demo" in result
    assert "Follow the demo instructions." in result


def test_native_tool_catalog_no_longer_advertises_workspace_requirements():
    bridge = RuntimeBridgeService(server_client=_ServerClientStub())

    catalog = bridge._build_native_tool_catalog()
    shell_tool = next(tool for tool in catalog if tool["qualified_id"] == "native::shell_execute")

    assert "workspace" not in shell_tool["description"].lower()
    assert shell_tool["input_schema"]["required"] == ["command"]
