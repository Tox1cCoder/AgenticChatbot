"""Sidecar shutdown must stop every MCP manager, not only the bridge's own."""

import pytest

from client_backend import main as main_module
from client_backend.schemas.mcp_config import MCPProfileScope
from client_backend.services import local_mcp_manager


class _Manager:
    def __init__(self) -> None:
        self.stopped = False

    async def shutdown(self) -> None:
        self.stopped = True


class _Bridge:
    async def stop(self) -> None:
        return None


class _InstallService:
    async def shutdown(self) -> None:
        return None


async def _noop() -> None:
    return None


@pytest.mark.asyncio
async def test_shutdown_stops_managers_the_bridge_never_owned(monkeypatch):
    manager = _Manager()
    scope = MCPProfileScope(user_id="user-a", device_identifier="device-a")
    monkeypatch.setitem(local_mcp_manager._mcp_managers, scope, manager)
    monkeypatch.setattr(main_module, "initialize_client_environment", lambda: None)
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    monkeypatch.setattr(main_module, "initialize_skills_registry", _noop)
    monkeypatch.setattr(main_module, "get_runtime_bridge", lambda: _Bridge())
    monkeypatch.setattr(main_module, "get_skill_installation_service", lambda: _InstallService())
    monkeypatch.setattr(main_module, "close_server_client", _noop)

    async with main_module.lifespan(main_module.app):
        pass

    assert manager.stopped is True
    assert scope not in local_mcp_manager._mcp_managers
