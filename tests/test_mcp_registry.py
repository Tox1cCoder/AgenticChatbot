from __future__ import annotations

import asyncio

import pytest

import app.ai.mcp_integration as mcp_integration
from app.ai.mcp_registry import MCPRegistry


class _FakeManager:
    def __init__(self, *args, **kwargs) -> None:
        self.reloads = 0

    async def initialize(self) -> None:
        await asyncio.sleep(0)

    async def get_tools(self) -> list:
        await asyncio.sleep(0)
        return []

    async def reload_tools(self) -> None:
        await asyncio.sleep(0)
        self.reloads += 1


@pytest.fixture
def registry(monkeypatch, tmp_path):
    config = tmp_path / "mcp_config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(MCPRegistry, "_instance", None)
    monkeypatch.setattr(MCPRegistry, "_initialized", False)
    monkeypatch.setattr(MCPRegistry, "_init_lock", asyncio.Lock())
    monkeypatch.setattr(MCPRegistry, "_config_mtime", 0.0)
    monkeypatch.setattr(MCPRegistry, "_config_path", str(config))
    monkeypatch.setattr(MCPRegistry, "_tools_generation", 0)
    monkeypatch.setattr(mcp_integration, "MCPManager", _FakeManager)
    return MCPRegistry


@pytest.mark.asyncio
async def test_a_caller_that_lost_the_init_race_to_a_config_change_does_not_deadlock(
    registry, monkeypatch
) -> None:
    """asyncio.Lock is not reentrant; reloading while holding it hung forever."""

    monkeypatch.setattr(registry, "_check_config_changed", classmethod(lambda cls: True))

    first, second = await asyncio.wait_for(
        asyncio.gather(registry.get_manager_async(), registry.get_manager_async()),
        timeout=2,
    )

    assert first is second
    assert first.reloads == 1


@pytest.mark.asyncio
async def test_concurrent_callers_reload_a_changed_config_once(registry) -> None:
    manager = await registry.get_manager_async()
    registry._config_mtime = 0.0  # the file now looks newer than the last load

    await asyncio.gather(*(registry.get_manager_async() for _ in range(3)))

    assert manager.reloads == 1
