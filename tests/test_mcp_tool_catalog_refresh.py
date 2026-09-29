from __future__ import annotations

import asyncio

import pytest

import app.ai.mcp_tool_catalog as catalog_module
from app.ai.mcp_tool_catalog import McpToolCatalog


class _Manager:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.release.set()
        self.tools = [
            {
                "name": "tavily_search",
                "server_name": "tavily",
                "description": "Search the web",
                "args_schema": {"properties": {"query": {}}, "required": ["query"]},
            }
        ]

    async def get_all_tools_info(self) -> list[dict]:
        await self.release.wait()
        await asyncio.sleep(0)
        return list(self.tools)

    def get_servers_status(self) -> dict:
        return {}


@pytest.mark.asyncio
async def test_overlapping_rebuilds_do_not_duplicate_descriptors(monkeypatch) -> None:
    monkeypatch.setattr(catalog_module, "get_mcp_tools_generation", lambda: 1)
    catalog = McpToolCatalog(_Manager())

    await asyncio.gather(catalog.refresh_if_needed(), catalog.refresh_if_needed())

    assert [tool.tool_name for tool in catalog.list_all()] == ["tavily_search"]


@pytest.mark.asyncio
async def test_a_search_during_a_rebuild_sees_the_previous_catalog(monkeypatch) -> None:
    generation = [1]
    monkeypatch.setattr(catalog_module, "get_mcp_tools_generation", lambda: generation[0])
    manager = _Manager()
    catalog = McpToolCatalog(manager)
    await catalog.refresh_if_needed()

    generation[0] = 2
    manager.release.clear()
    rebuild = asyncio.create_task(catalog.refresh_if_needed())
    await asyncio.sleep(0)

    assert [tool.tool_name for tool in catalog.list_all()] == ["tavily_search"]

    manager.release.set()
    assert await rebuild is True
    assert catalog.tool_count == 1
