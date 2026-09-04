"""Raw web providers are unreachable through ordinary tool discovery.

The product tools own the turn budget, the date anchoring, and the bounds on
what reaches context. A model that can find ``tavily_search`` through
``tool_search`` bypasses all three, so discovery has to refuse to name it.

Exclusion is by tool name, which is what ``excluded_tool_names`` compares. The
qualified catalog id assertions are the second half of the same claim: a raw
tool must not reach the model under either identity.
"""

from __future__ import annotations

import json

import pytest

from app.ai.deferred_tool_binding import RAW_WEB_TOOL_NAMES
from app.ai.mcp_tool_catalog import ToolDescriptor
from app.ai.tool_context import ToolContext
from app.ai.tool_search_tool import _execute_tool_search

_CATALOG = [
    ToolDescriptor(
        "tavily_search", "tavily", "Search the web for current facts.", ["query"], ["query"], "f1"
    ),
    ToolDescriptor(
        "tavily_extract", "tavily", "Extract page content from URLs.", ["urls"], ["urls"], "f2"
    ),
    ToolDescriptor(
        "brave_image_search",
        "brave",
        "Search the web for images of a subject.",
        ["query"],
        ["query"],
        "f3",
    ),
    ToolDescriptor(
        "tavily_map", "tavily", "Discover URLs on a website.", ["url"], ["url"], "f4"
    ),
]


class _FakeServerCatalog:
    def search(self, query=None, top_k=5, server_name=None, allowlist=None):
        return list(_CATALOG)

    def search_scored(self, query=None, top_k=5, server_name=None, allowlist=None):
        return list(_CATALOG)

    def is_ambiguous(self, tool_name):
        return False

    def get_server_inventory(self, allowlist=None):
        return [
            {"server_name": "tavily", "tool_count": 3},
            {"server_name": "brave", "tool_count": 1},
        ]


@pytest.fixture(autouse=True)
def _catalog(monkeypatch):
    async def _manager():
        return object()

    async def _tool_catalog(_manager):
        return _FakeServerCatalog()

    monkeypatch.setattr("app.ai.tool_search_tool.get_global_mcp_manager", _manager)
    monkeypatch.setattr("app.ai.tool_search_tool.get_tool_catalog", _tool_catalog)
    monkeypatch.setattr("app.ai.tool_search_tool.get_tool_context", lambda: ToolContext())


def test_the_policy_set_uses_unqualified_names():
    """``excluded_tool_names`` compares bare names, so a qualified spec in this
    set would silently match nothing and hide none of them."""
    assert set(RAW_WEB_TOOL_NAMES) == {"tavily_search", "tavily_extract", "brave_image_search"}
    assert not any("::" in name for name in RAW_WEB_TOOL_NAMES)


@pytest.mark.asyncio
async def test_ordinary_discovery_cannot_return_a_raw_web_tool():
    result = await _execute_tool_search(
        query="search the web for current news",
        excluded_tool_names=RAW_WEB_TOOL_NAMES,
    )

    returned = {item["tool_name"] for item in result.get("results", [])}
    assert returned.isdisjoint(RAW_WEB_TOOL_NAMES)


@pytest.mark.asyncio
async def test_the_qualified_catalog_ids_are_absent_from_the_result_too():
    result = await _execute_tool_search(
        query="extract the content of a page",
        excluded_tool_names=RAW_WEB_TOOL_NAMES,
    )
    serialized = json.dumps(result, default=str)

    for qualified in (
        "tavily::tavily_search",
        "tavily::tavily_extract",
        "brave::brave_image_search",
    ):
        assert qualified not in serialized


@pytest.mark.asyncio
async def test_a_raw_web_tool_is_never_autoloaded_by_an_excluded_search():
    result = await _execute_tool_search(
        query="search the web for images of a subject",
        excluded_tool_names=RAW_WEB_TOOL_NAMES,
    )

    recommended = result.get("recommended_tool") or {}
    assert recommended.get("tool_name") not in RAW_WEB_TOOL_NAMES
    assert result.get("loaded_count", 0) == 0


@pytest.mark.asyncio
async def test_unexcluded_web_tools_still_reach_discovery():
    """Only the three product tools replace are hidden. ``tavily_map`` has no
    product equivalent, so hiding it would remove a capability rather than
    route it."""
    result = await _execute_tool_search(
        query="discover the URLs on a website",
        excluded_tool_names=RAW_WEB_TOOL_NAMES,
    )

    returned = {item["tool_name"] for item in result.get("results", [])}
    assert "tavily_map" in returned


@pytest.mark.asyncio
async def test_an_authorized_diagnostic_scope_can_still_reach_the_raw_tools():
    """Operators debugging a provider need the real thing. Passing no exclusion
    set is the explicit opt-in, and only server code can do it."""
    result = await _execute_tool_search(query="search the web for current news")

    returned = {item["tool_name"] for item in result.get("results", [])}
    assert "tavily_search" in returned
