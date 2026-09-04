"""Production MCP resolution for the product web tools' dependencies.

Every one of them resolves its provider lazily from the MCP registry when
nothing was injected, and none of them touches the registry at all under
client-only scope. Migrated from the combined research tool's coverage: the
tools changed, the boundary did not.
"""

from __future__ import annotations

import json

import pytest

from app.ai import web_tools
from app.ai.research_budget import reset_research_budget
from app.ai.selected_image_sink import selected_image_sink
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.web_tools import (
    create_image_search_tool,
    create_web_open_tool,
    create_web_search_tool,
)

CONVERSATION_ID = "33333333-3333-3333-3333-333333333333"

TAVILY_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "index": 1,
                "title": "T1 roster",
                "url": "https://sheepesports.example/t1",
                "content": "T1 finalized its roster.",
                "score": 0.9,
            }
        ],
        "total_results": 1,
        "provider": "tavily",
        "operation": "search",
        "query": "T1 roster 2026",
    }
)

EXTRACT_PAYLOAD = json.dumps(
    {
        "provider": "tavily",
        "operation": "extract",
        "urls": ["https://sheepesports.example/t1"],
        "results": [
            {
                "url": "https://sheepesports.example/t1",
                "raw_content": "The roster was finalized on 2 January 2026.",
            }
        ],
        "failed_results": [],
    }
)


class _NamedTool:
    def __init__(self, name: str, payload: str = "{}"):
        self.name = name
        self.payload = payload
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict) -> str:
        self.calls.append(dict(args))
        return self.payload


class _FakeManager:
    """Stands in for MCPManager. Only reachable through an awaited coroutine."""

    def __init__(self, tools_by_server: dict[str, list]):
        self.tools_by_server = tools_by_server
        self.requested: list[str] = []

    async def get_server_tools(self, server_name: str) -> list:
        self.requested.append(server_name)
        return list(self.tools_by_server.get(server_name, []))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)
    monkeypatch.setattr(
        web_tools.settings, "remote_image_enrichment_enabled", True, raising=False
    )
    monkeypatch.setattr(web_tools.settings, "inline_rich_response_enabled", True, raising=False)
    yield
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)


def _patch_manager(monkeypatch, manager: _FakeManager) -> None:
    async def _factory():
        return manager

    monkeypatch.setattr("app.ai.mcp_registry.get_global_mcp_manager", _factory)


async def _run(tool, **kwargs):
    with (
        tool_execution_context(conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"),
        selected_image_sink() as sink,
    ):
        raw = await tool.ainvoke(kwargs)
    return json.loads(raw), sink


def _brave_payload() -> str:
    return json.dumps(
        {
            "images": [
                {
                    "url": "https://imgs.search.brave.com/team.jpg",
                    "original_image_url": "https://origin.example/team.jpg",
                    "thumbnail_url": "https://imgs.search.brave.com/team-thumb.jpg",
                    "confidence": "high",
                    "result_rank": 1,
                    "provider": "brave_image_search",
                    "mime_type": "image/jpeg",
                    "title": "T1 roster",
                    "description": "T1 roster",
                    "width": 995,
                    "height": 565,
                    "source_url": "https://sheepesports.example/t1",
                }
            ]
        }
    )


@pytest.mark.asyncio
async def test_web_search_resolves_tavily_from_mcp_when_it_was_not_injected(monkeypatch):
    tavily = _NamedTool("tavily_search", TAVILY_PAYLOAD)
    manager = _FakeManager({"tavily": [_NamedTool("tavily_extract"), tavily]})
    _patch_manager(monkeypatch, manager)

    payload, _ = await _run(
        create_web_search_tool(), query="T1 roster 2026", objective="Find the roster"
    )

    assert manager.requested == ["tavily"]
    assert len(tavily.calls) == 1
    assert payload["results"][0]["url"] == "https://sheepesports.example/t1"
    assert "status" not in payload


@pytest.mark.asyncio
async def test_web_open_resolves_the_extractor_from_mcp(monkeypatch):
    extract = _NamedTool("tavily_extract", EXTRACT_PAYLOAD)
    manager = _FakeManager({"tavily": [_NamedTool("tavily_search"), extract]})
    _patch_manager(monkeypatch, manager)

    payload, _ = await _run(
        create_web_open_tool(),
        urls=["https://sheepesports.example/t1"],
        question="When was the roster finalized?",
    )

    assert manager.requested == ["tavily"]
    assert len(extract.calls) == 1
    assert "2 January 2026" in " ".join(item["text"] for item in payload["excerpts"])


@pytest.mark.asyncio
async def test_image_search_resolves_brave_from_mcp(monkeypatch):
    brave = _NamedTool("brave_image_search", _brave_payload())
    manager = _FakeManager({"brave_image_search": [brave]})
    _patch_manager(monkeypatch, manager)

    _, sink = await _run(create_image_search_tool(), query="T1 League of Legends team photo")

    assert brave.calls == [{"query": "T1 League of Legends team photo"}]
    assert sink


@pytest.mark.asyncio
async def test_an_unresolvable_provider_still_reports_a_provider_error(monkeypatch):
    _patch_manager(monkeypatch, _FakeManager({}))

    payload, _ = await _run(
        create_web_search_tool(), query="T1 roster 2026", objective="Find the roster"
    )

    assert payload["status"] == "error"
    assert payload["error_type"] == "provider_error"


@pytest.mark.asyncio
async def test_injected_dependencies_are_never_overridden_by_resolution(monkeypatch):
    """Injection must short-circuit resolution, or every test would hit MCP."""

    manager = _FakeManager({"tavily": [_NamedTool("tavily_search", "{}")]})
    _patch_manager(monkeypatch, manager)
    injected = _NamedTool("tavily_search", TAVILY_PAYLOAD)

    payload, _ = await _run(
        create_web_search_tool(tavily_tool=injected),
        query="T1 roster 2026",
        objective="Find the roster",
    )

    assert manager.requested == []
    assert len(injected.calls) == 1
    assert payload["results"][0]["title"] == "T1 roster"


@pytest.mark.parametrize("device_id", ["device-a", None])
@pytest.mark.asyncio
async def test_client_only_direct_invocation_never_calls_server_dependencies(
    monkeypatch, device_id
):
    manager = _FakeManager({"tavily": [_NamedTool("tavily_search", TAVILY_PAYLOAD)]})
    _patch_manager(monkeypatch, manager)
    tavily = _NamedTool("tavily_search", TAVILY_PAYLOAD)
    extract = _NamedTool("tavily_extract", EXTRACT_PAYLOAD)
    brave = _NamedTool("brave_image_search", _brave_payload())

    invocations = (
        (
            create_web_search_tool(tavily_tool=tavily),
            {"query": "T1 roster", "objective": "Find the roster"},
        ),
        (
            create_web_open_tool(extract_tool=extract),
            {"urls": ["https://sheepesports.example/t1"], "question": "When was it finalized?"},
        ),
        (create_image_search_tool(brave_tool=brave), {"query": "T1 team photo"}),
    )

    with (
        tool_execution_context(
            conversation_id=CONVERSATION_ID,
            user_id="u1",
            agent_key="search",
            device_id=device_id,
            tool_scope="client_only",
        ),
        selected_image_sink() as sink,
    ):
        for tool, args in invocations:
            payload = json.loads(await tool.ainvoke(args))
            assert payload["status"] == "error"
            assert payload["error_type"] == "permission_error"
            assert payload["retryable"] is False

    assert manager.requested == []
    assert tavily.calls == []
    assert extract.calls == []
    assert brave.calls == []
    assert sink == []


class _McpShapedTool:
    """An MCP tool as ``load_mcp_tools`` actually returns it."""

    def __init__(self, name: str, payload: str):
        self.name = name
        self.payload = payload
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict) -> list[dict]:
        self.calls.append(dict(args))
        return [{"type": "text", "text": self.payload}]


@pytest.mark.asyncio
async def test_mcp_content_blocks_are_unwrapped_into_the_search_payload(monkeypatch):
    tavily = _McpShapedTool("tavily_search", TAVILY_PAYLOAD)
    _patch_manager(monkeypatch, _FakeManager({"tavily": [tavily]}))

    payload, _ = await _run(
        create_web_search_tool(), query="T1 roster 2026", objective="Find the roster"
    )

    assert payload["results"][0]["url"] == "https://sheepesports.example/t1"
    assert payload["reused"] is False
    assert payload["searches_used"] == 1


@pytest.mark.asyncio
async def test_mcp_content_blocks_from_brave_still_yield_selected_candidates():
    """Brave's payload is unwrapped before deterministic selection."""

    from app.ai.image_discovery_flow import discover_images

    selected = await discover_images(
        brave_tool=_McpShapedTool("brave_image_search", _brave_payload()),
        image_query="T1 team photo",
    )

    assert len(selected) == 1
