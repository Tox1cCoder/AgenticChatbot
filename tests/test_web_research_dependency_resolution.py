"""Production MCP resolution for ``web_research`` dependencies."""

from __future__ import annotations

import json

import pytest

from app.ai import web_research_tool
from app.ai.research_budget import reset_research_budget
from app.ai.selected_image_sink import selected_image_sink
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.web_research_tool import create_web_research_tool

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
        "answer": "T1 is a South Korean esports organization.",
        "provider": "tavily",
        "operation": "search",
        "query": "T1 roster 2026",
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
        web_research_tool.settings, "remote_image_enrichment_enabled", True, raising=False
    )
    yield
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)


def _patch_manager(monkeypatch, manager: _FakeManager) -> None:
    async def _factory():
        return manager

    monkeypatch.setattr("app.ai.mcp_registry.get_global_mcp_manager", _factory)


async def _run(tool, **kwargs):
    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), selected_image_sink() as sink:
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
async def test_tavily_resolves_from_mcp_when_it_was_not_injected(monkeypatch):
    tavily = _NamedTool("tavily_search", TAVILY_PAYLOAD)
    manager = _FakeManager({"tavily": [_NamedTool("tavily_extract"), tavily]})
    _patch_manager(monkeypatch, manager)

    payload, _ = await _run(create_web_research_tool(), query="T1 roster 2026")

    assert sorted(manager.requested) == ["brave_image_search", "tavily"]
    assert len(tavily.calls) == 1
    assert payload["answer"].startswith("T1 is a South Korean")
    assert "status" not in payload


@pytest.mark.asyncio
async def test_an_unresolvable_tavily_still_reports_a_provider_error(monkeypatch):
    _patch_manager(monkeypatch, _FakeManager({}))

    payload, _ = await _run(create_web_research_tool(), query="T1 roster 2026")

    assert payload["status"] == "error"
    assert payload["error_type"] == "provider_error"


@pytest.mark.asyncio
async def test_brave_resolves_when_it_was_not_injected(monkeypatch):
    brave = _NamedTool("brave_image_search", _brave_payload())
    manager = _FakeManager(
        {
            "tavily": [_NamedTool("tavily_search", TAVILY_PAYLOAD)],
            "brave_image_search": [brave],
        }
    )
    _patch_manager(monkeypatch, manager)

    _, sink = await _run(
        create_web_research_tool(),
        query="T1 roster 2026",
        image_query="T1 League of Legends team photo",
    )

    assert brave.calls == [{"query": "T1 League of Legends team photo"}]
    assert sink


@pytest.mark.asyncio
async def test_injected_dependencies_are_never_overridden_by_resolution(monkeypatch):
    """Injection must short-circuit resolution, or every test would hit MCP."""

    manager = _FakeManager({"tavily": [_NamedTool("tavily_search", "{}")]})
    _patch_manager(monkeypatch, manager)
    injected = _NamedTool("tavily_search", TAVILY_PAYLOAD)

    payload, _ = await _run(
        create_web_research_tool(
            tavily_tool=injected,
            brave_tool=_NamedTool("brave_image_search", json.dumps({"images": []})),
        ),
        query="T1 roster 2026",
    )

    assert manager.requested == []
    assert len(injected.calls) == 1
    assert payload["answer"].startswith("T1 is a South Korean")


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
async def test_mcp_content_blocks_are_unwrapped_into_the_research_payload(monkeypatch):
    tavily = _McpShapedTool("tavily_search", TAVILY_PAYLOAD)
    _patch_manager(monkeypatch, _FakeManager({"tavily": [tavily]}))

    payload, _ = await _run(create_web_research_tool(), query="T1 roster 2026")

    assert payload["answer"].startswith("T1 is a South Korean")
    assert payload["results"][0]["url"] == "https://sheepesports.example/t1"
    assert payload["research"] == {"reused": False, "searches_used": 1}


@pytest.mark.asyncio
async def test_mcp_content_blocks_from_brave_still_yield_selected_candidates():
    """Brave's payload is unwrapped before deterministic selection."""

    from app.ai.image_discovery_flow import discover_images

    selected = await discover_images(
        brave_tool=_McpShapedTool("brave_image_search", _brave_payload()),
        image_query="T1 team photo",
    )

    assert len(selected) == 1
