"""Production wiring for ``web_research``'s injected-in-tests dependencies.

Every other ``web_research`` test injects ``tavily_tool``, ``brave_tool`` and
``web_image_service`` directly, so nothing exercised the ``None`` defaults that
``base_agent`` actually binds with. Those defaults are the production path: if
they resolve to nothing, research reports "tavily_search is unavailable" and the
image path silently returns text-only forever, and no existing test notices.
"""

from __future__ import annotations

import json

import pytest

from app.ai import web_research_tool
from app.ai.research_budget import reset_research_budget
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.verified_image_sink import verified_image_sink
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
def _clean():
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)
    yield
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)


def _patch_manager(monkeypatch, manager: _FakeManager) -> None:
    """Patch the *async* factory, exactly as production defines it.

    ``get_global_mcp_manager`` is ``async def``; a caller that forgets to await
    it holds a coroutine, and ``coroutine.get_server_tools`` raises
    ``AttributeError`` inside ``_resolve_tool``'s blanket ``except``. Patching
    with an async function is what makes that mistake visible here.
    """

    async def _factory():
        return manager

    monkeypatch.setattr("app.ai.mcp_registry.get_global_mcp_manager", _factory)


async def _run(tool, **kwargs):
    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), verified_image_sink() as sink:
        raw = await tool.ainvoke(kwargs)
    return json.loads(raw), sink


@pytest.mark.asyncio
async def test_tavily_resolves_from_mcp_when_it_was_not_injected(monkeypatch):
    tavily = _NamedTool("tavily_search", TAVILY_PAYLOAD)
    manager = _FakeManager({"tavily": [_NamedTool("tavily_extract"), tavily]})
    _patch_manager(monkeypatch, manager)

    payload, _ = await _run(create_web_research_tool(), query="T1 roster 2026")

    # Images are default-on, so the un-injected image path resolves too.
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
async def test_brave_and_the_image_service_resolve_when_they_were_not_injected(
    monkeypatch,
):
    """The image path must reach real dependencies, not silently no-op.

    Both were left at ``None`` by ``base_agent``, and
    ``discover_and_verify_images`` returns ``[]`` when either is missing — so
    an enabled verifier produced text-only answers with no error anywhere.
    """

    brave = _NamedTool("brave_image_search", json.dumps({"images": []}))
    manager = _FakeManager(
        {
            "tavily": [_NamedTool("tavily_search", TAVILY_PAYLOAD)],
            "brave_image_search": [brave],
        }
    )
    _patch_manager(monkeypatch, manager)
    monkeypatch.setattr(
        web_research_tool.settings, "vision_image_verification_enabled", True, raising=False
    )

    seen: dict[str, object] = {}

    async def _capture(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(
        "app.ai.image_verification_flow.discover_and_verify_images", _capture
    )

    await _run(
        create_web_research_tool(),
        query="T1 roster 2026",
        image_query="T1 League of Legends team photo",
    )

    assert seen["brave_tool"] is brave
    assert seen["web_image_service"] is not None
    assert hasattr(seen["web_image_service"], "fetch_url")


@pytest.mark.asyncio
async def test_injected_dependencies_are_never_overridden_by_resolution(monkeypatch):
    """Injection must short-circuit resolution, or every test would hit MCP."""

    class _Service:
        async def fetch_url(self, url: str, *, provider: str = "other"):
            raise AssertionError("no candidate should be discovered")

    manager = _FakeManager({"tavily": [_NamedTool("tavily_search", "{}")]})
    _patch_manager(monkeypatch, manager)
    injected = _NamedTool("tavily_search", TAVILY_PAYLOAD)

    payload, _ = await _run(
        create_web_research_tool(
            tavily_tool=injected,
            brave_tool=_NamedTool("brave_image_search", json.dumps({"images": []})),
            web_image_service=_Service(),
        ),
        query="T1 roster 2026",
    )

    assert manager.requested == []
    assert len(injected.calls) == 1
    assert payload["answer"].startswith("T1 is a South Korean")


class _McpShapedTool:
    """An MCP tool as ``load_mcp_tools`` actually returns it.

    ``ainvoke`` yields a list of content blocks, not the JSON string the tool
    printed. Every other test in the suite fakes a bare string, so nothing
    covered the shape production sees.
    """

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
    # research metadata only lands when the payload parsed as JSON, so its
    # presence is what proves the block wrapper was stripped rather than
    # str()-ed into a Python repr.
    assert payload["research"] == {"reused": False, "searches_used": 1}


@pytest.mark.asyncio
async def test_mcp_content_blocks_from_brave_still_yield_image_candidates(monkeypatch):
    """Brave's payload is unwrapped too, or discovery finds zero candidates."""

    from app.ai.image_verification_flow import discover_and_verify_images

    brave_payload = json.dumps(
        {
            "query": "T1 team photo",
            "provider": "brave_image_search",
            "images": [
                {
                    "url": "https://cdn.example/team.jpg",
                    "provider": "brave_image_search",
                    "mime_type": "image/jpeg",
                    "title": "T1 roster",
                    "description": "T1 roster",
                    "width": 995,
                    "height": 565,
                    "source_url": "https://sheepesports.example/t1",
                }
            ],
            "total_results": 1,
        }
    )
    seen: list[str] = []

    class _Service:
        async def fetch_url(self, url: str, *, provider: str = "other"):
            seen.append(url)
            raise RuntimeError("stop after discovery")

    await discover_and_verify_images(
        brave_tool=_McpShapedTool("brave_image_search", brave_payload),
        web_image_service=_Service(),
        verifier_model=None,
        user_request="T1 roster 2026",
        image_query="T1 team photo",
        factual_query="T1 roster 2026",
    )

    assert seen == ["https://cdn.example/team.jpg"]
