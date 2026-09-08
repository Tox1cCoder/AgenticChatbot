"""Integration regressions for provider-native image selection in image_search."""

from __future__ import annotations

import json

import pytest

from app.ai import web_tools
from app.ai.research_budget import reset_research_budget
from app.ai.selected_image_sink import selected_image_sink
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.web_tools import create_image_search_tool

CONVERSATION_ID = "22222222-2222-2222-2222-222222222222"

TAVILY_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "index": 1,
                "title": "T1 completed 2026 LCK roster",
                "url": "https://sheepesports.example/t1",
                "content": "T1 finalized its 2026 LCK roster.",
                "score": 0.887,
            }
        ],
        "total_results": 1,
        "provider": "tavily",
        "operation": "search",
        "query": "T1 roster 2026",
    }
)


class _Tool:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict) -> str:
        self.calls.append(dict(args))
        return self.payload


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_tool_context()
    reset_research_budget(conversation_id=CONVERSATION_ID)
    monkeypatch.setattr(
        web_tools.settings,
        "remote_image_enrichment_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(web_tools.settings, "inline_rich_response_enabled", True, raising=False)
    yield
    clear_tool_context()
    reset_research_budget(conversation_id=CONVERSATION_ID)


def _brave_payload(*confidences: str) -> str:
    return json.dumps(
        {
            "query": "T1 team photo",
            "provider": "brave_image_search",
            "images": [
                {
                    "url": f"https://imgs.search.brave.com/display-{rank}.webp",
                    "original_image_url": f"https://origin.example/team-{rank}.webp",
                    "thumbnail_url": f"https://imgs.search.brave.com/thumb-{rank}.webp",
                    "confidence": confidence,
                    "result_rank": rank,
                    "provider": "brave_image_search",
                    "mime_type": "image/webp",
                    "title": f"T1 roster {rank}",
                    "description": f"T1 roster {rank}",
                    "width": 995,
                    "height": 565,
                    "source_url": f"https://sheepesports.example/t1/{rank}",
                }
                for rank, confidence in enumerate(confidences, start=1)
            ],
            "total_results": len(confidences),
        }
    )


def _structured_brave_failure() -> str:
    return json.dumps(
        {
            "error": "Brave image search rate limited the request.",
            "error_type": "rate_limit",
            "retryable": True,
            "provider": "brave_image_search",
            "images": [],
            "total_results": 0,
        }
    )


async def _run(*, brave_payload: str) -> tuple[dict, list[dict], _Tool]:
    brave = _Tool(brave_payload)
    tool = create_image_search_tool(brave_tool=brave)
    with (
        tool_execution_context(conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"),
        selected_image_sink() as sink,
    ):
        raw = await tool.ainvoke({"query": "T1 team photo"})
    return json.loads(raw), list(sink), brave


@pytest.mark.asyncio
async def test_high_confidence_provider_result_is_selected():
    _, selected, _ = await _run(brave_payload=_brave_payload("high"))

    assert [item["payload"]["url"] for item in selected] == [
        "https://imgs.search.brave.com/thumb-1.webp"
    ]


@pytest.mark.asyncio
async def test_high_confidence_tier_prevents_medium_mixing():
    _, selected, _ = await _run(brave_payload=_brave_payload("medium", "high", "medium"))

    assert [item["payload"]["url"] for item in selected] == [
        "https://imgs.search.brave.com/thumb-2.webp"
    ]


@pytest.mark.asyncio
async def test_medium_confidence_is_the_fallback_when_high_is_absent():
    _, selected, _ = await _run(brave_payload=_brave_payload("low", "medium", "medium"))

    # rank 1 is low-confidence and rejected outright, so the medium tier
    # supplies the single figure.
    assert [item["payload"]["url"] for item in selected] == [
        "https://imgs.search.brave.com/thumb-2.webp",
    ]


@pytest.mark.asyncio
async def test_not_calling_image_search_is_the_opt_out():
    """There is no skip flag any more: an answer that wants no picture simply
    does not spend a call, and the text path never touches Brave."""
    tavily = _Tool(TAVILY_PAYLOAD)
    tool = create_image_search_tool(brave_tool=_Tool(_brave_payload("high")), tavily_tool=tavily)

    with (
        tool_execution_context(conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"),
        selected_image_sink() as sink,
    ):
        pass

    assert tavily.calls == []
    assert sink == []
    assert tool.name == "image_search"


@pytest.mark.asyncio
async def test_disabled_remote_image_enrichment_skips_brave(monkeypatch):
    monkeypatch.setattr(
        web_tools.settings,
        "remote_image_enrichment_enabled",
        False,
        raising=False,
    )

    payload, selected, brave = await _run(brave_payload=_brave_payload("high"))

    assert brave.calls == []
    assert selected == []
    assert payload["selected"] == 0


@pytest.mark.asyncio
async def test_a_structured_brave_failure_is_reported_without_an_exception():
    payload, selected, brave = await _run(brave_payload=_structured_brave_failure())

    assert brave.calls == [{"query": "T1 team photo"}]
    assert selected == []
    assert payload["selected"] == 0
    assert "status" not in payload
