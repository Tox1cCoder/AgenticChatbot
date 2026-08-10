"""Trace-shaped regressions for provider-native T1 image selection."""

from __future__ import annotations

import json

import pytest

from app.ai import web_research_tool
from app.ai.research_budget import reset_research_budget
from app.ai.selected_image_sink import selected_image_sink
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.web_research_tool import create_web_research_tool

CONVERSATION_ID = "22222222-2222-2222-2222-222222222222"
PORTRAIT_ORIGINAL_URL = "https://origin.example/moi-1200x1600.jpg?private=1"
TEAM_ORIGINAL_URL = "https://origin.example/t1-team-995x565.webp?private=1"
TEAM_THUMBNAIL_URL = "https://imgs.search.brave.com/t1-team-thumb.webp"

TAVILY_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "index": 1,
                "title": "LoL: T1 completed 2026 LCK roster",
                "url": "https://sheepesports.example/t1",
                "content": "T1 finalized its 2026 LCK roster.",
                "score": 0.887,
            }
        ],
        "total_results": 1,
        "answer": "T1 is a South Korean esports organization.",
        "provider": "tavily",
        "operation": "search",
        "query": "T1 League of Legends Esports team news roster 2026",
    }
)

BRAVE_PAYLOAD = json.dumps(
    {
        "query": "T1 League of Legends team photo",
        "provider": "brave_image_search",
        "images": [
            {
                "url": "https://imgs.search.brave.com/moi-display.jpg",
                "original_image_url": PORTRAIT_ORIGINAL_URL,
                "thumbnail_url": "https://imgs.search.brave.com/moi-thumb.jpg",
                "confidence": "low",
                "result_rank": 1,
                "provider": "brave_image_search",
                "mime_type": "image/jpeg",
                "title": "Moi",
                "description": "Moi",
                "width": 1200,
                "height": 1600,
                "source_url": "https://sheepesports.example/t1",
            },
            {
                "url": "https://imgs.search.brave.com/t1-team-display.webp",
                "original_image_url": TEAM_ORIGINAL_URL,
                "thumbnail_url": TEAM_THUMBNAIL_URL,
                "confidence": "high",
                "result_rank": 2,
                "provider": "brave_image_search",
                "mime_type": "image/webp",
                "title": "T1 2026 roster",
                "description": "T1 2026 roster",
                "width": 995,
                "height": 565,
                "source_url": "https://sheepesports.example/t1",
            },
        ],
        "total_results": 2,
    }
)


class _Tool:
    def __init__(self, payload: str):
        self.payload = payload

    async def ainvoke(self, args: dict) -> str:
        return self.payload


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)
    monkeypatch.setattr(
        web_research_tool.settings,
        "remote_image_enrichment_enabled",
        True,
        raising=False,
    )
    yield
    clear_tool_context()
    reset_research_budget(CONVERSATION_ID)


async def _run(brave_payload: str = BRAVE_PAYLOAD) -> tuple[str, list[dict]]:
    tool = create_web_research_tool(
        tavily_tool=_Tool(TAVILY_PAYLOAD),
        brave_tool=_Tool(brave_payload),
    )
    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), selected_image_sink() as sink:
        raw = await tool.ainvoke(
            {
                "query": "T1 League of Legends Esports team news roster 2026",
                "image_query": "T1 League of Legends team photo",
            }
        )
    return raw, list(sink)


@pytest.mark.asyncio
async def test_high_confidence_team_image_excludes_the_low_confidence_portrait():
    raw, selected = await _run()

    serialized = json.dumps(selected)
    assert len(selected) == 1
    assert selected[0]["payload"]["url"] == TEAM_THUMBNAIL_URL
    assert "Moi" not in serialized
    assert PORTRAIT_ORIGINAL_URL not in serialized
    assert PORTRAIT_ORIGINAL_URL not in raw


@pytest.mark.asyncio
async def test_selected_candidate_keeps_dimensions_and_digests_the_original_url():
    _, selected = await _run()

    payload = selected[0]["payload"]
    assert payload["width"] == 995
    assert payload["height"] == 565
    assert payload["mime_type"] == "image/webp"
    assert TEAM_ORIGINAL_URL not in json.dumps(selected)
    digests = selected[0]["provenance"]["original_image_digests"]
    assert digests[TEAM_THUMBNAIL_URL] != TEAM_ORIGINAL_URL
    assert len(digests[TEAM_THUMBNAIL_URL]) == 64


@pytest.mark.asyncio
async def test_only_low_confidence_results_offer_no_selected_image():
    low_only = json.loads(BRAVE_PAYLOAD)
    low_only["images"] = low_only["images"][:1]

    _, selected = await _run(json.dumps(low_only))

    assert selected == []
