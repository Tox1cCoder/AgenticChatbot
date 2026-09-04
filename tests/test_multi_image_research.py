"""An answer gets as many distinct pictures as it needs, one subject per call.

A "what is X" answer often wants the thing's identity art *and* a shot of it in
use; a how-to answer usually wants one exact screenshot. Both are the same
mechanism: the model asks for one subject per call and stops when the answer is
served. What it must never do is take a deeper slice of a single query, which
returns the same subject twice.

The second call reuses the turn's research text, so a second picture costs one
image request and no extra web search.
"""

from __future__ import annotations

import json

import pytest

from app.ai import web_research_tool
from app.ai.research_budget import reset_research_budget
from app.ai.selected_image_sink import selected_image_sink
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.web_research_tool import create_web_research_tool

CONVERSATION_ID = "44444444-4444-4444-4444-444444444444"

TAVILY_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "index": 1,
                "title": "Pokemon Unite",
                "url": "https://example.test/unite",
                "content": "Pokemon Unite is a MOBA.",
            }
        ],
        "total_results": 1,
        "provider": "tavily",
        "operation": "search",
        "query": "pokemon unite",
    }
)


def _brave_payload(slug: str) -> str:
    return json.dumps(
        {
            "query": slug,
            "provider": "brave_image_search",
            "images": [
                {
                    "url": f"https://imgs.search.brave.com/{slug}-display.webp",
                    "original_image_url": f"https://origin.example/{slug}.webp",
                    "thumbnail_url": f"https://imgs.search.brave.com/{slug}-thumb.webp",
                    "confidence": "high",
                    "result_rank": 1,
                    "provider": "brave_image_search",
                    "mime_type": "image/webp",
                    "title": slug,
                    "description": slug,
                    "width": 995,
                    "height": 565,
                    "source_url": f"https://publisher.example/{slug}",
                }
            ],
            "total_results": 1,
        }
    )


class _Tavily:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict) -> str:
        self.calls.append(dict(args))
        return TAVILY_PAYLOAD


class _Brave:
    """Returns a different image per image query, as the real provider would."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict) -> str:
        self.calls.append(dict(args))
        slug = "logo" if "logo" in str(args.get("query", "")).lower() else "gameplay"
        return _brave_payload(slug)


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


@pytest.mark.asyncio
async def test_two_distinct_subjects_yield_two_distinct_images():
    tavily, brave = _Tavily(), _Brave()
    tool = create_web_research_tool(tavily_tool=tavily, brave_tool=brave)

    with (
        tool_execution_context(conversation_id=CONVERSATION_ID, user_id="u1", agent_key="chat"),
        selected_image_sink() as sink,
    ):
        await tool.ainvoke({"query": "pokemon unite", "image_query": "Pokemon Unite logo"})
        await tool.ainvoke(
            {"query": "pokemon unite", "image_query": "Pokemon Unite gameplay screenshot"}
        )

    assert [item["payload"]["url"] for item in sink] == [
        "https://imgs.search.brave.com/logo-thumb.webp",
        "https://imgs.search.brave.com/gameplay-thumb.webp",
    ]


@pytest.mark.asyncio
async def test_a_second_picture_does_not_buy_a_second_web_search():
    """The same factual query reuses the turn's research text, so asking for
    another subject costs one image request and nothing else."""
    tavily, brave = _Tavily(), _Brave()
    tool = create_web_research_tool(tavily_tool=tavily, brave_tool=brave)

    with (
        tool_execution_context(conversation_id=CONVERSATION_ID, user_id="u1", agent_key="chat"),
        selected_image_sink(),
    ):
        await tool.ainvoke({"query": "pokemon unite", "image_query": "Pokemon Unite logo"})
        await tool.ainvoke(
            {"query": "pokemon unite", "image_query": "Pokemon Unite gameplay screenshot"}
        )

    assert len(tavily.calls) == 1
    assert len(brave.calls) == 2


@pytest.mark.asyncio
async def test_asking_twice_for_the_same_subject_searches_once():
    tavily, brave = _Tavily(), _Brave()
    tool = create_web_research_tool(tavily_tool=tavily, brave_tool=brave)

    with (
        tool_execution_context(conversation_id=CONVERSATION_ID, user_id="u1", agent_key="chat"),
        selected_image_sink(),
    ):
        await tool.ainvoke({"query": "pokemon unite", "image_query": "Pokemon Unite logo"})
        await tool.ainvoke({"query": "pokemon unite", "image_query": "Pokemon Unite logo"})

    assert len(brave.calls) == 1
