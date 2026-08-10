"""Integration regressions for provider-native image selection in web research."""

from __future__ import annotations

import json

import pytest

from app.ai import web_research_tool
from app.ai.research_budget import reset_research_budget
from app.ai.selected_image_sink import selected_image_sink
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.web_research_tool import create_web_research_tool

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


async def _run(
    *,
    brave_payload: str,
    skip_images: bool = False,
) -> tuple[dict, list[dict], _Tool]:
    brave = _Tool(brave_payload)
    tool = create_web_research_tool(
        tavily_tool=_Tool(TAVILY_PAYLOAD),
        brave_tool=brave,
    )
    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), selected_image_sink() as sink:
        raw = await tool.ainvoke(
            {
                "query": "T1 roster 2026",
                "image_query": "T1 team photo",
                "skip_images": skip_images,
            }
        )
    return json.loads(raw), list(sink), brave


@pytest.mark.asyncio
async def test_high_confidence_provider_result_is_selected():
    _, selected, _ = await _run(brave_payload=_brave_payload("high"))

    assert [item["payload"]["url"] for item in selected] == [
        "https://imgs.search.brave.com/thumb-1.webp"
    ]


@pytest.mark.asyncio
async def test_high_confidence_tier_prevents_medium_mixing():
    _, selected, _ = await _run(
        brave_payload=_brave_payload("medium", "high", "medium")
    )

    assert [item["payload"]["url"] for item in selected] == [
        "https://imgs.search.brave.com/thumb-2.webp"
    ]


@pytest.mark.asyncio
async def test_medium_confidence_is_the_fallback_when_high_is_absent():
    _, selected, _ = await _run(
        brave_payload=_brave_payload("low", "medium", "medium")
    )

    assert [item["payload"]["url"] for item in selected] == [
        "https://imgs.search.brave.com/thumb-2.webp",
        "https://imgs.search.brave.com/thumb-3.webp",
    ]


@pytest.mark.asyncio
async def test_explicit_image_opt_out_skips_brave():
    payload, selected, brave = await _run(
        brave_payload=_brave_payload("high"), skip_images=True
    )

    assert brave.calls == []
    assert selected == []
    assert payload["results"]


@pytest.mark.asyncio
async def test_disabled_remote_image_enrichment_skips_brave(monkeypatch):
    monkeypatch.setattr(
        web_research_tool.settings,
        "remote_image_enrichment_enabled",
        False,
        raising=False,
    )

    payload, selected, brave = await _run(brave_payload=_brave_payload("high"))

    assert brave.calls == []
    assert selected == []
    assert payload["results"]


@pytest.mark.asyncio
async def test_structured_brave_failure_leaves_tavily_text_successful():
    payload, selected, brave = await _run(brave_payload=_structured_brave_failure())

    assert brave.calls == [{"query": "T1 team photo"}]
    assert selected == []
    assert payload["results"][0]["content"] == "T1 finalized its 2026 LCK roster."
    assert "status" not in payload
