"""The full chain from a provider-selected image to a model-copyable marker."""

from __future__ import annotations

import json

import pytest

from app.ai import web_research_tool
from app.ai.prompts import build_rich_response_guidance
from app.ai.research_budget import reset_research_budget
from app.ai.rich_image_selection import apply_rich_image_selection
from app.ai.selected_image_sink import selected_image_sink
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.tool_execution import _attach_rich_candidates_to_artifact
from app.ai.web_research_tool import create_web_research_tool
from app.ai.workflow.tool_loop import ToolLoopMixin

CONVERSATION_ID = "77777777-7777-7777-7777-777777777777"
TEAM_ORIGINAL_URL = "https://origin.example/t1-team.webp?private=1"
TEAM_THUMBNAIL_URL = "https://imgs.search.brave.com/t1-team-thumbnail.webp"

TAVILY_PAYLOAD = json.dumps(
    {
        "results": [
            {
                "index": 1,
                "title": "T1 completed 2026 LCK roster",
                "url": "https://sheepesports.example/t1",
                "content": "T1 finalized its 2026 LCK roster.",
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

BRAVE_PAYLOAD = json.dumps(
    {
        "query": "T1 League of Legends team photo",
        "provider": "brave_image_search",
        "images": [
            {
                "url": "https://imgs.search.brave.com/t1-team-display.webp",
                "original_image_url": TEAM_ORIGINAL_URL,
                "thumbnail_url": TEAM_THUMBNAIL_URL,
                "confidence": "high",
                "result_rank": 1,
                "provider": "brave_image_search",
                "mime_type": "image/webp",
                "title": "T1 2026 roster",
                "description": "T1 2026 roster",
                "width": 995,
                "height": 565,
                "source_url": "https://sheepesports.example/t1",
            }
        ],
        "total_results": 1,
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


async def _selected_candidates() -> list[dict]:
    tool = create_web_research_tool(
        tavily_tool=_Tool(TAVILY_PAYLOAD),
        brave_tool=_Tool(BRAVE_PAYLOAD),
    )
    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"
    ), selected_image_sink() as sink:
        await tool.ainvoke({"query": "cho t thông tin về T1"})
    return list(sink)


@pytest.mark.asyncio
async def test_selected_image_becomes_a_marker_the_model_can_copy():
    selected = await _selected_candidates()
    assert selected, "provider selection produced no candidate for the chain"

    artifact: dict = {}
    _attach_rich_candidates_to_artifact(
        artifact,
        raw_result=None,
        result_text="{}",
        render=None,
        tool_call_id="call-1",
        tool_name="web_research",
        selected_images=selected,
    )

    context: dict = {}
    ToolLoopMixin._lift_rich_candidates(context, [artifact])
    apply_rich_image_selection(context)

    candidates = context.get("rich_item_candidates") or []
    assert candidates, "the selected image was dropped before rich-item selection"

    guidance = build_rich_response_guidance(
        candidates=candidates, enabled=True, capability=True
    )

    item_id = candidates[0]["id"]
    assert f"<!--rich:{item_id}-->" in guidance
    assert "AVAILABLE RICH ITEMS" in guidance


@pytest.mark.asyncio
async def test_selected_marker_carries_dimensions_without_the_private_original_url():
    selected = await _selected_candidates()

    payload = selected[0]["payload"]
    assert payload["url"] == TEAM_THUMBNAIL_URL
    assert payload["width"] == 995
    assert payload["height"] == 565
    assert payload["mime_type"] == "image/webp"
    assert TEAM_ORIGINAL_URL not in json.dumps(selected)


@pytest.mark.asyncio
async def test_no_inventory_is_offered_when_the_remote_image_flag_is_off(monkeypatch):
    monkeypatch.setattr(
        web_research_tool.settings,
        "remote_image_enrichment_enabled",
        False,
        raising=False,
    )

    assert await _selected_candidates() == []
    assert build_rich_response_guidance(candidates=[], enabled=True, capability=True) == ""
