"""The full chain from a provider-selected image to a model-copyable marker."""

from __future__ import annotations

import json

import pytest

from app.ai import web_tools
from app.ai.prompts import build_rich_response_guidance
from app.ai.research_budget import reset_research_budget
from app.ai.rich_image_selection import apply_rich_image_selection
from app.ai.selected_image_sink import selected_image_sink
from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.tool_execution import _attach_rich_candidates_to_artifact
from app.ai.web_tools import create_image_search_tool
from app.ai.workflow.tool_loop import ToolLoopMixin
from app.core.rich_response import sanitize_public_rich_item

CONVERSATION_ID = "77777777-7777-7777-7777-777777777777"
ORIGINAL_IMAGE_URL = "https://origin.example/t1-team.webp?private=1"
THUMBNAIL_URL = "https://imgs.search.brave.com/t1-team-thumbnail.webp"


class _Brave:
    def __init__(self, *, confidence: str) -> None:
        self.confidence = confidence

    async def ainvoke(self, args: dict) -> str:
        return json.dumps(
            {
                "query": args["query"],
                "provider": "brave_image_search",
                "images": [
                    {
                        "url": "https://imgs.search.brave.com/t1-team-display.webp",
                        "original_image_url": ORIGINAL_IMAGE_URL,
                        "thumbnail_url": THUMBNAIL_URL,
                        "confidence": self.confidence,
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


async def _provider_selected_candidates() -> list[dict]:
    tool = create_image_search_tool(brave_tool=_Brave(confidence="high"))
    with (
        tool_execution_context(conversation_id=CONVERSATION_ID, user_id="u1", agent_key="search"),
        selected_image_sink() as sink,
    ):
        await tool.ainvoke({"query": "T1 team photo"})
    return list(sink)


@pytest.mark.asyncio
async def test_a_provider_selected_image_becomes_a_marker_the_model_can_copy():
    selected = await _provider_selected_candidates()
    assert selected, "provider selection produced no candidate for the chain"

    artifact: dict = {}
    _attach_rich_candidates_to_artifact(
        artifact,
        raw_result=None,
        result_text="{}",
        render=None,
        tool_call_id="call-1",
        tool_name="image_search",
        selected_images=selected,
    )
    context: dict = {}
    ToolLoopMixin._lift_rich_candidates(context, [artifact])
    apply_rich_image_selection(context)

    candidates = context.get("rich_item_candidates") or []
    assert candidates[0]["source"] == "image_search"

    guidance = build_rich_response_guidance(candidates=candidates, enabled=True, capability=True)
    assert f"<!--rich:{candidates[0]['id']}-->" in guidance
    assert "AVAILABLE RICH ITEMS" in guidance


@pytest.mark.asyncio
async def test_public_selected_image_metadata_uses_the_thumbnail_without_original_url():
    selected = await _provider_selected_candidates()

    public = sanitize_public_rich_item(selected[0])
    assert public["payload"]["url"] == THUMBNAIL_URL
    assert public["payload"]["width"] == 995
    assert public["payload"]["height"] == 565
    assert public["payload"]["mime_type"] == "image/webp"
    assert "original_image_digests" not in public["provenance"]
    assert ORIGINAL_IMAGE_URL not in json.dumps(public)
