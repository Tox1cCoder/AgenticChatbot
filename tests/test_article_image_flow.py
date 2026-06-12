"""End-to-end guard for the image search → inline display path.

Covers the "cannot send images" complaint: a Tavily-shaped tool result must
produce image candidates that survive into the persisted message as inline
rich items even when the model writes no marker itself (auto-placement).
"""

import json
from types import SimpleNamespace

from app.ai.tool_execution import build_image_candidates_from_tool_result
from app.core.config import settings
from app.core.response_constants import build_bot_metadata
from app.core.rich_placement import finalize_article_content

TAVILY_RESULT = json.dumps(
    {
        "query": "eiffel tower at night",
        "answer": "The Eiffel Tower is lit nightly.",
        "images": [
            {
                "url": "https://example.com/eiffel.jpg",
                "description": "Eiffel Tower illuminated at night in Paris",
            }
        ],
        "results": [],
        "total_results": 0,
    }
)

ANSWER = (
    "The Eiffel Tower is stunning at night, illuminated by thousands of lamps "
    "across Paris.\n\nIt was completed in 1889 for the World's Fair."
)


def _workflow_response(content, candidates):
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        metadata={
            "_inline_rich_response_v1": True,
            "_rich_item_candidates": candidates,
        },
        tool_artifacts=None,
        error=None,
    )


def test_tavily_image_lands_inline_in_persisted_message(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)

    candidates = build_image_candidates_from_tool_result(
        TAVILY_RESULT, tool_call_id="call-1", tool_name="tavily_search"
    )
    assert candidates, "tavily-shaped results must produce image candidates"
    assert candidates[0]["id"] == "image:tool:call-1:0"

    response = _workflow_response(ANSWER, candidates)
    content = finalize_article_content(response, ANSWER)
    assert "<!--rich:image:tool:call-1:0-->" in content

    metadata = build_bot_metadata(response)
    rich_items = metadata.get("rich_items") or []
    image_items = [item for item in rich_items if item.get("type") == "image"]
    assert len(image_items) == 1
    assert image_items[0]["id"] == "image:tool:call-1:0"
    assert image_items[0]["payload"]["url"] == "https://example.com/eiffel.jpg"
    assert metadata.get("rich_reference_warnings") == []


def test_irrelevant_image_stays_dropped(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)

    irrelevant = json.dumps(
        {"images": [{"url": "https://example.com/cat.jpg",
                     "description": "A cat sleeping on a windowsill"}]}
    )
    candidates = build_image_candidates_from_tool_result(
        irrelevant, tool_call_id="call-2", tool_name="tavily_search"
    )
    response = _workflow_response(ANSWER, candidates)
    content = finalize_article_content(response, ANSWER)
    assert "<!--rich:" not in content

    metadata = build_bot_metadata(response)
    assert all(item.get("type") != "image" for item in metadata.get("rich_items") or [])
