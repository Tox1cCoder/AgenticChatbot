"""End-to-end guard for the image search → inline display path.

Covers the "cannot send images" complaint: a Tavily-shaped tool result must
produce image candidates that survive into the persisted message as inline
rich items even when the model writes no marker itself (auto-placement).
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.ai.tool_execution import build_image_candidates_from_tool_result
from app.core.config import settings
from app.core.response_constants import build_bot_metadata
from app.core.rich_placement import finalize_article_content
from app.services.event_streaming.ai_sdk_projection import (
    attach_image_parts_to_message,
    project_ai_sdk_message_event,
)
from app.services.message_service import MessageService

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

BRAVE_RESULT = json.dumps(
    {
        "query": "sagrada familia exterior",
        "provider": "brave_image_search",
        "images": [
            {
                "url": "https://images.example/original.jpg",
                "thumbnail_url": "https://images.example/thumbnail.jpg",
                "source_url": "https://publisher.example/sagrada-familia",
                "title": "Sagrada Familia exterior",
                "width": 1200,
                "height": 800,
            }
        ],
    }
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
        {
            "images": [
                {
                    "url": "https://example.com/cat.jpg",
                    "description": "A cat sleeping on a windowsill",
                }
            ]
        }
    )
    candidates = build_image_candidates_from_tool_result(
        irrelevant, tool_call_id="call-2", tool_name="tavily_search"
    )
    response = _workflow_response(ANSWER, candidates)
    content = finalize_article_content(response, ANSWER)
    assert "<!--rich:" not in content

    metadata = build_bot_metadata(response)
    assert all(item.get("type") != "image" for item in metadata.get("rich_items") or [])


@pytest.mark.asyncio
async def test_brave_selection_persists_one_protected_visual_and_projects_once(monkeypatch):
    """Cover discovery through the persisted AI SDK terminal-message shape."""
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    answer = "The Sagrada Familia exterior combines dramatic towers and detailed stonework."
    candidates = build_image_candidates_from_tool_result(
        BRAVE_RESULT,
        tool_call_id="brave-call",
        tool_name="brave_image_search",
    )
    assert candidates[0]["payload"]["url"] == "https://images.example/thumbnail.jpg"

    response = _workflow_response(answer, candidates)
    content = finalize_article_content(response, answer)
    metadata = build_bot_metadata(response)

    reference_id = uuid4()
    web_images = AsyncMock()
    web_images.register.return_value = SimpleNamespace(id=reference_id)
    service = MessageService.__new__(MessageService)
    service.web_image_service = web_images
    stored_content, stored_metadata = await service._externalize_remote_rich_images(
        content,
        metadata,
        uuid4(),
        uuid4(),
    )

    marker = "<!--rich:image:tool:brave-call:0-->"
    assert marker in stored_content
    assert stored_content.replace(marker, "").strip() == answer
    assert "_rich_item_candidates" not in stored_metadata
    assert len(stored_metadata["rich_items"]) == 1
    image = stored_metadata["rich_items"][0]
    protected_url = f"/web-images/{reference_id}"
    assert image["payload"]["url"] == protected_url
    assert image["payload"]["source_url"] == (
        "https://publisher.example/sagrada-familia"
    )
    web_images.fetch.assert_not_called()

    persisted = attach_image_parts_to_message(
        {
            "id": "assistant-1",
            "sender": 2,
            "content": stored_content,
            "message_metadata": stored_metadata,
            "parts": [
                {"type": "text", "text": stored_content},
                {"type": "file", "url": protected_url, "mediaType": "image/jpeg"},
            ],
        }
    )
    projected = project_ai_sdk_message_event(persisted, include_content=True)
    file_parts = [part for part in projected["parts"] if part.get("type") == "file"]
    assert [part["url"] for part in file_parts] == [protected_url]
    assert projected["content"] == stored_content


@pytest.mark.asyncio
async def test_reference_registration_failure_keeps_complete_text_answer(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    answer = "The Sagrada Familia exterior is richly decorated."
    candidates = build_image_candidates_from_tool_result(
        BRAVE_RESULT,
        tool_call_id="brave-call",
        tool_name="brave_image_search",
    )
    response = _workflow_response(answer, candidates)
    content = finalize_article_content(response, answer)
    metadata = build_bot_metadata(response)
    web_images = AsyncMock()
    web_images.register.side_effect = RuntimeError("registration unavailable")
    service = MessageService.__new__(MessageService)
    service.web_image_service = web_images

    stored_content, stored_metadata = await service._externalize_remote_rich_images(
        content,
        metadata,
        uuid4(),
        uuid4(),
    )

    assert stored_content == answer
    assert stored_metadata["rich_items"] == []
    assert "<!--rich:" not in stored_content
