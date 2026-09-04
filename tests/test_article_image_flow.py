"""End-to-end guard for the image search → inline display path.

Covers the "cannot send images" complaint: a dedicated-search (Brave image
search) tool result must produce image candidates that survive into the
persisted message as inline rich items even when the model writes no marker
itself (auto-placement). Tavily no longer discovers images at all — page-
scraped images cannot establish relevance from page metadata (see
``build_image_candidates_from_tool_result``), so that provider is out of
scope for this guard.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.ai.tool_execution import (
    build_image_candidates_from_tool_result,
    execute_tool_calls,
)
from app.ai.workflow.tool_loop import ToolLoopMixin
from app.core.config import settings
from app.core.response_constants import build_bot_metadata
from app.core.rich_placement import finalize_article_content
from app.services.event_streaming.ai_sdk_projection import (
    attach_image_parts_to_message,
    project_ai_sdk_message_event,
)
from app.services.message_service import MessageService

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


class _SearchPayloadTool:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def ainvoke(self, _args):
        return json.dumps(self.payload)


@pytest.mark.asyncio
async def test_trace_shaped_parallel_search_selects_brave_without_legacy_gallery(
    monkeypatch,
):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    tavily_payload = {
        "provider": "tavily",
        "query": "T1 roster history",
        "images": [
            {
                "url": f"https://pages.example/assets/{index}.jpg",
                "source_url": f"https://pages.example/articles/{index}",
                "description": "team article illustration",
                "result_rank": index,
            }
            for index in range(114)
        ],
    }
    brave_payload = {
        "provider": "brave_image_search",
        "query": "T1 roster players",
        "images": [
            {
                "url": f"https://media.example/original-{index}.jpg",
                "thumbnail_url": f"https://media.example/roster-{index}.jpg",
                "source_url": f"https://publisher.example/roster-{index}",
                "description": f"T1 roster players on stage {index}",
                "width": 1200,
                "height": 800,
                "provider": "brave_image_search",
            }
            for index in range(3)
        ],
    }

    _outputs, artifacts, legacy_images = await execute_tool_calls(
        tool_calls=[
            {
                "id": "call-tavily",
                "name": "tavily_search",
                "args": {"query": "T1 roster history"},
            },
            {
                "id": "call-brave",
                "name": "brave_image_search",
                "args": {"query": "T1 roster players"},
            },
        ],
        tool_map={
            "tavily_search": _SearchPayloadTool(tavily_payload),
            "brave_image_search": _SearchPayloadTool(brave_payload),
        },
    )
    context: dict = {}
    ToolLoopMixin._lift_rich_candidates(context, artifacts)

    assert legacy_images == []
    assert context["rich_item_candidates"][0]["id"] == ("imagegroup:tool:call-brave")

    answer = (
        "The T1 roster combined Faker with Zeus, Oner, Gumayusi, and Keria "
        "during one of the team's defining competitive eras."
    )
    response = _workflow_response(answer, context["rich_item_candidates"])
    response.metadata["images"] = legacy_images
    finalize_article_content(response, answer)
    metadata = build_bot_metadata(response)

    assert metadata.get("images") in (None, [])
    assert metadata["rich_items"][0]["id"] == "imagegroup:tool:call-brave"


def test_tavily_results_never_land_inline(monkeypatch):
    """Tavily results carry no image candidates at all now, relevant-looking
    or not — there is no metadata-relevance filter left to bypass."""
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)

    payload = json.dumps(
        {
            "query": "eiffel tower at night",
            "images": [
                {
                    "url": "https://example.com/eiffel.jpg",
                    "description": "Eiffel Tower illuminated at night in Paris",
                }
            ],
            "results": [],
        }
    )
    candidates = build_image_candidates_from_tool_result(
        payload, tool_call_id="call-1", tool_name="tavily_search"
    )
    assert candidates == []

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
    assert image["payload"]["source_url"] == ("https://publisher.example/sagrada-familia")
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
