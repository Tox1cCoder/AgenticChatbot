from __future__ import annotations

from typing import Any

import pytest

from app.ai.prompts import build_rich_response_guidance
from app.ai.rag_tool_actions import register_document_image_candidates
from app.ai.rich_image_selection import apply_rich_image_selection
from app.ai.workflow.tool_loop import ToolLoopMixin
from app.core.config import settings
from app.core.rich_placement import _image_anchor_entries


def _web_image(index: int) -> dict[str, Any]:
    return {
        "id": f"image:tavily:{index}",
        "type": "image",
        "source": "web_search",
        "display_policy": "inline_only",
        "alt_text": "team article image",
        "payload": {
            "url": f"https://pages.example/assets/{index}.jpg",
            "source_url": f"https://pages.example/articles/{index}",
            "description": "team article image",
            "width": 1200,
            "height": 800,
            "mime_type": "image/jpeg",
        },
        "provenance": {
            "provider": "tavily",
            "query": "T1 roster",
            "result_rank": index,
        },
    }


def _image_search_group() -> dict[str, Any]:
    return {
        "id": "imagegroup:brave:0",
        "type": "image_group",
        "source": "image_search",
        "display_policy": "inline_only",
        "alt_text": "T1 roster players",
        "payload": {
            "items": [
                {
                    "url": "https://media.example/t1-roster.jpg",
                    "description": "T1 roster Faker Zeus Oner Gumayusi Keria",
                    "width": 1200,
                    "height": 800,
                    "mime_type": "image/jpeg",
                },
                {
                    "url": "https://media.example/t1-stage.jpg",
                    "description": "T1 players on stage",
                    "width": 1200,
                    "height": 800,
                    "mime_type": "image/jpeg",
                },
            ]
        },
        "provenance": {
            "provider": "brave_image_search",
            "query": "T1 roster",
        },
    }


def test_lift_selects_from_complete_parallel_tool_batch() -> None:
    context: dict[str, Any] = {"rich_item_candidates": []}
    artifacts = [
        {"_rich_item_candidates": [_web_image(index) for index in range(8)]},
        {"_rich_item_candidates": [_image_search_group()]},
    ]

    ToolLoopMixin._lift_rich_candidates(context, artifacts)

    images = [
        item
        for item in context["rich_item_candidates"]
        if item["type"] in {"image", "image_group"}
    ]
    assert images[0]["id"] == "imagegroup:brave:0"
    assert len(images) <= settings.rich_auto_place_max_images
    assert all("_rich_item_candidates" not in artifact for artifact in artifacts)


def test_later_dedicated_search_replaces_earlier_web_candidate() -> None:
    context: dict[str, Any] = {"rich_item_candidates": []}
    ToolLoopMixin._lift_rich_candidates(
        context,
        [{"_rich_item_candidates": [_web_image(0), _web_image(1)]}],
    )
    ToolLoopMixin._lift_rich_candidates(
        context,
        [{"_rich_item_candidates": [_image_search_group()]}],
    )

    assert context["rich_item_candidates"][0]["id"] == "imagegroup:brave:0"
    assert len(context["rich_item_candidates"]) == settings.rich_auto_place_max_images


def test_selector_failure_keeps_widget_and_drops_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    widget = {
        "id": "widget:weather:0",
        "type": "live_widget",
        "payload": {},
    }
    context = {"rich_item_candidates": [widget, _web_image(0)]}

    def _raise(*args: object, **kwargs: object) -> list[dict[str, object]]:
        raise RuntimeError("synthetic selector failure")

    monkeypatch.setattr(
        "app.ai.rich_image_selection.select_rich_item_candidates",
        _raise,
    )

    apply_rich_image_selection(context)

    assert context["rich_item_candidates"] == [widget]


def test_document_registration_reuses_canonical_selector() -> None:
    context: dict[str, Any] = {
        "rich_item_candidates": [_web_image(0), _web_image(1)]
    }

    added = register_document_image_candidates(
        context=context,
        images=[
            {
                "id": "figure-3",
                "mime_type": "image/png",
                "data": "QUJDRA==",
                "caption": "Figure from the cited document",
                "page_number": 3,
            }
        ],
    )

    assert added == 1
    assert context["rich_item_candidates"][0]["id"] == "image:document:figure-3"
    assert len(context["rich_item_candidates"]) == settings.rich_auto_place_max_images


def test_prompt_and_placement_preserve_canonical_image_order() -> None:
    context: dict[str, Any] = {"rich_item_candidates": []}
    ToolLoopMixin._lift_rich_candidates(
        context,
        [
            {"_rich_item_candidates": [_web_image(0), _web_image(1)]},
            {"_rich_item_candidates": [_image_search_group()]},
        ],
    )
    selected_ids = [item["id"] for item in context["rich_item_candidates"]]

    guidance = build_rich_response_guidance(
        candidates=context["rich_item_candidates"],
        enabled=True,
        capability=True,
    )
    anchor_ids = [
        entry.item_id
        for entry in _image_anchor_entries(
            {"_rich_item_candidates": context["rich_item_candidates"]},
            image_max_items=settings.rich_auto_place_max_images,
        )
    ]

    assert selected_ids == ["imagegroup:brave:0", "image:tavily:0"]
    assert guidance.index(selected_ids[0]) < guidance.index(selected_ids[1])
    assert anchor_ids == selected_ids
