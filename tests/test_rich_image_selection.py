from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

from app.core.rich_image_selection import (
    ImageSelectionPolicy,
    select_rich_item_candidates,
)

POLICY = ImageSelectionPolicy(
    max_items=2,
    min_width_px=320,
    min_height_px=180,
    min_aspect_ratio=0.2,
    max_aspect_ratio=5.0,
)


def _image(
    item_id: str,
    *,
    source: str,
    url: str,
    query: str = "T1 roster",
    result_rank: int = 0,
    width: int | None = 1200,
    height: int | None = 800,
    source_url: str | None = None,
    description: str = "T1 roster players",
    query_level: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "url": url,
        "description": description,
        "width": width,
        "height": height,
    }
    if source_url is not None:
        payload["source_url"] = source_url
    return {
        "id": item_id,
        "type": "image",
        "source": source,
        "payload": payload,
        "provenance": {
            "provider": "test",
            "query": query,
            "result_rank": result_rank,
            "query_level": query_level,
        },
    }


def test_dedicated_search_outranks_eight_source_bound_web_images() -> None:
    tavily = [
        _image(
            f"image:tavily:{index}",
            source="web_search",
            url=f"https://pages.example/assets/{index}.jpg",
            result_rank=index,
            source_url=f"https://pages.example/article/{index}",
            description="team article illustration",
        )
        for index in range(8)
    ]
    brave = _image(
        "image:brave:0",
        source="image_search",
        url="https://media.example/t1-roster.jpg",
        description="T1 roster players Faker Zeus Oner Gumayusi Keria",
    )
    candidates = [*tavily, brave]
    original = deepcopy(candidates)

    selected = select_rich_item_candidates(candidates, policy=POLICY)

    assert [item["id"] for item in selected] == [
        "image:brave:0",
        "image:tavily:0",
    ]
    assert candidates == original


def test_query_level_web_asset_is_excluded() -> None:
    candidate = _image(
        "image:tavily:query",
        source="web_search",
        url="https://search.example/generic.png",
        query_level=True,
    )

    assert select_rich_item_candidates([candidate], policy=POLICY) == []


def test_direct_document_image_remains_intentional() -> None:
    candidates = [
        _image(
            "image:web:0",
            source="image_search",
            url="https://media.example/result.jpg",
        ),
        _image(
            "image:document:0",
            source="rag_document",
            url="https://files.example/figure-3.png",
            description="figure from the cited document",
        ),
    ]

    selected = select_rich_item_candidates(
        candidates,
        policy=replace(POLICY, max_items=1),
    )

    assert [item["id"] for item in selected] == ["image:document:0"]


def test_non_image_items_are_preserved_in_original_order() -> None:
    widget = {"id": "widget:weather:0", "type": "weather", "payload": {}}

    selected = select_rich_item_candidates(
        [
            widget,
            _image(
                "image:web:0",
                source="image_search",
                url="https://media.example/a.jpg",
            ),
        ],
        policy=replace(POLICY, max_items=0),
    )

    assert selected == [widget]
