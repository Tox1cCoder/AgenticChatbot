from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

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
        "mime_type": "image/jpeg",
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


def _cell(
    url: str,
    *,
    width: int | None = 1200,
    height: int | None = 800,
    description: str = "T1 roster players",
) -> dict[str, Any]:
    return {
        "url": url,
        "mime_type": "image/jpeg",
        "width": width,
        "height": height,
        "description": description,
    }


def _group(
    item_id: str,
    cells: list[dict[str, Any]],
    *,
    query: str = "T1 roster",
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "image_group",
        "source": "image_search",
        "alt_text": f"Images of {query}",
        "payload": {"items": cells},
        "provenance": {
            "provider": "brave_image_search",
            "query": query,
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


def test_known_tiny_image_is_rejected() -> None:
    candidate = _image(
        "image:tiny",
        source="image_search",
        url="https://media.example/tiny.png",
        width=45,
        height=30,
    )

    assert select_rich_item_candidates([candidate], policy=POLICY) == []


@pytest.mark.parametrize("source", ["rag_document", "tool_image", "generated_image"])
def test_direct_sources_keep_their_existing_dimension_contract(source: str) -> None:
    candidate = _image(
        f"image:{source}:tiny",
        source=source,
        url=f"https://media.example/{source}.png",
        width=45,
        height=45,
    )

    selected = select_rich_item_candidates([candidate], policy=POLICY)

    assert [item["id"] for item in selected] == [f"image:{source}:tiny"]


def test_tiny_resize_url_is_rejected_when_dimensions_are_unknown() -> None:
    candidate = _image(
        "image:tiny-url",
        source="image_search",
        url="https://cdn.example/scale-to-width-down/45/logo.png",
        width=None,
        height=None,
    )

    assert select_rich_item_candidates([candidate], policy=POLICY) == []


def test_unknown_dimensions_rank_after_known_usable_dimensions() -> None:
    unknown = _image(
        "image:unknown",
        source="image_search",
        url="https://media.example/unknown.jpg",
        width=None,
        height=None,
    )
    known = _image(
        "image:known",
        source="image_search",
        url="https://media.example/known.jpg",
        result_rank=1,
    )

    selected = select_rich_item_candidates(
        [unknown, known],
        policy=replace(POLICY, max_items=1),
    )

    assert [item["id"] for item in selected] == ["image:known"]


@pytest.mark.parametrize(
    "invalid_payload_update",
    [
        {"mime_type": "text/html"},
        {"data": "QUJDRA=="},
        {"source_url": "javascript:alert(1)"},
        {"unexpected": "provider-private-field"},
    ],
)
def test_invalid_top_ranked_payload_does_not_displace_valid_runner_up(
    invalid_payload_update: dict[str, Any],
) -> None:
    invalid = _image(
        "image:invalid-top",
        source="image_search",
        url="https://media.example/invalid.jpg",
        result_rank=0,
    )
    invalid["payload"].update(invalid_payload_update)
    valid = _image(
        "image:valid-runner-up",
        source="image_search",
        url="https://media.example/valid.jpg",
        result_rank=1,
    )

    selected = select_rich_item_candidates(
        [invalid, valid],
        policy=replace(POLICY, max_items=1),
    )

    assert [item["id"] for item in selected] == ["image:valid-runner-up"]


def test_malformed_url_does_not_abort_valid_runner_up() -> None:
    malformed = _image(
        "image:malformed-url",
        source="image_search",
        url="https://[malformed",
        result_rank=0,
    )
    valid = _image(
        "image:valid-after-malformed",
        source="image_search",
        url="https://media.example/valid.jpg",
        result_rank=1,
    )

    selected = select_rich_item_candidates(
        [malformed, valid],
        policy=replace(POLICY, max_items=1),
    )

    assert [item["id"] for item in selected] == ["image:valid-after-malformed"]


def test_invalid_group_cell_is_removed_before_it_consumes_the_group() -> None:
    invalid = _cell("https://media.example/invalid.svg")
    invalid["mime_type"] = "image/svg+xml"
    group = _group(
        "imagegroup:brave:0",
        [invalid, _cell("https://media.example/valid.jpg")],
    )

    [selected] = select_rich_item_candidates([group], policy=POLICY)

    assert selected["type"] == "image"
    assert selected["payload"]["url"] == "https://media.example/valid.jpg"


def test_invalid_inline_data_does_not_displace_valid_direct_image() -> None:
    invalid = _image(
        "image:invalid-data",
        source="tool_image",
        url="https://media.example/invalid.jpg",
        result_rank=0,
    )
    invalid["payload"].pop("url")
    invalid["payload"]["data"] = "not-valid-base64!!!"
    invalid["payload"]["mime_type"] = "image/png"
    valid = _image(
        "image:valid-direct",
        source="tool_image",
        url="https://media.example/valid.jpg",
        result_rank=1,
    )

    selected = select_rich_item_candidates(
        [invalid, valid],
        policy=replace(POLICY, max_items=1),
    )

    assert [item["id"] for item in selected] == ["image:valid-direct"]


@pytest.mark.parametrize(
    ("item_id", "description"),
    [
        ("image:flag", "high-resolution national flag"),
        ("image:logo", "high-resolution organization logo"),
        ("image:portrait", "official portrait"),
        ("image:map", "regional map"),
        ("image:diagram", "system architecture diagram"),
    ],
)
def test_content_category_alone_does_not_reject(
    item_id: str,
    description: str,
) -> None:
    candidate = _image(
        item_id,
        source="image_search",
        url=f"https://media.example/{item_id.removeprefix('image:')}.png",
        description=description,
    )

    selected = select_rich_item_candidates([candidate], policy=POLICY)

    assert [item["id"] for item in selected] == [item_id]


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not-a-url",
        "http://media.example/insecure.jpg",
        "https://media.example/tracking-pixel.gif",
        "https://media.example/assets/sprite-v2.png",
        "https://media.example/avatars/default.png",
    ],
)
def test_structurally_invalid_or_junk_web_locator_is_rejected(url: str) -> None:
    candidate = _image(
        "image:invalid",
        source="image_search",
        url=url,
    )

    assert select_rich_item_candidates([candidate], policy=POLICY) == []


def test_real_high_resolution_avatar_is_not_treated_as_placeholder() -> None:
    candidate = _image(
        "image:avatar",
        source="image_search",
        url="https://media.example/users/avatar/12.jpg",
        description="official player portrait",
    )

    selected = select_rich_item_candidates([candidate], policy=POLICY)

    assert [item["id"] for item in selected] == ["image:avatar"]


def test_duplicate_locator_is_emitted_once_in_rank_order() -> None:
    source_bound = _image(
        "image:tavily:0",
        source="web_search",
        url="https://media.example/shared.jpg",
        source_url="https://pages.example/story",
    )
    dedicated = _image(
        "image:brave:0",
        source="image_search",
        url="https://media.example/shared.jpg",
    )

    selected = select_rich_item_candidates(
        [source_bound, dedicated],
        policy=POLICY,
    )

    assert [item["id"] for item in selected] == ["image:brave:0"]


def test_group_with_one_valid_cell_collapses_to_image() -> None:
    group = _group(
        "imagegroup:brave:0",
        [
            _cell("https://media.example/valid.jpg"),
            _cell("https://media.example/tiny.jpg", width=45, height=45),
        ],
    )

    selected = select_rich_item_candidates([group], policy=POLICY)

    assert selected[0]["id"] == "imagegroup:brave:0"
    assert selected[0]["type"] == "image"
    assert selected[0]["payload"]["url"] == "https://media.example/valid.jpg"
    assert selected[0]["provenance"]["provider"] == "brave_image_search"


def test_group_with_two_valid_cells_preserves_cell_order() -> None:
    group = _group(
        "imagegroup:brave:0",
        [
            _cell("https://media.example/first.jpg"),
            _cell("https://media.example/second.jpg"),
            _cell("https://media.example/tiny.jpg", width=1, height=1),
        ],
    )

    [selected] = select_rich_item_candidates([group], policy=POLICY)

    assert selected["type"] == "image_group"
    assert [cell["url"] for cell in selected["payload"]["items"]] == [
        "https://media.example/first.jpg",
        "https://media.example/second.jpg",
    ]


def test_group_with_no_valid_cells_is_removed() -> None:
    group = _group(
        "imagegroup:brave:0",
        [
            _cell("https://media.example/one.gif", width=1, height=1),
            _cell("http://media.example/insecure.jpg"),
        ],
    )

    assert select_rich_item_candidates([group], policy=POLICY) == []


def test_group_deduplicates_cells_claimed_by_a_stronger_candidate() -> None:
    dedicated = _image(
        "image:direct:0",
        source="tool_image",
        url="https://media.example/shared.jpg",
    )
    group = _group(
        "imagegroup:brave:0",
        [
            _cell("https://media.example/shared.jpg"),
            _cell("https://media.example/other.jpg"),
        ],
    )

    selected = select_rich_item_candidates([group, dedicated], policy=POLICY)

    assert [item["id"] for item in selected] == [
        "image:direct:0",
        "imagegroup:brave:0",
    ]
    assert selected[1]["type"] == "image"
    assert selected[1]["payload"]["url"] == "https://media.example/other.jpg"


@pytest.mark.parametrize(
    ("width", "height"),
    [(320, 1600), (1000, 200)],
)
def test_aspect_ratio_policy_boundaries_are_inclusive(
    width: int,
    height: int,
) -> None:
    candidate = _image(
        "image:boundary",
        source="image_search",
        url="https://media.example/boundary.jpg",
        width=width,
        height=height,
    )

    assert select_rich_item_candidates([candidate], policy=POLICY)


def test_repeated_selection_is_deterministic() -> None:
    candidates = [
        _image(
            "image:web:0",
            source="web_search",
            url="https://media.example/web.jpg",
            source_url="https://pages.example/story",
        ),
        _group(
            "imagegroup:brave:0",
            [_cell("https://media.example/search.jpg")],
        ),
    ]

    first = select_rich_item_candidates(candidates, policy=POLICY)
    second = select_rich_item_candidates(candidates, policy=POLICY)

    assert first == second


def test_relevance_overlap_ignores_provider_text_beyond_the_bound() -> None:
    oversized = _image(
        "image:oversized-description",
        source="image_search",
        url="https://media.example/oversized.jpg",
        query="needle",
        result_rank=0,
        description=f"{'x' * 5000} needle",
    )
    bounded_match = _image(
        "image:bounded-match",
        source="image_search",
        url="https://media.example/match.jpg",
        query="needle",
        result_rank=1,
        description="needle",
    )

    selected = select_rich_item_candidates(
        [oversized, bounded_match],
        policy=replace(POLICY, max_items=1),
    )

    assert [item["id"] for item in selected] == ["image:bounded-match"]
