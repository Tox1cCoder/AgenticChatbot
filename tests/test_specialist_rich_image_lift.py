"""A turn must not declare an image presentation before any image can exist.

``_presented_rich_image_ids`` is the set of images the model was actually shown
in its rich-item inventory. Placement treats a *present* list as definitive and
drops every candidate outside it, so the key is meaningful only once an
inventory has been built.

A specialist builds its request — and therefore its prompt kwargs — before
running a single tool, so the inventory step runs when the candidate pool is
necessarily empty. Writing an empty list there poisons the rest of the turn:
the web-search tools then discover images, ``_record_specialist_tool_results``
lifts them into the turn context correctly, and placement discards all of them
against the stale empty set. The answer describes pictures that never render.
"""

from __future__ import annotations

import pytest

from app.ai.graph import _build_inline_rich_inventory_for_state
from app.core.rich_placement import _image_anchor_entries

IMAGE_CANDIDATE = {
    "id": "img-eiffel-1",
    "type": "image",
    "source": "image_search",
    "payload": {
        "url": "https://example.com/eiffel.jpg",
        "mime_type": "image/jpeg",
        "width": 1200,
        "height": 800,
        "description": "Eiffel Tower at dusk",
    },
    "provenance": {"provider": "brave", "query": "Eiffel Tower"},
}


def test_inventory_leaves_presentation_unset_when_no_candidates_exist():
    """Before any tool has run there is nothing to present, so nothing is claimed."""
    context: dict = {"inline_rich_response_v1": True}

    assert _build_inline_rich_inventory_for_state(context) == ""
    assert "_presented_rich_image_ids" not in context, (
        "an empty presentation recorded before the tools ran discards every "
        "image the turn goes on to discover"
    )


def test_inventory_clears_a_previous_presentation():
    """A stale value from an earlier pass must not survive into this one."""
    context: dict = {
        "inline_rich_response_v1": True,
        "_presented_rich_image_ids": ["img-from-an-earlier-pass"],
    }

    _build_inline_rich_inventory_for_state(context)

    assert "_presented_rich_image_ids" not in context


@pytest.mark.parametrize(
    ("metadata", "expected_ids"),
    [
        pytest.param(
            {"_rich_item_candidates": [IMAGE_CANDIDATE]},
            ["img-eiffel-1"],
            id="no_presentation_recorded_falls_back_to_the_cap",
        ),
        pytest.param(
            {
                "_rich_item_candidates": [IMAGE_CANDIDATE],
                "_presented_rich_image_ids": ["img-eiffel-1"],
            },
            ["img-eiffel-1"],
            id="presented_image_is_anchorable",
        ),
        pytest.param(
            {"_rich_item_candidates": [IMAGE_CANDIDATE], "_presented_rich_image_ids": []},
            [],
            id="an_empty_presentation_drops_everything",
        ),
    ],
)
def test_placement_reads_the_presentation_as_definitive(metadata, expected_ids):
    """Pins why the empty list is harmful: it is a filter, not a default."""
    entries = _image_anchor_entries(metadata, image_max_items=2)

    assert [entry.item_id for entry in entries] == expected_ids


def test_discovered_image_survives_a_specialist_style_turn():
    """The end-to-end ordering this regression is about.

    The inventory step runs first (no candidates yet), tools then discover an
    image, and the discovered image must still be anchorable afterwards.
    """
    context: dict = {"inline_rich_response_v1": True}

    _build_inline_rich_inventory_for_state(context)
    context["rich_item_candidates"] = [IMAGE_CANDIDATE]

    metadata: dict = {"_rich_item_candidates": list(context["rich_item_candidates"])}
    presented = context.get("_presented_rich_image_ids")
    if isinstance(presented, list):
        metadata["_presented_rich_image_ids"] = list(presented)

    entries = _image_anchor_entries(metadata, image_max_items=2)
    assert [entry.item_id for entry in entries] == ["img-eiffel-1"]
