"""Tests for deterministic article-style placement of rich items."""

from types import SimpleNamespace

from app.core.config import settings
from app.core.rich_placement import (
    ImageAnchorEntry,
    _repair_unprefixed_markers,
    anchor_image_items_by_query,
    auto_place_rich_items,
    finalize_article_content,
)
from app.core.rich_response import parse_inline_rich_references

IMAGE = "image"
WIDGET = "live_widget"


def test_places_image_after_matching_paragraph():
    content = (
        "# Paris travel guide\n\n"
        "The Eiffel Tower is stunning at night, lit by thousands of lamps.\n\n"
        "The Louvre houses the Mona Lisa and countless other works."
    )
    items = [("image:tool:c1:0", IMAGE, "Eiffel Tower illuminated at night in Paris")]
    new_content, placed = auto_place_rich_items(content, items=items, max_images=3, min_score=0.25)
    assert placed == ["image:tool:c1:0"]
    lines = new_content.split("\n")
    eiffel_line = next(i for i, ln in enumerate(lines) if "Eiffel Tower is stunning" in ln)
    marker_line = next(i for i, ln in enumerate(lines) if ln == "<!--rich:image:tool:c1:0-->")
    louvre_line = next(i for i, ln in enumerate(lines) if "Louvre" in ln)
    assert eiffel_line < marker_line < louvre_line


def test_skips_items_below_min_score():
    content = "A paragraph about quarterly revenue growth and profit margins."
    items = [("image:tool:c1:0", IMAGE, "A cat sleeping on a windowsill")]
    new_content, placed = auto_place_rich_items(content, items=items, max_images=3, min_score=0.25)
    assert placed == []
    assert new_content == content


def test_respects_max_images_cap():
    content = (
        "Solar panels convert sunlight into electricity.\n\n"
        "Wind turbines harvest kinetic energy from moving air.\n\n"
        "Hydroelectric dams use falling water to spin turbines."
    )
    items = [
        ("img:0", IMAGE, "solar panels sunlight electricity"),
        ("img:1", IMAGE, "wind turbines kinetic energy air"),
        ("img:2", IMAGE, "hydroelectric dams falling water turbines"),
    ]
    new_content, placed = auto_place_rich_items(content, items=items, max_images=2, min_score=0.25)
    assert len(placed) == 2
    assert new_content.count("<!--rich:") == 2


def test_one_item_per_paragraph():
    content = "Solar panels convert sunlight into electricity using semiconductors."
    items = [
        ("img:0", IMAGE, "solar panels sunlight electricity"),
        ("img:1", IMAGE, "solar panels converting sunlight semiconductors"),
    ]
    _, placed = auto_place_rich_items(content, items=items, max_images=3, min_score=0.25)
    assert placed == ["img:0"]


def test_skips_already_referenced_items():
    content = "Solar panels convert sunlight into electricity.\n\n<!--rich:img:0-->\n"
    items = [("img:0", IMAGE, "solar panels sunlight electricity")]
    new_content, placed = auto_place_rich_items(content, items=items, max_images=3, min_score=0.25)
    assert placed == []
    assert new_content == content


def test_never_places_inside_code_blocks():
    content = (
        "```python\n"
        "# solar panels sunlight electricity semiconductors\n"
        "print('solar panels sunlight electricity')\n"
        "```"
    )
    items = [("img:0", IMAGE, "solar panels sunlight electricity")]
    new_content, placed = auto_place_rich_items(content, items=items, max_images=3, min_score=0.25)
    assert placed == []
    assert new_content == content


def test_places_widget_near_matching_paragraph():
    content = (
        "Here is the revenue comparison between the two quarters.\n\nOverall the trend is positive."
    )
    items = [("widget:w1", WIDGET, "Quarterly revenue comparison chart")]
    new_content, placed = auto_place_rich_items(content, items=items, max_images=0, min_score=0.25)
    assert placed == ["widget:w1"]
    assert "<!--rich:widget:w1-->" in new_content


def test_widget_cap_independent_of_image_cap():
    content = "Quarterly revenue comparison for the two business units."
    items = [("widget:w1", WIDGET, "quarterly revenue comparison chart")]
    _, placed = auto_place_rich_items(content, items=items, max_images=0, min_score=0.25)
    assert placed == ["widget:w1"]


def test_empty_inputs_are_safe():
    result_empty = auto_place_rich_items("", items=[("a", IMAGE, "x")], max_images=3, min_score=0.2)
    assert result_empty == ("", [])
    assert auto_place_rich_items("text", items=[], max_images=3, min_score=0.2) == ("text", [])


def test_inserted_marker_is_parseable():
    content = "Solar panels convert sunlight into electricity using semiconductors."
    items = [("img:0", IMAGE, "solar panels sunlight electricity")]
    new_content, placed = auto_place_rich_items(content, items=items, max_images=3, min_score=0.25)
    assert parse_inline_rich_references(new_content) == ["img:0"]


def _make_response(content, *, candidates=None, artifacts=None, capable=True):
    metadata = {"_inline_rich_response_v1": capable}
    if candidates is not None:
        metadata["_rich_item_candidates"] = candidates
    if artifacts is not None:
        metadata["tool_artifacts"] = artifacts
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        metadata=metadata,
        tool_artifacts=None,
    )


def _image_candidate(item_id="image:tool:c1:0", description="Eiffel Tower at night in Paris"):
    return {
        "id": item_id,
        "type": "image",
        "display_policy": "inline_only",
        "alt_text": description,
        "payload": {
            "url": "https://example.com/eiffel.jpg",
            "mime_type": "image/jpeg",
            "description": description,
        },
    }


def test_finalize_places_image_and_mutates_response_message(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    content = "The Eiffel Tower is stunning at night, lit by thousands of lamps."
    response = _make_response(content, candidates=[_image_candidate()])
    new_content = finalize_article_content(response, content)
    assert "<!--rich:image:tool:c1:0-->" in new_content
    assert response.message.content == new_content


def test_finalize_noop_when_feature_disabled(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", False)
    content = "The Eiffel Tower is stunning at night."
    response = _make_response(content, candidates=[_image_candidate()])
    assert finalize_article_content(response, content) == content


def test_finalize_noop_when_auto_place_disabled(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", False)
    content = "The Eiffel Tower is stunning at night."
    response = _make_response(content, candidates=[_image_candidate()])
    assert finalize_article_content(response, content) == content


def test_finalize_noop_without_capability(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    content = "The Eiffel Tower is stunning at night."
    response = _make_response(content, candidates=[_image_candidate()], capable=False)
    assert finalize_article_content(response, content) == content


def test_finalize_places_widget_from_artifacts(monkeypatch):
    import json

    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    artifact = {
        "tool": "widget_create",
        "status": "success",
        "output": json.dumps(
            {
                "widget_id": "w1",
                "session_id": "s1",
                "widget_type": "chart",
                "title": "Quarterly revenue comparison",
                "status": "active",
                "version": 1,
            }
        ),
    }
    content = "Here is the quarterly revenue comparison between both units."
    response = _make_response(content, artifacts=[artifact])
    new_content = finalize_article_content(response, content)
    assert "<!--rich:widget:w1-->" in new_content


def test_finalize_handles_none_response():
    assert finalize_article_content(None, "text") == "text"


def test_places_image_when_paragraph_directly_precedes_fence():
    content = (
        "The Eiffel Tower glows at night in Paris.\n```python\nprint('hi')\n```\n\nUnrelated note."
    )
    items = [("img:0", IMAGE, "Eiffel Tower glowing at night Paris")]
    new_content, placed = auto_place_rich_items(content, items=items, max_images=3, min_score=0.25)
    assert placed == ["img:0"]
    lines = new_content.split("\n")
    prose_line = next(i for i, ln in enumerate(lines) if "glows at night" in ln)
    marker_line = next(i for i, ln in enumerate(lines) if ln == "<!--rich:img:0-->")
    fence_line = next(i for i, ln in enumerate(lines) if ln.startswith("```python"))
    assert prose_line < marker_line < fence_line
    assert parse_inline_rich_references(new_content) == ["img:0"]


def test_places_image_when_paragraph_directly_follows_fence():
    content = "```python\nprint('hi')\n```\nThe Eiffel Tower glows at night in Paris."
    items = [("img:0", IMAGE, "Eiffel Tower glowing at night Paris")]
    new_content, placed = auto_place_rich_items(content, items=items, max_images=3, min_score=0.25)
    assert placed == ["img:0"]
    assert parse_inline_rich_references(new_content) == ["img:0"]


def test_skips_items_with_unparseable_ids():
    content = "Here is the quarterly revenue comparison between both units."
    items = [("widget:bad id with spaces & <stuff>", WIDGET, "quarterly revenue comparison")]
    new_content, placed = auto_place_rich_items(content, items=items, max_images=0, min_score=0.25)
    assert placed == []
    assert new_content == content


# ---------------------------------------------------------------------------
# Repair of model-authored markers that drop the required ``rich:`` prefix.
# A model that writes ``<!--widget:<id>-->`` (id already namespaced) instead of
# ``<!--rich:widget:<id>-->`` makes the marker invisible to every consumer: it
# leaks as literal text and the item falls to the append-after-body fallback.
# ---------------------------------------------------------------------------


def test_repairs_bare_widget_marker_for_known_id():
    content = "Intro paragraph.\n\n<!--widget:w1-->\n\nMore text."
    out = _repair_unprefixed_markers(content, {"widget:w1"})
    assert "<!--rich:widget:w1-->" in out
    assert "\n<!--widget:w1-->\n" not in out


def test_repairs_bare_image_marker_for_known_id():
    content = "See below.\n\n<!--image:tool:c1:0-->"
    out = _repair_unprefixed_markers(content, {"image:tool:c1:0"})
    assert out.rstrip().endswith("<!--rich:image:tool:c1:0-->")


def test_repair_leaves_unknown_comments_untouched():
    content = "Intro.\n\n<!--widget:w1-->\n\n<!--TODO revisit this later-->"
    out = _repair_unprefixed_markers(content, {"widget:w2"})
    assert out == content


def test_repair_ignores_already_correct_markers():
    content = "Intro.\n\n<!--rich:widget:w1-->\n\nEnd."
    out = _repair_unprefixed_markers(content, {"widget:w1"})
    assert out == content


def test_repair_skips_markers_inside_code_fence():
    content = "```\n<!--widget:w1-->\n```"
    out = _repair_unprefixed_markers(content, {"widget:w1"})
    assert out == content


def test_repair_noop_without_known_ids():
    content = "Intro.\n\n<!--widget:w1-->"
    assert _repair_unprefixed_markers(content, set()) == content


def test_repaired_marker_becomes_referenced():
    content = "Here is the breakdown.\n\n<!--widget:w1-->\n\nThat is the trend."
    out = _repair_unprefixed_markers(content, {"widget:w1"})
    assert parse_inline_rich_references(out) == ["widget:w1"]


def test_image_placement_entries_skips_signalless_candidates():
    from app.core.rich_placement import _image_placement_entries
    from app.core.rich_response import GENERIC_IMAGE_ALT_TEXT

    metadata = {
        "_rich_item_candidates": [
            {  # only the generic fallback → no real signal
                "id": "image:tool:c1:4",
                "type": "image",
                "alt_text": GENERIC_IMAGE_ALT_TEXT,
                "payload": {"url": "https://lookaside.instagram.com/seo/crawler"},
            },
            {  # genuine description → eligible
                "id": "image:tool:c1:0",
                "type": "image",
                "alt_text": "Eiffel Tower at night",
                "payload": {"description": "Eiffel Tower at night in Paris"},
            },
        ]
    }
    ids = [entry[0] for entry in _image_placement_entries(metadata)]
    assert ids == ["image:tool:c1:0"]


def test_generic_alt_text_image_is_not_auto_placed(monkeypatch):
    """Regression: a description-less junk image matched a paragraph via the
    generic fallback's 'tool'/'result' tokens and was auto-placed, rendering a
    broken placeholder. Such images carry no signal and must never be placed."""
    from app.core.rich_response import GENERIC_IMAGE_ALT_TEXT

    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    content = "Here are the latest match results and tool output for the league."
    candidate = {
        "id": "image:tool:c1:4",
        "type": "image",
        "display_policy": "inline_only",
        "alt_text": GENERIC_IMAGE_ALT_TEXT,
        "payload": {"url": "https://lookaside.instagram.com/seo/crawler", "mime_type": "image/png"},
    }
    response = _make_response(content, candidates=[candidate])
    assert finalize_article_content(response, content) == content


def test_finalize_repairs_model_authored_bare_widget_marker(monkeypatch):
    import json

    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    artifact = {
        "tool": "widget_create",
        "status": "success",
        "output": json.dumps(
            {
                "widget_id": "w1",
                "session_id": "s1",
                "widget_type": "chart",
                "title": "Quarterly revenue",
                "status": "active",
                "version": 1,
            }
        ),
    }
    # The model placed the marker itself but dropped the ``rich:`` prefix.
    content = "Here is the breakdown.\n\n<!--widget:w1-->\n\nThat is the trend."
    response = _make_response(content, artifacts=[artifact])
    new_content = finalize_article_content(response, content)
    assert "<!--rich:widget:w1-->" in new_content
    assert "\n<!--widget:w1-->\n" not in new_content
    assert response.message.content == new_content
    assert parse_inline_rich_references(new_content) == ["widget:w1"]


# ---------------------------------------------------------------------------
# Query-anchored placement: anchors an unreferenced image item on the model's
# own image-search query instead of the provider description. A model-authored
# marker still wins; anchoring only fills in when the model wrote none.
# ---------------------------------------------------------------------------

BODY = (
    "Apple Inc reported record services revenue this quarter, driven by "
    "subscriptions, advertising, and payments across its installed base of "
    "active devices.\n"
    "\n"
    "Apple Park in Cupertino remains the company headquarters and cost "
    "about five billion dollars to build.\n"
    "\n"
    "The fruit industry is unrelated to this discussion entirely.\n"
)


def test_anchors_after_the_block_matching_the_image_query():
    content, outcomes = anchor_image_items_by_query(
        BODY,
        entries=[
            ImageAnchorEntry(
                item_id="imagegroup:tool:c1",
                query="Apple Park Cupertino headquarters",
                origin="image_search",
            )
        ],
        min_score=0.34,
        max_images=2,
    )
    lines = content.split("\n")
    marker_index = lines.index("<!--rich:imagegroup:tool:c1-->")
    assert "Cupertino" in lines[marker_index - 2]
    assert outcomes["imagegroup:tool:c1"] == "query_anchored"


def test_image_search_falls_back_to_first_prose_block_when_nothing_matches():
    content, outcomes = anchor_image_items_by_query(
        BODY,
        entries=[
            ImageAnchorEntry(
                item_id="imagegroup:tool:c1",
                query="quantum chromodynamics lattice diagram",
                origin="image_search",
            )
        ],
        min_score=0.34,
        max_images=2,
    )
    assert "<!--rich:imagegroup:tool:c1-->" in content
    assert outcomes["imagegroup:tool:c1"] == "fallback_anchored"


def test_source_bound_tavily_image_has_no_fallback():
    content, outcomes = anchor_image_items_by_query(
        BODY,
        entries=[
            ImageAnchorEntry(
                item_id="image:tool:c1:0",
                query="quantum chromodynamics lattice diagram",
                origin="web_search_source_bound",
            )
        ],
        min_score=0.34,
        max_images=2,
    )
    assert content == BODY
    assert outcomes["image:tool:c1:0"] == "unplaced"


def test_query_level_image_is_never_anchored():
    content, outcomes = anchor_image_items_by_query(
        BODY,
        entries=[
            ImageAnchorEntry(
                item_id="image:tool:c1:0",
                query="Apple Park Cupertino headquarters",
                origin="web_search_query_level",
                anchorable=False,
            )
        ],
        min_score=0.34,
        max_images=2,
    )
    assert content == BODY
    assert outcomes["image:tool:c1:0"] == "unplaced"


def test_existing_marker_wins_and_is_never_duplicated():
    body = BODY + "\n<!--rich:imagegroup:tool:c1-->\n"
    content, outcomes = anchor_image_items_by_query(
        body,
        entries=[
            ImageAnchorEntry(
                item_id="imagegroup:tool:c1",
                query="Apple Park Cupertino headquarters",
                origin="image_search",
            )
        ],
        min_score=0.34,
        max_images=2,
    )
    assert content == body
    assert content.count("<!--rich:imagegroup:tool:c1-->") == 1
    assert outcomes["imagegroup:tool:c1"] == "marker"


def test_max_images_cap_is_respected():
    entries = [
        ImageAnchorEntry(
            item_id=f"imagegroup:tool:c{n}",
            query="Apple Park Cupertino",
            origin="image_search",
        )
        for n in range(3)
    ]
    content, outcomes = anchor_image_items_by_query(
        BODY, entries=entries, min_score=0.34, max_images=2
    )
    assert sum(1 for v in outcomes.values() if v != "unplaced") == 2


def test_never_anchors_inside_a_fenced_code_block():
    body = "```\nApple Park Cupertino headquarters\n```\n"
    content, outcomes = anchor_image_items_by_query(
        body,
        entries=[
            ImageAnchorEntry(
                item_id="imagegroup:tool:c1",
                query="Apple Park Cupertino headquarters",
                origin="image_search",
            )
        ],
        min_score=0.34,
        max_images=2,
    )
    assert content == body
    assert outcomes["imagegroup:tool:c1"] == "unplaced"


def test_invalid_item_id_is_never_inserted():
    content, outcomes = anchor_image_items_by_query(
        BODY,
        entries=[
            ImageAnchorEntry(item_id="bad id!", query="Apple Park Cupertino", origin="image_search")
        ],
        min_score=0.34,
        max_images=2,
    )
    assert content == BODY
    assert outcomes["bad id!"] == "unplaced"
