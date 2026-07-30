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

WIDGET = "live_widget"

# ---------------------------------------------------------------------------
# ``auto_place_rich_items`` places widget-class items only: image items are
# anchored on the model's own image query by ``anchor_image_items_by_query``,
# which owns the per-answer image cap. These tests therefore exercise the
# shared placement mechanics — best-block match, one item per paragraph, fence
# safety, id validation — with widget-class items.
# ---------------------------------------------------------------------------


def test_places_item_after_its_best_matching_paragraph():
    content = (
        "# Paris travel guide\n\n"
        "The Eiffel Tower is stunning at night, lit by thousands of lamps.\n\n"
        "The Louvre houses the Mona Lisa and countless other works."
    )
    items = [("widget:w1", WIDGET, "Eiffel Tower illuminated at night in Paris")]
    new_content, placed = auto_place_rich_items(content, items=items, min_score=0.25)
    assert placed == ["widget:w1"]
    lines = new_content.split("\n")
    eiffel_line = next(i for i, ln in enumerate(lines) if "Eiffel Tower is stunning" in ln)
    marker_line = next(i for i, ln in enumerate(lines) if ln == "<!--rich:widget:w1-->")
    louvre_line = next(i for i, ln in enumerate(lines) if "Louvre" in ln)
    assert eiffel_line < marker_line < louvre_line


def test_skips_items_below_min_score():
    content = "A paragraph about quarterly revenue growth and profit margins."
    items = [("widget:w1", WIDGET, "A cat sleeping on a windowsill")]
    new_content, placed = auto_place_rich_items(content, items=items, min_score=0.25)
    assert placed == []
    assert new_content == content


def test_places_every_matching_item_in_its_own_paragraph():
    """The narrowed function carries no per-type cap: widgets place until the
    matching paragraphs run out. The per-answer image cap lives in the
    anchoring path, not here."""
    content = (
        "Solar panels convert sunlight into electricity.\n\n"
        "Wind turbines harvest kinetic energy from moving air.\n\n"
        "Hydroelectric dams use falling water to spin turbines."
    )
    items = [
        ("widget:w0", WIDGET, "solar panels sunlight electricity"),
        ("widget:w1", WIDGET, "wind turbines kinetic energy air"),
        ("widget:w2", WIDGET, "hydroelectric dams falling water turbines"),
    ]
    new_content, placed = auto_place_rich_items(content, items=items, min_score=0.25)
    assert placed == ["widget:w0", "widget:w1", "widget:w2"]
    assert new_content.count("<!--rich:") == 3


def test_one_item_per_paragraph():
    content = "Solar panels convert sunlight into electricity using semiconductors."
    items = [
        ("widget:w0", WIDGET, "solar panels sunlight electricity"),
        ("widget:w1", WIDGET, "solar panels converting sunlight semiconductors"),
    ]
    _, placed = auto_place_rich_items(content, items=items, min_score=0.25)
    assert placed == ["widget:w0"]


def test_skips_already_referenced_items():
    content = "Solar panels convert sunlight into electricity.\n\n<!--rich:widget:w0-->\n"
    items = [("widget:w0", WIDGET, "solar panels sunlight electricity")]
    new_content, placed = auto_place_rich_items(content, items=items, min_score=0.25)
    assert placed == []
    assert new_content == content


def test_never_places_inside_code_blocks():
    content = (
        "```python\n"
        "# solar panels sunlight electricity semiconductors\n"
        "print('solar panels sunlight electricity')\n"
        "```"
    )
    items = [("widget:w0", WIDGET, "solar panels sunlight electricity")]
    new_content, placed = auto_place_rich_items(content, items=items, min_score=0.25)
    assert placed == []
    assert new_content == content


def test_places_widget_near_matching_paragraph():
    content = (
        "Here is the revenue comparison between the two quarters.\n\nOverall the trend is positive."
    )
    items = [("widget:w1", WIDGET, "Quarterly revenue comparison chart")]
    new_content, placed = auto_place_rich_items(content, items=items, min_score=0.25)
    assert placed == ["widget:w1"]
    assert "<!--rich:widget:w1-->" in new_content


def test_empty_inputs_are_safe():
    result_empty = auto_place_rich_items("", items=[("a", WIDGET, "x")], min_score=0.2)
    assert result_empty == ("", [])
    assert auto_place_rich_items("text", items=[], min_score=0.2) == ("text", [])


def test_inserted_marker_is_parseable():
    content = "Solar panels convert sunlight into electricity using semiconductors."
    items = [("widget:w0", WIDGET, "solar panels sunlight electricity")]
    new_content, placed = auto_place_rich_items(content, items=items, min_score=0.25)
    assert parse_inline_rich_references(new_content) == ["widget:w0"]


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
    """A tool-produced image candidate (e.g. a Brave/MCP result), which is the
    real-world shape: it carries a ``source`` and no image-search query, so it
    reaches placement only through the ``tool_image`` fallback-anchor origin."""
    return {
        "id": item_id,
        "type": "image",
        "source": "tool_image",
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
    # Long enough (>= _FALLBACK_MIN_BLOCK_TOKENS non-stopword tokens) for the
    # tool_image fallback anchor to accept the block: this candidate has no
    # query to score against a paragraph, so fallback is its only path.
    content = "The Eiffel Tower is stunning at night, lit by thousands of golden lamps."
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


def test_places_item_when_paragraph_directly_precedes_fence():
    content = (
        "The Eiffel Tower glows at night in Paris.\n```python\nprint('hi')\n```\n\nUnrelated note."
    )
    items = [("widget:w0", WIDGET, "Eiffel Tower glowing at night Paris")]
    new_content, placed = auto_place_rich_items(content, items=items, min_score=0.25)
    assert placed == ["widget:w0"]
    lines = new_content.split("\n")
    prose_line = next(i for i, ln in enumerate(lines) if "glows at night" in ln)
    marker_line = next(i for i, ln in enumerate(lines) if ln == "<!--rich:widget:w0-->")
    fence_line = next(i for i, ln in enumerate(lines) if ln.startswith("```python"))
    assert prose_line < marker_line < fence_line
    assert parse_inline_rich_references(new_content) == ["widget:w0"]


def test_places_item_when_paragraph_directly_follows_fence():
    content = "```python\nprint('hi')\n```\nThe Eiffel Tower glows at night in Paris."
    items = [("widget:w0", WIDGET, "Eiffel Tower glowing at night Paris")]
    new_content, placed = auto_place_rich_items(content, items=items, min_score=0.25)
    assert placed == ["widget:w0"]
    assert parse_inline_rich_references(new_content) == ["widget:w0"]


def test_skips_items_with_unparseable_ids():
    content = "Here is the quarterly revenue comparison between both units."
    items = [("widget:bad id with spaces & <stuff>", WIDGET, "quarterly revenue comparison")]
    new_content, placed = auto_place_rich_items(content, items=items, min_score=0.25)
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


def test_image_anchor_entries_tool_image_without_signal_is_not_anchorable():
    """A tool_image candidate with no provenance query and only the generic
    alt-text placeholder carries no genuine signal at all. It must not be
    anchorable — an auto-placed junk image (e.g. a crawler/SEO thumbnail) is
    exactly the failure this placement system exists to prevent, and the
    reasoning for tool_image's fallback eligibility ("the tool call implies
    display") does not license placing something with nothing to show."""
    from app.core.rich_placement import _image_anchor_entries
    from app.core.rich_response import GENERIC_IMAGE_ALT_TEXT

    metadata = {
        "_rich_item_candidates": [
            {
                "id": "image:tool:c1:4",
                "type": "image",
                "source": "tool_image",
                "alt_text": GENERIC_IMAGE_ALT_TEXT,
                "payload": {"url": "https://lookaside.instagram.com/seo/crawler"},
            }
        ]
    }
    [entry] = _image_anchor_entries(metadata)
    assert entry.origin == "tool_image"
    assert entry.anchorable is False


def test_image_anchor_entries_tool_image_with_description_is_anchorable():
    """A tool_image candidate with no query but a genuine description (title,
    real alt_text, or payload description) is anchorable — it can be placed
    via the fallback anchor even though it has no query to score against a
    paragraph."""
    from app.core.rich_placement import _image_anchor_entries

    metadata = {
        "_rich_item_candidates": [
            {
                "id": "image:tool:c1:0",
                "type": "image",
                "source": "tool_image",
                "alt_text": "Eiffel Tower at night",
                "payload": {"description": "Eiffel Tower at night in Paris"},
            }
        ]
    }
    [entry] = _image_anchor_entries(metadata)
    assert entry.origin == "tool_image"
    assert entry.anchorable is True


def test_generic_alt_text_image_is_not_auto_placed(monkeypatch):
    """Regression: a description-less junk image matched a paragraph via the
    generic fallback's 'tool'/'result' tokens and was auto-placed, rendering a
    broken placeholder. Such images carry no signal and must never be placed.

    Carries ``source: "tool_image"`` — the real production shape (a
    description-less tool image, e.g. a crawler/SEO thumbnail Brave returned,
    is stamped with this source by ``build_image_candidates_from_tool_result``
    per ``app/ai/tool_execution.py``) rather than the source-less shape,
    which does not occur in production. The content is one clause longer
    than the original regression case so the block clears
    ``_FALLBACK_MIN_BLOCK_TOKENS`` and the test genuinely exercises the
    descriptive-signal guard in ``_image_anchor_entries`` rather than passing
    only because the block was too short for any fallback anchor."""
    from app.core.rich_response import GENERIC_IMAGE_ALT_TEXT

    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    content = (
        "Here are the latest match results and tool output for the entire "
        "football league season."
    )
    candidate = {
        "id": "image:tool:c1:4",
        "type": "image",
        "source": "tool_image",
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


def test_tool_image_falls_back_to_first_prose_block_with_no_query():
    """A tool-produced image (chart, rendered diagram) carries no image query
    by construction, so its score against every block is always zero. Origin
    ``tool_image`` must still reach the fallback anchor, the same as a
    deliberate image search — the tool call itself implies display."""
    content, outcomes = anchor_image_items_by_query(
        BODY,
        entries=[
            ImageAnchorEntry(item_id="image:tool:c1:0", query="", origin="tool_image")
        ],
        min_score=0.34,
        max_images=2,
    )
    assert "<!--rich:image:tool:c1:0-->" in content
    assert outcomes["image:tool:c1:0"] == "fallback_anchored"


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


# ---------------------------------------------------------------------------
# Wiring: finalize_article_content routes image items to
# anchor_image_items_by_query — the single image-placement path — while widgets
# keep flowing through auto_place_rich_items untouched.
# ---------------------------------------------------------------------------


def test_finalize_uses_query_anchoring(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    candidate = {
        "id": "imagegroup:tool:c1",
        "type": "image_group",
        "source": "image_search",
        "payload": {"items": []},
        "provenance": {
            "query": "Apple Park Cupertino headquarters",
            "provider": "brave_image_search",
        },
    }
    response = _make_response(BODY, candidates=[candidate])
    content = finalize_article_content(response, BODY)
    assert "<!--rich:imagegroup:tool:c1-->" in content
    assert response.message.content == content


def test_widgets_still_auto_place_alongside_image_anchoring(monkeypatch):
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
                "title": "Apple Park cost breakdown",
                "status": "active",
                "version": 1,
            }
        ),
    }
    content = "Apple Park in Cupertino cost about five billion dollars to build."
    response = _make_response(content, artifacts=[artifact])
    assert "<!--rich:widget:w1-->" in finalize_article_content(response, content)


# ---------------------------------------------------------------------------
# Fallback-anchor origin contract at the finalize level: a tool-produced image
# (``tool_image``) and a deliberate image search anchor with a fallback; a
# source-bound web-search image (``web_search``/Tavily) with no query and no
# model marker must stay unplaced. This is the stricter half of the contract
# that ``test_finalize_places_image_and_mutates_response_message`` (tool_image,
# placed) is the positive counterpart to.
# ---------------------------------------------------------------------------


def test_finalize_leaves_source_bound_web_search_image_unplaced_without_query(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    candidate = {
        "id": "image:tool:c1:0",
        "type": "image",
        "source": "web_search",
        "display_policy": "inline_only",
        "alt_text": "Eiffel Tower at night",
        "payload": {
            "url": "https://example.com/eiffel.jpg",
            "mime_type": "image/jpeg",
        },
    }
    content = "The Eiffel Tower is stunning at night, lit by thousands of golden lamps."
    response = _make_response(content, candidates=[candidate])
    assert finalize_article_content(response, content) == content


def test_single_brave_image_result_is_placed_via_fallback(monkeypatch):
    """Pins the decision-2 fix: a single eligible Brave candidate (not grouped,
    since grouping needs two) is built with ``source: "tool_image"`` by
    ``build_image_candidates_from_tool_result``. Before ``tool_image`` joined
    _FALLBACK_ANCHOR_ORIGINS, such a candidate had no fallback path and was
    silently dropped whenever its search query did not textually match a
    paragraph, breaking the project's own acceptance criterion that a
    deliberate ``brave_image_search`` with at least one eligible candidate
    always results in a displayed image."""
    import json

    from app.ai.tool_execution import build_image_candidates_from_tool_result

    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_auto_place_enabled", True)
    payload = json.dumps(
        {
            "provider": "brave_image_search",
            "query": "red panda photo",
            "images": [
                {
                    "url": "https://e.com/0.jpg",
                    "thumbnail_url": "https://cdn.brave.com/0.jpg",
                    "mime_type": "image/jpeg",
                    "width": 1200,
                    "height": 800,
                    "source_url": "https://e.com/page-0",
                    "description": "red panda 0",
                }
            ],
        }
    )
    [candidate] = build_image_candidates_from_tool_result(
        payload, tool_call_id="c1", tool_name="brave_image_search"
    )
    assert candidate["type"] == "image"
    assert candidate["source"] == "tool_image"

    # Deliberately shares no tokens with "red panda photo" so the query-match
    # path scores zero and only the fallback anchor can place the image.
    content = (
        "Quarterly revenue grew across every product line this period, driven "
        "by strong subscription renewals and enterprise contract expansion."
    )
    response = _make_response(content, candidates=[candidate])
    new_content = finalize_article_content(response, content)
    assert f"<!--rich:{candidate['id']}-->" in new_content


# ---------------------------------------------------------------------------
# Cleanup guards: exactly one image-placement path survives, with no rollback
# scaffolding. A rollback flag that outlives its window is a permanent second
# code path, and a setting that survives cleanup unread is itself dead weight.
# ---------------------------------------------------------------------------


def test_legacy_image_placement_helper_is_gone():
    import app.core.rich_placement as rp

    assert not hasattr(rp, "_image_placement_entries")


def test_auto_place_no_longer_takes_max_images():
    import inspect

    assert "max_images" not in inspect.signature(auto_place_rich_items).parameters


def test_rollback_flag_is_removed():
    from app.core.config import Settings

    assert not hasattr(Settings(), "rich_query_anchored_images_enabled")


def test_retained_settings_still_have_live_readers():
    from app.core.config import Settings

    settings_obj = Settings()
    assert settings_obj.rich_auto_place_max_images >= 1
    assert settings_obj.rich_auto_place_min_score >= 0


def test_descriptive_signal_helper_survives_the_legacy_path_removal():
    """``_descriptive_signal_text`` is shared: the deleted legacy path used it,
    but so does the live ``tool_image`` anchor origin. Removing it with the
    legacy path would let signal-less junk tool images be auto-placed again."""
    from app.core.rich_placement import _descriptive_signal_text

    assert _descriptive_signal_text({"title": "Eiffel Tower"}) == "Eiffel Tower"
