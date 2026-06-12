"""Tests for deterministic article-style placement of rich items."""

from types import SimpleNamespace

from app.core.config import settings
from app.core.rich_placement import auto_place_rich_items, finalize_article_content
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
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=3, min_score=0.25
    )
    assert placed == ["image:tool:c1:0"]
    lines = new_content.split("\n")
    eiffel_line = next(i for i, ln in enumerate(lines) if "Eiffel Tower is stunning" in ln)
    marker_line = next(i for i, ln in enumerate(lines) if ln == "<!--rich:image:tool:c1:0-->")
    louvre_line = next(i for i, ln in enumerate(lines) if "Louvre" in ln)
    assert eiffel_line < marker_line < louvre_line


def test_skips_items_below_min_score():
    content = "A paragraph about quarterly revenue growth and profit margins."
    items = [("image:tool:c1:0", IMAGE, "A cat sleeping on a windowsill")]
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=3, min_score=0.25
    )
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
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=2, min_score=0.25
    )
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
    content = (
        "Solar panels convert sunlight into electricity.\n\n"
        "<!--rich:img:0-->\n"
    )
    items = [("img:0", IMAGE, "solar panels sunlight electricity")]
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=3, min_score=0.25
    )
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
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=3, min_score=0.25
    )
    assert placed == []
    assert new_content == content


def test_places_widget_near_matching_paragraph():
    content = (
        "Here is the revenue comparison between the two quarters.\n\n"
        "Overall the trend is positive."
    )
    items = [("widget:w1", WIDGET, "Quarterly revenue comparison chart")]
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=0, min_score=0.25
    )
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
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=3, min_score=0.25
    )
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
        "payload": {"url": "https://example.com/eiffel.jpg", "mime_type": "image/jpeg",
                    "description": description},
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
            {"widget_id": "w1", "session_id": "s1", "widget_type": "chart",
             "title": "Quarterly revenue comparison", "status": "active", "version": 1}
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
        "The Eiffel Tower glows at night in Paris.\n"
        "```python\n"
        "print('hi')\n"
        "```\n\n"
        "Unrelated note."
    )
    items = [("img:0", IMAGE, "Eiffel Tower glowing at night Paris")]
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=3, min_score=0.25
    )
    assert placed == ["img:0"]
    lines = new_content.split("\n")
    prose_line = next(i for i, ln in enumerate(lines) if "glows at night" in ln)
    marker_line = next(i for i, ln in enumerate(lines) if ln == "<!--rich:img:0-->")
    fence_line = next(i for i, ln in enumerate(lines) if ln.startswith("```python"))
    assert prose_line < marker_line < fence_line
    assert parse_inline_rich_references(new_content) == ["img:0"]


def test_places_image_when_paragraph_directly_follows_fence():
    content = (
        "```python\n"
        "print('hi')\n"
        "```\n"
        "The Eiffel Tower glows at night in Paris."
    )
    items = [("img:0", IMAGE, "Eiffel Tower glowing at night Paris")]
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=3, min_score=0.25
    )
    assert placed == ["img:0"]
    assert parse_inline_rich_references(new_content) == ["img:0"]


def test_skips_items_with_unparseable_ids():
    content = "Here is the quarterly revenue comparison between both units."
    items = [("widget:bad id with spaces & <stuff>", WIDGET, "quarterly revenue comparison")]
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=0, min_score=0.25
    )
    assert placed == []
    assert new_content == content
