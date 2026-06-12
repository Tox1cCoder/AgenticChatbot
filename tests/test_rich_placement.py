"""Tests for deterministic article-style placement of rich items."""

from app.core.rich_placement import auto_place_rich_items

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
    from app.core.rich_response import parse_inline_rich_references

    content = "Solar panels convert sunlight into electricity using semiconductors."
    items = [("img:0", IMAGE, "solar panels sunlight electricity")]
    new_content, placed = auto_place_rich_items(
        content, items=items, max_images=3, min_score=0.25
    )
    assert parse_inline_rich_references(new_content) == ["img:0"]
