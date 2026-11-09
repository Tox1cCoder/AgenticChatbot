"""Static answer visuals and labelled source figures as Streamlit HTML.

Pure markup tests for ``app.ui.rich_response``. The rules come from
``docs/frontend/answer-visuals-and-figures-contract.md`` sections 4-5: one
shared zero-based bar scale, visible basis and assumptions, assessment badges
that carry their text label, labelled figure cells with their own sources, one
caption per figure, and a visible note when a comparison lost an image.
"""

from __future__ import annotations

import re
from typing import Any

from app.ui.rich_response import build_answer_visual_html, build_inline_image_group_html


def _visual(payload: dict[str, Any]) -> dict[str, Any]:
    common = {"caption_citations": [], "evidence": [], "fallback_markdown": "**fallback**"}
    return {
        "id": "visual:abc",
        "type": "answer_visual",
        "display_policy": "inline_only",
        "payload": {**common, **payload},
    }


def _bar_widths(html: str) -> list[float]:
    return [
        float(width)
        for width in re.findall(r'data-role="bar-fill"[^>]*?width:([0-9.]+)%', html)
    ]


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------


def _steps() -> dict[str, Any]:
    return _visual(
        {
            "variant": "steps",
            "title": "Make Google the default",
            "steps": [
                {"title": "Open Settings", "citations": []},
                {
                    "title": "Pick Google",
                    "note": "Appears after one search.",
                    "citations": [1],
                },
            ],
            "evidence": [
                {"title": "Help", "url": "https://support.example/edge", "label": "support.example"}
            ],
        }
    )


def test_steps_render_as_an_ordered_list_with_notes():
    html = build_answer_visual_html(_steps())

    assert html is not None
    assert "<ol" in html
    assert html.index("Open Settings") < html.index("Pick Google")
    assert "Appears after one search." in html
    assert "Make Google the default" in html


def test_steps_link_each_citation_to_its_evidence_label():
    html = build_answer_visual_html(_steps())

    assert '<a href="https://support.example/edge"' in html
    assert ">support.example</a>" in html


def test_steps_never_draw_a_done_state():
    html = build_answer_visual_html(_steps())

    for forbidden in ("✓", "✔", "check_circle", "done"):
        assert forbidden not in html


def test_a_citation_outside_the_evidence_list_is_dropped_not_guessed():
    item = _steps()
    item["payload"]["steps"][0]["citations"] = [0, 7]

    html = build_answer_visual_html(item)

    assert html.count("<a ") == 1


def test_a_non_http_evidence_url_is_never_linked():
    item = _steps()
    item["payload"]["evidence"][0]["url"] = "javascript:alert(1)"

    html = build_answer_visual_html(item)

    assert "javascript:" not in html


def test_visual_text_is_escaped_not_rendered_as_html():
    item = _steps()
    item["payload"]["title"] = '<img src=x onerror="alert(1)">'
    item["payload"]["steps"][0]["note"] = "<script>alert(1)</script>"

    html = build_answer_visual_html(item)

    assert "<script>" not in html
    assert "<img" not in html
    assert "&lt;script&gt;" in html


def test_a_blank_line_in_model_text_cannot_end_the_html_block():
    """Streamlit parses Markdown around raw HTML; a blank line would end the block."""
    item = _steps()
    item["payload"]["steps"][0]["note"] = "first\n\nsecond"

    html = build_answer_visual_html(item)

    assert "\n\n" not in html


def test_dollar_signs_cannot_start_streamlit_math():
    item = _steps()
    item["payload"]["steps"][0]["title"] = "Costs $60 to $80"

    html = build_answer_visual_html(item)

    assert "$" not in html
    assert "&#36;60" in html


# ---------------------------------------------------------------------------
# bars
# ---------------------------------------------------------------------------


def _bars(**overrides: Any) -> dict[str, Any]:
    payload = {
        "variant": "bars",
        "title": "Average FPS",
        "unit": "FPS",
        "basis": "calculated",
        "assumptions": "Assumes FPS scales with clock.",
        "scale": {"min": 0, "max": 101.0},
        "bars": [
            {"label": "Stock", "value": 100.0, "citations": []},
            {"label": "Overclocked", "value": 101.0, "citations": []},
        ],
    }
    payload.update(overrides)
    return _visual(payload)


def test_bars_print_label_value_and_unit_on_every_bar():
    html = build_answer_visual_html(_bars())

    assert "Stock" in html
    assert "Overclocked" in html
    assert "100 FPS" in html
    assert "101 FPS" in html


def test_whole_number_floats_print_without_a_decimal_point():
    html = build_answer_visual_html(_bars())

    assert "100.0 FPS" not in html
    assert "0 to 101.0" not in html
    assert "0 to 101 FPS" in html


def test_fractional_values_keep_their_digits():
    html = build_answer_visual_html(
        _bars(
            unit="V",
            scale={"min": 0, "max": 1.32},
            bars=[
                {"label": "Stock", "value": 1.25, "citations": []},
                {"label": "Overclocked", "value": 1.32, "citations": []},
            ],
        )
    )

    assert "1.25 V" in html
    assert "1.32 V" in html


def test_100_and_101_look_almost_the_same_length():
    widths = _bar_widths(build_answer_visual_html(_bars()))

    assert len(widths) == 2
    assert widths[1] == 100.0
    assert 98.5 < widths[0] < 100.0


def test_bars_share_the_payload_scale_rather_than_the_largest_bar():
    html = build_answer_visual_html(
        _bars(
            scale={"min": 0, "max": 200},
            bars=[
                {"label": "A", "value": 50, "citations": []},
                {"label": "B", "value": 100, "citations": []},
            ],
        )
    )

    assert _bar_widths(html) == [25.0, 50.0]


def test_basis_and_assumptions_are_visible():
    html = build_answer_visual_html(_bars())

    assert "Basis: calculated" in html
    assert "Assumes FPS scales with clock." in html


def test_measured_bars_still_show_their_basis():
    html = build_answer_visual_html(_bars(basis="measured", assumptions=None))

    assert "Basis: measured" in html


def test_bars_without_a_positive_scale_fall_back():
    assert build_answer_visual_html(_bars(scale={"min": 0, "max": 0})) is None


# ---------------------------------------------------------------------------
# comparison_table
# ---------------------------------------------------------------------------


def _table() -> dict[str, Any]:
    return _visual(
        {
            "variant": "comparison_table",
            "title": "Windows gaming tweaks",
            "columns": ["Claimed effect"],
            "rows": [
                {
                    "label": "Disable HPET",
                    "cells": [{"text": "Lower input latency", "citations": []}],
                    "assessment": {
                        "status": "caution",
                        "label": "Skip",
                        "reason": "No measurable change.",
                        "citations": [1],
                    },
                },
                {
                    "label": "Plain row",
                    "cells": [{"text": "Something", "citations": []}],
                },
            ],
            "evidence": [{"url": "https://lab.example/tweaks", "label": "lab.example"}],
        }
    )


def test_comparison_table_is_a_semantic_table_with_an_assessment_column():
    html = build_answer_visual_html(_table())

    assert "<table" in html
    assert '<th scope="col">Claimed effect</th>' in html
    assert '<th scope="col">Assessment</th>' in html
    assert '<th scope="row">Disable HPET</th>' in html
    assert "Lower input latency" in html


def test_assessment_shows_its_text_label_and_reason_not_colour_alone():
    html = build_answer_visual_html(_table())

    assert ">Skip</span>" in html
    assert "No measurable change." in html
    assert 'data-status="caution"' in html
    assert ">lab.example</a>" in html


def test_a_row_without_assessment_leaves_that_cell_empty():
    html = build_answer_visual_html(_table())
    plain_row = html.split('<th scope="row">Plain row</th>', 1)[1].split("</tr>", 1)[0]

    assert plain_row.count("<td") == 2
    assert plain_row.endswith("<td></td>")


def test_wide_tables_scroll_inside_their_own_container():
    html = build_answer_visual_html(_table())

    assert "overflow-x:auto" in html


def test_no_assessment_column_when_no_row_has_one():
    item = _table()
    item["payload"]["rows"][0].pop("assessment")

    html = build_answer_visual_html(item)

    assert "Assessment" not in html


# ---------------------------------------------------------------------------
# unknown or malformed
# ---------------------------------------------------------------------------


def test_unknown_variant_returns_none_so_the_caller_renders_the_fallback():
    assert build_answer_visual_html(_visual({"variant": "pie", "title": "x"})) is None


def test_a_non_visual_item_returns_none():
    assert build_answer_visual_html({"type": "image", "payload": {}}) is None


# ---------------------------------------------------------------------------
# labelled source figures
# ---------------------------------------------------------------------------


def _figure_cells() -> list[dict[str, Any]]:
    return [
        {
            "url": "data:image/jpeg;base64,QUJD",
            "mime_type": "image/jpeg",
            "source_url": "https://screens1.example/graphics",
            "label": "Shadows: High",
            "alt_text": "Car with a soft contact shadow",
        },
        {
            "url": "data:image/jpeg;base64,REVG",
            "mime_type": "image/jpeg",
            "source_url": "https://screens2.example/graphics",
            "label": "Shadows: Low",
            "alt_text": "Same car with no shadow",
        },
    ]


def test_figure_cells_show_escaped_labels_and_their_own_sources():
    html = build_inline_image_group_html(_figure_cells(), alt_text="Compare shadows")

    assert html.count('data-role="cell-label"') == 2
    assert "Shadows: High" in html
    assert "Shadows: Low" in html
    assert "https://screens1.example/graphics" in html
    assert "https://screens2.example/graphics" in html


def test_cell_alt_text_is_the_img_alt_and_not_visible_text():
    html = build_inline_image_group_html(_figure_cells(), alt_text="Compare shadows")

    assert 'alt="Car with a soft contact shadow"' in html
    assert html.count("Car with a soft contact shadow") == 1


def test_figure_shows_one_caption_followed_by_its_evidence():
    html = build_inline_image_group_html(
        _figure_cells(),
        alt_text="Compare shadows",
        caption="Low shadows make the car float.",
        caption_evidence=[{"url": "https://guide.example/shadows", "label": "guide.example"}],
    )

    assert html.count("Low shadows make the car float.") == 1
    caption_at = html.index("Low shadows make the car float.")
    assert html.index(">guide.example</a>") > caption_at


def test_figure_cells_keep_the_lightbox_and_per_cell_failure():
    html = build_inline_image_group_html(_figure_cells(), alt_text="x")

    assert html.count('class="img-thumb"') == 2
    assert 'data-role="cell-fallback"' in html
    assert "onerror" in html


def test_figure_cells_stack_on_narrow_screens():
    html = build_inline_image_group_html(_figure_cells(), alt_text="x")

    assert "flex-wrap:wrap" in html
    assert "min-width:min(" in html


def test_hostile_figure_labels_and_captions_are_escaped():
    cells = _figure_cells()
    cells[0]["label"] = "<script>alert(1)</script>"
    html = build_inline_image_group_html(
        cells, alt_text="x", caption='<img src=x onerror="alert(2)">'
    )

    assert "<script>" not in html
    assert "&lt;img src=x" in html
    assert 'onerror="alert(2)"' not in html


def test_comparison_unavailable_keeps_the_single_survivor_labelled_with_a_note():
    html = build_inline_image_group_html(
        _figure_cells()[:1],
        alt_text="Comparison unavailable: Shadows: High",
        caption="A stale conclusion",
        comparison_unavailable=True,
    )

    assert html.count("<img") == 1
    assert "Shadows: High" in html
    assert "https://screens1.example/graphics" in html
    assert "Comparison unavailable" in html
    assert "A stale conclusion" not in html


def test_a_plain_single_cell_group_still_renders_as_one_image():
    """Legacy groups (no figure fields) keep their existing single-image shape."""
    cells = [{"url": "/web-images/1", "mime_type": "image/jpeg"}]

    html = build_inline_image_group_html(cells, alt_text="x")

    assert html.count("<img") == 1
    assert 'data-role="cell-label"' not in html
