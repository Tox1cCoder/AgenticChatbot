"""Tests for the HTML-only widget contract helper.

The contract is intentionally minimal: state must be a JSON object, html
content must be non-empty, and the iframe height must be present, numeric, and
bounded. No state-field aliases (``document``/``content``/``srcdoc``/
``min_height``/``minHeight``) and no widget-type aliases (``iframe``/
``micro_app``) are part of the contract.
"""

from __future__ import annotations

import pytest

from app.services.widget_contract import (
    MAX_WIDGET_HEIGHT,
    MIN_WIDGET_HEIGHT,
    render_action_template,
    resolve_widget_action_message,
    validate_html_widget_state,
)


# ---------------------------------------------------------------------------
# HTML state contract
# ---------------------------------------------------------------------------
def test_valid_html_state_passes():
    validate_html_widget_state({"html": "<!doctype html><div>hi</div>", "height": 620})


def test_non_object_state_is_rejected():
    with pytest.raises(ValueError, match="JSON object"):
        validate_html_widget_state("not-a-dict")


def test_empty_html_content_is_rejected():
    with pytest.raises(ValueError, match="html content"):
        validate_html_widget_state({"html": "   ", "height": 540})


def test_missing_html_content_is_rejected():
    with pytest.raises(ValueError, match="html content"):
        validate_html_widget_state({"height": 540})


def test_missing_height_is_rejected():
    with pytest.raises(ValueError, match="height"):
        validate_html_widget_state({"html": "<div>hi</div>"})


def test_non_numeric_height_is_rejected():
    with pytest.raises(ValueError, match="numeric"):
        validate_html_widget_state({"html": "<div>hi</div>", "height": "tall"})


def test_out_of_range_height_low_is_rejected():
    with pytest.raises(ValueError, match="between"):
        validate_html_widget_state({"html": "<div>hi</div>", "height": MIN_WIDGET_HEIGHT - 1})


def test_out_of_range_height_high_is_rejected():
    with pytest.raises(ValueError, match="between"):
        validate_html_widget_state({"html": "<div>hi</div>", "height": MAX_WIDGET_HEIGHT + 1})


def test_document_alias_is_not_accepted_as_html():
    with pytest.raises(ValueError, match="html content"):
        validate_html_widget_state({"document": "<div>hi</div>", "height": 540})


@pytest.mark.parametrize("alias", ["content", "srcdoc", "iframe", "micro_app"])
def test_html_state_field_aliases_are_not_accepted(alias):
    with pytest.raises(ValueError, match="html content"):
        validate_html_widget_state({alias: "<div>hi</div>", "height": 540})


@pytest.mark.parametrize("alias", ["min_height", "minHeight"])
def test_height_aliases_are_not_accepted(alias):
    with pytest.raises(ValueError, match="height"):
        validate_html_widget_state({"html": "<div>hi</div>", alias: 540})


# ---------------------------------------------------------------------------
# Action template resolution (moved verbatim from widget_quality)
# ---------------------------------------------------------------------------
def test_render_action_template_substitutes_state_and_inputs():
    rendered = render_action_template(
        "caption={{state.caption}} note={{input_values.note}}",
        {"caption": "demo"},
        {"note": "ok"},
    )
    assert rendered == "caption=demo note=ok"


def test_resolve_widget_action_message_renders_assistant_action():
    state = {
        "html": "<div>hi</div>",
        "height": 540,
        "actions": [
            {
                "key": "explain",
                "type": "assistant_message",
                "message_template": "Explain {{input_values.topic}}",
            }
        ],
    }
    assert resolve_widget_action_message(state, "explain", {"topic": "SHM"}) == "Explain SHM"


def test_resolve_widget_action_message_unknown_action_raises():
    with pytest.raises(KeyError):
        resolve_widget_action_message({"actions": []}, "missing", {})
