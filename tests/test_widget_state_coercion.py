"""Widget state coercion — models routinely stringify the nested state object.

The reproduced production failure is a pretty-printed JSON *string* whose
``html`` value carries raw (unescaped) newlines. Strict ``json.loads`` rejects
it with "Invalid control character", which is why FastMCP's ``pre_parse_json``
fell through and the tool call died on the first attempt.
"""

from __future__ import annotations

import json

import pytest

from app.services.widget_contract import coerce_widget_state_object

_HTML = "<!doctype html>\n<html>\n<body>\n<h1>Hi</h1>\n</body>\n</html>\n"


def test_dict_passes_through_unchanged():
    state = {"html": _HTML, "height": 620}
    assert coerce_widget_state_object(state) is state


def test_properly_escaped_json_string_is_parsed():
    raw = json.dumps({"height": 620, "html": _HTML})
    assert coerce_widget_state_object(raw) == {"height": 620, "html": _HTML}


def test_raw_unescaped_newlines_inside_html_are_recovered():
    """The exact production payload shape: raw control chars in a string value."""
    raw = '{\n  "height": 620,\n  "html": "' + _HTML + '"\n}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)

    state = coerce_widget_state_object(raw)
    assert state["height"] == 620
    assert state["html"] == _HTML


def test_markdown_fenced_json_is_recovered():
    raw = "```json\n" + json.dumps({"height": 300, "html": "<p>x</p>"}) + "\n```"
    assert coerce_widget_state_object(raw) == {"height": 300, "html": "<p>x</p>"}


def test_python_literal_state_is_recovered():
    raw = "{'height': 300, 'html': '<p>x</p>', 'caption': None}"
    state = coerce_widget_state_object(raw)
    assert state == {"height": 300, "html": "<p>x</p>", "caption": None}


def test_non_object_json_is_rejected_with_field_name():
    with pytest.raises(ValueError, match="initial_state"):
        coerce_widget_state_object("[1, 2, 3]", field="initial_state")


def test_unrecoverable_string_raises_actionable_error():
    with pytest.raises(ValueError) as excinfo:
        coerce_widget_state_object('{"html": "unterminated', field="state")
    message = str(excinfo.value)
    assert "state" in message
    assert "double-quoted" in message.lower()


def test_none_is_rejected():
    with pytest.raises(ValueError, match="must be an object"):
        coerce_widget_state_object(None)
