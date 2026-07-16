from __future__ import annotations

from app.ai.agents.canvas_agent import _extract_artifact, _strip_code_block


def test_truncated_code_block_still_extracts_artifact():
    """A generation cut off before the closing fence (length cap, provider
    stop) must still yield an artifact — otherwise the raw code dump persists
    as chat content and no canvas renders (regression: 2026-07-02)."""

    text = (
        "Here is your calculator page.\n\n"
        "```html\n"
        "<!DOCTYPE html>\n"
        "<html>\n"
        "  <head><title>Calculator</title></head>\n"
        "  <body>\n"
        "    <button onclick=\"append('8')\">8</button>\n"
        '    <button onclick="append'
    )

    artifact = _extract_artifact(text)

    assert artifact is not None
    assert artifact["language"] == "html"
    assert artifact["title"] == "Calculator"
    assert artifact["truncated"] is True
    assert artifact["content"].startswith("<!DOCTYPE html>")
    assert artifact["content"].endswith('<button onclick="append')


def test_truncated_code_block_strip_keeps_description_only():
    text = "Here is your calculator page.\n\n```html\n<!DOCTYPE html>\n<body>partial"

    assert _strip_code_block(text) == "Here is your calculator page."


def test_unclosed_fence_with_no_code_is_not_an_artifact():
    assert _extract_artifact("Some text.\n\n```html\n") is None


def test_complete_code_block_is_not_marked_truncated():
    artifact = _extract_artifact("Intro.\n\n```html\n<!doctype html><body>ok</body>\n```")

    assert artifact is not None
    assert "truncated" not in artifact


def test_canvas_artifact_response_shape_omits_editable_hint():
    artifact = _extract_artifact(
        """
Built a small page.

```html
<!doctype html>
<html>
  <head><title>Demo Canvas</title></head>
  <body>Hello</body>
</html>
```
"""
    )

    assert artifact == {
        "content": (
            "<!doctype html>\n"
            "<html>\n"
            "  <head><title>Demo Canvas</title></head>\n"
            "  <body>Hello</body>\n"
            "</html>"
        ),
        "language": "html",
        "title": "Demo Canvas",
    }
