"""Live Streamlit markdown rendering safety guards.

These tests cover the narrow entity-normalization helper used by the live
streaming path (``response_placeholder.markdown(...)`` in ``demo.py``).
The persisted message path goes through ``sanitize_message_content`` and
is exercised separately.

``demo.py`` re-exports ``normalize_stream_markdown_text`` from
``app.ui.stream_markdown`` so the helper stays importable without
``streamlit`` installed.
"""

from __future__ import annotations


def test_normalize_stream_markdown_text_unescapes_quotes_only():
    """Quote entities (``&quot;``, ``&#34;``, ``&#x22;``, ``&#39;``,
    ``&#x27;``) must be unescaped. Angle-bracket entities must be left
    as-is — broad ``html.unescape`` in the live path would let escaped
    HTML render and is the wrong tradeoff."""
    from app.ui.stream_markdown import normalize_stream_markdown_text

    raw = (
        "model: &quot;gemini-3.1-pro&quot;, provider: &#34;gemini&#34;: "
        "&lt;b&gt;safe text&lt;/b&gt;"
    )

    assert normalize_stream_markdown_text(raw) == (
        'model: "gemini-3.1-pro", provider: "gemini": &lt;b&gt;safe text&lt;/b&gt;'
    )


def test_normalize_stream_markdown_text_handles_apostrophe_entities():
    from app.ui.stream_markdown import normalize_stream_markdown_text

    raw = "It&#39;s &#x27;safe&#x27; for the user&apos;s prompt"

    # &apos; is NOT in the narrow allowlist — only the common quote forms
    # are. The renderer keeps it as-is so we don't expand the surface
    # area beyond what the reported bug actually needed.
    assert normalize_stream_markdown_text(raw) == "It's 'safe' for the user&apos;s prompt"


def test_normalize_stream_markdown_text_handles_empty_and_non_string():
    from app.ui.stream_markdown import normalize_stream_markdown_text

    assert normalize_stream_markdown_text("") == ""
    assert normalize_stream_markdown_text(None) == ""  # type: ignore[arg-type]
    assert normalize_stream_markdown_text(123) == ""  # type: ignore[arg-type]


def test_normalize_stream_markdown_text_strips_inline_rich_markers():
    """Standalone ``<!--rich:<id>-->`` lines are stripped from the live
    placeholder so they do not appear as visible text while the stream is
    in flight. The post-stream rerun resolves the markers and renders the
    referenced rich item inline via ``build_rich_response_view``.
    """
    from app.ui.stream_markdown import normalize_stream_markdown_text

    raw = "Intro paragraph.\n\n<!--rich:widget:abc-123-->\n\nClosing paragraph."

    assert normalize_stream_markdown_text(raw) == ("Intro paragraph.\n\n\n\nClosing paragraph.")


def test_normalize_stream_markdown_text_keeps_inline_marker_in_code():
    """A marker that appears inside an inline-code span is part of prose,
    not a block-level marker line, and must be preserved verbatim."""
    from app.ui.stream_markdown import normalize_stream_markdown_text

    raw = "See `<!--rich:foo-->` for the contract grammar."

    assert normalize_stream_markdown_text(raw) == raw


def test_group_segment_renders_group_html():
    """Characterization test: the view model already routes an
    ``image_group`` rich item as a ``rich`` segment via plain id lookup, with
    no special-casing by type. This pins existing behavior in
    ``build_rich_response_view`` — it is not new behavior added by this test
    — so that a later reader does not mistake it for a regression guard on
    the Streamlit rendering branch (which is exercised separately)."""
    from app.ui.rich_response import build_rich_response_view

    metadata = {
        "rich_items_version": 1,
        "rich_items": [
            {
                "id": "imagegroup:tool:c1",
                "type": "image_group",
                "alt_text": "Images of red panda",
                "payload": {
                    "items": [
                        {"url": "/web-images/1", "mime_type": "image/jpeg"},
                        {"url": "/web-images/2", "mime_type": "image/jpeg"},
                    ]
                },
            }
        ],
    }
    view = build_rich_response_view("Body\n\n<!--rich:imagegroup:tool:c1-->\n", metadata)
    rich_segments = [s for s in view.segments if s.kind == "rich"]
    assert len(rich_segments) == 1
    assert rich_segments[0].item["type"] == "image_group"
