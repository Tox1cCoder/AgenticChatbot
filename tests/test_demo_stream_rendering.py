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

    raw = 'model: &quot;gemini-3.1-pro&quot;, provider: &#34;gemini&#34;: &lt;b&gt;safe text&lt;/b&gt;'

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
