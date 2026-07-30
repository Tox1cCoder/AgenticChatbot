"""Live-stream markdown helpers shared between ``demo.py`` and tests.

These helpers must be importable without ``streamlit`` installed so the
test suite can exercise them in headless environments.
"""

from __future__ import annotations

import re

_STREAM_MARKDOWN_ENTITY_MAP = {
    "&quot;": '"',
    "&#34;": '"',
    "&#x22;": '"',
    "&#39;": "'",
    "&#x27;": "'",
}

# Standalone inline rich-response marker line. Matches the contract defined in
# ``app.core.rich_response`` (block-level HTML comment, optional leading up to
# three spaces, optional trailing whitespace) but is duplicated here to keep
# this helper free of any heavy imports.
_RICH_MARKER_LINE_RE = re.compile(
    r"^[ ]{0,3}<!--rich:[A-Za-z0-9_\-.:]+-->[ \t]*$",
    re.MULTILINE,
)

_MARKDOWN_PROTECTED_SPAN_RE = re.compile(
    r"(?ms)"
    r"(?:^[ ]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^[ ]{0,3}(?P=fence)[ \t]*$)"
    r"|(?:\$\$.*?\$\$)"
    r"|(?:(?P<inline_ticks>`+).*?(?P=inline_ticks))"
)
_CURRENCY_AMOUNT_RE = r"\d+(?:,\d{3})*(?:\.\d+)?"
_CURRENCY_SEPARATOR_RE = r"(?:\s*[–—-]\s*|\s*,\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+)"
_CURRENCY_PRICE_RUN_RE = re.compile(
    rf"(?<!\\)\${_CURRENCY_AMOUNT_RE}(?!\d|\.\d|,\d)"
    rf"(?:{_CURRENCY_SEPARATOR_RE}(?<!\\)\${_CURRENCY_AMOUNT_RE}(?!\d|\.\d|,\d))+"
)


def _escape_currency_run(match: re.Match[str]) -> str:
    """Escape every unescaped dollar marker in one recognised price run."""
    return re.sub(r"(?<!\\)\$", r"\\$", match.group(0))


def escape_markdown_currency(text: str) -> str:
    """Escape unambiguous dollar-price runs outside protected Markdown spans.

    Streamlit renders ``$...$`` as inline LaTeX. This display-only transform
    protects price ranges and lists while leaving code and intentional math
    untouched. Lone dollar amounts are intentionally not guessed because they
    are ambiguous with numeric LaTeX.
    """
    if not isinstance(text, str) or not text:
        return ""

    pieces: list[str] = []
    cursor = 0
    for protected_span in _MARKDOWN_PROTECTED_SPAN_RE.finditer(text):
        renderable_span = text[cursor : protected_span.start()]
        pieces.append(_CURRENCY_PRICE_RUN_RE.sub(_escape_currency_run, renderable_span))
        pieces.append(protected_span.group(0))
        cursor = protected_span.end()
    pieces.append(_CURRENCY_PRICE_RUN_RE.sub(_escape_currency_run, text[cursor:]))
    return "".join(pieces)


def normalize_stream_markdown_text(content: str) -> str:
    """Unescape quote entities and hide raw inline rich-item markers.

    Live token rendering calls ``st.markdown(...)`` directly without going
    through the HTML sanitizer used for persisted messages, so models that
    emit ``&quot;`` end up showing the literal entity. We unescape quote
    entities only — broad ``html.unescape`` would turn ``&lt;script&gt;``
    into actual HTML in the streamed output, which is the wrong tradeoff
    for the live path.

    We also strip standalone ``<!--rich:<id>-->`` marker lines from the live
    placeholder. They are the inline rich-response contract markers; they're
    invisible in a CommonMark renderer but Streamlit's safe-mode markdown
    leaves them as literal text. The post-stream rerun resolves the markers
    via ``app.ui.rich_response.build_rich_response_view`` and renders the
    referenced rich item (widget, image, tool view, etc.) at its inline
    position, so suppressing the raw token here only affects what the user
    sees while the stream is still in flight.
    """
    if not isinstance(content, str) or not content:
        return ""
    normalized = content
    for entity, replacement in _STREAM_MARKDOWN_ENTITY_MAP.items():
        normalized = normalized.replace(entity, replacement)
    if "<!--rich:" in normalized:
        normalized = _RICH_MARKER_LINE_RE.sub("", normalized)
    return escape_markdown_currency(normalized)
