"""Live-stream markdown helpers shared between ``demo.py`` and tests.

These helpers must be importable without ``streamlit`` installed so the
test suite can exercise them in headless environments.
"""

from __future__ import annotations

_STREAM_MARKDOWN_ENTITY_MAP = {
    "&quot;": '"',
    "&#34;": '"',
    "&#x22;": '"',
    "&#39;": "'",
    "&#x27;": "'",
}


def normalize_stream_markdown_text(content: str) -> str:
    """Unescape only quote entities in live-stream markdown text.

    Live token rendering calls ``st.markdown(...)`` directly without going
    through the HTML sanitizer used for persisted messages, so models that
    emit ``&quot;`` end up showing the literal entity. We unescape quote
    entities only — broad ``html.unescape`` would turn ``&lt;script&gt;``
    into actual HTML in the streamed output, which is the wrong tradeoff
    for the live path.
    """
    if not isinstance(content, str) or not content:
        return ""
    normalized = content
    for entity, replacement in _STREAM_MARKDOWN_ENTITY_MAP.items():
        normalized = normalized.replace(entity, replacement)
    return normalized
