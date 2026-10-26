"""Per-feature projection of a stored rich message for one client.

Stored content and ``rich_items`` are canonical. A client that does not
render a newer item family gets that family's stored fallback instead:

- an answer visual becomes its ``fallback_markdown`` at the marker position
  and leaves the registry;
- a composed source figure puts its source-linked text before the marker and
  stays in the registry as an exact legacy image payload (a non-rich client
  keeps only the text).

The AI SDK projection and the Streamlit view model both call this module, so
there is one fallback path, not one per client.
"""

from __future__ import annotations

from typing import Any

from .answer_visuals import ANSWER_VISUAL_ID_PREFIX, ANSWER_VISUAL_TYPE
from .rich_response import (
    expand_inline_rich_references,
    parse_inline_rich_references,
    remove_inline_rich_reference,
    strip_inline_rich_markers,
)
from .source_figures import is_source_figure, legacy_source_figure


def _fallbacks(items: list[Any], predicate: Any) -> dict[str, str]:
    """Stored fallback Markdown for each matching item, keyed by id."""
    fallbacks: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict) or not predicate(item):
            continue
        fallback = (item.get("payload") or {}).get("fallback_markdown")
        if isinstance(item.get("id"), str) and isinstance(fallback, str) and fallback:
            fallbacks[item["id"]] = fallback
    return fallbacks


def is_answer_visual(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == ANSWER_VISUAL_TYPE


def _blocks(
    items: list[Any], *, rich: bool, answer_visuals: bool, source_figures: bool
) -> dict[str, str]:
    blocks: dict[str, str] = {}
    if not (rich and answer_visuals):
        blocks.update(_fallbacks(items, is_answer_visual))
    if not (rich and source_figures):
        for item_id, fallback in _fallbacks(items, is_source_figure).items():
            blocks[item_id] = f"{fallback}\n\n<!--rich:{item_id}-->" if rich else fallback
    return blocks


def needs_projection(
    content: Any, items: list[Any], *, rich: bool, answer_visuals: bool, source_figures: bool
) -> bool:
    """Whether a rich client would see anything differ after projection."""
    if not rich:
        return True
    if _blocks(items, rich=rich, answer_visuals=answer_visuals, source_figures=source_figures):
        return True
    return (
        not answer_visuals
        and isinstance(content, str)
        and f"<!--rich:{ANSWER_VISUAL_ID_PREFIX}" in content
    )


def project_rich_content(
    content: str, items: list[Any], *, rich: bool, answer_visuals: bool, source_figures: bool
) -> str:
    """Content as one client should read it; ``items`` is the stored registry."""
    if not content or "<!--rich:" not in content:
        return content
    blocks = _blocks(items, rich=rich, answer_visuals=answer_visuals, source_figures=source_figures)
    projected = expand_inline_rich_references(content, blocks)
    if not (rich and answer_visuals):
        # A visual marker with no descriptor names nothing this client can show.
        for item_id in parse_inline_rich_references(projected):
            if item_id.startswith(ANSWER_VISUAL_ID_PREFIX) and item_id not in blocks:
                projected = remove_inline_rich_reference(projected, item_id)
    if not rich:
        projected = strip_inline_rich_markers(projected)
    return projected


def project_rich_items(
    items: list[Any], *, answer_visuals: bool, source_figures: bool
) -> list[Any]:
    """The registry a rich client should receive. Never mutates ``items``."""
    projected: list[Any] = []
    for item in items:
        if is_answer_visual(item) and not answer_visuals:
            continue
        if is_source_figure(item) and not source_figures:
            projected.append(legacy_source_figure(item))
            continue
        projected.append(item)
    return projected


__all__ = [
    "is_answer_visual",
    "needs_projection",
    "project_rich_content",
    "project_rich_items",
]
