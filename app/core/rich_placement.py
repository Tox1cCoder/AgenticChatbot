"""Deterministic article-style placement of unreferenced non-image rich items."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from .config import settings
from .response_constants import extract_live_widgets_from_artifacts
from .rich_response import (
    _ITEM_ID_PATTERN,  # noqa: PLC2701  # deliberate same-package reuse: inserter must mirror parser id rules
    RICH_ITEM_ID_MAX_LENGTH,
    RichItemType,
    _strip_fenced_code_blocks,  # noqa: PLC2701  # deliberate same-package reuse of CommonMark fence semantics
    parse_inline_rich_references,
    remove_inline_rich_reference,
    strip_malformed_rich_markers,
)

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]{3,}")

#: A standalone HTML-comment line whose inner text uses the rich-item id
#: charset. Used to detect markers the model wrote without the ``rich:``
#: prefix so they can be repaired against the known-id set.
_BARE_MARKER_LINE_RE = re.compile(r"^([ ]{0,3})<!--([A-Za-z0-9_\-.:]+)-->([ \t]*)$")

#: Generic words that carry no placement signal for media descriptions.
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "its",
        "but",
        "not",
        "you",
        "your",
        "they",
        "their",
        "about",
        "into",
        "over",
        "after",
        "before",
        "between",
        "image",
        "photo",
        "picture",
        "stock",
        "view",
    }
)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS)


@dataclass
class _Block:
    """A contiguous run of non-blank markdown lines outside fenced code."""

    end_line: int
    tokens: frozenset[str]
    is_code: bool


def _segment_blocks(lines: list[str]) -> list[_Block]:
    in_fence = _strip_fenced_code_blocks(lines)
    blocks: list[_Block] = []
    current: list[str] = []
    current_code = False
    current_end = -1
    prev_fenced = False
    for idx, (line, fenced) in enumerate(zip(lines, in_fence, strict=False)):
        if not line.strip():
            if current:
                blocks.append(_Block(current_end, _tokens(" ".join(current)), current_code))
                current, current_code = [], False
            prev_fenced = False
            continue
        if current and fenced != prev_fenced:
            # CommonMark fences need no surrounding blank lines; split here so
            # adjacent prose stays matchable instead of being treated as code.
            blocks.append(_Block(current_end, _tokens(" ".join(current)), current_code))
            current, current_code = [], False
        current.append(line)
        current_end = idx
        current_code = current_code or fenced or line.startswith(("    ", "\t"))
        prev_fenced = fenced
    if current:
        blocks.append(_Block(current_end, _tokens(" ".join(current)), current_code))
    return blocks


def _score(item_tokens: frozenset[str], block_tokens: frozenset[str]) -> float:
    if not item_tokens or not block_tokens:
        return 0.0
    return len(item_tokens & block_tokens) / len(item_tokens)


def _find_best_block(
    tokens: frozenset[str], blocks: list[_Block], insertions: dict[int, str]
) -> tuple[int, float]:
    """Return ``(end_line, score)`` of the best-scoring free block.

    Returns ``(-1, 0.0)`` when no block is free or nothing scores above zero.
    Blocks already carrying an insertion are skipped, which is what keeps
    placement to at most one item per paragraph.
    """
    best_line, best = -1, 0.0
    for block in blocks:
        if block.end_line in insertions:
            continue
        score = _score(tokens, block.tokens)
        if score > best:
            best_line, best = block.end_line, score
    return best_line, best


def _apply_insertions(lines: list[str], insertions: dict[int, str]) -> str:
    """Rebuild the document with each marker on its own line after its block."""
    out: list[str] = []
    for idx, line in enumerate(lines):
        out.append(line)
        marker = insertions.get(idx)
        if marker is not None:
            out.append("")
            out.append(marker)
    return "\n".join(out)


def auto_place_rich_items(
    content: str,
    *,
    items: list[tuple[str, str]],
    min_score: float,
) -> tuple[str, list[str]]:
    """Insert markers for unreferenced widget-class items after their best block.

    ``items`` holds ``(item_id, descriptive_text)`` tuples in priority order. At
    most one item is placed per paragraph. Items already referenced in
    ``content`` or scoring below ``min_score`` are skipped. Returns
    ``(new_content, placed_ids)``; content is returned unchanged when nothing
    places.

    Image items never travel through here. Web images are rendered only when
    the answer model selects a validated candidate with ``[[image:I#]]``.
    """
    if not content or not items:
        return content, []
    referenced = set(parse_inline_rich_references(content))
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    blocks = [b for b in _segment_blocks(lines) if not b.is_code]
    if not blocks:
        return content, []

    insertions: dict[int, str] = {}
    placed: list[str] = []
    for item_id, text in items:
        if item_id in referenced:
            continue
        if len(item_id) > RICH_ITEM_ID_MAX_LENGTH or not _ITEM_ID_PATTERN.match(item_id):
            # The parser rejects such markers; inserting one would persist a raw
            # comment with no matching rich_items entry.
            continue
        best_line, best = _find_best_block(_tokens(text or ""), blocks, insertions)
        if best_line < 0 or best < min_score:
            continue
        insertions[best_line] = f"<!--rich:{item_id}-->"
        placed.append(item_id)

    if not placed:
        return content, []
    return _apply_insertions(lines, insertions), placed


def _repair_unprefixed_markers(content: str, known_ids: set[str]) -> str:
    """Restore the ``rich:`` prefix on model-authored markers that dropped it.

    Some models write ``<!--widget:<id>-->`` instead of the contract's
    ``<!--rich:widget:<id>-->`` because the rich-item id already carries a
    namespace (``widget:``/``image:``/...), so the canonical marker looks
    redundant and the ``rich:`` prefix gets dropped. Every consumer (the marker
    parser, the live-stream normalizer, the renderer) requires that prefix, so
    such a marker leaks as literal text and leaves its item unreferenced — the
    image is discarded and the widget falls to the append-after-body fallback.

    A bare standalone marker whose inner text exactly matches a known available
    rich-item id is rewritten to the canonical ``rich:`` form. Ids are
    server-constructed, so an exact match is unambiguous: incidental HTML
    comments never collide. Markers inside fenced code are left untouched,
    mirroring ``parse_inline_rich_references``.
    """
    if not content or not known_ids or "<!--" not in content:
        return content
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    in_fence = _strip_fenced_code_blocks(lines)
    changed = False
    for idx, (line, fenced) in enumerate(zip(lines, in_fence, strict=False)):
        if fenced:
            continue
        match = _BARE_MARKER_LINE_RE.match(line)
        if match is None:
            continue
        inner = match.group(2)
        if inner.startswith("rich:") or inner not in known_ids:
            continue
        lines[idx] = f"{match.group(1)}<!--rich:{inner}-->{match.group(3)}"
        changed = True
    return "\n".join(lines) if changed else content


def _widget_placement_entries(
    metadata: dict[str, Any], response_artifacts: list[dict[str, Any]] | None
) -> list[tuple[str, str]]:
    artifacts: list[dict[str, Any]] = []
    meta_artifacts = metadata.get("tool_artifacts")
    if isinstance(meta_artifacts, list):
        artifacts.extend(a for a in meta_artifacts if isinstance(a, dict))
    if response_artifacts:
        artifacts.extend(a for a in response_artifacts if isinstance(a, dict))
    entries: list[tuple[str, str]] = []
    for widget in extract_live_widgets_from_artifacts(artifacts):
        widget_id = widget.get("widget_id")
        if not widget_id:
            continue
        text = str(widget.get("title") or "")
        entries.append((f"widget:{widget_id}", text))
    return entries


def finalize_article_content(response: Any, content: str) -> str:
    """Apply article-style placement to the final assistant markdown.

    Mutates ``response.message.content`` to the placed content so
    ``build_bot_metadata()`` resolves the exact marker set the persisted
    message carries. Returns the (possibly updated) content. Image-marker
    integrity runs whenever inline rich response is enabled and capable;
    automatic placement additionally requires its own setting.

    Never raises. Placement is an optional enhancement on the persistence path,
    so any unexpected failure returns the original content and the answer is
    persisted without inline media. Making that structural here means every
    caller inherits it rather than each having to remember a guard.
    """
    try:
        return _finalize_article_content(response, content)
    except Exception:
        logger.warning("Inline rich placement skipped code=rich_placement_failed")
        return content


def _finalize_article_content(response: Any, content: str) -> str:
    if not content or response is None:
        return content
    if not getattr(settings, "inline_rich_response_enabled", False):
        return content
    metadata = getattr(response, "metadata", None)
    if not isinstance(metadata, dict) or not metadata.get("_inline_rich_response_v1"):
        return content

    image_types = {RichItemType.image.value, RichItemType.image_group.value}
    allowed_image_ids = frozenset(
        str(candidate.get("id"))
        for candidate in metadata.get("_rich_item_candidates") or []
        if isinstance(candidate, dict)
        and candidate.get("type") in image_types
        and candidate.get("id")
    )
    raw_presented_ids = metadata.get("_presented_rich_image_ids")
    if isinstance(raw_presented_ids, list):
        allowed_image_ids = allowed_image_ids.intersection(
            str(item_id) for item_id in raw_presented_ids
        )
    # Unresolvable markers go first. The authorization sweep below can only
    # remove what the strict grammar can parse, so a marker with a space in its
    # id survives every check and is rendered to the reader verbatim.
    cleaned_content = strip_malformed_rich_markers(content)
    for reference in parse_inline_rich_references(cleaned_content):
        if reference.startswith(("image:", "imagegroup:")) and reference not in allowed_image_ids:
            cleaned_content = remove_inline_rich_reference(cleaned_content, reference)
    if cleaned_content != content:
        message = getattr(response, "message", None)
        if message is not None and isinstance(getattr(message, "content", None), str):
            message.content = cleaned_content
    if not getattr(settings, "rich_auto_place_enabled", False):
        return cleaned_content

    items = _widget_placement_entries(metadata, getattr(response, "tool_artifacts", None))
    if not items:
        return cleaned_content

    # Repair markers the model authored without the ``rich:`` prefix first, so
    # the now-canonical marker is recognized as a reference (the item renders
    # inline) and auto-placement does not place a second copy of it.
    known_ids = {entry[0] for entry in items}
    repaired = _repair_unprefixed_markers(cleaned_content, known_ids)
    new_content, _placed = auto_place_rich_items(
        repaired,
        items=items,
        min_score=settings.rich_auto_place_min_score,
    )
    if new_content == cleaned_content:
        return cleaned_content

    message = getattr(response, "message", None)
    if message is not None and isinstance(getattr(message, "content", None), str):
        message.content = new_content
    return new_content
