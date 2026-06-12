"""Deterministic article-style placement of unreferenced rich items.

Inserts ``<!--rich:<id>-->`` markers into the final assistant markdown for
relevant rich items the model did not place itself, so answers read like an
article with inline media instead of silently dropping images. Placement is
keyword-overlap based and runs once at persistence time — no extra model
calls and no prompt-context cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .rich_response import (
    RichItemType,
    _strip_fenced_code_blocks,  # noqa: PLC2701  # deliberate same-package reuse of CommonMark fence semantics
    parse_inline_rich_references,
)

_WORD_RE = re.compile(r"[a-z0-9]{3,}")

#: Generic words that carry no placement signal for media descriptions.
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "are", "was",
        "were", "has", "have", "had", "its", "but", "not", "you", "your",
        "they", "their", "about", "into", "over", "after", "before",
        "between", "image", "photo", "picture", "stock", "view",
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
    for idx, (line, fenced) in enumerate(zip(lines, in_fence, strict=False)):
        if not line.strip():
            if current:
                blocks.append(_Block(current_end, _tokens(" ".join(current)), current_code))
                current, current_code = [], False
            continue
        current.append(line)
        current_end = idx
        current_code = current_code or fenced or line.startswith(("    ", "\t"))
    if current:
        blocks.append(_Block(current_end, _tokens(" ".join(current)), current_code))
    return blocks


def _score(item_tokens: frozenset[str], block_tokens: frozenset[str]) -> float:
    if not item_tokens or not block_tokens:
        return 0.0
    return len(item_tokens & block_tokens) / len(item_tokens)


def auto_place_rich_items(
    content: str,
    *,
    items: list[tuple[str, str, str]],
    max_images: int,
    min_score: float,
) -> tuple[str, list[str]]:
    """Insert markers for unreferenced items after their best-matching paragraph.

    ``items`` holds ``(item_id, item_type, descriptive_text)`` tuples in
    priority order. At most one item is placed per paragraph and at most
    ``max_images`` image items overall. Items already referenced in
    ``content`` or scoring below ``min_score`` are skipped. Returns
    ``(new_content, placed_ids)``; content is returned unchanged when nothing
    places.
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
    images_placed = 0
    for item_id, item_type, text in items:
        if item_id in referenced:
            continue
        is_image = item_type == RichItemType.image.value
        if is_image and images_placed >= max_images:
            continue
        item_tokens = _tokens(text or "")
        best_line, best = -1, 0.0
        for block in blocks:
            if block.end_line in insertions:
                continue
            score = _score(item_tokens, block.tokens)
            if score > best:
                best_line, best = block.end_line, score
        if best_line < 0 or best < min_score:
            continue
        insertions[best_line] = f"<!--rich:{item_id}-->"
        placed.append(item_id)
        if is_image:
            images_placed += 1

    if not placed:
        return content, []

    out: list[str] = []
    for idx, line in enumerate(lines):
        out.append(line)
        marker = insertions.get(idx)
        if marker is not None:
            out.append("")
            out.append(marker)
    return "\n".join(out), placed
