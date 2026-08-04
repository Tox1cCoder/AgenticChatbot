"""Deterministic article-style placement of unreferenced rich items.

Inserts ``<!--rich:<id>-->`` markers into the final assistant markdown for
relevant rich items the model did not place itself, so answers read like an
article with inline media instead of silently dropping images. Placement is
keyword-overlap based and runs once at persistence time — no extra model
calls and no prompt-context cost.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from .config import settings
from .response_constants import extract_live_widgets_from_artifacts
from .rich_response import (
    _ITEM_ID_PATTERN,  # noqa: PLC2701  # deliberate same-package reuse: inserter must mirror parser id rules
    GENERIC_IMAGE_ALT_TEXT,
    RICH_ITEM_ID_MAX_LENGTH,
    RichItemType,
    _strip_fenced_code_blocks,  # noqa: PLC2701  # deliberate same-package reuse of CommonMark fence semantics
    parse_inline_rich_references,
    provenance_provider,
    remove_inline_rich_reference,
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

    Image items never travel through here: they are anchored on the model's own
    image query by ``anchor_image_items_by_query``, which owns the per-answer
    image cap and the fallback-anchor rules.
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


#: Origins allowed to anchor without a scoring match. Running an image search is
#: itself the intent to display, so a missed keyword match must not silently
#: discard the result. A tool-produced image carries the same intent: the tool
#: call itself implies display. Such an image may or may not carry a query — a
#: chart or rendered diagram has none, while a single ungrouped Brave
#: image-search candidate is stamped ``tool_image`` and does carry its
#: provenance query — so the fallback exists for the queryless case and for the
#: case where a real query simply matched no paragraph.
_FALLBACK_ANCHOR_ORIGINS = frozenset({"image_search", "tool_image"})

#: Minimum post-stopword token count for a block to accept a fallback anchor, so
#: the image never lands under a bare heading or a two-word line.
_FALLBACK_MIN_BLOCK_TOKENS = 8


@dataclass(frozen=True)
class ImageAnchorEntry:
    """One unreferenced image item eligible for query anchoring."""

    item_id: str
    query: str
    origin: str
    anchorable: bool = True


def _first_fallback_block_line(blocks: list[_Block], insertions: dict[int, str]) -> int:
    for block in blocks:
        if block.end_line in insertions:
            continue
        if len(block.tokens) >= _FALLBACK_MIN_BLOCK_TOKENS:
            return block.end_line
    return -1


def anchor_image_items_by_query(
    content: str,
    *,
    entries: list[ImageAnchorEntry],
    min_score: float,
    max_images: int,
) -> tuple[str, dict[str, str]]:
    """Insert markers for unreferenced image items using their image query.

    Returns ``(new_content, outcomes)`` where ``outcomes`` maps each entry id to
    ``"marker"``, ``"query_anchored"``, ``"fallback_anchored"``, or
    ``"unplaced"``. A model-authored marker always wins: its position is
    authoritative and nothing is inserted for that item.
    """
    if not entries:
        return content, {}
    outcomes: dict[str, str] = {}
    if not content:
        return content, {entry.item_id: "unplaced" for entry in entries}

    referenced = set(parse_inline_rich_references(content))
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    blocks = [b for b in _segment_blocks(lines) if not b.is_code]

    insertions: dict[int, str] = {}
    placed = 0
    for entry in entries:
        if entry.item_id in referenced:
            outcomes[entry.item_id] = "marker"
            continue
        if (
            not entry.anchorable
            or not blocks
            or placed >= max_images
            or len(entry.item_id) > RICH_ITEM_ID_MAX_LENGTH
            or not _ITEM_ID_PATTERN.match(entry.item_id)
        ):
            outcomes[entry.item_id] = "unplaced"
            continue

        best_line, best = _find_best_block(_tokens(entry.query or ""), blocks, insertions)

        if best_line >= 0 and best >= min_score:
            outcome = "query_anchored"
        elif entry.origin in _FALLBACK_ANCHOR_ORIGINS:
            best_line = _first_fallback_block_line(blocks, insertions)
            outcome = "fallback_anchored" if best_line >= 0 else "unplaced"
        else:
            outcome = "unplaced"

        if outcome == "unplaced":
            outcomes[entry.item_id] = "unplaced"
            continue
        insertions[best_line] = f"<!--rich:{entry.item_id}-->"
        outcomes[entry.item_id] = outcome
        placed += 1

    if not insertions:
        return content, outcomes
    return _apply_insertions(lines, insertions), outcomes


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


def _descriptive_signal_text(candidate: dict[str, Any]) -> str:
    """Join a candidate's genuine descriptive text: title, alt_text, description.

    The generic alt-text fallback is excluded — it is not a real description,
    and using it as a placement signal lets junk images (e.g. crawler/SEO
    URLs) match a paragraph via tokens like "tool"/"result" and render
    broken. Used by the ``tool_image`` fallback-anchor origin in
    ``_image_anchor_entries`` to decide "this candidate carries no genuine
    signal, never auto-place it."
    """
    payload = candidate.get("payload")
    description = payload.get("description") if isinstance(payload, dict) else None
    alt_text = candidate.get("alt_text")
    if alt_text == GENERIC_IMAGE_ALT_TEXT:
        alt_text = None
    return " ".join(
        str(part) for part in (candidate.get("title"), alt_text, description) if part
    )


def _image_anchor_entries(
    metadata: dict[str, Any], *, image_max_items: int
) -> list[ImageAnchorEntry]:
    """Map turn-scoped image candidates to anchoring entries.

    Origin decides fallback eligibility: a deliberate image search may anchor
    without a keyword match; a tool-produced image anchors the same way when
    it carries a genuine signal (an image-search query, or real descriptive
    text — never the generic alt-text placeholder), because the tool call
    itself implies display but a signal-less image is exactly the junk-image
    failure this placement system exists to prevent; a source-bound
    web-search image may not fall back; and a query-level image is never
    anchored because it carries no page provenance.

    Only the first ``image_max_items`` image candidates are considered, matching
    the cap the model-facing inventory applies. Without that bound the two stages
    read different sets and an image the model was never shown could be anchored
    into the answer. Repeated ids are collapsed, because inserting one id twice
    would place two markers and overwrite its own outcome.
    """
    image_types = {RichItemType.image.value, RichItemType.image_group.value}
    entries: list[ImageAnchorEntry] = []
    seen_ids: set[str] = set()
    raw_presented_ids = metadata.get("_presented_rich_image_ids")
    presented_ids = (
        {str(item_id) for item_id in raw_presented_ids}
        if isinstance(raw_presented_ids, list)
        else None
    )
    for candidate in metadata.get("_rich_item_candidates") or []:
        if not isinstance(candidate, dict) or candidate.get("type") not in image_types:
            continue
        item_id = candidate.get("id")
        if not isinstance(item_id, str) or not item_id or item_id in seen_ids:
            continue
        if presented_ids is not None and item_id not in presented_ids:
            continue
        if len(entries) >= image_max_items:
            break
        seen_ids.add(item_id)
        provenance = candidate.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        query = str(provenance.get("query") or "").strip()
        source = candidate.get("source")
        if source == "image_search":
            origin, anchorable = "image_search", True
        elif source == "tool_image":
            origin = "tool_image"
            anchorable = bool(query) or bool(_descriptive_signal_text(candidate).strip())
        elif provenance.get("query_level"):
            origin, anchorable = "web_search_query_level", False
        else:
            origin, anchorable = "web_search_source_bound", bool(query)
        entries.append(
            ImageAnchorEntry(
                item_id=item_id, query=query, origin=origin, anchorable=anchorable
            )
        )
    return entries


def _record_anchor_outcomes(metadata: dict[str, Any], outcomes: dict[str, str]) -> None:
    """Record anchor outcomes. Never raises: telemetry must not fail an answer."""
    if not outcomes:
        return
    try:
        from app.observability.rich_images import rich_image_metrics

        providers = {
            candidate["id"]: provenance_provider(candidate)
            for candidate in metadata.get("_rich_item_candidates") or []
            if isinstance(candidate, dict) and isinstance(candidate.get("id"), str)
        }
        for item_id, outcome in outcomes.items():
            rich_image_metrics.record_anchor(
                provider=providers.get(item_id, "other"), outcome=outcome
            )
    except Exception:  # noqa: BLE001  # telemetry is best-effort by contract
        return


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
    cleaned_content = content
    for reference in parse_inline_rich_references(content):
        if (
            reference.startswith(("image:", "imagegroup:"))
            and reference not in allowed_image_ids
        ):
            cleaned_content = remove_inline_rich_reference(cleaned_content, reference)
    if cleaned_content != content:
        message = getattr(response, "message", None)
        if message is not None and isinstance(getattr(message, "content", None), str):
            message.content = cleaned_content
    if not getattr(settings, "rich_auto_place_enabled", False):
        return cleaned_content

    items = _widget_placement_entries(metadata, getattr(response, "tool_artifacts", None))
    # One cap, one candidate set: the inventory the model saw and the anchoring
    # pass must bound the same list, or an unseen image can be placed.
    max_images = int(getattr(settings, "rich_auto_place_max_images", 2))
    anchor_entries = _image_anchor_entries(metadata, image_max_items=max_images)
    if not items and not anchor_entries:
        return cleaned_content

    # Repair markers the model authored without the ``rich:`` prefix first, so
    # the now-canonical marker is recognized as a reference (the item renders
    # inline) and auto-placement does not place a second copy of it.
    known_ids = {entry[0] for entry in items} | {e.item_id for e in anchor_entries}
    repaired = _repair_unprefixed_markers(cleaned_content, known_ids)
    new_content, _placed = auto_place_rich_items(
        repaired,
        items=items,
        min_score=settings.rich_auto_place_min_score,
    )
    if anchor_entries:
        new_content, outcomes = anchor_image_items_by_query(
            new_content,
            entries=anchor_entries,
            min_score=float(getattr(settings, "rich_image_anchor_min_score", 0.34)),
            max_images=max_images,
        )
        _record_anchor_outcomes(metadata, outcomes)
    if new_content == cleaned_content:
        return cleaned_content

    message = getattr(response, "message", None)
    if message is not None and isinstance(getattr(message, "content", None), str):
        message.content = new_content
    return new_content
