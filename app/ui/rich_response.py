"""Pure presentation view model for ordered inline rich-response rendering.

This module deliberately avoids importing Streamlit so the policy behavior is
unit-testable. The Streamlit renderer in ``demo.py`` consumes
``build_rich_response_view()`` and translates segments into widgets, images,
markdown blocks, etc.

See response_format.md Task 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.rich_response import (
    RICH_ITEMS_VERSION,
    RichDisplayPolicy,
    parse_inline_rich_references,
)

# ---------------------------------------------------------------------------
# Public view model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RichSegment:
    """One rendered segment of the assistant body.

    ``kind`` is one of:

    * ``"markdown"`` — render ``text`` as ordinary markdown.
    * ``"rich"`` — render ``item`` (a public rich-item record) using the
      type-specific renderer.
    * ``"unavailable"`` — the marker referenced ``item_id`` but no record was
      found in the registry; render a neutral unavailable-content block.
    """

    kind: str
    text: str | None = None
    item: dict[str, Any] | None = None
    item_id: str | None = None


@dataclass
class RichResponseView:
    """Ordered rendering view for a single assistant message."""

    segments: list[RichSegment] = field(default_factory=list)
    append_items: list[dict[str, Any]] = field(default_factory=list)
    referenced_ids: set[str] = field(default_factory=set)
    is_v1: bool = False
    #: True when the message is *not* a v1 rich-items message and contains a
    #: legacy ``metadata["images"]`` gallery; the Streamlit caller should fall
    #: back to its existing gallery rendering for backwards compatibility.
    use_legacy_image_gallery: bool = False


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _split_body_into_segments(
    body: str, items_by_id: dict[str, dict[str, Any]]
) -> tuple[list[RichSegment], set[str]]:
    """Split ``body`` into ordered markdown and rich segments using the marker
    parser. Each successful marker resolves to either a ``rich`` or
    ``unavailable`` segment depending on whether the id exists in
    ``items_by_id``.
    """
    if not body:
        return [], set()
    from app.core.rich_response import _MARKER_LINE_RE, _strip_fenced_code_blocks

    normalized = body.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    in_fence = _strip_fenced_code_blocks(lines)

    segments: list[RichSegment] = []
    referenced: set[str] = set()
    buffer: list[str] = []

    def _flush_markdown() -> None:
        if not buffer:
            return
        text = "\n".join(buffer).strip("\n")
        # Trim leading/trailing blank lines but preserve internal structure.
        if text:
            segments.append(RichSegment(kind="markdown", text=text))
        buffer.clear()

    for line, inside in zip(lines, in_fence, strict=False):
        if inside:
            buffer.append(line)
            continue
        if line.startswith("    ") or line.startswith("\t"):
            buffer.append(line)
            continue
        match = _MARKER_LINE_RE.match(line)
        if not match:
            buffer.append(line)
            continue
        candidate_id = match.group(1)
        _flush_markdown()
        referenced.add(candidate_id)
        item = items_by_id.get(candidate_id)
        if item is not None:
            segments.append(RichSegment(kind="rich", item=item))
        else:
            segments.append(RichSegment(kind="unavailable", item_id=candidate_id))
    _flush_markdown()
    return segments, referenced


def build_rich_response_view(
    content: str | None,
    metadata: dict[str, Any] | None,
) -> RichResponseView:
    """Return an ordered presentation view for an assistant message.

    Behavior:

    * If ``metadata.get("rich_items_version") == 1``, the body is split using
      block-level markers; selected images appear at their marker, unknown
      ids produce ``unavailable`` segments, and unreferenced items with
      ``display_policy == inline_or_append`` are returned in ``append_items``.
      Unreferenced image items (``inline_only``) are never appended.
    * If the message is not a v1 message but carries legacy
      ``metadata["images"]``, the caller is told to fall back to its existing
      image-gallery renderer via ``use_legacy_image_gallery=True``.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    body = content or ""
    is_v1 = metadata.get("rich_items_version") == RICH_ITEMS_VERSION
    raw_items = metadata.get("rich_items") or []
    items_by_id: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            items_by_id[item_id] = item

    if is_v1:
        segments, referenced = _split_body_into_segments(body, items_by_id)
        append_items: list[dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if item_id in referenced:
                continue
            policy = item.get("display_policy")
            if policy != RichDisplayPolicy.inline_or_append.value:
                continue
            append_items.append(item)
        if not segments:
            # Empty content with no markers still needs a placeholder so
            # callers can iterate uniformly.
            segments = [RichSegment(kind="markdown", text=body)]
        return RichResponseView(
            segments=segments,
            append_items=append_items,
            referenced_ids=referenced,
            is_v1=True,
            use_legacy_image_gallery=False,
        )

    # Non-v1 path: emit a single markdown segment and let the caller decide
    # whether to render the legacy gallery from ``metadata["images"]``.
    segments = [RichSegment(kind="markdown", text=body)] if body else []
    has_legacy_images = bool(metadata.get("images"))
    return RichResponseView(
        segments=segments,
        append_items=[],
        referenced_ids=set(),
        is_v1=False,
        use_legacy_image_gallery=has_legacy_images,
    )


# ---------------------------------------------------------------------------
# Live stream state
# ---------------------------------------------------------------------------


@dataclass
class RichStreamState:
    """Mutable container holding the in-progress live-stream state.

    The Streamlit caller appends token deltas to ``accumulated_text``, merges
    incoming ``rich_items`` upserts via ``apply_rich_items_upsert()``, and on
    each refresh calls ``build_view()`` to get the ordered presentation view
    rendered against the current registry. On terminal `complete`, the caller
    calls ``replace_with_finalized()`` to swap in the authoritative
    ``rich_items`` from persisted message metadata.
    """

    accumulated_text: str = ""
    items_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest: bool = False

    def append_text(self, delta: str) -> None:
        if delta:
            self.accumulated_text += delta

    def apply_rich_items_upsert(self, items: list[dict[str, Any]]) -> None:
        for item in items or []:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id:
                continue
            self.items_by_id[item_id] = item

    def replace_with_finalized(self, rich_items: list[dict[str, Any]]) -> None:
        """Replace the transient registry with the authoritative final items.

        Items in the new list overwrite same-id transient records; ids absent
        from the final list are dropped (they were never persisted).
        """
        self.items_by_id = {}
        for item in rich_items or []:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id:
                self.items_by_id[item_id] = item

    def build_view(self) -> RichResponseView:
        metadata = {
            "rich_items_version": RICH_ITEMS_VERSION,
            "rich_items": list(self.items_by_id.values()),
        }
        return build_rich_response_view(self.accumulated_text, metadata)


# Re-export the parser for clients that want raw reference IDs.
__all__ = [
    "RichResponseView",
    "RichSegment",
    "RichStreamState",
    "build_rich_response_view",
    "parse_inline_rich_references",
]
