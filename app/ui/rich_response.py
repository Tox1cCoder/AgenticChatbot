"""Pure presentation view model for ordered inline rich-response rendering.

This module deliberately avoids importing Streamlit so the policy behavior is
unit-testable. The Streamlit renderer in ``demo.py`` consumes
``build_rich_response_view()`` and translates segments into widgets, images,
markdown blocks, etc.

See response_format.md Task 6.
"""

from __future__ import annotations

import hashlib
import html as _html
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from app.core.rich_response import (
    RICH_ITEMS_VERSION,
    RichDisplayPolicy,
    parse_inline_rich_references,
)

#: Maximum display width for an inline article image. Images render at their
#: natural size up to this cap and are never upscaled, so small source images
#: stay crisp instead of being stretched to the full chat-column width.
INLINE_IMAGE_MAX_WIDTH_PX: int = 480


def build_inline_image_html(
    src: str,
    *,
    alt_text: str | None = None,
    caption: str | None = None,
    source_url: str | None = None,
    width: int | None = None,
    height: int | None = None,
    max_width_px: int = INLINE_IMAGE_MAX_WIDTH_PX,
) -> str:
    """Return one responsive loading/loaded/failed article-image component."""
    escaped_src = _html.escape(src or "", quote=True)
    escaped_caption = _html.escape(caption, quote=True) if caption else ""
    escaped_alt = _html.escape(alt_text or caption or "", quote=True)
    source_link = _source_link_html(source_url)
    footer_parts = []
    if escaped_caption:
        footer_parts.append(f'<span class="rich-image-caption">{escaped_caption}</span>')
    if source_link:
        footer_parts.append(source_link)
    footer = (
        '<figcaption style="color:#64748b;font-size:13px;margin-top:4px;">'
        + " · ".join(footer_parts)
        + "</figcaption>"
        if footer_parts
        else ""
    )
    known_width = width if isinstance(width, int) and width > 0 else None
    known_height = height if isinstance(height, int) and height > 0 else None
    display_width = min(known_width, int(max_width_px)) if known_width else int(max_width_px)
    aspect_style = (
        f"aspect-ratio:{known_width} / {known_height};"
        if known_width and known_height
        else "min-height:120px;"
    )
    wrapper_style = (
        f"position:relative;display:inline-block;width:min({display_width}px, 100%);"
        f"{aspect_style}"
    )
    img_style = (
        f"max-width:min({int(max_width_px)}px, 100%);width:auto;height:auto;"
        "position:relative;display:block;margin-left:auto;margin-right:auto;"
        "border-radius:8px;cursor:zoom-in;"
    )
    component_id = hashlib.sha256((src or "").encode("utf-8")).hexdigest()[:12]
    skeleton = (
        '<div data-role="skeleton" aria-hidden="true" style="position:absolute;inset:0;'
        "border-radius:8px;background:linear-gradient(90deg,#f1f5f9,#e2e8f0,#f1f5f9);"
        '"></div>'
    )
    onload = (
        "const f=this.closest('figure');f.dataset.state='loaded';"
        "const s=f.querySelector('[data-role=skeleton]');if(s)s.remove();"
        "if(!this.dataset.ratio)this.parentElement.style.minHeight='0'"
    )
    onerror = "this.closest('figure').remove()"
    img = (
        f'<img src="{escaped_src}" alt="{escaped_alt}" class="img-thumb" '
        f'data-ratio="{"known" if known_width and known_height else ""}" '
        f'loading="lazy" title="Click to view full size" style="{img_style}" '
        f'onload="{onload}" onerror="{onerror}" />'
    )
    return (
        f'<figure id="rich-image-{component_id}" data-state="loading" '
        'style="margin:8px 0;text-align:center;">'
        f'<div class="rich-image-media" style="{wrapper_style}">{skeleton}{img}</div>'
        f"{footer}</figure>"
    )


#: Maximum columns in one inline group row. Additional approved cells wrap to
#: later rows instead of being discarded.
INLINE_IMAGE_GROUP_MAX_CELLS: int = 3


def build_inline_image_group_html(
    cells: list[dict[str, Any]],
    *,
    alt_text: str | None = None,
    max_width_px: int = INLINE_IMAGE_MAX_WIDTH_PX,
) -> str:
    """Return one responsive figure holding a row of source-linked image cells.

    A single-cell group renders as one ordinary image. A cell whose image fails
    to load is swapped for a neutral in-place block, so sibling cells and the
    surrounding prose are unaffected.
    """
    usable = [
        cell
        for cell in (cells or [])
        if isinstance(cell, dict)
        and (str(cell.get("url") or "").strip() or cell.get("_load_failed") is True)
    ]
    if not usable:
        return ""
    if len(usable) == 1:
        cell = usable[0]
        if cell.get("_load_failed") is True:
            group_alt = _html.escape(alt_text or "", quote=True)
            return (
                f'<figure data-state="failed" aria-label="{group_alt}" '
                'style="margin:8px 0;width:min(480px, 100%);">'
                '<div data-role="cell" data-state="failed" style="min-width:0;">'
                '<div data-role="cell-fallback" style="display:block;padding:12px;'
                'border-radius:8px;background:#f1f5f9;color:#64748b;font-size:12px;'
                'text-align:center;">Visual unavailable</div></div></figure>'
            )
        return build_inline_image_html(
            str(cell.get("url") or ""),
            alt_text=str(cell.get("description") or alt_text or ""),
            source_url=cell.get("source_url"),
            width=cell.get("width"),
            height=cell.get("height"),
            max_width_px=max_width_px,
        )

    group_alt = _html.escape(alt_text or "", quote=True)
    # Per-cell failure reveals only that cell's fallback in place. It also removes
    # the failed cell's source attribution so no caption is stranded without its
    # image. The row never removes or reorders sibling cells.
    onerror = (
        "const c=this.closest('[data-role=cell]');"
        "c.dataset.state='failed';"
        "c.querySelector('[data-role=cell-fallback]').style.display='block';"
        "const cap=c.querySelector('[data-role=cell-caption]');"
        "if(cap)cap.remove();"
        "this.remove();"
    )
    rendered: list[str] = []
    cell_basis = 100 / INLINE_IMAGE_GROUP_MAX_CELLS
    for cell in usable:
        load_failed = cell.get("_load_failed") is True
        fallback_display = "block" if load_failed else "none"
        fallback = (
            f'<div data-role="cell-fallback" style="display:{fallback_display};padding:12px;'
            "border-radius:8px;background:#f1f5f9;color:#64748b;font-size:12px;"
            'text-align:center;">Visual unavailable</div>'
        )
        if load_failed:
            rendered.append(
                '<div data-role="cell" data-state="failed" '
                f'style="flex:1 1 calc({cell_basis:.3f}% - 8px);min-width:0;">'
                f"{fallback}</div>"
            )
            continue
        src = _html.escape(str(cell.get("url") or ""), quote=True)
        cell_alt = _html.escape(str(cell.get("description") or alt_text or ""), quote=True)
        link = _source_link_html(cell.get("source_url"))
        caption = (
            f'<div data-role="cell-caption" '
            f'style="color:#64748b;font-size:12px;margin-top:4px;">{link}</div>'
            if link
            else ""
        )
        rendered.append(
            f'<div data-role="cell" '
            f'style="flex:1 1 calc({cell_basis:.3f}% - 8px);min-width:0;">'
            f'<img src="{src}" alt="{cell_alt}" class="img-thumb" loading="lazy" '
            f'style="width:100%;height:auto;border-radius:8px;cursor:zoom-in;" '
            f'onerror="{onerror}" />{fallback}{caption}</div>'
        )
    row = "".join(rendered)
    return (
        f'<figure data-state="loaded" aria-label="{group_alt}" '
        f'style="margin:8px 0;display:flex;flex-wrap:wrap;gap:8px;'
        f'align-items:flex-start;'
        f'width:min({int(max_width_px) * 2}px, 100%);">{row}</figure>'
    )


def _source_link_html(source_url: str | None) -> str:
    if not isinstance(source_url, str) or not source_url.strip():
        return ""
    parsed = urlsplit(source_url.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return ""
    escaped_url = _html.escape(source_url.strip(), quote=True)
    escaped_label = _html.escape(f"Source: {parsed.hostname}")
    return (
        f'<a href="{escaped_url}" target="_blank" rel="noopener noreferrer">'
        f"{escaped_label}</a>"
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
    from app.core.rich_response import _iter_inline_rich_marker_matches, _strip_fenced_code_blocks

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
        matches = list(_iter_inline_rich_marker_matches(line))
        if not matches:
            buffer.append(line)
            continue

        cursor = 0
        for match in matches:
            before = line[cursor : match.start()]
            if before:
                buffer.append(before)
            _flush_markdown()
            candidate_id = match.group(1)
            referenced.add(candidate_id)
            item = items_by_id.get(candidate_id)
            if item is not None:
                segments.append(RichSegment(kind="rich", item=item))
            else:
                segments.append(RichSegment(kind="unavailable", item_id=candidate_id))
            cursor = match.end()
        trailing = line[cursor:]
        if trailing:
            buffer.append(trailing)
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
    "INLINE_IMAGE_GROUP_MAX_CELLS",
    "RichResponseView",
    "RichSegment",
    "RichStreamState",
    "build_inline_image_group_html",
    "build_inline_image_html",
    "build_rich_response_view",
    "parse_inline_rich_references",
]
