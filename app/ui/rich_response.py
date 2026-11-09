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
import math
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from app.core.rich_projection import project_rich_content, project_rich_items
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
        f"position:relative;display:inline-block;width:min({display_width}px, 100%);{aspect_style}"
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


#: Narrowest a group cell may shrink before it wraps to its own row, so a
#: comparison stacks on a phone instead of shrinking to thumbnails.
INLINE_IMAGE_GROUP_MIN_CELL_PX: int = 220

_GROUP_FALLBACK_STYLE = (
    "padding:12px;border-radius:8px;background:#f1f5f9;color:#64748b;"
    "font-size:12px;text-align:center;"
)
_GROUP_NOTE_STYLE = "flex-basis:100%;color:#64748b;font-size:13px;margin-top:4px;"


def build_inline_image_group_html(
    cells: list[dict[str, Any]],
    *,
    alt_text: str | None = None,
    max_width_px: int = INLINE_IMAGE_MAX_WIDTH_PX,
    caption: str | None = None,
    caption_evidence: list[dict[str, Any]] | None = None,
    comparison_unavailable: bool = False,
) -> str:
    """Return one responsive figure holding a row of source-linked image cells.

    A plain single-cell group renders as one ordinary image. A cell whose image
    fails to load is swapped for a neutral in-place block, so sibling cells and
    the surrounding prose are unaffected.

    Source-figure fields: each cell's ``label`` is shown with its image and its
    ``alt_text`` becomes the img ``alt``; ``caption`` is shown once for the
    whole figure, followed by ``caption_evidence`` links. A figure marked
    ``comparison_unavailable`` drops the caption (its conclusion no longer
    holds) and says so in a visible note.
    """
    usable = [
        cell
        for cell in (cells or [])
        if isinstance(cell, dict)
        and (str(cell.get("url") or "").strip() or cell.get("_load_failed") is True)
    ]
    if not usable:
        return ""
    is_figure = (
        comparison_unavailable
        or bool(_visual_text(caption))
        or bool(caption_evidence)
        or any(_visual_text(cell.get("label")) for cell in usable)
    )
    if is_figure:
        return _labelled_figure_html(
            usable,
            alt_text=alt_text,
            max_width_px=max_width_px,
            caption=caption,
            caption_evidence=caption_evidence,
            comparison_unavailable=comparison_unavailable,
        )
    if len(usable) == 1:
        cell = usable[0]
        if cell.get("_load_failed") is True:
            group_alt = _html.escape(alt_text or "", quote=True)
            return (
                f'<figure data-state="failed" aria-label="{group_alt}" '
                'style="margin:8px 0;width:min(480px, 100%);">'
                '<div data-role="cell" data-state="failed" style="min-width:0;">'
                '<div data-role="cell-fallback" style="display:block;padding:12px;'
                "border-radius:8px;background:#f1f5f9;color:#64748b;font-size:12px;"
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
        f"align-items:flex-start;"
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
    return f'<a href="{escaped_url}" target="_blank" rel="noopener noreferrer">{escaped_label}</a>'


# Per-cell failure reveals only that cell's fallback in place and removes the
# failed cell's source attribution, so no source is stranded without its image.
# The label stays: it names what is missing. Siblings are never touched.
_GROUP_CELL_ONERROR = (
    "const c=this.closest('[data-role=cell]');"
    "c.dataset.state='failed';"
    "c.querySelector('[data-role=cell-fallback]').style.display='block';"
    "const cap=c.querySelector('[data-role=cell-caption]');"
    "if(cap)cap.remove();"
    "this.remove();"
)


def _labelled_figure_cell_html(cell: dict[str, Any], *, alt_text: str | None, style: str) -> str:
    label = _visual_text(cell.get("label"))
    label_html = (
        f'<div data-role="cell-label" style="font-size:13px;font-weight:600;margin-top:4px;">'
        f"{label}</div>"
        if label
        else ""
    )
    if cell.get("_load_failed") is True:
        return (
            f'<div data-role="cell" data-state="failed" style="{style}">'
            f'<div data-role="cell-fallback" style="display:block;{_GROUP_FALLBACK_STYLE}">'
            f"Image unavailable</div>{label_html}</div>"
        )
    src = _html.escape(str(cell.get("url") or ""), quote=True)
    cell_alt = _visual_text(cell.get("alt_text") or cell.get("description") or alt_text)
    link = _source_link_html(cell.get("source_url"))
    source_html = (
        f'<div data-role="cell-caption" style="color:#64748b;font-size:12px;">{link}</div>'
        if link
        else ""
    )
    return (
        f'<div data-role="cell" style="{style}">'
        f'<img src="{src}" alt="{cell_alt}" class="img-thumb" loading="lazy" '
        'style="width:100%;height:auto;border-radius:8px;cursor:zoom-in;" '
        f'onerror="{_GROUP_CELL_ONERROR}" />'
        f'<div data-role="cell-fallback" style="display:none;{_GROUP_FALLBACK_STYLE}">'
        f"Image unavailable</div>{label_html}{source_html}</div>"
    )


def _labelled_figure_html(
    cells: list[dict[str, Any]],
    *,
    alt_text: str | None,
    max_width_px: int,
    caption: str | None,
    caption_evidence: list[dict[str, Any]] | None,
    comparison_unavailable: bool,
) -> str:
    """A source figure: labelled, source-linked cells and one figure caption."""
    cell_basis = 100 / INLINE_IMAGE_GROUP_MAX_CELLS
    cell_style = (
        f"flex:1 1 calc({cell_basis:.3f}% - 8px);"
        f"min-width:min({INLINE_IMAGE_GROUP_MIN_CELL_PX}px, 100%);"
        f"max-width:{int(max_width_px)}px;"
    )
    row = "".join(
        _labelled_figure_cell_html(cell, alt_text=alt_text, style=cell_style) for cell in cells
    )
    if comparison_unavailable:
        footer = (
            f'<figcaption data-role="comparison-unavailable" style="{_GROUP_NOTE_STYLE}">'
            "Comparison unavailable: one of the compared images could not be kept."
            "</figcaption>"
        )
    else:
        caption_text = _visual_text(caption)
        links = _evidence_links_html(caption_evidence)
        footer = (
            f'<figcaption data-role="figure-caption" style="{_GROUP_NOTE_STYLE}">'
            f"{caption_text}{links}</figcaption>"
            if caption_text or links
            else ""
        )
    group_alt = _visual_text(alt_text)
    return (
        f'<figure data-state="loaded" data-role="source-figure" aria-label="{group_alt}" '
        'style="margin:8px 0;display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start;'
        f'width:min({int(max_width_px) * 2}px, 100%);">{row}{footer}</figure>'
    )


# ---------------------------------------------------------------------------
# Answer visuals: static steps, bars and comparison tables
# ---------------------------------------------------------------------------
#
# Everything here is built from escaped text on one line. Streamlit parses the
# Markdown around raw HTML: a blank line would end the HTML block and turn the
# rest into Markdown, and a ``$`` could start math. Colours are translucent or
# inherited so the same markup reads in light and dark themes.

_VISUAL_BOX_STYLE = (
    "border:1px solid rgba(128,128,128,0.35);border-radius:10px;padding:12px 14px;"
    "margin:8px 0;max-width:100%;box-sizing:border-box;"
)
_VISUAL_TITLE_STYLE = "font-weight:600;margin:0 0 8px 0;"
_VISUAL_MUTED_STYLE = "font-size:13px;opacity:0.8;"
_BAR_ROW_STYLE = (
    "display:grid;grid-template-columns:minmax(72px, 28%) minmax(60px, 1fr) auto;"
    "gap:4px 10px;align-items:center;margin:6px 0;"
)
_BAR_TRACK_STYLE = (
    "display:block;height:14px;border-radius:4px;overflow:hidden;"
    "background:rgba(128,128,128,0.2);"
)
_BAR_FILL_STYLE = "display:block;height:100%;border-radius:4px;background:#3b82f6;"
#: RGB triples for assessment badges. The badge always prints its label, so
#: colour only reinforces meaning and never carries it alone.
_ASSESSMENT_RGB: dict[str, str] = {
    "positive": "34,197,94",
    "neutral": "148,163,184",
    "caution": "245,158,11",
    "negative": "239,68,68",
}


def _visual_text(value: Any) -> str:
    """Escaped single-line text, safe inside Streamlit's raw HTML block."""
    if value is None:
        return ""
    collapsed = " ".join(str(value).split())
    return _html.escape(collapsed, quote=True).replace("$", "&#36;")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def format_visual_number(value: float) -> str:
    """Whole numbers without a decimal point, as the stored fallback prints them."""
    return str(int(value)) if value.is_integer() else format(value, ".15g")


def _http_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or any(char.isspace() for char in candidate):
        return None
    parsed = urlsplit(candidate)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    return candidate


def _evidence_link_html(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    url = _http_url(entry.get("url"))
    if url is None:
        return ""
    label = _visual_text(entry.get("label")) or _visual_text(urlsplit(url).hostname)
    title = _visual_text(entry.get("title"))
    title_attr = f' title="{title}"' if title else ""
    return (
        f'<a href="{_visual_text(url)}"{title_attr} target="_blank" '
        f'rel="noopener noreferrer">{label}</a>'
    )


def _join_links(links: list[str]) -> str:
    present = [link for link in links if link]
    if not present:
        return ""
    return ' <span data-role="citations" style="font-size:12px;">' + ", ".join(present) + "</span>"


def _evidence_links_html(entries: Any) -> str:
    """Links for a figure's ``caption_evidence`` list."""
    return _join_links([_evidence_link_html(entry) for entry in entries or []])


def _citation_links_html(citations: Any, evidence: list[Any]) -> str:
    """Each 1-based citation ``n`` links ``evidence[n - 1]`` by its label."""
    links: list[str] = []
    seen: set[int] = set()
    for index in citations if isinstance(citations, list) else []:
        if isinstance(index, bool) or not isinstance(index, int) or index in seen:
            continue
        if not 1 <= index <= len(evidence):
            continue
        seen.add(index)
        links.append(_evidence_link_html(evidence[index - 1]))
    return _join_links(links)


def _steps_body_html(payload: dict[str, Any], evidence: list[Any]) -> str | None:
    items: list[str] = []
    for step in payload.get("steps") or []:
        if not isinstance(step, dict):
            continue
        title = _visual_text(step.get("title"))
        if not title:
            continue
        note = _visual_text(step.get("note"))
        note_html = f'<div style="{_VISUAL_MUTED_STYLE}">{note}</div>' if note else ""
        cites = _citation_links_html(step.get("citations"), evidence)
        items.append(
            f'<li style="margin:4px 0;"><span style="font-weight:500;">{title}</span>'
            f"{cites}{note_html}</li>"
        )
    if not items:
        return None
    return f'<ol style="margin:0;padding-left:1.5em;">{"".join(items)}</ol>'


def _bar_row_html(bar: Any, *, maximum: float, unit: str, evidence: list[Any]) -> str:
    if not isinstance(bar, dict):
        return ""
    label = _visual_text(bar.get("label"))
    value = _finite_number(bar.get("value"))
    if not label or value is None:
        return ""
    width = min(max(value / maximum, 0.0), 1.0) * 100
    value_text = format_visual_number(value) + (f" {unit}" if unit else "")
    cites = _citation_links_html(bar.get("citations"), evidence)
    return (
        f'<div data-role="bar" style="{_BAR_ROW_STYLE}">'
        f'<span style="overflow-wrap:anywhere;">{label}</span>'
        f'<span aria-hidden="true" style="{_BAR_TRACK_STYLE}">'
        f'<span data-role="bar-fill" style="{_BAR_FILL_STYLE}width:{width:.2f}%;"></span></span>'
        f'<span style="font-variant-numeric:tabular-nums;">{value_text}{cites}</span></div>'
    )


def _bars_body_html(payload: dict[str, Any], evidence: list[Any]) -> str | None:
    """Every bar on one zero-based scale taken from ``payload.scale``.

    The scale is never derived from the bars, so 100 and 101 look nearly equal
    and the visual cannot exaggerate a difference the data does not have.
    """
    maximum = _finite_number((payload.get("scale") or {}).get("max"))
    basis = _visual_text(payload.get("basis"))
    if maximum is None or maximum <= 0 or not basis:
        return None
    unit = _visual_text(payload.get("unit"))
    rows = "".join(
        _bar_row_html(bar, maximum=maximum, unit=unit, evidence=evidence)
        for bar in payload.get("bars") or []
    )
    if not rows:
        return None
    scale_text = f"0 to {format_visual_number(maximum)}" + (f" {unit}" if unit else "")
    assumptions = _visual_text(payload.get("assumptions"))
    assumptions_html = (
        f'<div data-role="bar-assumptions" style="{_VISUAL_MUTED_STYLE}">'
        f"Assumptions: {assumptions}</div>"
        if assumptions
        else ""
    )
    return (
        f'<div role="list">{rows}</div>'
        f'<div data-role="bar-basis" style="{_VISUAL_MUTED_STYLE}margin-top:6px;">'
        f"Basis: {basis} · Scale: {scale_text}</div>{assumptions_html}"
    )


def _assessment_cell_html(assessment: Any, evidence: list[Any]) -> str:
    if not isinstance(assessment, dict):
        return "<td></td>"
    status = str(assessment.get("status") or "")
    if status not in _ASSESSMENT_RGB:
        status = "neutral"
    rgb = _ASSESSMENT_RGB[status]
    label = _visual_text(assessment.get("label")) or _visual_text(status.capitalize())
    reason = _visual_text(assessment.get("reason"))
    cites = _citation_links_html(assessment.get("citations"), evidence)
    badge = (
        f'<span data-role="assessment-badge" data-status="{status}" title="{status}" '
        "style=\"display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;"
        f'font-weight:600;background:rgba({rgb},0.18);border:1px solid rgba({rgb},0.6);">'
        f"{label}</span>"
    )
    return f"<td>{badge} {reason}{cites}</td>"


def _table_row_html(
    row: dict[str, Any], *, column_count: int, has_assessment: bool, evidence: list[Any]
) -> str:
    cells = [cell for cell in row.get("cells") or [] if isinstance(cell, dict)]
    rendered = [f'<th scope="row">{_visual_text(row.get("label"))}</th>']
    for index in range(column_count):
        cell = cells[index] if index < len(cells) else {}
        text = _visual_text(cell.get("text"))
        rendered.append(f"<td>{text}{_citation_links_html(cell.get('citations'), evidence)}</td>")
    if has_assessment:
        rendered.append(_assessment_cell_html(row.get("assessment"), evidence))
    return f"<tr>{''.join(rendered)}</tr>"


def _comparison_table_body_html(payload: dict[str, Any], evidence: list[Any]) -> str | None:
    columns = [_visual_text(column) for column in payload.get("columns") or []]
    rows = [row for row in payload.get("rows") or [] if isinstance(row, dict)]
    if not columns or not rows:
        return None
    has_assessment = any(isinstance(row.get("assessment"), dict) for row in rows)
    headers = ['<th scope="col">Item</th>'] + [f'<th scope="col">{c}</th>' for c in columns]
    if has_assessment:
        headers.append('<th scope="col">Assessment</th>')
    body = "".join(
        _table_row_html(
            row, column_count=len(columns), has_assessment=has_assessment, evidence=evidence
        )
        for row in rows
    )
    region_label = _visual_text(payload.get("title")) or "Comparison table"
    return (
        f'<div data-role="table-scroll" role="region" aria-label="{region_label}" tabindex="0" '
        'style="overflow-x:auto;max-width:100%;">'
        '<table style="border-collapse:collapse;margin:0;">'
        f"<thead><tr>{''.join(headers)}</tr></thead><tbody>{body}</tbody></table></div>"
    )


_ANSWER_VISUAL_BODIES = {
    "steps": _steps_body_html,
    "bars": _bars_body_html,
    "comparison_table": _comparison_table_body_html,
}


def build_answer_visual_html(item: Any) -> str | None:
    """Static HTML for one ``answer_visual`` item, or ``None``.

    ``None`` means "render this item's ``fallback_markdown`` instead": an
    unknown variant, a payload this renderer cannot draw faithfully, or a record
    that is not an answer visual at all.
    """
    if not isinstance(item, dict) or item.get("type") != "answer_visual":
        return None
    payload = item.get("payload")
    if not isinstance(payload, dict):
        return None
    variant = payload.get("variant")
    body_builder = _ANSWER_VISUAL_BODIES.get(variant) if isinstance(variant, str) else None
    if body_builder is None:
        return None
    # Citations index this list, so it is never filtered or reordered.
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
    body = body_builder(payload, evidence)
    if not body:
        return None
    title = _visual_text(payload.get("title") or item.get("title"))
    title_html = (
        f'<div data-role="visual-title" style="{_VISUAL_TITLE_STYLE}">{title}</div>'
        if title
        else ""
    )
    caption = _visual_text(payload.get("caption"))
    caption_cites = _citation_links_html(payload.get("caption_citations"), evidence)
    caption_html = (
        f'<div data-role="visual-caption" style="{_VISUAL_MUTED_STYLE}margin-top:8px;">'
        f"{caption}{caption_cites}</div>"
        if caption or caption_cites
        else ""
    )
    return (
        f'<div class="rich-answer-visual" data-variant="{variant}" '
        f'style="{_VISUAL_BOX_STYLE}">{title_html}{body}{caption_html}</div>'
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


def _renderable_rich_items(metadata: dict[str, Any]) -> list[Any]:
    raw_items = metadata.get("rich_items") or []
    if not metadata.get("fork_history"):
        return raw_items
    # A row copied into a fork is history: it never opens a live widget
    # connection, even if a descriptor reached it by a path the fork snapshot
    # missed.
    return [
        item
        for item in raw_items
        if not (isinstance(item, dict) and item.get("type") == "live_widget")
    ]


def build_rich_response_view(
    content: str | None,
    metadata: dict[str, Any] | None,
    *,
    answer_visuals: bool = False,
    source_figures: bool = False,
) -> RichResponseView:
    """Return an ordered presentation view for an assistant message.

    Behavior:

    * If ``metadata.get("rich_items_version") == 1``, the body is split using
      block-level markers; selected images appear at their marker, unknown
      ids produce ``unavailable`` segments, and unreferenced items with
      ``display_policy == inline_or_append`` are returned in ``append_items``.
      Unreferenced image items (``inline_only``) are never appended.
    * Item families the renderer does not support natively -- by default
      answer visuals and composed figures -- arrive through the shared
      fallback: Markdown for a visual, text plus a legacy image for a figure.
    * If the message is not a v1 message but carries legacy
      ``metadata["images"]``, the caller is told to fall back to its existing
      image-gallery renderer via ``use_legacy_image_gallery=True``.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    body = content or ""
    is_v1 = metadata.get("rich_items_version") == RICH_ITEMS_VERSION
    raw_items = _renderable_rich_items(metadata)
    if is_v1:
        capabilities = {"answer_visuals": answer_visuals, "source_figures": source_figures}
        body = project_rich_content(body, raw_items, rich=True, **capabilities)
        raw_items = project_rich_items(raw_items, **capabilities)
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
    #: Item families the caller renders natively; see ``build_rich_response_view``.
    answer_visuals: bool = False
    source_figures: bool = False

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
        return build_rich_response_view(
            self.accumulated_text,
            metadata,
            answer_visuals=self.answer_visuals,
            source_figures=self.source_figures,
        )


# Re-export the parser for clients that want raw reference IDs.
__all__ = [
    "INLINE_IMAGE_GROUP_MAX_CELLS",
    "RichResponseView",
    "RichSegment",
    "RichStreamState",
    "build_answer_visual_html",
    "build_inline_image_group_html",
    "build_inline_image_html",
    "build_rich_response_view",
    "format_visual_number",
    "parse_inline_rich_references",
]
