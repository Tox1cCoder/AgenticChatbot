"""Composed source figures in the existing ``image`` / ``image_group`` family.

A figure is one inspected web image, or an ordered comparison of up to four,
with a visible label per cell, alt text, and one explanatory caption. Every
cell keeps its own source. Media are the turn's protected ``/web-images/``
references; a figure never names an upstream URL.

When a comparison later loses a cell -- externalization, a fork, selection --
:func:`degrade_source_figure` removes the comparison-wide caption, keeps each
surviving cell's label and source, and marks the comparison unavailable, so a
before/after conclusion never stands over a single image.

This module must not import ``rich_response`` or ``rich_image_selection``;
both import it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .answer_visuals import ResolvedEvidence, _source_label, _text

SOURCE_FIGURE_SOURCE = "source_figure"
SOURCE_FIGURE_ID_PREFIX = "figure:"
SOURCE_FIGURE_MAX_CELLS = 4

_LABEL_MAX = 160
_PROSE_MAX = 500
#: Payload fields only a figure-capable client understands.
_ENRICHED_PAYLOAD_KEYS = frozenset(
    {"label", "caption_evidence", "fallback_markdown", "comparison_unavailable"}
)
_ENRICHED_GROUP_KEYS = _ENRICHED_PAYLOAD_KEYS | {"caption"}
_ENRICHED_CELL_KEYS = frozenset({"label", "alt_text"})
_UNAVAILABLE_NOTE = (
    "_The comparison is unavailable because one of its images could not be kept._"
)


class SourceFigureError(ValueError):
    """A composition the tool must refuse, with a message the model can act on."""


@dataclass(frozen=True)
class FigureCell:
    """One resolved cell: protected media plus the source it came from."""

    url: str
    mime_type: str
    source_url: str
    label: str
    alt_text: str
    source_title: str | None = None
    width: int | None = None
    height: int | None = None


def is_source_figure(item: Any) -> bool:
    return isinstance(item, dict) and item.get("source") == SOURCE_FIGURE_SOURCE


def build_source_figure(
    cells: Sequence[FigureCell],
    *,
    caption: str | None,
    caption_evidence: Sequence[ResolvedEvidence],
    turn_key: str,
) -> dict[str, Any]:
    """Return the public rich item for a figure of one to four cells.

    Raises:
        SourceFigureError: the cell count, a repeated cell, or a text bound.
    """
    _check(cells, caption)
    evidence = [
        {"title": source.title, "url": source.url, "label": _source_label(source.url)}
        for source in caption_evidence
    ]
    public_cells = [_public_cell(cell) for cell in cells]
    digest = hashlib.sha256(
        json.dumps(
            {
                "turn": turn_key,
                "cells": [asdict(cell) for cell in cells],
                "caption": caption,
                "evidence": evidence,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()[:24]
    item: dict[str, Any] = {
        "id": f"{SOURCE_FIGURE_ID_PREFIX}{digest}",
        "source": SOURCE_FIGURE_SOURCE,
        "display_policy": "inline_only",
        "provenance": {"provider": SOURCE_FIGURE_SOURCE},
    }
    if caption:
        item["title"] = caption
    if len(public_cells) == 1:
        payload = {key: value for key, value in public_cells[0].items() if key != "alt_text"}
        item.update(type="image", alt_text=cells[0].alt_text)
    else:
        payload = {"items": public_cells}
        item.update(
            type="image_group",
            alt_text=caption or "Comparison: " + ", ".join(cell.label for cell in cells),
        )
    if caption:
        payload["caption"] = caption
    if evidence:
        payload["caption_evidence"] = evidence
    payload["fallback_markdown"] = figure_fallback_markdown(
        payload, image_alt_text=item["alt_text"]
    )
    item["payload"] = payload
    return item


def _check(cells: Sequence[FigureCell], caption: str | None) -> None:
    if not 1 <= len(cells) <= SOURCE_FIGURE_MAX_CELLS:
        raise SourceFigureError(
            f"a figure needs 1 to {SOURCE_FIGURE_MAX_CELLS} images; got {len(cells)}"
        )
    if len({cell.url for cell in cells}) != len(cells):
        raise SourceFigureError("use each image once per figure")
    for index, cell in enumerate(cells):
        if not cell.label.strip() or len(cell.label) > _LABEL_MAX:
            raise SourceFigureError(f"cells[{index}].label must be 1 to {_LABEL_MAX} characters")
        if not cell.alt_text.strip() or len(cell.alt_text) > _PROSE_MAX:
            raise SourceFigureError(
                f"cells[{index}].alt_text must be 1 to {_PROSE_MAX} characters"
            )
    if caption is not None and len(caption) > _PROSE_MAX:
        raise SourceFigureError(f"caption must be at most {_PROSE_MAX} characters")


def _public_cell(cell: FigureCell) -> dict[str, Any]:
    out: dict[str, Any] = {
        "url": cell.url,
        "mime_type": cell.mime_type,
        "source_url": cell.source_url,
        "label": cell.label,
        "alt_text": cell.alt_text,
    }
    if cell.width:
        out["width"] = cell.width
    if cell.height:
        out["height"] = cell.height
    return out


def figure_fallback_markdown(
    payload: dict[str, Any], *, image_alt_text: str | None = None
) -> str:
    """Source-linked text for a figure, built from its public payload.

    A group payload carries alt text per cell; a single image's alt text lives
    on the item, so the caller passes it as ``image_alt_text``.
    """
    if "items" in payload:
        cells = [cell for cell in payload.get("items") or () if isinstance(cell, dict)]
        alt_texts = [cell.get("alt_text") for cell in cells]
    else:
        cells = [payload]
        alt_texts = [image_alt_text]
    caption = payload.get("caption")
    header = "**Figure:**"
    if caption:
        header = f"{header} {_text(caption)}"
        links = [
            f"[{source['label']}]({_url(source['url'])})"
            for source in payload.get("caption_evidence") or ()
        ]
        if links:
            header = f"{header} {' '.join(links)}"
    else:
        header = "**Figure**"
    lines = []
    for cell, alt_text in zip(cells, alt_texts, strict=False):
        text = _text(cell.get("label") or "Image")
        if alt_text:
            text = f"{text} — {_text(alt_text)}"
        source = cell.get("source_url")
        if source:
            text = f"{text} ([{_source_label(source)}]({_url(source)}))"
        lines.append(f"- {text}")
    blocks = [header, "\n".join(lines)]
    if payload.get("comparison_unavailable"):
        blocks.append(_UNAVAILABLE_NOTE)
    return "\n\n".join(block for block in blocks if block)


def _url(url: str) -> str:
    return str(url).replace("(", "%28").replace(")", "%29").replace(" ", "%20")


def degrade_source_figure(
    item: dict[str, Any], surviving_cells: Sequence[dict[str, Any]]
) -> dict[str, Any] | None:
    """Apply the readable-failure rule to a group that lost cells.

    Returns ``None`` when no cell survives, the item itself when none was
    lost, and otherwise a copy holding only the survivors. A source figure's
    copy also loses its comparison-wide caption and says the comparison is
    unavailable; a plain image group just keeps its survivors.
    """
    if not surviving_cells:
        return None
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    original = payload.get("items") or []
    if len(surviving_cells) == len(original):
        return item
    degraded = copy.deepcopy(item)
    degraded_payload = degraded["payload"]
    degraded_payload["items"] = [copy.deepcopy(cell) for cell in surviving_cells]
    if not is_source_figure(item):
        return degraded
    for key in ("caption", "caption_evidence"):
        degraded_payload.pop(key, None)
    degraded.pop("title", None)
    degraded_payload["comparison_unavailable"] = True
    degraded["alt_text"] = "Comparison unavailable: " + ", ".join(
        str(cell.get("label") or "image") for cell in surviving_cells
    )
    degraded_payload["fallback_markdown"] = figure_fallback_markdown(degraded_payload)
    return degraded


def figure_group_as_image(item: dict[str, Any]) -> dict[str, Any]:
    """Turn a one-cell figure group into an ``image`` keeping that cell's meaning."""
    payload = item.get("payload") or {}
    [cell] = payload.get("items") or [{}]
    image = copy.deepcopy(item)
    image["type"] = "image"
    image["alt_text"] = str(cell.get("alt_text") or item.get("alt_text") or "Image")
    image_payload = {key: value for key, value in cell.items() if key != "alt_text"}
    for key in ("caption", "caption_evidence", "comparison_unavailable"):
        if key in payload:
            image_payload[key] = copy.deepcopy(payload[key])
    image_payload["fallback_markdown"] = figure_fallback_markdown(
        image_payload, image_alt_text=image["alt_text"]
    )
    image["payload"] = image_payload
    return image


def legacy_source_figure(item: dict[str, Any]) -> dict[str, Any]:
    """The figure as a pre-figure client reads it: exact legacy payload fields.

    A group of one is only valid as a declared comparison remnant, which an
    older client cannot read, so it becomes an ordinary image.
    """
    cells = (item.get("payload") or {}).get("items")
    if item.get("type") == "image_group" and isinstance(cells, list) and len(cells) < 2:
        item = figure_group_as_image(item)
    legacy = copy.deepcopy(item)
    payload = legacy.get("payload") or {}
    if legacy.get("type") == "image_group":
        for key in _ENRICHED_GROUP_KEYS:
            payload.pop(key, None)
        payload["items"] = [
            {key: value for key, value in cell.items() if key not in _ENRICHED_CELL_KEYS}
            for cell in payload.get("items") or ()
            if isinstance(cell, dict)
        ]
    else:
        for key in _ENRICHED_PAYLOAD_KEYS:
            payload.pop(key, None)
    return legacy


__all__ = [
    "SOURCE_FIGURE_ID_PREFIX",
    "SOURCE_FIGURE_MAX_CELLS",
    "SOURCE_FIGURE_SOURCE",
    "FigureCell",
    "SourceFigureError",
    "build_source_figure",
    "degrade_source_figure",
    "figure_fallback_markdown",
    "figure_group_as_image",
    "is_source_figure",
    "legacy_source_figure",
]
