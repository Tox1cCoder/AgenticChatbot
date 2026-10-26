"""``compose_source_figure``: a labelled figure from inspected web images.

The model names candidates it inspected (``I#``), a label and alt text per
cell, and an optional caption. The server resolves each id to the turn's
protected media and the cell's own source, pins those media until the turn
finishes, and returns a marker. Upstream URLs never enter the figure.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from app.core.answer_visuals import EvidenceRef
from app.core.source_figures import (
    SOURCE_FIGURE_MAX_CELLS,
    FigureCell,
    SourceFigureError,
    build_source_figure,
)

from .answer_visual_tools import RICH_ITEM_RESULT_KEY, _error, _resolve_web_evidence
from .tool_context import get_tool_context

COMPOSE_SOURCE_FIGURE_TOOL_NAME = "compose_source_figure"

COMPOSE_SOURCE_FIGURE_DESCRIPTION = (
    "Compose a figure from web image candidates you have inspected this turn and get "
    "its marker. One image gives a captioned figure; two to four give an ordered "
    "side-by-side comparison. Use it when the reader should compare what images show, "
    "such as two settings or a before and after; comparable views only, not two "
    "pictures that merely share a topic.\n\n"
    "Give each cell an I# id you inspected, a short visible label, and alt text "
    "describing what it shows. The caption explains what to notice; cite factual "
    "caption claims with caption_evidence [{kind:'web', id:'S#'}]. Each cell keeps "
    "its own source automatically.\n\n"
    "Place the returned marker on its own line beside the prose it supports instead "
    "of writing [[image:I#]] for those images, and keep the conclusion in prose."
)


class FigureCellInput(BaseModel):
    image_id: str = Field(pattern=r"^I[1-9][0-9]*$", description="An inspected I# candidate.")
    label: str = Field(min_length=1, max_length=160, description="Visible cell label.")
    alt_text: str = Field(min_length=1, max_length=500, description="What the image shows.")


class ComposeSourceFigureInput(BaseModel):
    cells: list[FigureCellInput] = Field(min_length=1, max_length=SOURCE_FIGURE_MAX_CELLS)
    caption: str | None = Field(default=None, max_length=500)
    caption_evidence: list[EvidenceRef] = Field(default_factory=list, max_length=8)


def _validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors()[:3]:
        location = ".".join(str(part) for part in error.get("loc", ()))
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return _error("; ".join(parts) or "invalid figure")


def create_compose_source_figure_tool() -> StructuredTool:
    async def compose_source_figure(
        cells: list[Any],
        caption: str | None = None,
        caption_evidence: list[Any] | None = None,
    ) -> str:
        context = get_tool_context()
        session = context.web_research_session
        if session is None:
            return _error("web research is unavailable, so there are no images to compose")
        requested = [FigureCellInput.model_validate(cell) for cell in cells]
        refs = [EvidenceRef.model_validate(ref) for ref in caption_evidence or ()]
        try:
            item = _compose(session, requested, caption, refs, context)
        except SourceFigureError as exc:
            return _error(str(exc))
        return json.dumps(
            {
                "status": "ok",
                "rich_item_id": item["id"],
                "marker": f"<!--rich:{item['id']}-->",
                "kind": "comparison" if item["type"] == "image_group" else "single",
                "next": (
                    "Put the marker on its own line beside the prose it supports. "
                    "Do not also select these images with [[image:I#]]."
                ),
                RICH_ITEM_RESULT_KEY: item,
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        coroutine=compose_source_figure,
        name=COMPOSE_SOURCE_FIGURE_TOOL_NAME,
        description=COMPOSE_SOURCE_FIGURE_DESCRIPTION,
        args_schema=ComposeSourceFigureInput,
        handle_validation_error=_validation_error,
        metadata={
            "tool_origin": "internal",
            "qualified_tool_id": f"internal::{COMPOSE_SOURCE_FIGURE_TOOL_NAME}",
        },
    )


def _compose(
    session: Any,
    requested: list[FigureCellInput],
    caption: str | None,
    refs: list[EvidenceRef],
    context: Any,
) -> dict[str, Any]:
    candidate_ids = [cell.image_id for cell in requested]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise SourceFigureError("use each image once per figure")
    images = session.figure_images(candidate_ids)
    evidence = []
    for ref in refs:
        resolved = _resolve_web_evidence(ref)
        if resolved is None:
            raise SourceFigureError(
                f"unknown evidence {ref.id}; cite only S# IDs web_search returned this turn"
            )
        evidence.append(resolved)
    figure_cells = []
    for cell, prepared in zip(requested, images, strict=True):
        payload = prepared.rich_item.get("payload") or {}
        figure_cells.append(
            FigureCell(
                url=str(payload["url"]),
                mime_type=str(payload["mime_type"]),
                source_url=str(payload["source_url"]),
                source_title=prepared.rich_item.get("title"),
                width=payload.get("width"),
                height=payload.get("height"),
                label=cell.label,
                alt_text=cell.alt_text,
            )
        )
    item = build_source_figure(
        figure_cells,
        caption=caption,
        caption_evidence=evidence,
        turn_key=context.logical_turn_id or context.conversation_id or "turn",
    )
    item["_internal"] = {
        "candidate_ids": candidate_ids,
        "reference_ids": [str(prepared.reference_id) for prepared in images],
        "web_source_ids": list(
            dict.fromkeys(
                [*(prepared.record.source_id for prepared in images), *(ref.id for ref in refs)]
            )
        ),
    }
    session.pin_figure(item["id"], candidate_ids)
    return item


__all__ = [
    "COMPOSE_SOURCE_FIGURE_DESCRIPTION",
    "COMPOSE_SOURCE_FIGURE_TOOL_NAME",
    "ComposeSourceFigureInput",
    "create_compose_source_figure_tool",
]
