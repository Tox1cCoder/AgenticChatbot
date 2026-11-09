"""``create_answer_visual``: the server-owned path to a static answer visual.

The tool validates the model's content, binds its evidence to this turn's
admitted web sources, and returns a marker. The validated descriptor rides
in the tool result under ``rich_item`` only until tool execution splits it
into the internal artifact; the model never reads it back.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from app.core.answer_visuals import (
    AnswerVisualError,
    AnswerVisualSpec,
    EvidenceRef,
    ResolvedEvidence,
    build_answer_visual,
)

from .tool_context import get_tool_context

CREATE_ANSWER_VISUAL_TOOL_NAME = "create_answer_visual"

#: Key under which the tool result carries its validated descriptor to tool
#: execution. Stripped before the result becomes model-facing text.
RICH_ITEM_RESULT_KEY = "rich_item"

CREATE_ANSWER_VISUAL_DESCRIPTION = (
    "Create a compact static visual for the answer and get its marker. Use it only "
    "when a visual says something prose or a plain Markdown table cannot: 'steps' "
    "for an ordered procedure, 'bars' for numbers on one shared unit and scale, "
    "'comparison_table' for rows that each need a status badge with a reason. "
    "Ordinary comparisons stay Markdown tables.\n\n"
    "Supply content, not styling. Include only the fields of the chosen variant. "
    "bars: one unit per visual (separate visuals for different units), values are "
    "nonnegative, the scale always starts at zero, and basis says whether values "
    "are measured, calculated, or illustrative; calculated and illustrative need "
    "assumptions. Measured values need evidence. Cite with evidence "
    "[{kind:'web', id:'S#'}] using only IDs web_search returned this turn.\n\n"
    "Limits: 12 steps, bars, or rows; 6 columns; labels 160 characters; notes, "
    "reasons, and captions 500. Place the returned marker on its own line beside "
    "the prose it supports, and keep the key conclusion in that prose."
)


class CreateAnswerVisualInput(BaseModel):
    visual: AnswerVisualSpec = Field(description="The visual's content.")


def _error(message: str) -> str:
    return json.dumps(
        {
            "status": "error",
            "error_type": "validation",
            "retryable": True,
            "message": message,
        },
        ensure_ascii=False,
    )


def _validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors()[:3]:
        location = ".".join(str(part) for part in error.get("loc", ()))
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return _error("; ".join(parts) or "invalid visual")


def _resolve_web_evidence(ref: EvidenceRef) -> ResolvedEvidence | None:
    session = get_tool_context().web_research_session
    registry = getattr(session, "source_registry", None)
    if registry is None:
        return None
    record = registry.resolve(ref.id)
    if record is None:
        return None
    return ResolvedEvidence(title=record.title, url=str(record.url))


def create_answer_visual_tool() -> StructuredTool:
    async def create_answer_visual(visual: Any) -> str:
        context = get_tool_context()
        turn_key = context.logical_turn_id or context.conversation_id or "turn"
        try:
            item = build_answer_visual(
                visual, resolve_evidence=_resolve_web_evidence, turn_key=turn_key
            )
        except AnswerVisualError as exc:
            return _error(str(exc))
        return json.dumps(
            {
                "status": "ok",
                "rich_item_id": item["id"],
                "marker": f"<!--rich:{item['id']}-->",
                "variant": item["payload"]["variant"],
                "next": (
                    "Put the marker on its own line beside the prose it supports. "
                    "Do not restate the visual's data as a second table."
                ),
                RICH_ITEM_RESULT_KEY: item,
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        coroutine=create_answer_visual,
        name=CREATE_ANSWER_VISUAL_TOOL_NAME,
        description=CREATE_ANSWER_VISUAL_DESCRIPTION,
        args_schema=CreateAnswerVisualInput,
        handle_validation_error=_validation_error,
        metadata={
            "tool_origin": "internal",
            "qualified_tool_id": f"internal::{CREATE_ANSWER_VISUAL_TOOL_NAME}",
        },
    )


def split_rich_item_result(result: Any) -> tuple[Any, dict[str, Any] | None]:
    """Separate the descriptor from a ``create_answer_visual`` result.

    Returns the model-facing result and the descriptor, or the result
    unchanged with ``None`` when it carries none (an error, or text that was
    already split).
    """
    if not isinstance(result, str):
        return result, None
    try:
        parsed = json.loads(result)
    except (TypeError, ValueError):
        return result, None
    if not isinstance(parsed, dict) or not isinstance(parsed.get(RICH_ITEM_RESULT_KEY), dict):
        return result, None
    descriptor = parsed.pop(RICH_ITEM_RESULT_KEY)
    return json.dumps(parsed, ensure_ascii=False), descriptor


__all__ = [
    "CREATE_ANSWER_VISUAL_DESCRIPTION",
    "CREATE_ANSWER_VISUAL_TOOL_NAME",
    "CreateAnswerVisualInput",
    "create_answer_visual_tool",
    "split_rich_item_result",
]
