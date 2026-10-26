from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

VisualKind = Literal[
    "steps", "bars", "comparison_table", "figure", "image", "markdown_table", "none"
]


class WebResearchEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    offered_source_ids: tuple[str, ...] = ()
    cited_source_ids: tuple[str, ...] = ()
    offered_image_ids: tuple[str, ...] = ()
    selected_image_ids: tuple[str, ...] = ()
    expect_zero_images: bool = False
    stream_source_ids: tuple[str, ...] = ()
    history_source_ids: tuple[str, ...] = ()


class VisualAnswerEvaluationCase(BaseModel):
    """One recorded answer judged for its visuals.

    ``expected_visual_kinds`` is the reviewer's judgement of the smallest
    useful presentation; ``none`` means prose only. Evidence support checks
    membership in the admitted sources, never whether a claim is true.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    answer_content: str = ""
    rich_items: tuple[dict[str, Any], ...] = ()
    expected_visual_kinds: tuple[VisualKind, ...] = Field(min_length=1)
    admitted_source_urls: tuple[str, ...] = ()
    selected_image_ids: tuple[str, ...] = ()
    irrelevant_image_ids: tuple[str, ...] = ()
    latency_ms: float | None = Field(default=None, ge=0)
    latency_budget_ms: float | None = Field(default=None, gt=0)
    tokens: int | None = Field(default=None, ge=0)
    token_budget: int | None = Field(default=None, gt=0)
