from __future__ import annotations

from pydantic import BaseModel, ConfigDict


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
