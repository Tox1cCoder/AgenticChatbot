from __future__ import annotations

from .contracts import WebResearchEvaluationCase


def evaluate_case(case: WebResearchEvaluationCase) -> dict[str, float]:
    sources = set(case.offered_source_ids)
    images = set(case.offered_image_ids)
    source_membership = all(source_id in sources for source_id in case.cited_source_ids)
    image_membership = all(image_id in images for image_id in case.selected_image_ids)
    zero_image_ok = not case.selected_image_ids if case.expect_zero_images else True
    return {
        "source_membership": float(source_membership),
        "image_membership": float(image_membership),
        "zero_image_behavior": float(zero_image_ok),
        "stream_history_parity": float(case.stream_source_ids == case.history_source_ids),
    }
