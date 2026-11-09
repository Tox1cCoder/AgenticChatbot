from __future__ import annotations

import math
import re
from typing import Any

from app.core.rich_response import parse_inline_rich_references
from app.core.source_figures import is_source_figure

from .contracts import VisualAnswerEvaluationCase, WebResearchEvaluationCase

_MARKDOWN_TABLE = re.compile(r"^\|.*\|[ \t]*\n\|[ \t:-]*-{3,}", re.MULTILINE)


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


def evaluate_visual_case(case: VisualAnswerEvaluationCase) -> dict[str, float]:
    """Score one recorded answer's visuals; latency and tokens only when recorded."""
    placed = _placed_items(case)
    metrics = {
        "visual_selection": float(_placed_kinds(case, placed) == set(case.expected_visual_kinds)),
        "irrelevant_image_omission": float(
            not set(case.selected_image_ids) & set(case.irrelevant_image_ids)
        ),
        "evidence_support": float(
            _cited_urls(placed) <= set(case.admitted_source_urls)
        ),
        "numerical_integrity": float(all(_bars_are_sound(item) for item in placed)),
    }
    if case.latency_ms is not None and case.latency_budget_ms is not None:
        metrics["latency_within_budget"] = float(case.latency_ms <= case.latency_budget_ms)
    if case.tokens is not None and case.token_budget is not None:
        metrics["token_cost_within_budget"] = float(case.tokens <= case.token_budget)
    return metrics


def _placed_items(case: VisualAnswerEvaluationCase) -> list[dict[str, Any]]:
    by_id = {item.get("id"): item for item in case.rich_items if isinstance(item, dict)}
    return [
        by_id[item_id]
        for item_id in dict.fromkeys(parse_inline_rich_references(case.answer_content))
        if item_id in by_id
    ]


def _placed_kinds(case: VisualAnswerEvaluationCase, placed: list[dict[str, Any]]) -> set[str]:
    kinds: set[str] = set()
    for item in placed:
        if item.get("type") == "answer_visual":
            kinds.add(str((item.get("payload") or {}).get("variant")))
        elif is_source_figure(item):
            kinds.add("figure")
        elif item.get("type") in {"image", "image_group"}:
            kinds.add("image")
    if _MARKDOWN_TABLE.search(case.answer_content):
        kinds.add("markdown_table")
    return kinds or {"none"}


def _cited_urls(placed: list[dict[str, Any]]) -> set[str]:
    urls: set[str] = set()
    for item in placed:
        payload = item.get("payload") or {}
        for key in ("evidence", "caption_evidence"):
            urls.update(
                str(source.get("url"))
                for source in payload.get(key) or ()
                if isinstance(source, dict) and source.get("url")
            )
    return urls


def _bars_are_sound(item: dict[str, Any]) -> bool:
    payload = item.get("payload") or {}
    if item.get("type") != "answer_visual" or payload.get("variant") != "bars":
        return True
    scale = payload.get("scale") or {}
    maximum = scale.get("max")
    if scale.get("min") != 0 or not isinstance(maximum, (int, float)) or maximum <= 0:
        return False
    if payload.get("basis") != "measured" and not payload.get("assumptions"):
        return False
    fallback = str(payload.get("fallback_markdown") or "")
    for bar in payload.get("bars") or ():
        value = bar.get("value")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
        if not 0 <= value <= maximum:
            return False
        shown = str(int(value)) if float(value).is_integer() else format(value, ".15g")
        if f"| {shown} |" not in fallback:
            return False
    return True
