from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import VisualAnswerEvaluationCase, WebResearchEvaluationCase
from .metrics import evaluate_case, evaluate_visual_case


def evaluate_cases(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [WebResearchEvaluationCase.model_validate(item) for item in raw]
    results = [{"case_id": case.case_id, "metrics": evaluate_case(case)} for case in cases]
    return _summary(results)


def evaluate_visual_cases(path: str | Path) -> dict[str, Any]:
    """Score recorded visual answers.

    A case may name an answer ``fixture`` (a JSON file with ``content`` and
    ``rich_items``, path relative to the working directory) instead of
    inlining ``answer_content`` and ``rich_items``.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [VisualAnswerEvaluationCase.model_validate(_with_fixture(item)) for item in raw]
    results = [{"case_id": case.case_id, "metrics": evaluate_visual_case(case)} for case in cases]
    return _summary(results)


def _with_fixture(item: dict[str, Any]) -> dict[str, Any]:
    fixture_path = item.get("fixture")
    if not fixture_path:
        return item
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    case = {key: value for key, value in item.items() if key != "fixture"}
    case.setdefault("answer_content", fixture["content"])
    case.setdefault("rich_items", fixture["rich_items"])
    return case


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = all(all(value == 1.0 for value in result["metrics"].values()) for result in results)
    return {"passed": passed, "case_count": len(results), "results": results}
