from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import WebResearchEvaluationCase
from .metrics import evaluate_case


def evaluate_cases(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [WebResearchEvaluationCase.model_validate(item) for item in raw]
    results = [{"case_id": case.case_id, "metrics": evaluate_case(case)} for case in cases]
    passed = all(all(value == 1.0 for value in result["metrics"].values()) for result in results)
    return {"passed": passed, "case_count": len(results), "results": results}
