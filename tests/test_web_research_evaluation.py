from app.evaluation.web_research.contracts import WebResearchEvaluationCase
from app.evaluation.web_research.harness import evaluate_cases
from app.evaluation.web_research.metrics import evaluate_case


def test_membership_rejects_invented_source_and_image_ids() -> None:
    metrics = evaluate_case(
        WebResearchEvaluationCase(
            case_id="invented",
            offered_source_ids=("S1",),
            cited_source_ids=("S2",),
            offered_image_ids=("I1",),
            selected_image_ids=("I2",),
        )
    )
    assert metrics["source_membership"] == 0.0
    assert metrics["image_membership"] == 0.0


def test_recorded_release_cases_pass() -> None:
    result = evaluate_cases("eval/web_research/cases.json")
    assert result["passed"] is True
