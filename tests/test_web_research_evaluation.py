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


# ---------------------------------------------------------------------------
# Answer-visual quality
# ---------------------------------------------------------------------------


def _bars_item(*, values=(100, 101), maximum=101, basis="measured", url=None) -> dict:
    return {
        "id": "visual:a",
        "type": "answer_visual",
        "payload": {
            "variant": "bars",
            "title": "FPS",
            "unit": "FPS",
            "basis": basis,
            "scale": {"min": 0, "max": maximum},
            "bars": [
                {"label": f"b{n}", "value": v, "citations": [1]} for n, v in enumerate(values)
            ],
            "evidence": [{"url": url or "https://bench.example/r", "label": "bench.example"}],
            "caption_citations": [],
            "fallback_markdown": "| b0 | 100 |\n| b1 | 101 |",
        },
    }


def _visual_case(**overrides):
    from app.evaluation.web_research.contracts import VisualAnswerEvaluationCase

    fields = {
        "case_id": "bars",
        "answer_content": "Result:\n\n<!--rich:visual:a-->",
        "rich_items": [_bars_item()],
        "expected_visual_kinds": ["bars"],
        "admitted_source_urls": ["https://bench.example/r"],
    }
    fields.update(overrides)
    return VisualAnswerEvaluationCase.model_validate(fields)


def _visual_metrics(**overrides) -> dict[str, float]:
    from app.evaluation.web_research.metrics import evaluate_visual_case

    return evaluate_visual_case(_visual_case(**overrides))


def test_a_sound_visual_answer_scores_every_metric() -> None:
    assert set(_visual_metrics().values()) == {1.0}


def test_visual_evidence_outside_the_admitted_sources_fails_support() -> None:
    metrics = _visual_metrics(rich_items=[_bars_item(url="https://invented.example/x")])
    assert metrics["evidence_support"] == 0.0


def test_a_value_beyond_the_shared_scale_fails_numerical_integrity() -> None:
    metrics = _visual_metrics(rich_items=[_bars_item(values=(100, 140), maximum=101)])
    assert metrics["numerical_integrity"] == 0.0


def test_fallback_disagreeing_with_component_values_fails_numerical_integrity() -> None:
    metrics = _visual_metrics(rich_items=[_bars_item(values=(100, 102), maximum=102)])
    assert metrics["numerical_integrity"] == 0.0


def test_unexpected_visual_on_a_simple_question_fails_selection() -> None:
    metrics = _visual_metrics(expected_visual_kinds=["none"])
    assert metrics["visual_selection"] == 0.0


def test_selecting_an_irrelevant_image_fails_omission() -> None:
    metrics = _visual_metrics(selected_image_ids=["I2"], irrelevant_image_ids=["I2"])
    assert metrics["irrelevant_image_omission"] == 0.0


def test_latency_and_token_budgets_are_scored_only_when_recorded() -> None:
    assert "latency_within_budget" not in _visual_metrics()
    slow = _visual_metrics(latency_ms=9000, latency_budget_ms=4000, tokens=900, token_budget=1000)
    assert slow["latency_within_budget"] == 0.0
    assert slow["token_cost_within_budget"] == 1.0


def test_recorded_visual_cases_pass() -> None:
    from app.evaluation.web_research.harness import evaluate_visual_cases

    result = evaluate_visual_cases("eval/rich_answer_visuals/cases.json")
    assert result["passed"] is True
    assert result["case_count"] >= 6
