"""Release gates must fail closed when baseline data is incomplete or regresses."""

from __future__ import annotations

import pytest

from app.evaluation.rag.release_gates import compare_release_gates


def test_release_gate_comparison_rejects_missing_baseline_metric():
    with pytest.raises(ValueError, match="missing baseline metrics: citation_validity"):
        compare_release_gates(
            baseline={},
            candidate={"citation_validity": 0.1},
            gates={"citation_validity": {"direction": "higher", "max_regression": 0.01}},
        )


def test_release_gate_comparison_marks_a_regression_as_failed():
    result = compare_release_gates(
        baseline={"citation_validity": 0.9},
        candidate={"citation_validity": 0.1},
        gates={"citation_validity": {"direction": "higher", "max_regression": 0.01}},
    )

    assert result[0].passed is False


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({"direction": "sideways", "max_regression": 0.1}, "direction"),
        ({"direction": "higher", "max_regression": -0.1}, "max_regression"),
        ({"direction": "higher", "max_regression": 0.1, "status": "measuring"}, "status"),
    ],
)
def test_release_gate_rules_are_validated(rule, message):
    with pytest.raises(ValueError, match=message):
        compare_release_gates({"metric": 1.0}, {"metric": 1.0}, {"metric": rule})


# ---------------------------------------------------------------------------
# Task 13: capacity/latency/cost gates with no evidence-based threshold yet
# must be structurally present but never treated as a pass.
# ---------------------------------------------------------------------------


def test_unmeasured_gate_does_not_require_baseline_or_candidate_metrics():
    gates = {
        "citation_validity": {"direction": "higher", "max_regression": 0.01},
        "p95_stage_latency_ms": {
            "direction": "lower",
            "max_regression": None,
            "status": "unmeasured",
        },
    }

    results = compare_release_gates(
        baseline={"citation_validity": 0.9},
        candidate={"citation_validity": 0.91},
        gates=gates,
    )

    by_metric = {result.metric: result for result in results}
    assert by_metric["citation_validity"].binding is True
    assert by_metric["p95_stage_latency_ms"].binding is False
    assert by_metric["p95_stage_latency_ms"].passed is None
    assert by_metric["p95_stage_latency_ms"].baseline is None
    assert by_metric["p95_stage_latency_ms"].candidate is None


def test_unmeasured_gate_is_excluded_from_the_overall_pass_fail_decision():
    gates = {
        "citation_validity": {"direction": "higher", "max_regression": 0.01},
        "cost_per_document_usd": {
            "direction": "lower",
            "max_regression": None,
            "status": "unmeasured",
        },
    }

    results = compare_release_gates(
        baseline={"citation_validity": 0.9},
        candidate={"citation_validity": 0.91},
        gates=gates,
    )

    assert all(result.passed for result in results if result.binding)


def test_a_measured_gate_still_uses_reported_values_when_both_are_present():
    gates = {
        "vector_memory_mb": {
            "direction": "lower",
            "max_regression": None,
            "status": "unmeasured",
        }
    }

    results = compare_release_gates(
        baseline={"vector_memory_mb": 100.0},
        candidate={"vector_memory_mb": 120.0},
        gates=gates,
    )

    assert results[0].baseline == 100.0
    assert results[0].candidate == 120.0
    assert results[0].binding is False
