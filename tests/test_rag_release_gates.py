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
    ],
)
def test_release_gate_rules_are_validated(rule, message):
    with pytest.raises(ValueError, match=message):
        compare_release_gates({"metric": 1.0}, {"metric": 1.0}, {"metric": rule})
