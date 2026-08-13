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
