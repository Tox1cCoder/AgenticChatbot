"""Release gates must fail closed when baseline data is incomplete or regresses."""

from __future__ import annotations

import pytest

from app.evaluation.rag.release_gates import compare_release_gates


def test_release_gate_comparison_rejects_missing_baseline_metric():
    with pytest.raises(ValueError, match="missing baseline metrics: citation_validity"):
        compare_release_gates(
            baseline={},
            candidate={"citation_validity": 0.1},
            gates={
                "citation_validity": {
                    "direction": "higher",
                    "max_regression": 0.01,
                    "status": "measured",
                }
            },
        )


def test_release_gate_comparison_marks_a_regression_as_failed():
    result = compare_release_gates(
        baseline={"citation_validity": 0.9},
        candidate={"citation_validity": 0.1},
        gates={
            "citation_validity": {
                "direction": "higher",
                "max_regression": 0.01,
                "status": "measured",
            }
        },
    )

    assert result[0].passed is False


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({"direction": "sideways", "max_regression": 0.1, "status": "measured"}, "direction"),
        ({"direction": "higher", "max_regression": -0.1, "status": "measured"}, "max_regression"),
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
        "citation_validity": {
            "direction": "higher",
            "max_regression": 0.01,
            "status": "measured",
        },
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
        "citation_validity": {
            "direction": "higher",
            "max_regression": 0.01,
            "status": "measured",
        },
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


def test_a_gate_missing_its_status_key_defaults_to_unmeasured_not_binding():
    """Item 6: the fail-direction must be safe-by-default. A gate silently
    missing its ``status`` key (a bad merge, a hand-edit, a copy-paste from
    a different gate) must never be treated as ``"measured"`` -- that would
    make it binding on no evidence, which is exactly the failure mode this
    plan exists to fix. It must default to ``"unmeasured"`` and require an
    explicit opt-in.
    """
    gates = {"citation_validity": {"direction": "higher", "max_regression": 0.01}}

    results = compare_release_gates(
        baseline={"citation_validity": 0.9},
        candidate={"citation_validity": 0.1},  # a real regression
        gates=gates,
    )

    assert results[0].binding is False
    assert results[0].passed is None, (
        "a status-less gate must never resolve to a pass or a fail -- "
        "silence must not read as evidence"
    )


def test_an_unmeasured_gate_still_reports_baseline_and_candidate_values_when_present():
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


# ---------------------------------------------------------------------------
# Round-1 review fix: eval/rag/release_gates.json inherited five gates from
# Task 1 stamped status="measured" with provenance naming an experiment
# ("pre-hardening-baseline") that was never run (see Task 1's own report and
# `git show 9c05ff5 -- eval/rag/release_gates.json`, which shows those five
# entries committed as bare placeholder numbers with no status/provenance
# field at all). This test locks the honest state of this branch: every gate
# in the real file is non-binding until a real baseline experiment records
# one. It must fail again if any gate is ever silently promoted to
# status="measured" without an experiment that actually produced its number.
# ---------------------------------------------------------------------------


def test_the_real_release_gates_file_has_no_gate_claiming_a_measurement_yet():
    from pathlib import Path

    from app.evaluation.rag.release_gates import load_release_gates

    gates = load_release_gates(Path("eval/rag/release_gates.json"))

    unmeasured = {
        metric: rule for metric, rule in gates.items() if rule.get("status") == "unmeasured"
    }
    assert unmeasured == gates, (
        "a gate in eval/rag/release_gates.json claims status='measured' with no "
        "baseline experiment on record; see docs/rag-scale-runbook.md for how to "
        "promote a gate honestly, one real result at a time"
    )
    for metric, rule in gates.items():
        assert rule["max_regression"] is None, (
            f"gate {metric} is unmeasured but carries a non-null max_regression"
        )
        assert rule["provenance"]["experiment"] is None, (
            f"gate {metric} is unmeasured but names a specific experiment in provenance"
        )
