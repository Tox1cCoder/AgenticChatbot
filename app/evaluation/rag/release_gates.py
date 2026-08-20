"""Baseline-relative release gate comparison.

A gate entry's ``status`` defaults to ``"unmeasured"``: a gate is
non-binding unless it explicitly opts in with ``"status": "measured"``,
which asserts it carries an evidence-based ``max_regression`` threshold
selected from a real baseline experiment. This fails closed on the
central failure mode this plan exists to fix -- a gate silently missing
its ``status`` key (a bad merge, a hand-edit, a copy-paste from a
different gate) must never be treated as binding. A gate may instead be
marked ``"status": "unmeasured"`` explicitly when no baseline benchmark
has produced a threshold for it yet (see ``eval/rag/release_gates.json``'s
Task 13 capacity/latency/cost slots) -- the plan's Global Constraints
forbid hard-coding a threshold that was not selected from evaluation
results. An unmeasured gate is structurally present (so its shape is
reviewed and its provenance is recorded) but is **non-binding**:
``compare_release_gates`` never requires it to be present in
``baseline``/``candidate`` and its result carries ``binding=False`` with
``passed=None`` -- never treated as either a pass or a failure of the
release decision.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_STATUS_MEASURED = "measured"
_STATUS_UNMEASURED = "unmeasured"
_VALID_STATUSES = {_STATUS_MEASURED, _STATUS_UNMEASURED}


@dataclass(frozen=True)
class ReleaseGateResult:
    metric: str
    baseline: float | None
    candidate: float | None
    passed: bool | None
    threshold: float | None
    binding: bool = True


def load_release_gates(path: str | Path) -> dict[str, dict[str, float]]:
    gates = json.loads(Path(path).read_text(encoding="utf-8"))["metrics"]
    _validate_gate_rules(gates)
    return gates


def _validate_gate_rules(gates: Mapping[str, Mapping[str, float]]) -> None:
    for metric, rule in gates.items():
        direction = rule.get("direction", "higher")
        status = rule.get("status", _STATUS_UNMEASURED)
        if direction not in {"higher", "lower"}:
            raise ValueError(f"gate {metric} direction must be exactly 'higher' or 'lower'")
        if status not in _VALID_STATUSES:
            raise ValueError(f"gate {metric} status must be one of {sorted(_VALID_STATUSES)}")
        if status == _STATUS_UNMEASURED:
            continue
        threshold = float(rule.get("max_regression", 0.0))
        if threshold < 0:
            raise ValueError(f"gate {metric} max_regression must be non-negative")


def compare_release_gates(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    gates: Mapping[str, Mapping[str, float]],
) -> list[ReleaseGateResult]:
    _validate_gate_rules(gates)
    binding_metrics = {
        metric: rule
        for metric, rule in gates.items()
        if rule.get("status", _STATUS_UNMEASURED) != _STATUS_UNMEASURED
    }
    missing_baseline = sorted(metric for metric in binding_metrics if metric not in baseline)
    missing_candidate = sorted(metric for metric in binding_metrics if metric not in candidate)
    if missing_baseline:
        raise ValueError(f"missing baseline metrics: {', '.join(missing_baseline)}")
    if missing_candidate:
        raise ValueError(f"missing candidate metrics: {', '.join(missing_candidate)}")

    results: list[ReleaseGateResult] = []
    for metric, rule in gates.items():
        if rule.get("status", _STATUS_UNMEASURED) == _STATUS_UNMEASURED:
            results.append(
                ReleaseGateResult(
                    metric=metric,
                    baseline=float(baseline[metric]) if metric in baseline else None,
                    candidate=float(candidate[metric]) if metric in candidate else None,
                    passed=None,
                    threshold=None,
                    binding=False,
                )
            )
            continue
        threshold = float(rule.get("max_regression", 0.0))
        direction = rule.get("direction", "higher")
        delta = float(candidate[metric]) - float(baseline[metric])
        passed = delta >= -threshold if direction == "higher" else delta <= threshold
        results.append(
            ReleaseGateResult(
                metric=metric,
                baseline=float(baseline[metric]),
                candidate=float(candidate[metric]),
                passed=passed,
                threshold=threshold,
                binding=True,
            )
        )
    return results
