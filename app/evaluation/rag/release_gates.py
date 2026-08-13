"""Baseline-relative release gate comparison."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReleaseGateResult:
    metric: str
    baseline: float
    candidate: float
    passed: bool
    threshold: float


def load_release_gates(path: str | Path) -> dict[str, dict[str, float]]:
    gates = json.loads(Path(path).read_text(encoding="utf-8"))["metrics"]
    _validate_gate_rules(gates)
    return gates


def _validate_gate_rules(gates: Mapping[str, Mapping[str, float]]) -> None:
    for metric, rule in gates.items():
        direction = rule.get("direction", "higher")
        threshold = float(rule.get("max_regression", 0.0))
        if direction not in {"higher", "lower"}:
            raise ValueError(f"gate {metric} direction must be exactly 'higher' or 'lower'")
        if threshold < 0:
            raise ValueError(f"gate {metric} max_regression must be non-negative")


def compare_release_gates(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    gates: Mapping[str, Mapping[str, float]],
) -> list[ReleaseGateResult]:
    _validate_gate_rules(gates)
    results: list[ReleaseGateResult] = []
    missing_baseline = sorted(metric for metric in gates if metric not in baseline)
    missing_candidate = sorted(metric for metric in gates if metric not in candidate)
    if missing_baseline:
        raise ValueError(f"missing baseline metrics: {', '.join(missing_baseline)}")
    if missing_candidate:
        raise ValueError(f"missing candidate metrics: {', '.join(missing_candidate)}")
    for metric, rule in gates.items():
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
            )
        )
    return results
