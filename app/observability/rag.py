"""Bounded-cardinality operational metrics for the RAG pipeline."""

from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Counter, generate_latest

_COMPONENTS = {"reranker"}
_FAILURE_CODES = {
    "timeout",
    "model_load_failure",
    "provider_exception",
    "score_count_mismatch",
    "invalid_score",
    "non_finite_score",
    "missing_candidate_id",
}


class RAGMetrics:
    """Process-local RAG counters with bounded, content-free labels."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.degraded_operations = Counter(
            "rag_degraded_operations_total",
            "RAG operations that returned a safe degraded result.",
            ("component", "failure_code"),
            registry=self.registry,
        )

    def degraded(self, component: str, failure_code: str) -> None:
        self.degraded_operations.labels(
            component=_bounded(component, _COMPONENTS),
            failure_code=_bounded(failure_code, _FAILURE_CODES),
        ).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)


def _bounded(value: Any, allowed: set[str]) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else "other"


rag_metrics = RAGMetrics()
