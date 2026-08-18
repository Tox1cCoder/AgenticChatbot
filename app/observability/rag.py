"""Bounded-cardinality operational metrics for the RAG pipeline."""

from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Counter, generate_latest

_COMPONENTS = {"reranker"}
_GROUNDING_MODES = {"shadow", "enforced"}
# "would_abstain" is shadow-only: the answer is never actually replaced in
# shadow mode, so it must read differently from a live "abstained" outcome.
_GROUNDING_OUTCOMES = {"accepted", "regenerated", "abstained", "would_abstain"}
_GROUNDING_REASON_CODES = {
    "none",
    "unknown_evidence_id",
    "citation_coverage_below_minimum",
    "answer_without_evidence",
    "insufficient_evidence",
    "unstructured_answer",
}
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

        self.grounded_answers = Counter(
            "rag_grounded_answers_total",
            "Grounded-answer gate decisions. mode=shadow decisions never change the "
            "final response — they measure whether today's answer *would* pass, which "
            "is a floor, not an estimate, while the citation prompt stays enforcement-only: "
            "the model is never asked to cite in shadow mode, so most shadow records read "
            "citation_coverage_below_minimum by construction. outcome=would_abstain is the "
            "shadow-mode equivalent of an enforced abstention; only outcome=abstained is a "
            "live one.",
            ("mode", "outcome", "reason_code"),
            registry=self.registry,
        )

    def grounded_answer(self, *, mode: str, outcome: str, reason_code: str) -> None:
        self.grounded_answers.labels(
            mode=_bounded(mode, _GROUNDING_MODES),
            outcome=_bounded(outcome, _GROUNDING_OUTCOMES),
            reason_code=_bounded(reason_code, _GROUNDING_REASON_CODES),
        ).inc()

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
