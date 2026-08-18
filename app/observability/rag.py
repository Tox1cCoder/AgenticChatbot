"""Bounded-cardinality operational metrics for the RAG pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

_COMPONENTS = {"reranker"}
# Task 12: stage/cache instrumentation. Every label below is a small closed
# enum; ``_bounded`` coerces anything else (including a caller's mistake, such
# as passing document text or an id) to "other" before it ever reaches a
# label value, and ``stage``/``cache_result`` only ever read these fixed keys
# out of a caller-supplied mapping -- any other key, whatever it is named or
# holds, is never inspected and never recorded.
_STAGES = {
    "parse",
    "caption",
    "embedding",
    "dense_retrieval",
    "lexical_retrieval",
    "sql_hydration",
    "reranking",
    "evidence_assembly",
    "generation",
    "validation",
}
# Round-1 fix (finding 6): providers span both the RAG-embedding domain
# (gemini, sentence_transformers) and the chat/generation domain (gemini,
# openai, anthropic -- see app/services/provider_service.py's
# SUPPORTED_PROVIDERS). Both domains share one closed set here because they
# share one label dimension.
_STAGE_PROVIDERS = {"gemini", "openai", "anthropic", "sentence_transformers", "n/a"}
# Round-1 fix (finding 6): "model was dropped entirely." Bounded to the
# handful of models this deployment's config can actually select for the
# stages that record one (embedding, reranking) -- see
# app/core/config.py's rag_embedding_model / rag_reranker_model defaults.
# Chat/generation model ids are open-ended (provider-hosted catalogs), so the
# generation stage intentionally leaves `model` at its "n/a" default rather
# than enumerating an unbounded set.
_STAGE_MODELS = {
    "gemini-embedding-2",
    "cross-encoder/ms-marco-minilm-l-6-v2",
    "n/a",
}
_STAGE_MODALITIES = {"text", "image", "n/a"}
_CACHE_NAMES = {"document_embedding", "query_embedding", "retrieval"}
_CACHE_RESULTS = {"hit", "miss", "disabled"}
_STAGE_CACHE_RESULTS = _CACHE_RESULTS | {"n/a"}
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
    # Round-1 fix (finding 4): infra-level failures (Qdrant, PostgreSQL) that
    # are not a specific reranker code above.
    "dependency_exception",
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

        self.stage_duration_seconds = Histogram(
            "rag_stage_duration_seconds",
            "Wall-clock duration of one RAG pipeline stage, recorded on both "
            "success and failure so p95/p99 are not biased downward by "
            "excluding slow failures. Labels are bounded, content-free enums "
            "only -- never a document id, filename or text.",
            ("stage", "provider", "model", "modality", "cache_result"),
            registry=self.registry,
        )

        self.stage_failures = Counter(
            "rag_stage_failures_total",
            "Failed attempts per RAG pipeline stage. Denominator for failure "
            "rate is this counter plus rag_stage_duration_seconds_count for "
            "the same stage.",
            ("stage", "failure_code"),
            registry=self.registry,
        )

        self.cache_operations = Counter(
            "rag_cache_operations_total",
            "Exact-cache lookups for RAG document/query embeddings and retrieval results.",
            ("cache", "result"),
            registry=self.registry,
        )

        self.evidence_pack_tokens = Histogram(
            "rag_evidence_pack_tokens",
            "Token count of the rendered evidence pack sent to the model for one turn.",
            buckets=(64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
            registry=self.registry,
        )

    def stage(
        self,
        stage: str,
        *,
        elapsed_seconds: float,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one pipeline-stage duration -- call on success AND failure.

        ``labels`` may be a caller's raw kwargs and can carry anything,
        including document content by mistake. Only the five fixed keys
        below are ever read from it; every other key -- and every value read
        here -- is bounded to a closed enum before export, so nothing else
        ever reaches the exported series. Callers on a failure path should
        still call this (so a slow failure is not invisible to p95/p99) and
        also call :meth:`stage_failure` (so failure rate is computable).
        """
        source = labels or {}
        self.stage_duration_seconds.labels(
            stage=_bounded(stage, _STAGES),
            provider=_bounded(source.get("provider", "n/a"), _STAGE_PROVIDERS),
            model=_bounded(source.get("model", "n/a"), _STAGE_MODELS),
            modality=_bounded(source.get("modality", "n/a"), _STAGE_MODALITIES),
            cache_result=_bounded(source.get("cache_result", "n/a"), _STAGE_CACHE_RESULTS),
        ).observe(max(0.0, float(elapsed_seconds)))

    def stage_failure(self, stage: str, failure_code: str) -> None:
        self.stage_failures.labels(
            stage=_bounded(stage, _STAGES),
            failure_code=_bounded(failure_code, _FAILURE_CODES),
        ).inc()

    def cache_result(self, cache: str, result: str) -> None:
        self.cache_operations.labels(
            cache=_bounded(cache, _CACHE_NAMES),
            result=_bounded(result, _CACHE_RESULTS),
        ).inc()

    def evidence_tokens(self, token_count: int) -> None:
        self.evidence_pack_tokens.observe(max(0, int(token_count)))

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
