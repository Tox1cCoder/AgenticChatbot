"""Bounded-cardinality operational metrics for the model-usage recorder.

Mirrors the mechanism used by ``app.observability.conversation_compaction``:
a ``prometheus_client`` registry whose label sets are deliberately bounded so
they can never encode tenant content. The recorder emits two kinds of signal:

* ``attempts`` -- one recorded provider attempt, labelled only by provider
  family, operation family, status, and usage source.
* ``persistence`` -- the outcome of writing an attempt to the ledger,
  labelled only by outcome and a bounded failure class.

The labels never carry user, conversation, trace, request, or model IDs; a
model identifier is not a label at all (only its provider family is), so the
ledger's raw event rows remain the sole place model identity is stored.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Counter, generate_latest

_PROVIDERS = {"openai", "gemini", "anthropic"}
_OPERATIONS = {
    "chat",
    "rag",
    "search",
    "router",
    "planning",
    "embedding",
    "reranking",
    "image_generation",
    "compaction",
    "document_processing",
    "form_filler",
    "suggestion",
    "title",
    "unknown",
}
_STATUSES = {"success", "error", "cancelled", "timeout"}
_SOURCES = {
    "provider_reported",
    "mixed_reported_estimated",
    "locally_estimated",
    "unavailable",
}
_PERSIST_OUTCOMES = {"stored", "duplicate", "retry_enqueued", "dropped"}


class ModelUsageMetrics:
    """Prometheus collectors whose labels never contain tenant content or IDs."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.attempts = Counter(
            "model_usage_attempts_total",
            "Recorded model-call attempts by bounded outcome dimensions.",
            ("provider", "operation", "status", "source"),
            registry=self.registry,
        )
        self.persistence = Counter(
            "model_usage_persistence_total",
            "Ledger-write persistence outcomes for recorded attempts.",
            ("outcome", "failure_class"),
            registry=self.registry,
        )

    def record_attempt(self, *, provider: str, operation: str, status: str, source: str) -> None:
        self.attempts.labels(
            provider=_provider_family(provider),
            operation=_operation_family(operation),
            status=_bounded(status, _STATUSES),
            source=_bounded(source, _SOURCES),
        ).inc()

    def record_persistence(self, outcome: str, *, failure_class: str | None = None) -> None:
        self.persistence.labels(
            outcome=_bounded(outcome, _PERSIST_OUTCOMES),
            failure_class=_failure_class(failure_class),
        ).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)


def _bounded(value: Any, allowed: set[str]) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else "other"


def _provider_family(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if "openai" in normalized or normalized.startswith("gpt"):
        return "openai"
    if "gemini" in normalized or "google" in normalized:
        return "gemini"
    if "anthropic" in normalized or "claude" in normalized:
        return "anthropic"
    return normalized if normalized in _PROVIDERS else "other"


def _operation_family(value: Any) -> str:
    return _bounded(value, _OPERATIONS)


def _failure_class(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return "none"
    markers = {
        "reference": "reference",
        "integrity": "reference",
        "timeout": "timeout",
        "rate": "rate_limit",
        "broker": "broker_publish",
        "connection": "connection",
        "operational": "connection",
    }
    return next((label for marker, label in markers.items() if marker in normalized), "other")


model_usage_metrics = ModelUsageMetrics()
