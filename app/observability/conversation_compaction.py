"""Bounded-cardinality metrics and aggregate health for conversation compaction."""

from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

_JOB_STATUSES = ("idle", "pending", "processing", "retry", "dead")
_CONTENT_CLASSES = {"text", "tools", "multimodal", "mixed"}
_PROVIDERS = {"openai", "gemini", "anthropic"}
_OUTCOMES = {
    "success",
    "skipped",
    "retry",
    "dead",
    "conflict",
    "timeout",
    "failure",
}


class ConversationCompactionMetrics:
    """Prometheus collectors whose labels never contain tenant content or IDs."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.jobs = Counter(
            "conversation_compaction_jobs_total",
            "Conversation compaction job outcomes.",
            ("outcome", "error_class"),
            registry=self.registry,
        )
        self.job_count = Gauge(
            "conversation_compaction_job_count",
            "Current durable jobs by status.",
            ("status",),
            registry=self.registry,
        )
        self.operations = Counter(
            "conversation_compaction_operations_total",
            "Background and emergency compaction operations.",
            ("mode", "outcome", "provider", "model", "content_class"),
            registry=self.registry,
        )
        common_labels = ("mode", "provider", "model", "content_class")
        self.duration = Histogram(
            "conversation_compaction_duration_seconds",
            "Compaction provider latency.",
            common_labels,
            registry=self.registry,
        )
        self.input_tokens = Histogram(
            "conversation_compaction_input_tokens",
            "Counted compaction input tokens.",
            common_labels,
            registry=self.registry,
        )
        self.output_tokens = Histogram(
            "conversation_compaction_output_tokens",
            "Validated compaction output tokens.",
            common_labels,
            registry=self.registry,
        )
        self.estimate_delta = Histogram(
            "conversation_compaction_estimate_delta_tokens",
            "Actual minus estimated request tokens.",
            ("provider", "model", "content_class"),
            registry=self.registry,
        )
        self.cost = Counter(
            "conversation_compaction_cost_total",
            "Provider-reported compaction cost.",
            ("provider", "model", "currency"),
            registry=self.registry,
        )
        self.deterministic_trim = Counter(
            "conversation_compaction_deterministic_trim_total",
            "Complete history groups removed by deterministic reduction.",
            registry=self.registry,
        )
        self.provider_overflow_retry = Counter(
            "conversation_compaction_provider_overflow_retry_total",
            "Single aggressive provider-overflow retries.",
            ("outcome",),
            registry=self.registry,
        )
        self.oldest_actionable_age = Gauge(
            "conversation_compaction_oldest_actionable_age_seconds",
            "Age of the oldest pending or retry job.",
            registry=self.registry,
        )
        self.expired_leases = Gauge(
            "conversation_compaction_expired_leases",
            "Current expired processing leases.",
            registry=self.registry,
        )
        self.sequence_lag = Gauge(
            "conversation_compaction_sequence_lag",
            "Aggregate requested-versus-summarized sequence lag.",
            ("kind",),
            registry=self.registry,
        )

    def record_job_outcome(self, outcome: str, *, error_code: str | None) -> None:
        self.jobs.labels(
            outcome=_bounded(outcome, _OUTCOMES),
            error_class=_error_class(error_code),
        ).inc()

    def record_compaction(
        self,
        *,
        mode: str,
        outcome: str,
        provider: str,
        model: str,
        content_class: str,
        input_tokens: int,
        output_tokens: int,
        duration_seconds: float,
        cost_amount: float | None = None,
        cost_currency: str | None = None,
    ) -> None:
        labels = {
            "mode": _bounded(mode, {"background", "emergency"}),
            "provider": _provider(provider),
            "model": _model_family(model),
            "content_class": _bounded(content_class, _CONTENT_CLASSES),
        }
        self.operations.labels(
            **labels,
            outcome=_bounded(outcome, _OUTCOMES),
        ).inc()
        self.duration.labels(**labels).observe(max(0.0, float(duration_seconds)))
        self.input_tokens.labels(**labels).observe(max(0, int(input_tokens)))
        self.output_tokens.labels(**labels).observe(max(0, int(output_tokens)))
        if cost_amount is not None and float(cost_amount) >= 0:
            self.cost.labels(
                provider=labels["provider"],
                model=labels["model"],
                currency=_currency(cost_currency),
            ).inc(float(cost_amount))

    def record_token_calibration(
        self,
        *,
        provider: str,
        model: str,
        content_class: str,
        estimated_tokens: int,
        actual_tokens: int,
    ) -> None:
        self.estimate_delta.labels(
            provider=_provider(provider),
            model=_model_family(model),
            content_class=_bounded(content_class, _CONTENT_CLASSES),
        ).observe(int(actual_tokens) - int(estimated_tokens))

    def record_deterministic_trim(self, *, removed_groups: int) -> None:
        if removed_groups > 0:
            self.deterministic_trim.inc(int(removed_groups))

    def record_provider_overflow_retry(self, outcome: str) -> None:
        self.provider_overflow_retry.labels(outcome=_bounded(outcome, {"success", "failure"})).inc()

    def update_health(self, snapshot: dict[str, Any]) -> None:
        counts = snapshot.get("job_counts") or {}
        for status in _JOB_STATUSES:
            self.job_count.labels(status=status).set(max(0, int(counts.get(status, 0))))
        self.oldest_actionable_age.set(
            max(0.0, float(snapshot.get("oldest_actionable_age_seconds", 0)))
        )
        self.expired_leases.set(max(0, int(snapshot.get("expired_lease_count", 0))))
        self.sequence_lag.labels(kind="max").set(max(0, int(snapshot.get("max_sequence_lag", 0))))
        self.sequence_lag.labels(kind="total").set(
            max(0, int(snapshot.get("total_sequence_lag", 0)))
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)


class ConversationCompactionHealthService:
    """Classify aggregate durable state without returning tenant identifiers."""

    def __init__(
        self,
        repository: Any,
        *,
        queue_age_degraded_seconds: float,
        lag_degraded_sequences: int,
    ) -> None:
        self.repository = repository
        self.queue_age_degraded_seconds = max(1.0, float(queue_age_degraded_seconds))
        self.lag_degraded_sequences = max(1, int(lag_degraded_sequences))

    def get_health(self) -> dict[str, Any]:
        raw = self.repository.get_compaction_health_snapshot()
        counts = {
            status: max(0, int((raw.get("job_counts") or {}).get(status, 0)))
            for status in _JOB_STATUSES
        }
        oldest_age = max(0.0, float(raw.get("oldest_actionable_age_seconds", 0)))
        expired = max(0, int(raw.get("expired_lease_count", 0)))
        max_lag = max(0, int(raw.get("max_sequence_lag", 0)))
        total_lag = max(0, int(raw.get("total_sequence_lag", 0)))
        if counts["dead"] > 0 or expired > 0:
            status = "unhealthy"
        elif (
            counts["retry"] > 0
            or oldest_age > self.queue_age_degraded_seconds
            or max_lag > self.lag_degraded_sequences
        ):
            status = "degraded"
        else:
            status = "healthy"
        return {
            "status": status,
            "job_counts": counts,
            "oldest_actionable_age_seconds": oldest_age,
            "expired_lease_count": expired,
            "max_sequence_lag": max_lag,
            "total_sequence_lag": total_lag,
        }


def _bounded(value: Any, allowed: set[str]) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else "other"


def _provider(value: Any) -> str:
    return _bounded(value, _PROVIDERS)


def _model_family(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    families = (
        "gpt-4o",
        "gpt-4.1",
        "gpt-5",
        "o1",
        "o3",
        "o4",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3-flash",
        "gemini-3-pro",
        "claude-opus",
        "claude-sonnet",
        "claude-haiku",
    )
    return next((family for family in families if normalized.startswith(family)), "other")


def _error_class(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return "none"
    if "provider" in normalized and "timeout" in normalized:
        return "provider_timeout"
    markers = {
        "rate": "rate_limit",
        "credential": "credential",
        "invalid": "validation",
        "cas": "cas_conflict",
        "lease": "lease_expired",
        "broker": "broker_publish",
        "context": "context_overflow",
        "timeout": "timeout",
    }
    return next((label for marker, label in markers.items() if marker in normalized), "other")


def _currency(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    return normalized if normalized in {"USD", "EUR"} else "OTHER"


conversation_compaction_metrics = ConversationCompactionMetrics()
