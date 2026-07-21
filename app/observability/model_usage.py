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

import hashlib
import hmac
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest

from app.observability.model_usage_failure_store import (
    FailureStoreSnapshot,
    ModelUsageFailureStore,
    UnavailableModelUsageFailureStore,
    create_redis_failure_store,
)

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
    "workflow",
    "vision",
    "image_user_response",
    "form_fill",
    "suggestions",
    "title_generation",
    "image_caption",
    "conversation_compaction",
    "document_index",
}
_STATUSES = {"success", "error", "cancelled", "timeout"}
_SOURCES = {
    "provider_reported",
    "mixed_reported_estimated",
    "locally_estimated",
    "unavailable",
}
_PERSIST_OUTCOMES = {"stored", "duplicate", "retry_enqueued", "dropped"}
_RECENT_FAILURE_BUFFER_MAX = 10_000


class ModelUsageMetrics:
    """Prometheus collectors whose labels never contain tenant content or IDs."""

    def __init__(
        self,
        registry: CollectorRegistry | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        failure_store: ModelUsageFailureStore | None = None,
    ) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.attempts = Counter(
            "model_usage_attempts_total",
            "Recorded model-call attempts by bounded outcome dimensions.",
            ("provider", "operation", "status", "source"),
            registry=self.registry,
        )
        self.persistence = Counter(
            "model_usage_persistence_total",
            "Process-local ledger-write persistence outcomes for recorded attempts.",
            ("outcome", "failure_class"),
            registry=self.registry,
        )
        self.health_status = Gauge(
            "model_usage_health_status",
            "Current model-usage health classification (healthy=0, degraded=1, unhealthy=2).",
            registry=self.registry,
        )
        self.health_rollup_lag = Gauge(
            "model_usage_health_rollup_lag_minutes",
            "Lag of the newest rollup in complete UTC minutes.",
            registry=self.registry,
        )
        self.health_rollup_gap = Gauge(
            "model_usage_health_rollup_gap_count",
            "Absolute recent difference between durable raw attempts and rollup requests.",
            registry=self.registry,
        )
        self.health_unattributed_rate = Gauge(
            "model_usage_health_unattributed_rate",
            "Recent fraction of durable attempts without a verified owner.",
            registry=self.registry,
        )
        self.health_persistence_failure_saturated = Gauge(
            "model_usage_health_persistence_failure_saturated",
            "Whether this process-local bounded persistence-failure buffer saturated.",
            registry=self.registry,
        )
        self.health_failure_store_available = Gauge(
            "model_usage_health_failure_store_available",
            "Whether the deployment-scoped persistence-failure store was readable.",
            registry=self.registry,
        )
        self.health_shared_persistence_failures = Gauge(
            "model_usage_health_shared_persistence_failures",
            "Deployment-scoped recent model-usage persistence failures.",
            registry=self.registry,
        )
        self.health_snapshot_timestamp = Gauge(
            "model_usage_health_snapshot_timestamp_seconds",
            "Unix timestamp of the latest successful durable health snapshot refresh.",
            registry=self.registry,
        )
        self._monotonic = monotonic
        self._failure_times: deque[float] = deque(maxlen=_RECENT_FAILURE_BUFFER_MAX)
        self._failure_buffer_saturated = False
        self._failure_lock = Lock()
        self._failure_store = failure_store or UnavailableModelUsageFailureStore()

    def record_attempt(self, *, provider: str, operation: str, status: str, source: str) -> None:
        self.attempts.labels(
            provider=_provider_family(provider),
            operation=_operation_family(operation),
            status=_bounded(status, _STATUSES),
            source=_bounded(source, _SOURCES),
        ).inc()

    def record_persistence(self, outcome: str, *, failure_class: str | None = None) -> None:
        bounded_outcome = _bounded(outcome, _PERSIST_OUTCOMES)
        self.persistence.labels(
            outcome=bounded_outcome,
            failure_class=_failure_class(failure_class),
        ).inc()
        if bounded_outcome in {"retry_enqueued", "dropped"}:
            with self._failure_lock:
                if len(self._failure_times) == _RECENT_FAILURE_BUFFER_MAX:
                    self._failure_buffer_saturated = True
                self._failure_times.append(self._monotonic())
            with suppress(Exception):
                self._failure_store.record_failure()

    def recent_persistence_failure_count(self, *, window_seconds: float = 300.0) -> int:
        """Return process-local failures in a bounded recent window.

        Durable raw/rollup comparison remains the authoritative cross-process
        signal. This short-lived counter adds immediate recorder feedback in
        the process serving the health endpoint without leaking error content.
        """
        cutoff = self._monotonic() - max(1.0, float(window_seconds))
        with self._failure_lock:
            while self._failure_times and self._failure_times[0] < cutoff:
                self._failure_times.popleft()
            if len(self._failure_times) < _RECENT_FAILURE_BUFFER_MAX:
                self._failure_buffer_saturated = False
            return len(self._failure_times)

    def persistence_failure_buffer_saturated(self) -> bool:
        with self._failure_lock:
            return self._failure_buffer_saturated

    def shared_persistence_failure_snapshot(self, *, window_seconds: float) -> FailureStoreSnapshot:
        try:
            return self._failure_store.recent_failure_count(window_seconds=window_seconds)
        except Exception:
            return FailureStoreSnapshot(count=0, available=False)

    def update_health(self, snapshot: dict[str, Any]) -> None:
        self.health_status.set(
            {"healthy": 0, "degraded": 1, "unhealthy": 2}.get(snapshot.get("status"), 2)
        )
        self.health_rollup_lag.set(max(0, int(snapshot.get("rollup_lag_minutes", 0))))
        self.health_rollup_gap.set(max(0, int(snapshot.get("rollup_gap_count", 0))))
        self.health_unattributed_rate.set(
            min(1.0, max(0.0, float(snapshot.get("unattributed_rate", 0.0))))
        )
        self.health_persistence_failure_saturated.set(
            1 if snapshot.get("persistence_failure_count_saturated") else 0
        )
        store_available = bool(snapshot.get("persistence_failure_store_available"))
        self.health_failure_store_available.set(1 if store_available else 0)
        self.health_shared_persistence_failures.set(
            max(0, int(snapshot.get("persistence_failure_count", 0)))
        )
        self.health_snapshot_timestamp.set(time.time())

    def render(self) -> bytes:
        return generate_latest(self.registry)

    def mark_health_refresh_failure(self) -> None:
        """Mark classification unavailable without fabricating snapshot freshness."""
        self.health_status.set(2)


class ModelUsageHealthService:
    """Classify durable aggregate usage health without tenant dimensions.

    Thresholds are explicit constructor inputs. A durable raw/rollup count gap
    is unhealthy, an unattributed rate above its threshold is degraded, and
    rollup lag is degraded/unhealthy above the configured minute thresholds.
    Recent process-local persistence failures degrade immediately; durable
    count reconciliation remains the cross-worker source of truth.
    """

    def __init__(
        self,
        repository: Any,
        *,
        metrics: ModelUsageMetrics,
        lookback_minutes: int = 60,
        unattributed_degraded_ratio: float = 0.10,
        rollup_lag_degraded_minutes: int = 2,
        rollup_lag_unhealthy_minutes: int = 5,
        persistence_failure_window_seconds: float = 300.0,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.repository = repository
        self.metrics = metrics
        self.lookback_minutes = max(1, int(lookback_minutes))
        self.unattributed_degraded_ratio = min(1.0, max(0.0, float(unattributed_degraded_ratio)))
        self.rollup_lag_degraded_minutes = max(0, int(rollup_lag_degraded_minutes))
        self.rollup_lag_unhealthy_minutes = max(
            self.rollup_lag_degraded_minutes,
            int(rollup_lag_unhealthy_minutes),
        )
        self.persistence_failure_window_seconds = max(
            1.0, float(persistence_failure_window_seconds)
        )
        self.clock = clock

    def get_health(self) -> dict[str, Any]:
        now = self.clock().astimezone(timezone.utc)
        raw = self.repository.get_model_usage_health_snapshot(
            now=now,
            lookback_minutes=self.lookback_minutes,
        )
        event_count = max(0, int(raw.get("raw_event_count", 0)))
        rollup_count = max(0, int(raw.get("rollup_request_count", 0)))
        unattributed_count = min(
            event_count,
            max(0, int(raw.get("unattributed_event_count", 0))),
        )
        unattributed_rate = unattributed_count / event_count if event_count else 0.0
        gap = abs(event_count - rollup_count)
        latest_event = raw.get("latest_event_minute")
        latest_rollup = raw.get("latest_rollup_minute")
        if (
            event_count
            and isinstance(latest_event, datetime)
            and isinstance(latest_rollup, datetime)
        ):
            lag = max(
                0,
                int(
                    (
                        latest_event.astimezone(timezone.utc)
                        - latest_rollup.astimezone(timezone.utc)
                    ).total_seconds()
                    // 60
                ),
            )
        elif event_count:
            lag = self.lookback_minutes
        else:
            lag = 0
        local_failures = self.metrics.recent_persistence_failure_count(
            window_seconds=self.persistence_failure_window_seconds
        )
        shared_failures = self.metrics.shared_persistence_failure_snapshot(
            window_seconds=self.persistence_failure_window_seconds
        )
        failures = max(local_failures, shared_failures.count)

        if gap > 0 or lag > self.rollup_lag_unhealthy_minutes:
            status = "unhealthy"
        elif (
            not shared_failures.available
            or failures > 0
            or unattributed_rate > self.unattributed_degraded_ratio
            or lag > self.rollup_lag_degraded_minutes
        ):
            status = "degraded"
        else:
            status = "healthy"
        result = {
            "status": status,
            "lookback_minutes": self.lookback_minutes,
            "raw_event_count": event_count,
            "rollup_request_count": rollup_count,
            "rollup_gap_count": gap,
            "unattributed_event_count": unattributed_count,
            "unattributed_rate": round(unattributed_rate, 6),
            "rollup_lag_minutes": lag,
            "persistence_failure_count": max(0, shared_failures.count),
            "process_persistence_failure_count": local_failures,
            "persistence_failure_scope": (
                "deployment" if shared_failures.available else "unavailable"
            ),
            "persistence_failure_store_available": shared_failures.available,
            "authoritative_health_scope": "database",
            "persistence_failure_count_saturated": (
                self.metrics.persistence_failure_buffer_saturated()
            ),
        }
        self.metrics.update_health(result)
        return result


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


def usage_user_hash(user_id: str | None) -> str | None:
    """Return a keyed HMAC-SHA256 of ``user_id`` for LangSmith correlation.

    Returns ``None`` when there is no user or no configured secret, so raw user
    ids never leak into trace metadata. The secret keys the hash so the value
    cannot be reversed by dictionary attack across tenants.
    """
    from app.core.config import settings

    if not user_id:
        return None
    secret = settings.model_usage_user_hash_secret or ""
    if not secret:
        return None
    digest = hmac.new(secret.encode("utf-8"), str(user_id).encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


def _default_failure_store() -> ModelUsageFailureStore:
    from app.core.config import settings

    return create_redis_failure_store(
        settings.redis_url,
        ttl_seconds=settings.model_usage_failure_store_ttl_seconds,
        timeout_seconds=settings.model_usage_failure_store_timeout_seconds,
    )


model_usage_metrics = ModelUsageMetrics(failure_store=_default_failure_store())
