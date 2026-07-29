"""Bounded, content-free telemetry for rich-image discovery and delivery."""

from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

_PROVIDERS = {"brave", "tavily"}
_SELECTION_OUTCOMES = {"selected", "rejected", "omitted"}
_FETCH_OUTCOMES = {
    "success",
    "timeout",
    "network",
    "status",
    "mime",
    "size",
    "dimensions",
    "decode",
    "scheme",
    "url",
    "dns",
    "private_address",
    "redirect_limit",
}


class RichImageMetrics:
    """Prometheus collectors whose labels cannot contain tenant content."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.discovery_results = Histogram(
            "rich_image_discovery_results",
            "Normalized image results returned per provider call.",
            ("provider",),
            registry=self.registry,
        )
        self.selections = Counter(
            "rich_image_selections_total",
            "Deterministic rich-image selection outcomes.",
            ("provider", "outcome"),
            registry=self.registry,
        )
        self.fetches = Counter(
            "rich_image_fetches_total",
            "Render-time rich-image fetch outcomes.",
            ("provider", "outcome"),
            registry=self.registry,
        )
        self.fetch_duration = Histogram(
            "rich_image_fetch_duration_seconds",
            "Render-time rich-image fetch duration.",
            ("provider", "outcome"),
            registry=self.registry,
        )

    def record_discovery(self, *, provider: str, result_count: int) -> None:
        self.discovery_results.labels(provider=_provider(provider)).observe(
            max(0, int(result_count))
        )

    def record_selection(self, *, provider: str, outcome: str) -> None:
        self.selections.labels(
            provider=_provider(provider),
            outcome=_bounded(outcome, _SELECTION_OUTCOMES),
        ).inc()

    def record_fetch(self, *, provider: str, outcome: str, duration_seconds: float) -> None:
        labels = {
            "provider": _provider(provider),
            "outcome": _bounded(outcome, _FETCH_OUTCOMES),
        }
        self.fetches.labels(**labels).inc()
        self.fetch_duration.labels(**labels).observe(max(0.0, float(duration_seconds)))

    def render(self) -> bytes:
        return generate_latest(self.registry)


def _provider(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("brave"):
        return "brave"
    return normalized if normalized in _PROVIDERS else "other"


def _bounded(value: Any, allowed: set[str]) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else "other"


rich_image_metrics = RichImageMetrics()
