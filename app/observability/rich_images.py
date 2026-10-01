"""Bounded, content-free telemetry for rich-image discovery and delivery."""

from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from app.observability.labels import bounded_label as _bounded

_PROVIDERS = {"brave", "tavily"}
_REGISTRATION_OUTCOMES = {"registered", "reused", "skipped_scheme", "failed"}
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
        self.selector_duration = Histogram(
            "rich_image_selector_duration_seconds",
            "Time spent selecting canonical rich image candidates.",
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
        self.presented = Counter(
            "rich_image_presented_total",
            "Image items included in the model-facing inventory.",
            ("provider",),
            registry=self.registry,
        )
        self.final_selections = Counter(
            "rich_image_final_selection_total",
            "Image items surviving finalization and persisted with the message.",
            ("provider",),
            registry=self.registry,
        )
        self.registrations = Counter(
            "rich_image_registrations_total",
            "Protected-reference registration outcomes per image or group cell.",
            ("provider", "outcome"),
            registry=self.registry,
        )

    def record_selection_duration(self, duration_seconds: float) -> None:
        self.selector_duration.observe(max(0.0, float(duration_seconds)))

    def record_presentation(self, *, provider: str, count: int) -> None:
        if count > 0:
            self.presented.labels(provider=_provider(provider)).inc(int(count))

    def record_final_selection(self, *, provider: str, count: int) -> None:
        if count > 0:
            self.final_selections.labels(provider=_provider(provider)).inc(int(count))

    def record_registration(self, *, provider: str, outcome: str) -> None:
        """Record one protected-reference registration attempt.

        Counted per image or per group cell, so a cell-level failure inside a
        group that keeps its siblings is visible rather than hidden behind the
        group's single final-selection count.
        """
        self.registrations.labels(
            provider=_provider(provider),
            outcome=_bounded(outcome, _REGISTRATION_OUTCOMES),
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


rich_image_metrics = RichImageMetrics()
