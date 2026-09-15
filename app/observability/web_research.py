"""Low-cardinality telemetry for canonical web research."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, generate_latest

_OPERATIONS = {"search", "open", "image_search", "image_fetch", "finish"}
_MODES = {"quick", "agentic"}
_OUTCOMES = {"success", "partial", "error", "reused", "selected", "released"}
_VISUAL_INTENTS = {"none", "figure", "comparison", "gallery"}


def _bounded(value: object, allowed: set[str]) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else "other"


class WebResearchMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.operations = Counter(
            "web_research_operations_total",
            "Canonical web research outcomes.",
            ("operation", "mode", "outcome", "visual_intent"),
            registry=self.registry,
        )

    def record(
        self,
        *,
        operation: str,
        mode: str,
        outcome: str,
        visual_intent: str = "none",
    ) -> None:
        self.operations.labels(
            operation=_bounded(operation, _OPERATIONS),
            mode=_bounded(mode, _MODES),
            outcome=_bounded(outcome, _OUTCOMES),
            visual_intent=_bounded(visual_intent, _VISUAL_INTENTS),
        ).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)


web_research_metrics = WebResearchMetrics()
