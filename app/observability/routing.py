"""Content-free routing, transition, and finalization metrics.

Nothing recorded here may carry user text, model-generated reason text, prompt
payloads, credentials, or unbounded identifiers. Metric labels are allowlisted
enums plus bounded provider/model/inventory identifiers; request, conversation,
user, custom-agent-instance, message, and evidence IDs belong only in
access-controlled sampled logs and traces.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.ai.workflow.inventory import CUSTOM_AGENT_PREFIX

logger = logging.getLogger(__name__)

__all__ = [
    "BASE_AGENT_KIND_LABELS",
    "RoutingMetricsRecorder",
    "bounded_agent_label",
]

# Bounded label space for route volume. A dynamic custom-agent ID must never
# become a metric label; every custom target collapses to one bucket.
BASE_AGENT_KIND_LABELS = frozenset(
    {
        "chat_agent",
        "rag_agent",
        "search_agent",
        "image_generator_agent",
        "planning_agent",
        "canvas_agent",
        "custom_agent",
        "unknown",
    }
)


def bounded_agent_label(agent_id: str | None) -> str:
    """Collapse an agent ID into an allowlisted metric label."""
    if not isinstance(agent_id, str) or not agent_id:
        return "unknown"
    if agent_id.startswith(CUSTOM_AGENT_PREFIX):
        return "custom_agent"
    return agent_id if agent_id in BASE_AGENT_KIND_LABELS else "unknown"


@dataclass
class RoutingMetricsRecorder:
    """In-process counters/histograms with an allowlisted label space.

    The dataclass keeps the surface testable; a deployment can subclass it and
    forward the same calls to its metrics backend.
    """

    counters: Counter = field(default_factory=Counter)
    latencies_ms: list[float] = field(default_factory=list)
    transition_depths: list[int] = field(default_factory=list)

    # -- routing ---------------------------------------------------------

    def routing_completed(
        self,
        *,
        agent_id: str | None,
        provider: str,
        model: str,
        inventory_version: str,
        attempts: int,
        latency_ms: float,
        schema_ok: bool,
    ) -> None:
        label = bounded_agent_label(agent_id)
        self.counters[f"routing.completed.{label}"] += 1
        self.counters[f"routing.provider.{provider}.{model}"] += 1
        self.counters[f"routing.inventory.{inventory_version[:16]}"] += 1
        self.counters[f"routing.attempts.{min(int(attempts), 2)}"] += 1
        self.counters[f"routing.schema.{'ok' if schema_ok else 'invalid'}"] += 1
        self.latencies_ms.append(float(latency_ms))

    def routing_failed(self, *, code: str, provider: str, model: str, attempts: int) -> None:
        self.counters[f"routing.failed.{code}"] += 1
        self.counters[f"routing.provider.{provider}.{model}"] += 1
        self.counters[f"routing.attempts.{min(int(attempts), 2)}"] += 1

    def routing_schema_invalid(self) -> None:
        self.counters["routing.schema.invalid"] += 1

    def routing_target_race(self) -> None:
        self.counters["routing.target_race"] += 1

    # -- transitions -----------------------------------------------------

    def transition_accepted(self, *, from_agent_id: str | None, to_agent_id: str, depth: int):
        self.counters[
            f"transition.accepted.{bounded_agent_label(from_agent_id)}"
            f".{bounded_agent_label(to_agent_id)}"
        ] += 1
        self.transition_depths.append(int(depth))

    def transition_rejected(self, *, reason: str) -> None:
        self.counters[f"transition.rejected.{reason}"] += 1

    # -- execution and finalization --------------------------------------

    def agent_execution_limit(self, *, agent_id: str | None, limit_kind: str) -> None:
        self.counters[f"execution.limit.{bounded_agent_label(agent_id)}.{limit_kind}"] += 1

    def worker_completed(self, *, agent_id: str | None, status: str, evidence_count: int) -> None:
        self.counters[f"worker.{status}.{bounded_agent_label(agent_id)}"] += 1
        self.counters["worker.evidence_total"] += max(0, int(evidence_count))

    def grounding_outcome(self, *, outcome: str) -> None:
        self.counters[f"grounding.{outcome}"] += 1

    def finalization_completed(self, *, policy_ids: tuple[str, ...]) -> None:
        self.counters["finalization.completed"] += 1
        for policy_id in policy_ids:
            self.counters[f"finalization.policy.{policy_id}"] += 1

    def finalization_failed(self, *, code: str) -> None:
        self.counters[f"finalization.failed.{code}"] += 1

    def terminal_error(self, *, code: str, retriable: bool) -> None:
        self.counters[f"workflow.error.{code}.{'retriable' if retriable else 'terminal'}"] += 1

    # -- export ----------------------------------------------------------

    def export(self) -> dict[str, Any]:
        return {
            "counters": dict(self.counters),
            "latencies_ms": list(self.latencies_ms),
            "transition_depths": list(self.transition_depths),
        }

    def reset(self) -> None:
        self.counters.clear()
        self.latencies_ms.clear()
        self.transition_depths.clear()


_recorder = RoutingMetricsRecorder()


def get_routing_metrics_recorder() -> RoutingMetricsRecorder:
    """Return the process-wide recorder."""
    return _recorder
