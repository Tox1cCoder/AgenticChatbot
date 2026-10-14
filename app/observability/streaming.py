"""Low-cardinality telemetry for AI SDK answer streaming.

Latency is measured twice, on purpose:

- ``first_answer``: from the adapter starting the response to the first frame
  carrying answer text (``text-delta`` or an answer preview). It includes
  routing, tool calls and provider latency -- it is what the reader waits.
- ``delta_to_wire``: from the adapter receiving that first answer event from
  the service to yielding its frame. It excludes provider and tool time -- it is
  the streaming layer's own overhead.

Keepalives, status, reasoning and sources never count as an answer.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from app.observability.labels import bounded_label as _bounded

_OPERATIONS = {"chat", "resume", "continue", "unknown"}
#: ``stopped`` is a service-published Stop; ``unknown`` is a stream that ended
#: with no terminal outcome (reported to the client as ``stream_incomplete``).
_OUTCOMES = {"complete", "paused", "interrupt", "stopped", "error", "disconnect", "unknown"}

_FIRST_ANSWER_BUCKETS = (0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120)
_DELTA_TO_WIRE_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1)


class StreamingMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.first_answer = Histogram(
            "ai_sdk_stream_first_answer_seconds",
            "Response start to the first answer-text frame (includes provider and tools).",
            ("operation",),
            buckets=_FIRST_ANSWER_BUCKETS,
            registry=self.registry,
        )
        self.delta_to_wire = Histogram(
            "ai_sdk_stream_delta_to_wire_seconds",
            "First answer event received from the service to its frame yielded.",
            ("operation",),
            buckets=_DELTA_TO_WIRE_BUCKETS,
            registry=self.registry,
        )
        self.reconciles = Counter(
            "ai_sdk_stream_reconcile_total",
            "Web-draft reconciliations, by whether the saved text differed from the draft.",
            ("changed",),
            registry=self.registry,
        )
        self.ends = Counter(
            "ai_sdk_stream_end_total",
            "AI SDK responses by how they ended.",
            ("operation", "outcome"),
            registry=self.registry,
        )

    def record_first_answer(self, seconds: float, *, operation: str) -> None:
        self.first_answer.labels(operation=_bounded(operation, _OPERATIONS)).observe(seconds)

    def record_delta_to_wire(self, seconds: float, *, operation: str) -> None:
        self.delta_to_wire.labels(operation=_bounded(operation, _OPERATIONS)).observe(seconds)

    def record_reconcile(self, *, changed: bool) -> None:
        self.reconciles.labels(changed="true" if changed else "false").inc()

    def record_end(self, *, operation: str, outcome: str) -> None:
        self.ends.labels(
            operation=_bounded(operation, _OPERATIONS),
            outcome=_bounded(outcome, _OUTCOMES),
        ).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)


streaming_metrics = StreamingMetrics()
