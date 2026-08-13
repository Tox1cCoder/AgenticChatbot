"""Lazy RAGAS collection adapters; RAGAS is never imported by the server."""

from __future__ import annotations

import importlib
from typing import Any

RAGAS_COLLECTION_METRICS = (
    "ContextPrecision",
    "ContextRecall",
    "NoiseSensitivity",
    "Faithfulness",
    "ResponseRelevancy",
    "MultimodalFaithfulness",
    "MultimodalRelevance",
)


def ragas_evaluators(enabled: bool) -> list[Any]:
    """Return requested RAGAS collection evaluators, importing only on demand."""
    if not enabled:
        return []
    collections = importlib.import_module("ragas.metrics.collections")
    evaluators: list[Any] = []
    missing: list[str] = []
    for metric_name in RAGAS_COLLECTION_METRICS:
        metric = getattr(collections, metric_name, None)
        if metric is None:
            missing.append(metric_name)
        else:
            evaluators.append(metric() if isinstance(metric, type) else metric)
    if missing:
        raise RuntimeError(
            f"Installed RAGAS does not expose collection metrics: {', '.join(missing)}"
        )
    return evaluators
