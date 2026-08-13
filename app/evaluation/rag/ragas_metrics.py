"""Lazy RAGAS collection adapters; RAGAS is never imported by the server."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

RAGAS_COLLECTION_METRICS = {
    "context_precision": ("ContextPrecision",),
    "context_recall": ("ContextRecall",),
    "noise_sensitivity": ("NoiseSensitivity",),
    "faithfulness": ("Faithfulness",),
    "answer_relevancy": ("AnswerRelevancy", "ResponseRelevancy"),
    "multimodal_faithfulness": ("MultiModalFaithfulness", "MultimodalFaithfulness"),
    "multimodal_relevance": ("MultiModalRelevance", "MultimodalRelevance"),
}


def _mapping(value: Any, attribute: str) -> Mapping[str, Any]:
    return getattr(value, attribute, None) or {}


def _sample(run: Any, example: Any) -> dict[str, Any]:
    outputs = _mapping(run, "outputs")
    inputs = _mapping(example, "inputs")
    reference = _mapping(example, "outputs")
    contexts = [
        item.get("content", "") for item in outputs.get("evidence", ()) if item.get("content")
    ]
    return {
        "user_input": inputs.get("question", ""),
        "response": outputs.get("answer", ""),
        "reference": reference.get("answer", ""),
        "retrieved_contexts": contexts,
        "reference_contexts": list(reference.get("relevant_document_ids", ())),
    }


def ragas_evaluators(
    enabled: bool,
    *,
    metric_factory: Callable[[str], Any] | None = None,
    llm: Any | None = None,
    embeddings: Any | None = None,
) -> list[Callable[[Any, Any], dict[str, Any]]]:
    """Build lazy LangSmith evaluators around RAGAS Collections metrics.

    Callers explicitly inject provider dependencies; the adapter never creates a
    model client or makes a network call during import or construction.
    """
    if not enabled:
        return []
    collections = importlib.import_module("ragas.metrics.collections")
    evaluators: list[Callable[[Any, Any], dict[str, Any]]] = []
    missing: list[str] = []
    for key, names in RAGAS_COLLECTION_METRICS.items():
        metric_name = next((name for name in names if hasattr(collections, name)), None)
        metric = metric_factory(key) if metric_factory else getattr(collections, metric_name, None)
        if metric is None:
            missing.append("/".join(names))
        else:
            if isinstance(metric, type):
                kwargs = {
                    name: value
                    for name, value in {"llm": llm, "embeddings": embeddings}.items()
                    if value
                }
                metric = metric(**kwargs)

            def evaluate(
                run: Any, example: Any, *, _metric: Any = metric, _key: str = key
            ) -> dict[str, Any]:
                result = _metric.score(**_sample(run, example))
                score = getattr(result, "value", result)
                return {"key": f"ragas_{_key}", "score": float(score)}

            evaluators.append(evaluate)
    if missing:
        raise RuntimeError(
            f"Installed RAGAS does not expose collection metrics: {', '.join(missing)}"
        )
    return evaluators
