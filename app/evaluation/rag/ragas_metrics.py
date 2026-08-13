"""Lazy RAGAS collection adapters; RAGAS is never imported by the server."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

RAGAS_COLLECTION_METRICS = {
    "context_precision": (
        ("ContextPrecision",),
        ("user_input", "reference", "retrieved_contexts"),
        True,
        False,
    ),
    "context_recall": (
        ("ContextRecall",),
        ("user_input", "reference", "retrieved_contexts"),
        True,
        False,
    ),
    "noise_sensitivity": (
        ("NoiseSensitivity",),
        ("user_input", "reference", "response", "retrieved_contexts"),
        True,
        False,
    ),
    "faithfulness": (("Faithfulness",), ("response", "retrieved_contexts"), True, False),
    "answer_relevancy": (
        ("AnswerRelevancy", "ResponseRelevancy"),
        ("user_input", "response"),
        True,
        True,
    ),
    "multimodal_faithfulness": (
        ("MultiModalFaithfulness", "MultimodalFaithfulness"),
        ("response", "retrieved_contexts"),
        True,
        False,
    ),
    "multimodal_relevance": (
        ("MultiModalRelevance", "MultimodalRelevance"),
        ("user_input", "response", "retrieved_contexts"),
        True,
        False,
    ),
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
    for key, (names, arguments, needs_llm, needs_embeddings) in RAGAS_COLLECTION_METRICS.items():
        metric_name = next((name for name in names if hasattr(collections, name)), None)
        metric = metric_factory(key) if metric_factory else getattr(collections, metric_name, None)
        if metric is None:
            missing.append("/".join(names))
        else:
            if isinstance(metric, type):
                if needs_llm and llm is None:
                    raise RuntimeError(f"RAGAS {key} requires an explicit llm dependency")
                if needs_embeddings and embeddings is None:
                    raise RuntimeError(f"RAGAS {key} requires an explicit embeddings dependency")
                kwargs = {}
                if needs_llm:
                    kwargs["llm"] = llm
                if needs_embeddings:
                    kwargs["embeddings"] = embeddings
                metric = metric(**kwargs)

            def evaluate(
                run: Any,
                example: Any,
                *,
                _metric: Any = metric,
                _key: str = key,
                _arguments: tuple[str, ...] = arguments,
            ) -> dict[str, Any]:
                sample = _sample(run, example)
                result = _metric.score(**{argument: sample[argument] for argument in _arguments})
                score = getattr(result, "value", result)
                return {"key": f"ragas_{_key}", "score": float(score)}

            evaluators.append(evaluate)
    if missing:
        raise RuntimeError(
            f"Installed RAGAS does not expose collection metrics: {', '.join(missing)}"
        )
    return evaluators
