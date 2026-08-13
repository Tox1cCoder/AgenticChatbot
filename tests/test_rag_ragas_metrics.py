"""RAGAS adapters must remain lazy and LangSmith-compatible."""

from __future__ import annotations

import sys
import types

import pytest

from app.evaluation.rag.ragas_metrics import ragas_evaluators


def test_ragas_adapter_maps_run_and_example_to_collection_metric(monkeypatch):
    observed: dict[str, object] = {}

    class ContextPrecisionMetric:
        def score(self, *, user_input, reference, retrieved_contexts):
            observed["context"] = (user_input, reference, retrieved_contexts)
            return type("MetricResult", (), {"value": 0.75})()

    class AnswerRelevancyMetric:
        def score(self, *, user_input, response):
            observed["answer"] = (user_input, response)
            return type("MetricResult", (), {"value": 0.5})()

    class MultiModalFaithfulnessMetric:
        def score(self, *, response, retrieved_contexts):
            observed["multimodal"] = (response, retrieved_contexts)
            return type("MetricResult", (), {"value": 1.0})()

    class MultiModalRelevanceMetric:
        def score(self, *, user_input, response, retrieved_contexts):
            observed["multimodal_relevance"] = (user_input, response, retrieved_contexts)
            return type("MetricResult", (), {"value": 1.0})()

    collections = types.ModuleType("ragas.metrics.collections")
    collections.ContextPrecision = ContextPrecisionMetric
    collections.ContextRecall = ContextPrecisionMetric
    collections.NoiseSensitivity = ContextPrecisionMetric
    collections.Faithfulness = MultiModalFaithfulnessMetric
    collections.AnswerRelevancy = AnswerRelevancyMetric
    collections.MultiModalFaithfulness = MultiModalFaithfulnessMetric
    collections.MultiModalRelevance = MultiModalRelevanceMetric
    ragas = types.ModuleType("ragas")
    metrics = types.ModuleType("ragas.metrics")
    monkeypatch.setitem(sys.modules, "ragas", ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics)
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", collections)

    class Run:
        outputs = {
            "answer": "answer",
            "evidence": [{"document_id": "doc-a", "content": "source evidence"}],
        }

    class Example:
        inputs = {"question": "question"}
        outputs = {"answer": "reference", "relevant_document_ids": ["doc-a"]}

    metrics_by_key = {
        "context_precision": ContextPrecisionMetric(),
        "context_recall": ContextPrecisionMetric(),
        "noise_sensitivity": ContextPrecisionMetric(),
        "faithfulness": MultiModalFaithfulnessMetric(),
        "answer_relevancy": AnswerRelevancyMetric(),
        "multimodal_faithfulness": MultiModalFaithfulnessMetric(),
        "multimodal_relevance": MultiModalRelevanceMetric(),
    }
    evaluators = ragas_evaluators(True, metric_factory=metrics_by_key.__getitem__)
    result = evaluators[0](Run(), Example())
    evaluators[4](Run(), Example())
    evaluators[5](Run(), Example())
    evaluators[6](Run(), Example())

    assert result == {"key": "ragas_context_precision", "score": 0.75}
    assert observed["context"] == ("question", "reference", ["source evidence"])
    assert observed["answer"] == ("question", "answer")
    assert observed["multimodal"] == ("answer", ["source evidence"])
    assert observed["multimodal_relevance"] == (
        "question",
        "answer",
        ["source evidence"],
    )


def test_ragas_collection_classes_require_explicit_dependencies(monkeypatch):
    class Metric:
        def __init__(self, *, llm):
            self.llm = llm

    collections = types.ModuleType("ragas.metrics.collections")
    for name in (
        "ContextPrecision",
        "ContextRecall",
        "NoiseSensitivity",
        "Faithfulness",
        "AnswerRelevancy",
        "MultiModalFaithfulness",
        "MultiModalRelevance",
    ):
        setattr(collections, name, Metric)
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", collections)

    with pytest.raises(RuntimeError, match="requires an explicit llm dependency"):
        ragas_evaluators(True)
