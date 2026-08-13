"""RAGAS adapters must remain lazy and LangSmith-compatible."""

from __future__ import annotations

import sys
import types

from app.evaluation.rag.ragas_metrics import ragas_evaluators


def test_ragas_adapter_maps_run_and_example_to_collection_metric(monkeypatch):
    observed: dict[str, object] = {}

    class FakeMetric:
        def score(self, **kwargs):
            observed["sample"] = kwargs
            return type("MetricResult", (), {"value": 0.75})()

    collections = types.ModuleType("ragas.metrics.collections")
    collections.ContextPrecision = FakeMetric
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

    result = ragas_evaluators(True, metric_factory=lambda _: FakeMetric())[0](Run(), Example())

    assert result == {"key": "ragas_context_precision", "score": 0.75}
    assert observed["sample"]["user_input"] == "question"
    assert observed["sample"]["retrieved_contexts"] == ["source evidence"]
