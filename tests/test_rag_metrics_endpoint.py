"""Round-1 fix (finding 5): the RAG metrics registry was write-only.

``rag_metrics.render()`` had no caller anywhere in ``app/`` before this fix,
so nothing ever scraped ``rag_stage_duration_seconds`` or the other Task 12
metrics. This mirrors the three existing siblings
(``/metrics/conversation-compaction``, ``/metrics/model-usage``,
``/metrics/rich-images``) in ``app/api/health.py``.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.health import create_health_router
from app.observability.rag import RAGMetrics


def test_rag_metrics_endpoint_renders_the_registry():
    metrics = RAGMetrics()
    metrics.stage("embedding", elapsed_seconds=0.1, labels={"provider": "gemini"})
    app = FastAPI()
    app.include_router(create_health_router(rag_metrics=metrics))

    response = TestClient(app).get("/metrics/rag")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "rag_stage_duration_seconds" in response.text


def test_rag_metrics_endpoint_defaults_to_the_shared_singleton():
    from app.observability.rag import rag_metrics as rag_metrics_singleton

    rag_metrics_singleton.degraded("reranker", "timeout")
    app = FastAPI()
    app.include_router(create_health_router())

    response = TestClient(app).get("/metrics/rag")

    assert response.status_code == 200
    assert "rag_degraded_operations_total" in response.text


def test_rag_metrics_route_is_registered_once():
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/metrics/rag" in paths
