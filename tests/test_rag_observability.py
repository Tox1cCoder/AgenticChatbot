"""Content-free stage/cache telemetry for the RAG pipeline.

Every assertion here is about what the exported Prometheus text does and does
not contain -- no live Redis/PostgreSQL/Qdrant/model provider involved.
"""

from __future__ import annotations

import pytest

from app.observability.rag import RAGMetrics


@pytest.fixture
def metrics() -> RAGMetrics:
    # A fresh registry per test avoids duplicate-metric-name collisions
    # across the module-level ``rag_metrics`` singleton used elsewhere.
    return RAGMetrics()


# ---------------------------------------------------------------------------
# Step 1 test: metrics never receive document content
# ---------------------------------------------------------------------------


def test_metrics_never_receive_document_content(metrics):
    metrics.stage("retrieval", elapsed_seconds=0.1, labels={"document_text": "secret"})
    # ``render()`` returns bytes (it feeds a Prometheus scrape response
    # directly, see app/api/health.py) -- decode before the containment check.
    assert "secret" not in metrics.render().decode("utf-8")


def test_stage_never_receives_document_id_or_filename(metrics):
    metrics.stage(
        "sql_hydration",
        elapsed_seconds=0.05,
        labels={
            "document_id": "11111111-1111-1111-1111-111111111111",
            "filename": "quarterly-earnings.pdf",
        },
    )
    rendered = metrics.render().decode("utf-8")
    assert "11111111-1111-1111-1111-111111111111" not in rendered
    assert "quarterly-earnings.pdf" not in rendered


def test_cache_result_never_receives_arbitrary_values(metrics):
    metrics.cache_result("document_embedding", "SECRET-CONTENT-HASH-LOOKS-LIKE-A-VALUE")
    rendered = metrics.render().decode("utf-8")
    assert "SECRET-CONTENT-HASH-LOOKS-LIKE-A-VALUE" not in rendered
    # Unknown values are bounded to "other" rather than dropped silently.
    assert 'result="other"' in rendered


# ---------------------------------------------------------------------------
# Bounded enums: unknown stage/provider/modality/cache names collapse to "other"
# ---------------------------------------------------------------------------


def test_unknown_stage_name_is_bounded_to_other(metrics):
    metrics.stage("totally-made-up-stage", elapsed_seconds=0.01)
    rendered = metrics.render().decode("utf-8")
    assert "totally-made-up-stage" not in rendered
    assert 'stage="other"' in rendered


def test_known_stage_name_is_recorded_verbatim(metrics):
    metrics.stage("embedding", elapsed_seconds=0.02, labels={"provider": "gemini"})
    rendered = metrics.render().decode("utf-8")
    assert 'stage="embedding"' in rendered
    assert 'provider="gemini"' in rendered


def test_unknown_provider_and_modality_bound_to_other(metrics):
    metrics.stage(
        "dense_retrieval",
        elapsed_seconds=0.03,
        labels={"provider": "some-unlisted-vendor", "modality": "audio"},
    )
    rendered = metrics.render().decode("utf-8")
    assert "some-unlisted-vendor" not in rendered
    assert "audio" not in rendered
    assert 'provider="other"' in rendered
    assert 'modality="other"' in rendered


def test_stage_without_labels_defaults_to_not_applicable(metrics):
    metrics.stage("reranking", elapsed_seconds=0.5)
    rendered = metrics.render().decode("utf-8")
    assert 'provider="n/a"' in rendered
    assert 'cache_result="n/a"' in rendered


def test_cache_result_bounds_unknown_cache_name(metrics):
    metrics.cache_result("some-other-cache", "hit")
    rendered = metrics.render().decode("utf-8")
    assert "some-other-cache" not in rendered
    assert 'cache="other"' in rendered


def test_cache_result_records_known_cache_and_result(metrics):
    metrics.cache_result("query_embedding", "hit")
    rendered = metrics.render().decode("utf-8")
    assert 'cache="query_embedding"' in rendered
    assert 'result="hit"' in rendered


# ---------------------------------------------------------------------------
# Existing gate/reranker signatures are untouched (extend, never rewrite)
# ---------------------------------------------------------------------------


def test_grounded_answer_and_degraded_signatures_are_unchanged(metrics):
    metrics.grounded_answer(mode="shadow", outcome="accepted", reason_code="none")
    metrics.degraded("reranker", "timeout")
    rendered = metrics.render().decode("utf-8")
    assert "rag_grounded_answers_total" in rendered
    assert "rag_degraded_operations_total" in rendered


def test_evidence_tokens_records_a_count_without_content(metrics):
    metrics.evidence_tokens(512)
    rendered = metrics.render().decode("utf-8")
    assert "rag_evidence_pack_tokens" in rendered
