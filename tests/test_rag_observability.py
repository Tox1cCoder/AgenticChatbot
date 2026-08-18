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


# ---------------------------------------------------------------------------
# Round-1 fix: the stage histogram must carry a real, bounded `model` label
# (finding 6 -- "model was dropped entirely").
# ---------------------------------------------------------------------------


def test_stage_records_known_model_label(metrics):
    metrics.stage(
        "embedding",
        elapsed_seconds=0.02,
        labels={"provider": "gemini", "model": "gemini-embedding-2"},
    )
    rendered = metrics.render().decode("utf-8")
    assert 'model="gemini-embedding-2"' in rendered


def test_stage_unknown_model_bounds_to_other(metrics):
    metrics.stage(
        "embedding",
        elapsed_seconds=0.02,
        labels={"provider": "gemini", "model": "some-unlisted-finetune"},
    )
    rendered = metrics.render().decode("utf-8")
    assert "some-unlisted-finetune" not in rendered
    assert 'model="other"' in rendered


def test_stage_without_model_label_defaults_to_not_applicable(metrics):
    metrics.stage("validation", elapsed_seconds=0.01)
    rendered = metrics.render().decode("utf-8")
    assert 'model="n/a"' in rendered


# ---------------------------------------------------------------------------
# Round-1 fix: failed stage attempts must be countable and their duration
# must still land in the histogram (finding 4 -- failures were invisible,
# biasing p95/p99 downward for exactly the slow/failing calls).
# ---------------------------------------------------------------------------


def test_stage_failure_is_recorded_and_bounded(metrics):
    metrics.stage_failure("reranking", "timeout")
    rendered = metrics.render().decode("utf-8")
    assert "rag_stage_failures_total" in rendered
    assert 'stage="reranking"' in rendered
    assert 'failure_code="timeout"' in rendered


def test_stage_failure_unknown_code_bounds_to_other(metrics):
    metrics.stage_failure("embedding", "some-never-seen-failure-mode")
    rendered = metrics.render().decode("utf-8")
    assert "some-never-seen-failure-mode" not in rendered
    assert 'failure_code="other"' in rendered


def test_stage_failure_unknown_stage_bounds_to_other(metrics):
    metrics.stage_failure("not-a-real-stage", "timeout")
    rendered = metrics.render().decode("utf-8")
    assert "not-a-real-stage" not in rendered
    assert 'stage="other"' in rendered


# ---------------------------------------------------------------------------
# Round-1 fix: provider now has real producers at the embedding and
# dense-retrieval sites, expanded to cover chat providers too (finding 6).
# ---------------------------------------------------------------------------


def test_stage_accepts_chat_providers_for_the_generation_stage(metrics):
    metrics.stage("generation", elapsed_seconds=1.2, labels={"provider": "anthropic"})
    rendered = metrics.render().decode("utf-8")
    assert 'provider="anthropic"' in rendered
