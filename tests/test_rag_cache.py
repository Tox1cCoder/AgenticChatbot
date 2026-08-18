"""Exact-cache key isolation and Redis-failure/dimension-validation guards.

No live Redis, PostgreSQL, Qdrant or model provider anywhere in this file --
``_FakeRedis`` is a deterministic in-process fake, and every "failure" case
below is a fake that raises on demand so the miss-not-error contract can be
exercised without a real network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.services.rag_cache import (
    NullRAGExactCache,
    RedisRAGExactCache,
    build_rag_exact_cache,
    document_embedding_key,
    normalize_query,
    query_embedding_key,
    retrieval_config_sha256,
    retrieval_key,
)


class _FakeRedis:
    """In-memory stand-in for a redis-py client, with optional failure modes."""

    def __init__(self, *, fail: bool = False) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.fail = fail

    def get(self, key: str) -> str | None:
        if self.fail:
            raise ConnectionError("redis unavailable")
        return self.store.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        if self.fail:
            raise ConnectionError("redis unavailable")
        self.store[key] = value
        if ex is not None:
            self.ttls[key] = ex

    def ping(self) -> bool:
        if self.fail:
            raise ConnectionError("redis unavailable")
        return True


def _settings(*, enabled: bool, redis_url: str = "redis://localhost:6379/0") -> SimpleNamespace:
    return SimpleNamespace(rag_exact_cache_enabled=enabled, redis_url=redis_url)


# ---------------------------------------------------------------------------
# Step 1 test: cache-key isolation across tenant and generation
# ---------------------------------------------------------------------------


def test_query_cache_key_changes_with_tenant_and_generation():
    a = retrieval_key(
        tenant="u1",
        conversation="c1",
        generation="g1",
        normalized_query="revenue?",
        retrieval_config_sha256="cfg",
    )
    b = retrieval_key(
        tenant="u2",
        conversation="c1",
        generation="g1",
        normalized_query="revenue?",
        retrieval_config_sha256="cfg",
    )
    c = retrieval_key(
        tenant="u1",
        conversation="c1",
        generation="g2",
        normalized_query="revenue?",
        retrieval_config_sha256="cfg",
    )
    assert len({a, b, c}) == 3


def test_document_embedding_key_is_tenant_scoped():
    """The document-embedding key must include tenant, not only content hash."""
    same_content_kwargs = dict(
        provider="gemini",
        model="gemini-embedding-2",
        dimension=768,
        format_version="doc-fmt-v1",
        content_sha256="abc123",
    )
    a = document_embedding_key(tenant="tenant-a", **same_content_kwargs)
    b = document_embedding_key(tenant="tenant-b", **same_content_kwargs)
    assert a != b


def test_query_embedding_key_changes_with_normalized_query_only():
    """Exact match only -- no fuzzy/semantic collapsing of distinct queries."""
    base = dict(tenant="u1", provider="gemini", model="m", dimension=8, task_prefix="search")
    a = query_embedding_key(normalized_query="what is revenue", **base)
    b = query_embedding_key(normalized_query="what is profit", **base)
    same_as_a = query_embedding_key(normalized_query="what is revenue", **base)
    assert a != b
    assert a == same_as_a


def test_normalize_query_collapses_whitespace_and_case_only():
    assert normalize_query("  Revenue   Growth  ") == "revenue growth"
    # Distinct wording is never collapsed -- exact match only.
    assert normalize_query("revenue growth") != normalize_query("growth in revenue")


def test_retrieval_config_sha256_changes_when_cache_settings_change():
    """Cache settings are part of the retrieval configuration hash."""
    base = dict(
        dense_candidate_limit=40,
        lexical_candidate_limit=40,
        rrf_k=60,
        hybrid_enabled=False,
        score_threshold=0.2,
        query_embedding_cache_ttl_seconds=300,
        retrieval_cache_ttl_seconds=60,
    )
    enabled = retrieval_config_sha256(cache_enabled=True, **base)
    disabled = retrieval_config_sha256(cache_enabled=False, **base)
    assert enabled != disabled


# ---------------------------------------------------------------------------
# Redis failure is a cache miss, never a propagated error
# ---------------------------------------------------------------------------


def test_redis_read_failure_is_a_miss_not_an_exception():
    cache = RedisRAGExactCache(_FakeRedis(fail=True))
    assert cache.get_document_embedding("some-key", dimension=4) is None
    assert cache.get_query_embedding("some-key", dimension=4) is None
    assert cache.get_retrieval("some-key") is None


def test_redis_write_failure_does_not_raise():
    cache = RedisRAGExactCache(_FakeRedis(fail=True))
    # None of these should raise even though every underlying call fails.
    cache.set_document_embedding("k", [1.0, 2.0])
    cache.set_query_embedding("k", [1.0, 2.0], ttl_seconds=300)
    cache.set_retrieval("k", {"fused": []}, ttl_seconds=60)


def test_build_rag_exact_cache_falls_back_when_redis_construction_raises(monkeypatch):
    import app.services.rag_cache as rag_cache_module

    class _ExplodingRedisModule:
        class Redis:
            @staticmethod
            def from_url(*_args, **_kwargs):
                raise ConnectionError("no redis here")

    monkeypatch.setitem(__import__("sys").modules, "redis", _ExplodingRedisModule)
    cache = rag_cache_module.build_rag_exact_cache(_settings(enabled=True))
    assert isinstance(cache, NullRAGExactCache)


# ---------------------------------------------------------------------------
# An empty redis_url means "cache disabled", independent of the feature flag
# ---------------------------------------------------------------------------


def test_empty_redis_url_disables_cache_even_when_flag_enabled():
    cache = build_rag_exact_cache(_settings(enabled=True, redis_url=""))
    assert isinstance(cache, NullRAGExactCache)


def test_feature_flag_off_disables_cache_even_with_redis_url():
    cache = build_rag_exact_cache(_settings(enabled=False, redis_url="redis://localhost:6379/0"))
    assert isinstance(cache, NullRAGExactCache)


def test_null_cache_every_read_is_a_miss_and_writes_are_dropped():
    cache = NullRAGExactCache()
    assert cache.enabled is False
    assert cache.get_document_embedding("k", dimension=4) is None
    assert cache.get_query_embedding("k", dimension=4) is None
    assert cache.get_retrieval("k") is None
    cache.set_document_embedding("k", [1.0])
    cache.set_query_embedding("k", [1.0], ttl_seconds=1)
    cache.set_retrieval("k", {}, ttl_seconds=1)
    assert cache.get_document_embedding("k", dimension=4) is None


# ---------------------------------------------------------------------------
# Cached vectors are JSON float arrays validated on read
# ---------------------------------------------------------------------------


def test_wrong_dimension_vector_is_a_miss_not_a_value():
    fake = _FakeRedis()
    fake.store["k"] = json.dumps([1.0, 2.0, 3.0])
    cache = RedisRAGExactCache(fake)
    assert cache.get_document_embedding("k", dimension=8) is None


def test_correct_dimension_vector_round_trips():
    fake = _FakeRedis()
    cache = RedisRAGExactCache(fake)
    cache.set_document_embedding("k", [1.0, 2.0, 3.0])
    assert cache.get_document_embedding("k", dimension=3) == [1.0, 2.0, 3.0]


def test_malformed_json_vector_is_a_miss():
    fake = _FakeRedis()
    fake.store["k"] = "not json at all"
    cache = RedisRAGExactCache(fake)
    assert cache.get_document_embedding("k", dimension=3) is None


def test_document_embedding_set_uses_no_ttl_and_query_embedding_uses_ttl():
    fake = _FakeRedis()
    cache = RedisRAGExactCache(fake)
    cache.set_document_embedding("doc-key", [1.0])
    cache.set_query_embedding("query-key", [1.0], ttl_seconds=300)
    assert "doc-key" not in fake.ttls
    assert fake.ttls["query-key"] == 300


def test_retrieval_cache_round_trips_json_payload_with_ttl():
    fake = _FakeRedis()
    cache = RedisRAGExactCache(fake)
    payload = {"fused": [{"candidate_id": "abc", "fused_score": 0.5}], "dense_scores": {}}
    cache.set_retrieval("retrieval-key", payload, ttl_seconds=60)
    assert cache.get_retrieval("retrieval-key") == payload
    assert fake.ttls["retrieval-key"] == 60
