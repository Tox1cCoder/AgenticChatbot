"""Exact-match Redis caches for RAG embeddings and retrieval results.

Exactly three caches exist, matching the approved plan -- no semantic
similarity matching, no answer caching:

* document embeddings, keyed by exact content SHA-256
* query embeddings, keyed by the exact normalized query text
* retrieval fusion results, keyed by the conversation's exact active
  generation fingerprint, with a short TTL

Every key is scoped to the requesting tenant, including the document-embedding
key. Redis failures (connection errors, timeouts, malformed payloads) always
degrade to a cache miss and are never raised to the caller, and an empty
``redis_url`` means "cache disabled" independently of the feature flag. This
follows the same shape as the two existing Redis-backed stores in this
codebase: ``app/services/client_runtime_store.py`` and
``app/observability/model_usage_failure_store.py``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from hashlib import sha256
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_DOCUMENT_EMBEDDING_PREFIX = "rag:doc-embedding:"
_QUERY_EMBEDDING_PREFIX = "rag:query-embedding:"
_RETRIEVAL_PREFIX = "rag:retrieval:"


def _digest(parts: Sequence[str]) -> str:
    encoded = "\x1f".join(parts).encode("utf-8")
    return sha256(encoded).hexdigest()


def document_embedding_key(
    *,
    tenant: str,
    provider: str,
    model: str,
    dimension: int,
    format_version: str,
    content_sha256: str,
) -> str:
    return _DOCUMENT_EMBEDDING_PREFIX + _digest(
        [tenant, provider, model, str(dimension), format_version, content_sha256]
    )


def query_embedding_key(
    *,
    tenant: str,
    provider: str,
    model: str,
    dimension: int,
    task_prefix: str,
    normalized_query: str,
) -> str:
    return _QUERY_EMBEDDING_PREFIX + _digest(
        [tenant, provider, model, str(dimension), task_prefix, normalized_query]
    )


def retrieval_key(
    *,
    tenant: str,
    conversation: str,
    generation: str,
    normalized_query: str,
    retrieval_config_sha256: str,
) -> str:
    return _RETRIEVAL_PREFIX + _digest(
        [tenant, conversation, generation, normalized_query, retrieval_config_sha256]
    )


def normalize_query(query: str) -> str:
    """Canonicalize whitespace and case only. No semantic normalization.

    Two queries are the same cache entry only when they are the same text up
    to whitespace and case -- there is no fuzzy or embedding-similarity match
    anywhere in this module.
    """
    return " ".join(str(query or "").split()).casefold()


def retrieval_config_sha256(
    *,
    dense_candidate_limit: int,
    lexical_candidate_limit: int,
    rrf_k: int,
    hybrid_enabled: bool,
    score_threshold: float | None,
    cache_enabled: bool,
    query_embedding_cache_ttl_seconds: int,
    retrieval_cache_ttl_seconds: int,
) -> str:
    """Hash every retrieval knob that changes what a cached result means.

    Cache settings themselves are part of this hash so flipping the exact
    cache on/off, or changing a TTL, never reuses another configuration's
    entries.
    """
    return _digest(
        [
            str(dense_candidate_limit),
            str(lexical_candidate_limit),
            str(rrf_k),
            str(bool(hybrid_enabled)),
            str(score_threshold),
            str(bool(cache_enabled)),
            str(query_embedding_cache_ttl_seconds),
            str(retrieval_cache_ttl_seconds),
        ]
    )


class RAGExactCache(Protocol):
    """Interface implemented by both the disabled and Redis-backed caches."""

    enabled: bool

    def get_document_embedding(self, key: str, *, dimension: int) -> list[float] | None: ...

    def set_document_embedding(self, key: str, vector: Sequence[float]) -> None: ...

    def get_query_embedding(self, key: str, *, dimension: int) -> list[float] | None: ...

    def set_query_embedding(
        self, key: str, vector: Sequence[float], *, ttl_seconds: int
    ) -> None: ...

    def get_retrieval(self, key: str) -> dict[str, Any] | None: ...

    def set_retrieval(self, key: str, payload: dict[str, Any], *, ttl_seconds: int) -> None: ...


class NullRAGExactCache:
    """No-op cache used when caching is disabled or Redis is unavailable.

    Every read is a miss and every write is dropped, so a caller wired
    against this implementation behaves exactly as if no cache existed.
    """

    enabled = False

    def get_document_embedding(self, key: str, *, dimension: int) -> list[float] | None:
        return None

    def set_document_embedding(self, key: str, vector: Sequence[float]) -> None:
        return None

    def get_query_embedding(self, key: str, *, dimension: int) -> list[float] | None:
        return None

    def set_query_embedding(self, key: str, vector: Sequence[float], *, ttl_seconds: int) -> None:
        return None

    def get_retrieval(self, key: str) -> dict[str, Any] | None:
        return None

    def set_retrieval(self, key: str, payload: dict[str, Any], *, ttl_seconds: int) -> None:
        return None


def _decode_vector(raw: Any, *, dimension: int) -> list[float] | None:
    """Parse a JSON float array and validate its dimension. A mismatch is a miss."""
    if not raw:
        return None
    try:
        values = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(values, list) or len(values) != dimension:
        return None
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return None


def _decode_payload(raw: Any) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


class RedisRAGExactCache:
    """Redis-backed exact cache. Any failure degrades to a miss, never an error.

    Takes an already-constructed client so tests can inject a deterministic
    fake instead of a live Redis connection; production construction goes
    through :func:`build_rag_exact_cache`.
    """

    enabled = True

    def __init__(self, client: Any) -> None:
        self._client = client

    def get_document_embedding(self, key: str, *, dimension: int) -> list[float] | None:
        return self._get_vector(key, dimension=dimension)

    def set_document_embedding(self, key: str, vector: Sequence[float]) -> None:
        # Content-addressed by an exact SHA-256: the same key can only ever
        # mean the same input text under a fixed provider/model/dimension, so
        # this entry never needs to expire.
        self._set(key, json.dumps([float(value) for value in vector]))

    def get_query_embedding(self, key: str, *, dimension: int) -> list[float] | None:
        return self._get_vector(key, dimension=dimension)

    def set_query_embedding(self, key: str, vector: Sequence[float], *, ttl_seconds: int) -> None:
        self._set(
            key, json.dumps([float(value) for value in vector]), ttl_seconds=ttl_seconds
        )

    def get_retrieval(self, key: str) -> dict[str, Any] | None:
        try:
            raw = self._client.get(key)
        except Exception:
            return None
        return _decode_payload(raw)

    def set_retrieval(self, key: str, payload: dict[str, Any], *, ttl_seconds: int) -> None:
        self._set(key, json.dumps(payload), ttl_seconds=ttl_seconds)

    def _get_vector(self, key: str, *, dimension: int) -> list[float] | None:
        try:
            raw = self._client.get(key)
        except Exception:
            return None
        return _decode_vector(raw, dimension=dimension)

    def _set(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        try:
            if ttl_seconds is None:
                self._client.set(key, value)
            else:
                self._client.set(key, value, ex=max(1, int(ttl_seconds)))
        except Exception:
            logger.warning("RAG exact cache write failed; continuing without cache")


def build_rag_exact_cache(settings: Any) -> RAGExactCache:
    """Build the shared exact cache from settings.

    Disabled (``NullRAGExactCache``) unless ``rag_exact_cache_enabled`` is
    true AND ``redis_url`` is non-empty -- an empty URL means "cache
    disabled" regardless of the feature flag. Any connection failure falls
    back to disabled rather than raising, matching every other Redis-backed
    store in this codebase.
    """
    if not bool(getattr(settings, "rag_exact_cache_enabled", False)):
        return NullRAGExactCache()
    redis_url = (getattr(settings, "redis_url", "") or "").strip()
    if not redis_url:
        return NullRAGExactCache()
    try:
        import redis

        client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.25,
            socket_timeout=0.25,
        )
        client.ping()
        return RedisRAGExactCache(client)
    except Exception:
        logger.warning("RAG exact cache falling back to disabled: Redis is unavailable")
        return NullRAGExactCache()
