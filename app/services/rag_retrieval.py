"""Tenant-scoped hybrid retrieval with SQL as the authorization authority."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal
from uuid import UUID

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from app.services.rag_cache import (
    NullRAGExactCache,
    RAGExactCache,
    normalize_query,
    query_embedding_key,
    retrieval_config_sha256,
    retrieval_key,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalScope:
    user_id: str
    conversation_id: UUID


@dataclass(frozen=True)
class RetrievalCandidate:
    document_id: UUID
    chunk_id: UUID | None
    image_id: UUID | None
    modality: Literal["text", "image"]
    content: str
    filename: str
    page_start: int | None
    page_end: int | None
    section_path: tuple[str, ...]
    dense_rank: int | None
    dense_score: float | None
    lexical_rank: int | None
    lexical_score: float | None
    fused_score: float
    rerank_score: float | None = None
    chunk_index: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class FusedRank:
    candidate_id: str
    dense_rank: int | None
    lexical_rank: int | None
    fused_score: float


def _first_ranks(candidate_ids) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, candidate_id in enumerate(candidate_ids, start=1):
        ranks.setdefault(str(candidate_id), rank)
    return ranks


def reciprocal_rank_fusion(*, dense, lexical, k: int = 60) -> list[FusedRank]:
    """Fuse source ranks without interpreting provider scores as probabilities."""
    bounded_k = max(1, int(k))
    dense_ranks = _first_ranks(dense)
    lexical_ranks = _first_ranks(lexical)
    candidate_ids = set(dense_ranks) | set(lexical_ranks)
    fused = [
        FusedRank(
            candidate_id=candidate_id,
            dense_rank=dense_ranks.get(candidate_id),
            lexical_rank=lexical_ranks.get(candidate_id),
            fused_score=(
                (1 / (bounded_k + dense_ranks[candidate_id]))
                if candidate_id in dense_ranks
                else 0.0
            )
            + (
                (1 / (bounded_k + lexical_ranks[candidate_id]))
                if candidate_id in lexical_ranks
                else 0.0
            ),
        )
        for candidate_id in candidate_ids
    ]
    return sorted(
        fused,
        key=lambda row: (
            -row.fused_score,
            min(row.dense_rank or 2**31, row.lexical_rank or 2**31),
            row.candidate_id,
        ),
    )


def active_generation_fingerprint(generation_ids) -> str:
    canonical = "\n".join(sorted(str(generation_id) for generation_id in generation_ids))
    return sha256(canonical.encode("utf-8")).hexdigest()


class RAGRetriever:
    def __init__(
        self,
        *,
        qdrant_client: Any,
        embedding_service: Any,
        chunk_repository: Any,
        collection_name: str,
        hybrid_enabled: bool = False,
        dense_candidate_limit: int = 40,
        lexical_candidate_limit: int = 40,
        rrf_k: int = 60,
        score_threshold: float | None = None,
        document_image_repository: Any | None = None,
        cache: RAGExactCache | None = None,
        query_embedding_cache_ttl_seconds: int = 300,
        retrieval_cache_ttl_seconds: int = 60,
        metrics: Any | None = None,
    ) -> None:
        self.qdrant_client = qdrant_client
        self.embedding_service = embedding_service
        self.chunk_repository = chunk_repository
        self.collection_name = collection_name
        self.hybrid_enabled = bool(hybrid_enabled)
        self.dense_candidate_limit = max(1, int(dense_candidate_limit))
        self.lexical_candidate_limit = max(1, int(lexical_candidate_limit))
        self.rrf_k = max(1, int(rrf_k))
        self.score_threshold = score_threshold
        # Task 11: native multimodal image points. Optional — only
        # ``search_images`` requires it, and that path is only ever called
        # behind ``rag_multimodal_image_embeddings_enabled``.
        self.document_image_repository = document_image_repository
        self.last_trace: dict[str, Any] = {}
        # Task 12: exact query-embedding and retrieval-result caches. Defaults
        # to a no-op so a caller that never wires a cache is byte-for-byte
        # identical to the pre-Task-12 code path.
        self.cache = cache if cache is not None else NullRAGExactCache()
        self.query_embedding_cache_ttl_seconds = max(1, int(query_embedding_cache_ttl_seconds))
        self.retrieval_cache_ttl_seconds = max(1, int(retrieval_cache_ttl_seconds))
        self.metrics = metrics

    def search(
        self,
        query: str,
        scope: RetrievalScope,
        *,
        dense_candidate_limit: int | None = None,
        lexical_candidate_limit: int | None = None,
        final_limit: int = 10,
        generation_fingerprint: str | None = None,
        active_generation_ids: Iterable[UUID] | None = None,
    ) -> list[RetrievalCandidate]:
        if not query or not scope.user_id or not scope.conversation_id:
            return []

        dense_limit = max(1, int(dense_candidate_limit or self.dense_candidate_limit))
        lexical_limit = max(1, int(lexical_candidate_limit or self.lexical_candidate_limit))
        output_limit = max(1, int(final_limit))
        if active_generation_ids is None:
            active_generation_ids = self.chunk_repository.get_active_generation_ids_for_scope(
                user_id=scope.user_id,
                conversation_id=scope.conversation_id,
            )
        generation_ids = tuple(
            sorted(
                (self._coerce_uuid(generation_id) for generation_id in active_generation_ids),
                key=str,
            )
        )
        if generation_fingerprint is None:
            generation_fingerprint = active_generation_fingerprint(generation_ids)
        self.last_trace = {
            "active_generation_fingerprint": generation_fingerprint,
            "dense_candidate_limit": dense_limit,
            "lexical_candidate_limit": lexical_limit,
            "final_limit": output_limit,
            "rrf_k": self.rrf_k,
            "hybrid_enabled": self.hybrid_enabled,
        }
        if not generation_ids:
            return []

        fused, dense_scores, lexical_scores, retrieval_cache_result = self._fused_candidates(
            query,
            scope,
            dense_limit=dense_limit,
            lexical_limit=lexical_limit,
            generation_ids=generation_ids,
            generation_fingerprint=generation_fingerprint,
        )
        if not fused:
            return []

        winner_ids = [UUID(row.candidate_id) for row in fused]
        hydration_t0 = time.monotonic()
        try:
            authorized = self.chunk_repository.get_active_by_ids_for_scope(
                winner_ids,
                user_id=scope.user_id,
                conversation_id=scope.conversation_id,
            )
        except Exception:
            self._record_stage(
                "sql_hydration",
                time.monotonic() - hydration_t0,
                modality="text",
                cache_result=retrieval_cache_result,
            )
            self._record_stage_failure("sql_hydration", "dependency_exception")
            raise
        self._record_stage(
            "sql_hydration",
            time.monotonic() - hydration_t0,
            modality="text",
            cache_result=retrieval_cache_result,
        )
        chunks_by_id = {str(chunk.id): chunk for chunk in authorized}

        results: list[RetrievalCandidate] = []
        for rank in fused:
            chunk = chunks_by_id.get(rank.candidate_id)
            if chunk is None:
                continue
            document = getattr(chunk, "document", None)
            results.append(
                RetrievalCandidate(
                    document_id=self._coerce_uuid(chunk.document_id),
                    chunk_id=self._coerce_uuid(chunk.id),
                    image_id=None,
                    modality="text",
                    content=str(chunk.content),
                    filename=str(getattr(document, "filename", None) or "unknown"),
                    page_start=getattr(chunk, "page_start", None),
                    page_end=getattr(chunk, "page_end", None),
                    section_path=tuple(getattr(chunk, "section_path", None) or ()),
                    dense_rank=rank.dense_rank,
                    dense_score=dense_scores.get(rank.candidate_id),
                    lexical_rank=rank.lexical_rank,
                    lexical_score=lexical_scores.get(rank.candidate_id),
                    fused_score=rank.fused_score,
                    chunk_index=getattr(chunk, "chunk_index", None),
                    metadata={
                        **dict(getattr(chunk, "chunk_metadata", None) or {}),
                        "block_provenance": list(getattr(chunk, "block_provenance", None) or []),
                    },
                )
            )
            if len(results) >= output_limit:
                break
        return results

    def _fused_candidates(
        self,
        query: str,
        scope: RetrievalScope,
        *,
        dense_limit: int,
        lexical_limit: int,
        generation_ids: tuple[UUID, ...],
        generation_fingerprint: str,
    ) -> tuple[list[FusedRank], dict[str, float], dict[str, float], str]:
        """Fuse dense + lexical ranks, using the exact retrieval cache when hit.

        A cache hit only replaces the Qdrant dense search, the SQL lexical
        search, and the RRF fusion arithmetic below -- it never replaces the
        SQL re-authorization step the caller performs afterward on the
        returned candidate ids, so a cached result cannot bypass tenant
        authorization. The fourth return value is the retrieval-cache
        hit/miss/disabled label, threaded back to the caller so the
        ``sql_hydration`` stage recording can carry it too.
        """
        cache_key = self._retrieval_cache_key(
            scope,
            generation_fingerprint,
            query,
            dense_limit=dense_limit,
            lexical_limit=lexical_limit,
        )
        cached = self.cache.get_retrieval(cache_key)
        cache_result = self._record_cache_result("retrieval", hit=cached is not None)
        if cached is not None:
            fused, dense_scores, lexical_scores = self._fused_from_cached_payload(cached)
            return fused, dense_scores, lexical_scores, cache_result

        provider = str(getattr(self.embedding_service, "provider", "") or "")
        model = str(getattr(self.embedding_service, "model_name", "") or "")
        dense_ids, dense_scores = self._timed_dense_search(
            query,
            scope,
            dense_limit,
            generation_ids=generation_ids,
            provider=provider,
            model=model,
        )

        lexical_ids: list[str] = []
        lexical_scores: dict[str, float] = {}
        if self.hybrid_enabled:
            lexical_rows = self._timed_lexical_search(query, scope, lexical_limit=lexical_limit)
            for chunk, raw_score in lexical_rows:
                candidate_key = str(chunk.id)
                if candidate_key in lexical_scores:
                    continue
                lexical_ids.append(candidate_key)
                lexical_scores[candidate_key] = float(raw_score)

        fused = reciprocal_rank_fusion(dense=dense_ids, lexical=lexical_ids, k=self.rrf_k)
        if fused:
            self.cache.set_retrieval(
                cache_key,
                {
                    "fused": [
                        {
                            "candidate_id": row.candidate_id,
                            "dense_rank": row.dense_rank,
                            "lexical_rank": row.lexical_rank,
                            "fused_score": row.fused_score,
                        }
                        for row in fused
                    ],
                    "dense_scores": dense_scores,
                    "lexical_scores": lexical_scores,
                },
                ttl_seconds=self.retrieval_cache_ttl_seconds,
            )
        return fused, dense_scores, lexical_scores, cache_result

    def _timed_dense_search(
        self,
        query: str,
        scope: RetrievalScope,
        dense_limit: int,
        *,
        generation_ids: tuple[UUID, ...],
        provider: str,
        model: str,
    ) -> tuple[list[str], dict[str, float]]:
        """Run the dense Qdrant search, recording duration on every outcome."""
        dense_t0 = time.monotonic()
        try:
            dense_points, query_cache_result = self._dense_search(
                query,
                scope,
                dense_limit,
                active_generation_ids=generation_ids,
                modality="text",
            )
        except Exception:
            self._record_stage(
                "dense_retrieval",
                time.monotonic() - dense_t0,
                provider=provider,
                model=model,
                modality="text",
            )
            self._record_stage_failure("dense_retrieval", "dependency_exception")
            raise
        self._record_stage(
            "dense_retrieval",
            time.monotonic() - dense_t0,
            provider=provider,
            model=model,
            modality="text",
            cache_result=query_cache_result,
        )
        dense_ids, dense_scores = self._dense_ids_and_scores(dense_points)
        return dense_ids, dense_scores

    def _timed_lexical_search(
        self,
        query: str,
        scope: RetrievalScope,
        *,
        lexical_limit: int,
    ) -> list[Any]:
        """Run the SQL lexical search, recording duration on every outcome."""
        lexical_t0 = time.monotonic()
        try:
            lexical_rows = self.chunk_repository.search_active_lexical_for_scope(
                query,
                user_id=scope.user_id,
                conversation_id=scope.conversation_id,
                limit=lexical_limit,
            )
        except Exception:
            self._record_stage("lexical_retrieval", time.monotonic() - lexical_t0, modality="text")
            self._record_stage_failure("lexical_retrieval", "dependency_exception")
            raise
        self._record_stage("lexical_retrieval", time.monotonic() - lexical_t0, modality="text")
        return lexical_rows

    @staticmethod
    def _fused_from_cached_payload(
        cached: dict[str, Any],
    ) -> tuple[list[FusedRank], dict[str, float], dict[str, float]]:
        fused = [
            FusedRank(
                candidate_id=str(row["candidate_id"]),
                dense_rank=row.get("dense_rank"),
                lexical_rank=row.get("lexical_rank"),
                fused_score=float(row["fused_score"]),
            )
            for row in cached.get("fused") or ()
        ]
        dense_scores = {str(k): float(v) for k, v in (cached.get("dense_scores") or {}).items()}
        lexical_scores = {str(k): float(v) for k, v in (cached.get("lexical_scores") or {}).items()}
        return fused, dense_scores, lexical_scores

    def _retrieval_cache_key(
        self,
        scope: RetrievalScope,
        generation_fingerprint: str,
        query: str,
        *,
        dense_limit: int,
        lexical_limit: int,
    ) -> str:
        config_hash = retrieval_config_sha256(
            dense_candidate_limit=dense_limit,
            lexical_candidate_limit=lexical_limit,
            rrf_k=self.rrf_k,
            hybrid_enabled=self.hybrid_enabled,
            score_threshold=self.score_threshold,
            cache_enabled=bool(getattr(self.cache, "enabled", False)),
            query_embedding_cache_ttl_seconds=self.query_embedding_cache_ttl_seconds,
            retrieval_cache_ttl_seconds=self.retrieval_cache_ttl_seconds,
        )
        return retrieval_key(
            tenant=str(scope.user_id),
            conversation=str(scope.conversation_id),
            generation=generation_fingerprint,
            normalized_query=normalize_query(query),
            retrieval_config_sha256=config_hash,
        )

    def _dense_ids_and_scores(self, points: list[Any]) -> tuple[list[str], dict[str, float]]:
        ids: list[str] = []
        scores: dict[str, float] = {}
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            candidate_id = self._valid_chunk_id(payload.get("chunk_id"))
            if candidate_id is None:
                continue
            candidate_key = str(candidate_id)
            if candidate_key in scores:
                continue
            ids.append(candidate_key)
            scores[candidate_key] = float(getattr(point, "score", 0.0))
        return ids, scores

    def _embed_query_cached(self, query: str, scope: RetrievalScope) -> tuple[list[float], str]:
        """Embed the query, using the exact cache when possible.

        Returns the vector and the cache-result label so the caller can
        attribute the ``dense_retrieval`` stage's duration to a hit/miss.
        """
        dimension = int(getattr(self.embedding_service, "dimension", 0) or 0)
        if dimension <= 0:
            # Round-1 fix (finding 8): fail closed. A key hashed with a
            # placeholder dimension can never validate on read (every read
            # would demand ``len(vector) == 0``), so skip the cache entirely
            # rather than write an entry that can never be served back.
            vector = list(self.embedding_service.embed_query(query))
            return vector, "disabled"

        key = query_embedding_key(
            tenant=str(scope.user_id),
            provider=str(getattr(self.embedding_service, "provider", "") or ""),
            model=str(getattr(self.embedding_service, "model_name", "") or ""),
            dimension=dimension,
            task_prefix=str(getattr(self.embedding_service, "query_task", "") or ""),
            normalized_query=normalize_query(query),
        )
        cached = self.cache.get_query_embedding(key, dimension=dimension)
        cache_result = self._record_cache_result("query_embedding", hit=cached is not None)
        if cached is not None:
            return cached, cache_result
        vector = list(self.embedding_service.embed_query(query))
        self.cache.set_query_embedding(
            key, vector, ttl_seconds=self.query_embedding_cache_ttl_seconds
        )
        return vector, cache_result

    def _record_stage(self, stage: str, elapsed_seconds: float, **labels: Any) -> None:
        recorder = getattr(self.metrics, "stage", None)
        if not callable(recorder):
            return
        try:
            recorder(stage, elapsed_seconds=elapsed_seconds, labels=labels)
        except Exception:
            logger.exception("Failed to record RAG stage metric for %s", stage)

    def _record_stage_failure(self, stage: str, failure_code: str) -> None:
        recorder = getattr(self.metrics, "stage_failure", None)
        if not callable(recorder):
            return
        try:
            recorder(stage, failure_code)
        except Exception:
            logger.exception("Failed to record RAG stage-failure metric for %s", stage)

    def _record_cache_result(self, cache_name: str, *, hit: bool) -> str:
        cache_enabled = bool(getattr(self.cache, "enabled", False))
        result = "hit" if hit else ("miss" if cache_enabled else "disabled")
        recorder = getattr(self.metrics, "cache_result", None)
        if callable(recorder):
            try:
                recorder(cache_name, result)
            except Exception:
                logger.exception("Failed to record RAG cache metric for %s", cache_name)
        return result

    def search_images(
        self,
        query: str,
        scope: RetrievalScope,
        *,
        limit: int = 10,
        active_generation_ids: Iterable[UUID] | None = None,
    ) -> list[RetrievalCandidate]:
        """Dense-search native ``modality="image"`` points and hydrate them.

        Qdrant payloads carry only lookup metadata; each candidate's
        ``DocumentImage`` row is re-authorized through its parent document via
        ``document_image_repository.get_by_id_for_scope`` before it is ever
        returned, exactly like text hydration re-checks scope in SQL rather
        than trusting the Qdrant payload. Caption-first retrieval never calls
        this method — it is additive and only meaningful when native image
        embeddings were indexed (``rag_multimodal_image_embeddings_enabled``).
        """
        if not query or not scope.user_id or not scope.conversation_id:
            return []
        if self.document_image_repository is None:
            return []

        if active_generation_ids is None:
            active_generation_ids = self.chunk_repository.get_active_generation_ids_for_scope(
                user_id=scope.user_id,
                conversation_id=scope.conversation_id,
            )
        generation_ids = tuple(
            self._coerce_uuid(generation_id) for generation_id in active_generation_ids
        )
        if not generation_ids:
            return []

        output_limit = max(1, int(limit))
        points, _query_cache_result = self._dense_search(
            query,
            scope,
            output_limit,
            active_generation_ids=generation_ids,
            modality="image",
        )

        results: list[RetrievalCandidate] = []
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            image_id = self._valid_chunk_id(payload.get("image_id"))
            if image_id is None:
                continue
            image = self.document_image_repository.get_by_id_for_scope(
                image_id,
                user_id=scope.user_id,
                conversation_id=scope.conversation_id,
            )
            if image is None:
                continue
            document_id = self._valid_chunk_id(getattr(image, "document_id", None))
            if document_id is None:
                continue
            page_number = getattr(image, "page_number", None)
            score = float(getattr(point, "score", 0.0))
            results.append(
                RetrievalCandidate(
                    document_id=document_id,
                    chunk_id=self._valid_chunk_id(getattr(image, "chunk_id", None)),
                    image_id=image_id,
                    modality="image",
                    content=str(getattr(image, "image_caption", None) or ""),
                    filename=str(
                        getattr(getattr(image, "document", None), "filename", None) or "unknown"
                    ),
                    page_start=page_number,
                    page_end=page_number,
                    section_path=tuple(getattr(image, "section_path", None) or ()),
                    dense_rank=len(results) + 1,
                    dense_score=score,
                    lexical_rank=None,
                    lexical_score=None,
                    fused_score=score,
                    metadata={"content_sha256": getattr(image, "content_sha256", None)},
                )
            )
            if len(results) >= output_limit:
                break
        return results

    def _dense_search(
        self,
        query: str,
        scope: RetrievalScope,
        limit: int,
        *,
        active_generation_ids: Iterable[UUID],
        modality: Literal["text", "image"] = "text",
    ) -> tuple[list[Any], str]:
        """Search Qdrant, returning the points and the query-embedding cache result."""
        generation_values = sorted(
            str(self._coerce_uuid(generation_id)) for generation_id in active_generation_ids
        )
        if not generation_values:
            return [], "n/a"
        query_embedding, query_cache_result = self._embed_query_cached(query, scope)
        search_filter = Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=str(scope.user_id))),
                FieldCondition(
                    key="conversation_id",
                    match=MatchValue(value=str(scope.conversation_id)),
                ),
                FieldCondition(key="modality", match=MatchValue(value=modality)),
                FieldCondition(
                    key="index_generation",
                    match=MatchAny(any=generation_values),
                ),
                FieldCondition(key="is_active", match=MatchValue(value=True)),
            ]
        )
        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            score_threshold=self.score_threshold,
            query_filter=search_filter,
        )
        return list(getattr(response, "points", None) or []), query_cache_result

    @staticmethod
    def _valid_chunk_id(value: Any) -> UUID | None:
        try:
            return UUID(str(value))
        except (TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def _coerce_uuid(value: Any) -> UUID:
        return value if isinstance(value, UUID) else UUID(str(value))
