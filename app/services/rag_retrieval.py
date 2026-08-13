"""Tenant-scoped hybrid retrieval with SQL as the authorization authority."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal
from uuid import UUID

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue


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


def reciprocal_rank_fusion(
    *, dense, lexical, k: int = 60
) -> list[FusedRank]:
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
        self.last_trace: dict[str, Any] = {}

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
        lexical_limit = max(
            1, int(lexical_candidate_limit or self.lexical_candidate_limit)
        )
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

        dense_points = self._dense_search(
            query,
            scope,
            dense_limit,
            active_generation_ids=generation_ids,
            modality="text",
        )
        dense_ids: list[str] = []
        dense_scores: dict[str, float] = {}
        for point in dense_points:
            payload = dict(getattr(point, "payload", None) or {})
            candidate_id = self._valid_chunk_id(payload.get("chunk_id"))
            if candidate_id is None:
                continue
            candidate_key = str(candidate_id)
            if candidate_key in dense_scores:
                continue
            dense_ids.append(candidate_key)
            dense_scores[candidate_key] = float(getattr(point, "score", 0.0))

        lexical_ids: list[str] = []
        lexical_scores: dict[str, float] = {}
        if self.hybrid_enabled:
            lexical_rows = self.chunk_repository.search_active_lexical_for_scope(
                query,
                user_id=scope.user_id,
                conversation_id=scope.conversation_id,
                limit=lexical_limit,
            )
            for chunk, raw_score in lexical_rows:
                candidate_key = str(chunk.id)
                if candidate_key in lexical_scores:
                    continue
                lexical_ids.append(candidate_key)
                lexical_scores[candidate_key] = float(raw_score)

        fused = reciprocal_rank_fusion(
            dense=dense_ids,
            lexical=lexical_ids,
            k=self.rrf_k,
        )
        if not fused:
            return []

        winner_ids = [UUID(row.candidate_id) for row in fused]
        authorized = self.chunk_repository.get_active_by_ids_for_scope(
            winner_ids,
            user_id=scope.user_id,
            conversation_id=scope.conversation_id,
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
                    metadata=dict(getattr(chunk, "chunk_metadata", None) or {}),
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
    ) -> list[Any]:
        generation_values = sorted(
            str(self._coerce_uuid(generation_id))
            for generation_id in active_generation_ids
        )
        if not generation_values:
            return []
        query_embedding = list(self.embedding_service.embed_query(query))
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id", match=MatchValue(value=str(scope.user_id))
                ),
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
        return list(getattr(response, "points", None) or [])

    @staticmethod
    def _valid_chunk_id(value: Any) -> UUID | None:
        try:
            return UUID(str(value))
        except (TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def _coerce_uuid(value: Any) -> UUID:
        return value if isinstance(value, UUID) else UUID(str(value))
