from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from qdrant_client.models import FieldCondition, MatchAny
from sqlalchemy import create_engine, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.base import Base
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_index_generation import DocumentIndexGeneration
from app.models.document_parse_artifact import DocumentParseArtifact
from app.models.user import User
from app.repositories.document_chunk import DocumentChunkRepository


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(_type, _compiler, **_kwargs):
    return "JSON"


class _EmbeddingStub:
    # Round-1 fix: a real embedding service always reports its own
    # provider/model/dimension/query_task; a stub without them made every
    # query-embedding cache key build with ``dimension=0``, which the
    # fail-closed guard (finding 8) always treats as uncacheable.
    provider = "gemini"
    model_name = "test-model"
    dimension = 2
    query_task = "search result"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query: str):
        self.queries.append(query)
        return [0.1, 0.2]


def _chunk(chunk_id: UUID, *, document_id: UUID | None = None, content: str = "body"):
    return SimpleNamespace(
        id=chunk_id,
        document_id=document_id or uuid4(),
        chunk_index=2,
        content=content,
        page_start=3,
        page_end=4,
        section_path=["Results"],
        chunk_metadata={"has_tables": True, "table_count": 1},
        document=SimpleNamespace(filename="report.pdf"),
    )


def _point(chunk_id: UUID, score: float, **payload):
    return SimpleNamespace(
        score=score,
        payload={
            "chunk_id": str(chunk_id),
            "document_id": str(payload.pop("document_id", uuid4())),
            "modality": "text",
            **payload,
        },
    )


def _retriever(*, hybrid_enabled: bool = True, points=None, hydrated=None, lexical=None):
    from app.services.rag_retrieval import RAGRetriever

    qdrant = MagicMock()
    qdrant.query_points.return_value = SimpleNamespace(points=list(points or []))
    embedding = _EmbeddingStub()
    repository = MagicMock()
    repository.get_active_by_ids_for_scope.return_value = list(hydrated or [])
    repository.search_active_lexical_for_scope.return_value = list(lexical or [])
    repository.get_active_generation_ids_for_scope.return_value = [
        UUID("00000000-0000-0000-0000-000000000002"),
        UUID("00000000-0000-0000-0000-000000000001"),
    ]
    retriever = RAGRetriever(
        qdrant_client=qdrant,
        embedding_service=embedding,
        chunk_repository=repository,
        collection_name="documents",
        hybrid_enabled=hybrid_enabled,
        dense_candidate_limit=40,
        lexical_candidate_limit=40,
        rrf_k=60,
        score_threshold=0.0,
    )
    return retriever, qdrant, embedding, repository


def test_rrf_combines_dense_and_lexical_ranks_deterministically():
    from app.services.rag_retrieval import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion(dense=["a", "b"], lexical=["b", "c"], k=60)

    assert [row.candidate_id for row in fused] == ["b", "a", "c"]
    assert fused[0].dense_rank == 2
    assert fused[0].lexical_rank == 1
    assert fused[0].fused_score == (1 / 62) + (1 / 61)


def test_rrf_deduplicates_repeated_source_candidates():
    from app.services.rag_retrieval import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion(dense=["a", "a", "b"], lexical=["a"], k=60)

    assert [row.candidate_id for row in fused] == ["a", "b"]
    assert fused[0].dense_rank == 1


def test_incomplete_scope_performs_no_embedding_qdrant_or_sql_work():
    from app.services.rag_retrieval import RetrievalScope

    retriever, qdrant, embedding, repository = _retriever()

    results = retriever.search(
        "query",
        RetrievalScope(user_id="", conversation_id=None),  # type: ignore[arg-type]
        final_limit=5,
    )

    assert results == []
    assert embedding.queries == []
    qdrant.query_points.assert_not_called()
    repository.search_active_lexical_for_scope.assert_not_called()
    repository.get_active_generation_ids_for_scope.assert_not_called()


def test_dense_query_filters_server_scope_active_generation_and_modality():
    from app.services.rag_retrieval import RetrievalScope

    user_id = str(uuid4())
    conversation_id = uuid4()
    retriever, qdrant, _, _ = _retriever(hybrid_enabled=False)

    retriever.search(
        "query",
        RetrievalScope(user_id=user_id, conversation_id=conversation_id),
        dense_candidate_limit=7,
        final_limit=3,
    )

    call = qdrant.query_points.call_args.kwargs
    assert call["limit"] == 7
    conditions = {
        condition.key: condition.match
        for condition in call["query_filter"].must
        if isinstance(condition, FieldCondition)
    }
    assert {key: match.value for key, match in conditions.items() if key != "index_generation"} == {
        "user_id": user_id,
        "conversation_id": str(conversation_id),
        "modality": "text",
        "is_active": True,
    }
    assert isinstance(conditions["index_generation"], MatchAny)
    assert conditions["index_generation"].any == [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]


def test_retired_points_cannot_starve_active_generation_from_dense_limit():
    from app.services.rag_retrieval import RetrievalScope

    active_generation_id = uuid4()
    retired_generation_id = uuid4()
    active_chunk_id = uuid4()
    active_row = _chunk(active_chunk_id, content="authorized active evidence")
    retired_points = [
        _point(
            uuid4(),
            1.0 - index / 1000,
            index_generation=str(retired_generation_id),
            is_active=True,
        )
        for index in range(40)
    ]
    active_point = _point(
        active_chunk_id,
        0.1,
        index_generation=str(active_generation_id),
        is_active=True,
    )
    retriever, qdrant, _, repository = _retriever(
        hybrid_enabled=False,
        hydrated=[active_row],
    )
    repository.get_active_generation_ids_for_scope.return_value = [active_generation_id]
    provider_points = retired_points + [active_point]

    def filtered_query_points(**kwargs):
        generation_match = next(
            condition.match
            for condition in kwargs["query_filter"].must
            if condition.key == "index_generation"
        )
        allowed = set(generation_match.any)
        filtered = [
            point
            for point in provider_points
            if point.payload["index_generation"] in allowed
        ]
        return SimpleNamespace(points=filtered[: kwargs["limit"]])

    qdrant.query_points.side_effect = filtered_query_points

    results = retriever.search(
        "active evidence",
        RetrievalScope(user_id=str(uuid4()), conversation_id=uuid4()),
        dense_candidate_limit=40,
        final_limit=1,
    )

    assert [candidate.chunk_id for candidate in results] == [active_chunk_id]


def test_no_active_generation_fails_closed_before_embedding_or_retrieval():
    from app.services.rag_retrieval import RetrievalScope

    retriever, qdrant, embedding, repository = _retriever()
    repository.get_active_generation_ids_for_scope.return_value = []

    results = retriever.search(
        "query",
        RetrievalScope(user_id=str(uuid4()), conversation_id=uuid4()),
        final_limit=5,
    )

    assert results == []
    assert embedding.queries == []
    qdrant.query_points.assert_not_called()
    repository.search_active_lexical_for_scope.assert_not_called()
    repository.get_active_by_ids_for_scope.assert_not_called()


def test_image_dense_filter_uses_the_same_active_generation_constraint():
    from app.services.rag_retrieval import RetrievalScope

    retriever, qdrant, _, repository = _retriever()
    generation_ids = repository.get_active_generation_ids_for_scope.return_value

    retriever._dense_search(
        "chart",
        RetrievalScope(user_id=str(uuid4()), conversation_id=uuid4()),
        4,
        active_generation_ids=generation_ids,
        modality="image",
    )

    matches = {
        condition.key: condition.match
        for condition in qdrant.query_points.call_args.kwargs["query_filter"].must
    }
    assert matches["modality"].value == "image"
    assert matches["index_generation"].any == sorted(
        str(generation_id) for generation_id in generation_ids
    )


def test_missing_or_stale_sql_rows_are_never_returned():
    from app.services.rag_retrieval import RetrievalScope

    chunk_id = uuid4()
    retriever, _, _, repository = _retriever(
        hybrid_enabled=False,
        points=[_point(chunk_id, 0.91)],
        hydrated=[],
    )

    results = retriever.search(
        "query",
        RetrievalScope(user_id=str(uuid4()), conversation_id=uuid4()),
        final_limit=5,
    )

    assert results == []
    repository.get_active_by_ids_for_scope.assert_called_once()


def test_hybrid_search_fuses_then_reauthorizes_winners_and_preserves_raw_scores():
    from app.services.rag_retrieval import RetrievalCandidate, RetrievalScope

    dense_only, shared, lexical_only = uuid4(), uuid4(), uuid4()
    rows = {
        dense_only: _chunk(dense_only, content="dense"),
        shared: _chunk(shared, content="shared"),
        lexical_only: _chunk(lexical_only, content="lexical"),
    }
    retriever, _, _, repository = _retriever(
        points=[_point(dense_only, 0.88), _point(shared, 0.52)],
        lexical=[(rows[shared], 0.031), (rows[lexical_only], 0.022)],
        hydrated=[rows[dense_only], rows[shared], rows[lexical_only]],
    )
    scope = RetrievalScope(user_id=str(uuid4()), conversation_id=uuid4())

    results = retriever.search("revenue", scope, final_limit=3)

    assert all(isinstance(row, RetrievalCandidate) for row in results)
    assert [row.chunk_id for row in results] == [shared, dense_only, lexical_only]
    assert results[0].dense_score == 0.52
    assert results[0].lexical_score == 0.031
    assert results[0].fused_score < 1
    repository.search_active_lexical_for_scope.assert_called_once_with(
        "revenue", user_id=scope.user_id, conversation_id=scope.conversation_id, limit=40
    )


def test_retrieval_candidate_carries_production_block_provenance() -> None:
    from app.services.rag_retrieval import RetrievalScope

    chunk_id = uuid4()
    row = _chunk(chunk_id, content="equation")
    row.block_provenance = [{"kind": "equation", "block_index": 3}]
    retriever, _, _, _ = _retriever(
        hybrid_enabled=False,
        points=[_point(chunk_id, 0.9)],
        hydrated=[row],
    )

    results = retriever.search(
        "equation",
        RetrievalScope(user_id=str(uuid4()), conversation_id=uuid4()),
        final_limit=1,
    )

    assert results[0].metadata["block_provenance"] == [
        {"kind": "equation", "block_index": 3}
    ]


def test_dense_only_rollback_path_remains_typed_and_authorized():
    from app.services.rag_retrieval import RetrievalCandidate, RetrievalScope

    chunk_id = uuid4()
    row = _chunk(chunk_id)
    retriever, _, _, repository = _retriever(
        hybrid_enabled=False,
        points=[_point(chunk_id, 0.42)],
        hydrated=[row],
    )

    results = retriever.search(
        "query",
        RetrievalScope(user_id=str(uuid4()), conversation_id=uuid4()),
        final_limit=1,
    )

    assert len(results) == 1
    assert isinstance(results[0], RetrievalCandidate)
    assert results[0].dense_score == 0.42
    assert results[0].lexical_score is None
    repository.search_active_lexical_for_scope.assert_not_called()
    repository.get_active_by_ids_for_scope.assert_called_once()


def test_active_generation_fingerprint_sorts_uuid_strings_before_hashing():
    from app.services.rag_retrieval import active_generation_fingerprint

    generation_ids = [
        UUID("00000000-0000-0000-0000-000000000002"),
        UUID("00000000-0000-0000-0000-000000000001"),
    ]
    expected = hashlib.sha256(
        b"00000000-0000-0000-0000-000000000001\n"
        b"00000000-0000-0000-0000-000000000002"
    ).hexdigest()

    assert active_generation_fingerprint(generation_ids) == expected
    assert active_generation_fingerprint(reversed(generation_ids)) == expected


def test_search_records_active_generation_fingerprint_and_experiment_inputs():
    from app.services.rag_retrieval import RetrievalScope

    retriever, _, _, _ = _retriever()

    retriever.search(
        "query",
        RetrievalScope(user_id=str(uuid4()), conversation_id=uuid4()),
        dense_candidate_limit=11,
        lexical_candidate_limit=13,
        final_limit=5,
    )

    assert retriever.last_trace == {
        "active_generation_fingerprint": hashlib.sha256(
            b"00000000-0000-0000-0000-000000000001\n"
            b"00000000-0000-0000-0000-000000000002"
        ).hexdigest(),
        "dense_candidate_limit": 11,
        "lexical_candidate_limit": 13,
        "final_limit": 5,
        "rrf_k": 60,
        "hybrid_enabled": True,
    }


def test_server_resolved_fingerprint_is_used_without_requerying_generations():
    from app.services.rag_retrieval import RetrievalScope

    retriever, _, _, repository = _retriever()

    retriever.search(
        "query",
        RetrievalScope(user_id=str(uuid4()), conversation_id=uuid4()),
        final_limit=5,
        generation_fingerprint="server-owned-fingerprint",
        active_generation_ids=[
            UUID("00000000-0000-0000-0000-000000000001")
        ],
    )

    assert retriever.last_trace["active_generation_fingerprint"] == (
        "server-owned-fingerprint"
    )
    repository.get_active_generation_ids_for_scope.assert_not_called()


def _sqlite_repository():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(
        engine,
        tables=[
            User.__table__,
            Conversation.__table__,
            Document.__table__,
            DocumentIndexGeneration.__table__,
            DocumentParseArtifact.__table__,
            DocumentChunk.__table__,
        ],
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return engine, factory, DocumentChunkRepository(factory)


def _stored_chunk(document_id: UUID, generation_id: UUID, content: str):
    return DocumentChunk(
        id=uuid4(),
        document_id=document_id,
        index_generation_id=generation_id,
        chunk_index=0,
        content=content,
        content_sha256="a" * 64,
        char_count=len(content),
        token_count=len(content.split()),
        section_path=[],
        block_provenance=[],
        chunk_metadata={},
    )


def test_lexical_sql_excludes_other_tenants_and_retired_generations():
    engine, factory, repository = _sqlite_repository()
    owner_id, other_owner_id = uuid4(), uuid4()
    conversation_id, other_conversation_id = uuid4(), uuid4()
    owned_document_id, other_document_id = uuid4(), uuid4()
    active_id, retired_id, other_active_id = uuid4(), uuid4(), uuid4()
    owned_active = _stored_chunk(owned_document_id, active_id, "quarterly revenue growth")
    owned_retired = _stored_chunk(owned_document_id, retired_id, "quarterly revenue secret")
    other_active = _stored_chunk(other_document_id, other_active_id, "quarterly revenue foreign")
    try:
        with factory.begin() as session:
            session.add_all(
                [
                    User(
                        id=owner_id,
                        username="owner",
                        email="owner@example.test",
                        password_hash="test",
                    ),
                    User(
                        id=other_owner_id,
                        username="other",
                        email="other@example.test",
                        password_hash="test",
                    ),
                    Conversation(id=conversation_id, owner_id=owner_id, title="owned"),
                    Conversation(
                        id=other_conversation_id,
                        owner_id=other_owner_id,
                        title="other",
                    ),
                ]
            )
            session.add_all(
                [
                    Document(
                        id=owned_document_id,
                        conversation_id=conversation_id,
                        filename="owned.pdf",
                        filename_key="owned.pdf",
                        file_type="application/pdf",
                        status=2,
                        upload_time=datetime.now(timezone.utc),
                    ),
                    Document(
                        id=other_document_id,
                        conversation_id=other_conversation_id,
                        filename="other.pdf",
                        filename_key="other.pdf",
                        file_type="application/pdf",
                        status=2,
                        upload_time=datetime.now(timezone.utc),
                    ),
                ]
            )
            session.add_all(
                [
                    DocumentIndexGeneration(
                        id=active_id,
                        document_id=owned_document_id,
                        status="active",
                        embedding_provider="test",
                        embedding_model="test",
                        embedding_dimension=2,
                        chunking_version="test",
                    ),
                    DocumentIndexGeneration(
                        id=retired_id,
                        document_id=owned_document_id,
                        status="retired",
                        embedding_provider="test",
                        embedding_model="test",
                        embedding_dimension=2,
                        chunking_version="test",
                    ),
                    DocumentIndexGeneration(
                        id=other_active_id,
                        document_id=other_document_id,
                        status="active",
                        embedding_provider="test",
                        embedding_model="test",
                        embedding_dimension=2,
                        chunking_version="test",
                    ),
                ]
            )
            session.add_all([owned_active, owned_retired, other_active])

        rows = repository.search_active_lexical_for_scope(
            "quarterly revenue",
            user_id=owner_id,
            conversation_id=conversation_id,
            limit=10,
        )

        assert [row.id for row, _score in rows] == [owned_active.id]
        assert repository.get_active_generation_ids_for_scope(
            user_id=owner_id,
            conversation_id=conversation_id,
        ) == [active_id]
        assert [
            row.id
            for row in repository.get_active_by_ids_for_scope(
                [owned_active.id, owned_retired.id, other_active.id],
                user_id=owner_id,
                conversation_id=conversation_id,
            )
        ] == [owned_active.id]
    finally:
        engine.dispose()


def test_postgresql_lexical_expression_uses_simple_language_tsvector():
    from app.repositories.document_chunk import postgres_lexical_expressions

    match, score = postgres_lexical_expressions("quarterly revenue")
    statement = select(DocumentChunk.id, score).where(match)
    compiled = str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "to_tsvector('simple', document_chunks.content)" in compiled
    assert "plainto_tsquery('simple', 'quarterly revenue')" in compiled
    assert "@@" in compiled


def test_lexical_migration_creates_simple_language_gin_expression_index():
    migration_path = (
        Path(__file__).parents[1]
        / "app"
        / "alembic"
        / "versions"
        / "d4e5f6a7b8c9_add_document_chunk_lexical_index.py"
    )
    spec = spec_from_file_location("lexical_migration", migration_path)
    assert spec and spec.loader
    migration = module_from_spec(spec)
    spec.loader.exec_module(migration)
    migration.op = MagicMock()

    migration.upgrade()

    call = migration.op.create_index.call_args
    assert call.args[:2] == (
        "idx_document_chunks_content_simple_fts",
        "document_chunks",
    )
    assert str(call.args[2][0]) == "to_tsvector('simple', content)"
    assert call.kwargs["postgresql_using"] == "gin"


def test_retrieval_config_defaults_are_shadow_safe():
    from app.core.config import Settings

    fields = Settings.model_fields
    assert fields["rag_hybrid_retrieval_enabled"].default is False
    assert fields["rag_dense_candidate_limit"].default == 40
    assert fields["rag_lexical_candidate_limit"].default == 40
    assert fields["rag_rrf_k"].default == 60


def test_container_exposes_configured_rag_retriever_provider():
    from app.core.container import Container

    assert "rag_retriever" in Container.providers


# ---------------------------------------------------------------------------
# Round-1 fix (finding 1): the cache-enabled paths had zero test coverage,
# including the security-critical "a hit still re-authorizes" property. Each
# fake below is local to this module and duck-types ``RAGExactCache``.
# ---------------------------------------------------------------------------


class _FakeExactCache:
    """Deterministic in-memory stand-in for ``RAGExactCache``."""

    enabled = True

    def __init__(self, *, retrieval_payload=None, query_vector=None):
        self._retrieval_payload = retrieval_payload
        self._query_vector = query_vector
        self.retrieval_get_keys: list[str] = []
        self.retrieval_set_calls: list[tuple[str, dict]] = []
        self.query_embedding_get_keys: list[str] = []
        self.query_embedding_set_calls: list[tuple[str, list]] = []

    def get_document_embedding(self, key, *, dimension):
        del key, dimension
        return None

    def set_document_embedding(self, key, vector):
        del key, vector

    def get_query_embedding(self, key, *, dimension):
        self.query_embedding_get_keys.append(key)
        if self._query_vector is not None and len(self._query_vector) == dimension:
            return list(self._query_vector)
        return None

    def set_query_embedding(self, key, vector, *, ttl_seconds):
        del ttl_seconds
        self.query_embedding_set_calls.append((key, list(vector)))

    def get_retrieval(self, key):
        self.retrieval_get_keys.append(key)
        return self._retrieval_payload

    def set_retrieval(self, key, payload, *, ttl_seconds):
        del ttl_seconds
        self.retrieval_set_calls.append((key, payload))


def test_retrieval_cache_hit_still_reauthorizes_through_sql_and_skips_qdrant():
    """The security-critical property: a cache hit must never bypass SQL
    re-authorization, and it must skip the Qdrant dense search it replaces.
    """
    from app.services.rag_retrieval import RetrievalScope

    chunk_id = uuid4()
    document_id = uuid4()
    hydrated_chunk = _chunk(chunk_id, document_id=document_id, content="cached body")
    retriever, qdrant, embedding, repository = _retriever(
        hybrid_enabled=False, hydrated=[hydrated_chunk]
    )
    payload = {
        "fused": [
            {
                "candidate_id": str(chunk_id),
                "dense_rank": 1,
                "lexical_rank": None,
                "fused_score": 0.5,
            }
        ],
        "dense_scores": {str(chunk_id): 0.9},
        "lexical_scores": {},
    }
    retriever.cache = _FakeExactCache(retrieval_payload=payload)
    scope = RetrievalScope(user_id="user-1", conversation_id=uuid4())

    results = retriever.search("revenue", scope, active_generation_ids=[uuid4()])

    repository.get_active_by_ids_for_scope.assert_called_once()
    assert repository.get_active_by_ids_for_scope.call_args.args[0] == [chunk_id]
    qdrant.query_points.assert_not_called()
    assert embedding.queries == []
    assert [candidate.chunk_id for candidate in results] == [chunk_id]
    assert results[0].dense_score == 0.9


def test_retrieval_cache_miss_populates_cache_for_next_call():
    from app.services.rag_retrieval import RetrievalScope

    chunk_id = uuid4()
    retriever, qdrant, embedding, repository = _retriever(
        hybrid_enabled=False,
        points=[_point(chunk_id, 0.77)],
        hydrated=[_chunk(chunk_id, content="fresh body")],
    )
    fake_cache = _FakeExactCache()
    retriever.cache = fake_cache
    scope = RetrievalScope(user_id="user-1", conversation_id=uuid4())

    results = retriever.search("revenue", scope, active_generation_ids=[uuid4()])

    assert len(results) == 1
    qdrant.query_points.assert_called_once()
    assert len(fake_cache.retrieval_set_calls) == 1
    stored_key, stored_payload = fake_cache.retrieval_set_calls[0]
    assert stored_payload["fused"][0]["candidate_id"] == str(chunk_id)
    # The same key must be used for the lookup that missed.
    assert fake_cache.retrieval_get_keys == [stored_key]


def test_retrieval_cache_key_differs_by_tenant_and_by_conversation():
    """Round-1 fix (finding 1): pin the retriever's *own* key construction,
    not only the standalone key-builder function.
    """
    from app.services.rag_retrieval import RetrievalScope

    conversation_id = uuid4()
    keys: list[str] = []
    for user_id in ("tenant-a", "tenant-b"):
        retriever, _, _, _ = _retriever(hybrid_enabled=False, hydrated=[])
        fake_cache = _FakeExactCache()
        retriever.cache = fake_cache
        retriever.search(
            "revenue",
            RetrievalScope(user_id=user_id, conversation_id=conversation_id),
            active_generation_ids=[uuid4()],
        )
        keys.append(fake_cache.retrieval_get_keys[0])

    assert keys[0] != keys[1]


def test_embed_query_cached_returns_cached_vector_without_calling_the_provider():
    from app.services.rag_retrieval import RetrievalScope

    chunk_id = uuid4()
    cached_vector = [0.5, 0.25]
    retriever, qdrant, embedding, repository = _retriever(
        hybrid_enabled=False,
        points=[_point(chunk_id, 0.77)],
        hydrated=[_chunk(chunk_id, content="body")],
    )
    fake_cache = _FakeExactCache(query_vector=cached_vector)
    retriever.cache = fake_cache
    scope = RetrievalScope(user_id="user-1", conversation_id=uuid4())

    retriever.search("revenue", scope, active_generation_ids=[uuid4()])

    # The embedding provider must never be called on a cache hit ...
    assert embedding.queries == []
    # ... and the cached vector must be exactly what reaches Qdrant.
    assert qdrant.query_points.call_args.kwargs["query"] == cached_vector


def test_zero_dimension_embedding_service_bypasses_query_embedding_cache():
    """Round-1 fix (finding 8): a dimension of 0 must skip the cache
    entirely rather than write an entry that can never validate on read.
    """
    from app.services.rag_retrieval import RetrievalScope

    class _ZeroDimEmbedding:
        provider = "gemini"
        model_name = "test-model"
        dimension = 0
        query_task = "search result"

        def __init__(self) -> None:
            self.queries: list[str] = []

        def embed_query(self, query: str):
            self.queries.append(query)
            return [0.1, 0.2]

    chunk_id = uuid4()
    retriever, qdrant, _, repository = _retriever(
        hybrid_enabled=False, points=[_point(chunk_id, 0.5)], hydrated=[_chunk(chunk_id)]
    )
    zero_dim_embedding = _ZeroDimEmbedding()
    retriever.embedding_service = zero_dim_embedding
    fake_cache = _FakeExactCache()
    retriever.cache = fake_cache
    scope = RetrievalScope(user_id="user-1", conversation_id=uuid4())

    retriever.search("revenue", scope, active_generation_ids=[uuid4()])

    assert fake_cache.query_embedding_get_keys == []
    assert fake_cache.query_embedding_set_calls == []
    assert zero_dim_embedding.queries == ["revenue"]


class _FakeStageMetrics:
    """Deterministic recorder matching the ``RAGMetrics`` stage/cache API."""

    def __init__(self) -> None:
        self.stage_calls: list[tuple[str, float, dict]] = []
        self.stage_failure_calls: list[tuple[str, str]] = []
        self.cache_result_calls: list[tuple[str, str]] = []

    def stage(self, stage, *, elapsed_seconds, labels=None):
        self.stage_calls.append((stage, elapsed_seconds, dict(labels or {})))

    def stage_failure(self, stage, failure_code):
        self.stage_failure_calls.append((stage, failure_code))

    def cache_result(self, cache, result):
        self.cache_result_calls.append((cache, result))


def test_dense_retrieval_failure_still_records_duration_and_failure_metric():
    """Round-1 fix (finding 4): a raising dense search must not be invisible
    to the duration histogram (which would bias p95/p99 downward) and must
    increment a countable failure metric. The exception still propagates --
    retrieval failures are not silently swallowed.
    """
    from app.services.rag_retrieval import RetrievalScope

    retriever, qdrant, _, _ = _retriever(hybrid_enabled=False, hydrated=[])
    qdrant.query_points.side_effect = RuntimeError("qdrant is down")
    fake_metrics = _FakeStageMetrics()
    retriever.metrics = fake_metrics
    scope = RetrievalScope(user_id="user-1", conversation_id=uuid4())

    with pytest.raises(RuntimeError):
        retriever.search("revenue", scope, active_generation_ids=[uuid4()])

    stage_names = [call[0] for call in fake_metrics.stage_calls]
    assert "dense_retrieval" in stage_names
    assert ("dense_retrieval", "dependency_exception") in fake_metrics.stage_failure_calls


def test_lexical_retrieval_failure_still_records_duration_and_failure_metric():
    from app.services.rag_retrieval import RetrievalScope

    retriever, _, _, repository = _retriever(hybrid_enabled=True, hydrated=[])
    repository.search_active_lexical_for_scope.side_effect = RuntimeError("db is down")
    fake_metrics = _FakeStageMetrics()
    retriever.metrics = fake_metrics
    scope = RetrievalScope(user_id="user-1", conversation_id=uuid4())

    with pytest.raises(RuntimeError):
        retriever.search("revenue", scope, active_generation_ids=[uuid4()])

    stage_names = [call[0] for call in fake_metrics.stage_calls]
    assert "lexical_retrieval" in stage_names
    assert ("lexical_retrieval", "dependency_exception") in fake_metrics.stage_failure_calls


def test_sql_hydration_failure_still_records_duration_and_failure_metric():
    from app.services.rag_retrieval import RetrievalScope

    chunk_id = uuid4()
    retriever, _, _, repository = _retriever(
        hybrid_enabled=False, points=[_point(chunk_id, 0.5)]
    )
    repository.get_active_by_ids_for_scope.side_effect = RuntimeError("db is down")
    fake_metrics = _FakeStageMetrics()
    retriever.metrics = fake_metrics
    scope = RetrievalScope(user_id="user-1", conversation_id=uuid4())

    with pytest.raises(RuntimeError):
        retriever.search("revenue", scope, active_generation_ids=[uuid4()])

    stage_names = [call[0] for call in fake_metrics.stage_calls]
    assert "sql_hydration" in stage_names
    assert ("sql_hydration", "dependency_exception") in fake_metrics.stage_failure_calls


def test_dense_retrieval_stage_records_provider_model_and_cache_result():
    """Round-1 fix (finding 6): provider/model/cache_result must have real
    producers instead of every series reading provider="n/a", cache_result="n/a".
    """
    from app.services.rag_retrieval import RetrievalScope

    chunk_id = uuid4()
    retriever, qdrant, embedding, repository = _retriever(
        hybrid_enabled=False, points=[_point(chunk_id, 0.5)], hydrated=[_chunk(chunk_id)]
    )
    fake_metrics = _FakeStageMetrics()
    retriever.metrics = fake_metrics
    retriever.cache = _FakeExactCache()
    scope = RetrievalScope(user_id="user-1", conversation_id=uuid4())

    retriever.search("revenue", scope, active_generation_ids=[uuid4()])

    dense_calls = [call for call in fake_metrics.stage_calls if call[0] == "dense_retrieval"]
    assert len(dense_calls) == 1
    _, _, dense_labels = dense_calls[0]
    assert dense_labels["provider"] == embedding.provider
    assert dense_labels["model"] == embedding.model_name
    assert dense_labels["cache_result"] == "miss"

    hydration_calls = [call for call in fake_metrics.stage_calls if call[0] == "sql_hydration"]
    assert len(hydration_calls) == 1
    _, _, hydration_labels = hydration_calls[0]
    assert hydration_labels["cache_result"] == "miss"
