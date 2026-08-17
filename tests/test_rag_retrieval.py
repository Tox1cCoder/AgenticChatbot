from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

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
