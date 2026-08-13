from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture()
def generation_db():
    from app.models.base import Base
    from app.models.document_index_generation import DocumentIndexGeneration

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine, tables=[DocumentIndexGeneration.__table__])
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine.dispose()


def _create(repository, document_id: UUID):
    return repository.create(
        document_id=document_id,
        embedding_provider="gemini",
        embedding_model="gemini-embedding-2",
        embedding_dimension=8,
        chunking_version="structure-v2",
    )


def test_activation_retires_previous_generation_atomically(generation_db):
    from app.repositories.document_index_generation import DocumentIndexGenerationRepository

    repository = DocumentIndexGenerationRepository(generation_db)
    document_id = uuid4()
    old = _create(repository, document_id)
    repository.activate(old.id)
    replacement = _create(repository, document_id)

    activated = repository.activate(replacement.id)

    generations = {row.id: row for row in repository.list_for_document(document_id)}
    assert activated.id == replacement.id
    assert generations[replacement.id].status == "active"
    assert generations[old.id].status == "retired"
    assert repository.get_active(document_id).id == replacement.id


def test_failure_code_is_bounded_without_retiring_active_generation(generation_db):
    from app.repositories.document_index_generation import DocumentIndexGenerationRepository

    repository = DocumentIndexGenerationRepository(generation_db)
    document_id = uuid4()
    active = _create(repository, document_id)
    repository.activate(active.id)
    failed = _create(repository, document_id)

    repository.mark_failed(failed.id, "x" * 500)

    assert repository.get_active(document_id).id == active.id
    latest_failed = repository.get_latest_failed(document_id)
    assert latest_failed.id == failed.id
    assert latest_failed.status == "failed"
    assert len(latest_failed.failure_code) <= 64


class _QdrantGenerationFake:
    def __init__(self) -> None:
        self.points: dict[str, SimpleNamespace] = {}
        self.events: list[tuple[str, object]] = []
        self.fail_upsert: Exception | None = None
        self.fail_old_cleanup = False

    def upsert(self, *, collection_name, points):
        self.events.append(("upsert", collection_name))
        if self.fail_upsert is not None:
            raise self.fail_upsert
        for point in points:
            self.points[str(point.id)] = SimpleNamespace(
                id=str(point.id), payload=dict(point.payload), vector=list(point.vector)
            )

    @staticmethod
    def _matches(payload, qdrant_filter) -> bool:
        for condition in qdrant_filter.must or []:
            if payload.get(condition.key) != condition.match.value:
                return False
        return True

    def count(self, *, collection_name, count_filter, exact):
        self.events.append(("count", exact))
        count = sum(self._matches(point.payload, count_filter) for point in self.points.values())
        return SimpleNamespace(count=count)

    def retrieve(self, *, collection_name, ids, with_payload, with_vectors):
        self.events.append(("retrieve", tuple(str(item) for item in ids)))
        return [self.points[str(point_id)] for point_id in ids if str(point_id) in self.points]

    def set_payload(self, *, collection_name, payload, points, wait=True):
        generation = next(
            condition.match.value
            for condition in points.filter.must
            if condition.key == "index_generation"
        )
        self.events.append(("set_active", (generation, payload["is_active"])))
        if not payload["is_active"] and self.fail_old_cleanup:
            raise RuntimeError("old cleanup unavailable")
        for point in self.points.values():
            if self._matches(point.payload, points.filter):
                point.payload.update(payload)

    def delete(self, **_kwargs):
        self.events.append(("delete", None))


class _GenerationRepositoryFake:
    def __init__(self, document_id: UUID, old_generation_id: UUID):
        self.rows = {
            old_generation_id: SimpleNamespace(
                id=old_generation_id,
                document_id=document_id,
                status="active",
                created_at=datetime.now(timezone.utc),
            )
        }
        self.activation_error: Exception | None = None
        self.events: list[tuple[str, UUID]] = []

    def create(self, *, document_id, **metadata):
        generation_id = uuid4()
        row = SimpleNamespace(
            id=generation_id,
            document_id=document_id,
            status="building",
            failure_code=None,
            created_at=datetime.now(timezone.utc),
            **metadata,
        )
        self.rows[generation_id] = row
        self.events.append(("create", generation_id))
        return row

    def get_active(self, document_id):
        return next(
            (
                row
                for row in self.rows.values()
                if row.document_id == document_id and row.status == "active"
            ),
            None,
        )

    def mark_ready(self, generation_id):
        self.rows[generation_id].status = "ready"
        self.events.append(("ready", generation_id))
        return self.rows[generation_id]

    def activate(self, generation_id):
        self.events.append(("activate", generation_id))
        if self.activation_error is not None:
            raise self.activation_error
        target = self.rows[generation_id]
        for row in self.rows.values():
            if row.document_id == target.document_id and row.status == "active":
                row.status = "retired"
        target.status = "active"
        return target

    def mark_failed(self, generation_id, failure_code):
        row = self.rows[generation_id]
        row.status = "failed"
        row.failure_code = failure_code[:64]
        self.events.append(("failed", generation_id))
        return row

    def get_latest_failed(self, document_id):
        return next(
            (
                row
                for row in reversed(list(self.rows.values()))
                if row.document_id == document_id and row.status == "failed"
            ),
            None,
        )


def _built_chunk():
    from app.services.document_chunk_builder import BuiltChunk

    return BuiltChunk(
        chunk_index=0,
        content="safe replacement",
        content_sha256="a" * 64,
        char_count=16,
        token_count=2,
        page_start=1,
        page_end=1,
        section_path=("Intro",),
        block_provenance=(),
        metadata={},
    )


def _index_service(document_id: UUID, old_generation_id: UUID):
    from app.services.document_index_service import DocumentIndexService

    generation_repository = _GenerationRepositoryFake(document_id, old_generation_id)
    chunk_repository = MagicMock()

    def create_chunks(doc_id, generation_id, rows):
        return [
            SimpleNamespace(
                id=row["id"],
                document_id=doc_id,
                index_generation_id=generation_id,
                chunk_index=row["chunk_index"],
                content=row["content"],
                content_sha256=row["content_sha256"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                section_path=row["section_path"],
                chunk_metadata=row["chunk_metadata"],
            )
            for row in rows
        ]

    chunk_repository.create_generation_chunks.side_effect = create_chunks
    qdrant = _QdrantGenerationFake()
    embedding = SimpleNamespace(
        provider="gemini",
        model_name="gemini-embedding-2",
        dimension=8,
        embed_documents=lambda texts, **_kwargs: [[0.0] * 8 for _ in texts],
    )
    service = DocumentIndexService(
        chunk_repository=chunk_repository,
        generation_repository=generation_repository,
        qdrant_client=qdrant,
        embedding_service=embedding,
        collection_name="documents",
        embedding_model_name="gemini-embedding-2",
        embedding_dimension=8,
        embedding_provider="gemini",
        chunking_version="structure-v2",
    )
    return service, generation_repository, chunk_repository, qdrant


def test_failed_reindex_keeps_previous_generation_active():
    document_id, old_generation_id = uuid4(), uuid4()
    service, generations, _chunks, qdrant = _index_service(document_id, old_generation_id)
    qdrant.fail_upsert = RuntimeError("qdrant unavailable")

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        service.index_document(
            document=SimpleNamespace(id=document_id, conversation_id=uuid4(), user_id=uuid4()),
            built_chunks=[_built_chunk()],
            parse_artifact_id=None,
        )

    assert generations.get_active(document_id).id == old_generation_id
    assert generations.get_latest_failed(document_id) is not None


def test_activation_occurs_only_after_count_dimension_and_scope_verification():
    document_id, old_generation_id = uuid4(), uuid4()
    service, generations, _chunks, qdrant = _index_service(document_id, old_generation_id)

    persisted = service.index_document(
        document=SimpleNamespace(id=document_id, conversation_id=uuid4(), user_id=uuid4()),
        built_chunks=[_built_chunk()],
        parse_artifact_id=None,
    )

    new_generation_id = persisted[0].index_generation_id
    assert generations.get_active(document_id).id == new_generation_id
    event_names = [event[0] for event in qdrant.events]
    assert event_names.index("count") < event_names.index("set_active")
    point = next(iter(qdrant.points.values()))
    assert len(point.vector) == 8
    assert point.payload["document_id"] == str(document_id)
    assert point.payload["index_generation"] == str(new_generation_id)
    assert point.payload["is_active"] is True
    assert all(event[0] != "delete" for event in qdrant.events)


def test_sql_activation_failure_restores_new_qdrant_points_to_inactive():
    document_id, old_generation_id = uuid4(), uuid4()
    service, generations, _chunks, qdrant = _index_service(document_id, old_generation_id)
    generations.activation_error = RuntimeError("commit rejected")

    with pytest.raises(RuntimeError, match="commit rejected"):
        service.index_document(
            document=SimpleNamespace(id=document_id, conversation_id=uuid4(), user_id=uuid4()),
            built_chunks=[_built_chunk()],
            parse_artifact_id=None,
        )

    assert generations.get_active(document_id).id == old_generation_id
    assert all(point.payload["is_active"] is False for point in qdrant.points.values())


def test_sql_chunk_build_failure_marks_new_generation_failed():
    document_id, old_generation_id = uuid4(), uuid4()
    service, generations, chunks, _qdrant = _index_service(document_id, old_generation_id)
    chunks.create_generation_chunks.side_effect = RuntimeError("chunk transaction failed")

    with pytest.raises(RuntimeError, match="chunk transaction failed"):
        service.index_document(
            document=SimpleNamespace(id=document_id, conversation_id=uuid4(), user_id=uuid4()),
            built_chunks=[_built_chunk()],
            parse_artifact_id=None,
        )

    assert generations.get_active(document_id).id == old_generation_id
    assert generations.get_latest_failed(document_id) is not None
