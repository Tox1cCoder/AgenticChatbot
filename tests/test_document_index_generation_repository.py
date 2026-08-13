from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(_type, _compiler, **_kwargs):
    return "JSON"


@pytest.fixture()
def generation_db():
    from app.models.base import Base
    from app.models.document_chunk import DocumentChunk
    from app.models.document_image import DocumentImage
    from app.models.document_index_generation import DocumentIndexGeneration

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(
        engine,
        tables=[
            DocumentIndexGeneration.__table__,
            DocumentChunk.__table__,
            DocumentImage.__table__,
        ],
    )
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
    assert generations[old.id].retired_at is not None
    assert repository.get_active(document_id).id == replacement.id


def test_recently_retired_old_generation_keeps_full_rollback_window(generation_db):
    from app.repositories.document_index_generation import DocumentIndexGenerationRepository

    repository = DocumentIndexGenerationRepository(generation_db)
    document_id = uuid4()
    old = _create(repository, document_id)
    repository.activate(old.id)
    with generation_db() as db:
        stored = db.get(type(old), old.id)
        stored.created_at = datetime.now(timezone.utc) - timedelta(days=30)
        db.commit()
    replacement = _create(repository, document_id)

    repository.activate(replacement.id)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=168)
    assert repository.retired_before(document_id, cutoff) == []


def test_activation_rebinds_images_to_replacement_chunk_before_retiring_old(
    generation_db,
):
    from app.models.document_chunk import DocumentChunk
    from app.models.document_image import DocumentImage
    from app.repositories.document_index_generation import DocumentIndexGenerationRepository

    repository = DocumentIndexGenerationRepository(generation_db)
    document_id = uuid4()
    old = _create(repository, document_id)
    old_chunk_id, old_chunk_2_id = uuid4(), uuid4()
    new_chunk_id, new_chunk_2_id = uuid4(), uuid4()
    image_id, image_2_id, image_3_id = uuid4(), uuid4(), uuid4()
    with generation_db() as db:
        db.add(
            DocumentChunk(
                id=old_chunk_id,
                document_id=document_id,
                index_generation_id=old.id,
                chunk_index=0,
                content="old",
                content_sha256="a" * 64,
                char_count=3,
                token_count=1,
                section_path=[],
                block_provenance=[],
                chunk_metadata={},
            )
        )
        db.add(
            DocumentImage(
                id=image_id,
                document_id=document_id,
                chunk_id=old_chunk_id,
                image_path="figure.png",
                mime_type="image/png",
            )
        )
        db.add(
            DocumentChunk(
                id=old_chunk_2_id,
                document_id=document_id,
                index_generation_id=old.id,
                chunk_index=1,
                content="old two",
                content_sha256="c" * 64,
                char_count=7,
                token_count=2,
                section_path=[],
                block_provenance=[],
                chunk_metadata={},
            )
        )
        db.add_all(
            [
                DocumentImage(
                    id=image_2_id,
                    document_id=document_id,
                    chunk_id=old_chunk_id,
                    image_path="figure-2.png",
                    mime_type="image/png",
                ),
                DocumentImage(
                    id=image_3_id,
                    document_id=document_id,
                    chunk_id=old_chunk_2_id,
                    image_path="figure-3.png",
                    mime_type="image/png",
                ),
            ]
        )
        db.commit()
    repository.activate(old.id)
    replacement = _create(repository, document_id)
    with generation_db() as db:
        db.add(
            DocumentChunk(
                id=new_chunk_id,
                document_id=document_id,
                index_generation_id=replacement.id,
                chunk_index=0,
                content="new",
                content_sha256="b" * 64,
                char_count=3,
                token_count=1,
                section_path=[],
                block_provenance=[],
                chunk_metadata={},
            )
        )
        db.add(
            DocumentChunk(
                id=new_chunk_2_id,
                document_id=document_id,
                index_generation_id=replacement.id,
                chunk_index=1,
                content="new two",
                content_sha256="d" * 64,
                char_count=7,
                token_count=2,
                section_path=[],
                block_provenance=[],
                chunk_metadata={},
            )
        )
        db.commit()

    repository.activate(replacement.id)

    with generation_db() as db:
        assert db.get(DocumentImage, image_id).chunk_id == new_chunk_id
        assert db.get(DocumentImage, image_2_id).chunk_id == new_chunk_id
        assert db.get(DocumentImage, image_3_id).chunk_id == new_chunk_2_id

    repository.delete(old.id)
    with generation_db() as db:
        assert db.get(DocumentImage, image_id).chunk_id == new_chunk_id
        assert db.get(DocumentImage, image_2_id).chunk_id == new_chunk_id
        assert db.get(DocumentImage, image_3_id).chunk_id == new_chunk_2_id


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
    assert latest_failed.failed_at is not None


def test_failed_generation_becomes_purgeable_by_failure_age_only(generation_db):
    from app.repositories.document_index_generation import DocumentIndexGenerationRepository

    repository = DocumentIndexGenerationRepository(generation_db)
    document_id = uuid4()
    active = _create(repository, document_id)
    repository.activate(active.id)
    failed = _create(repository, document_id)
    repository.mark_failed(failed.id, "INDEX_BUILD_RUNTIMEERROR")
    with generation_db() as db:
        stored = db.get(type(failed), failed.id)
        stored.failed_at = datetime.now(timezone.utc) - timedelta(days=8)
        db.commit()

    purgeable = repository.purgeable_before(
        document_id, datetime.now(timezone.utc) - timedelta(days=7)
    )

    assert [row.id for row in purgeable] == [failed.id]
    assert all(row.id != active.id for row in purgeable)


class _QdrantGenerationFake:
    def __init__(self) -> None:
        self.points: dict[str, SimpleNamespace] = {}
        self.events: list[tuple[str, object]] = []
        self.fail_upsert: Exception | None = None
        self.fail_old_cleanup = False
        self.on_inactive_payload = None

    def get_collections(self):
        return SimpleNamespace(collections=[])

    def create_collection(self, **_kwargs):
        self.events.append(("create_collection", None))

    def create_payload_index(self, *, field_name, **_kwargs):
        self.events.append(("create_payload_index", field_name))

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
        for condition in qdrant_filter.must_not or []:
            if payload.get(condition.key) == condition.match.value:
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
            (
                condition.match.value
                for condition in points.filter.must
                if condition.key == "index_generation"
            ),
            None,
        )
        self.events.append(("set_active", (generation, payload["is_active"])))
        if not payload["is_active"] and self.on_inactive_payload is not None:
            callback, self.on_inactive_payload = self.on_inactive_payload, None
            callback()
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


def test_successful_activation_reconciles_all_document_payloads():
    document_id, old_generation_id = uuid4(), uuid4()
    service, _generations, _chunks, qdrant = _index_service(
        document_id, old_generation_id
    )

    service.index_document(
        document=SimpleNamespace(id=document_id, conversation_id=uuid4(), user_id=uuid4()),
        built_chunks=[_built_chunk()],
        parse_artifact_id=None,
    )

    set_events = [event for event in qdrant.events if event[0] == "set_active"]
    assert ("set_active", (None, False)) in set_events
    new_generation_id = next(iter(qdrant.points.values())).payload["index_generation"]
    assert ("set_active", (new_generation_id, True)) in set_events


def test_ambiguous_commit_confirmed_active_keeps_new_payload_visible():
    document_id, old_generation_id = uuid4(), uuid4()
    service, generations, _chunks, qdrant = _index_service(document_id, old_generation_id)

    def commit_then_raise(generation_id):
        target = generations.rows[generation_id]
        for row in generations.rows.values():
            if row.document_id == document_id and row.status == "active":
                row.status = "retired"
        target.status = "active"
        raise RuntimeError("connection lost after commit")

    generations.activate = commit_then_raise

    persisted = service.index_document(
        document=SimpleNamespace(id=document_id, conversation_id=uuid4(), user_id=uuid4()),
        built_chunks=[_built_chunk()],
        parse_artifact_id=None,
    )

    assert generations.get_active(document_id).id == persisted[0].index_generation_id
    assert next(iter(qdrant.points.values())).payload["is_active"] is True
    assert generations.get_latest_failed(document_id) is None


def test_reconciliation_cleanup_failure_does_not_hide_active_generation():
    document_id, old_generation_id = uuid4(), uuid4()
    service, generations, _chunks, qdrant = _index_service(document_id, old_generation_id)
    qdrant.fail_old_cleanup = True

    persisted = service.index_document(
        document=SimpleNamespace(id=document_id, conversation_id=uuid4(), user_id=uuid4()),
        built_chunks=[_built_chunk()],
        parse_artifact_id=None,
    )

    new_generation_id = persisted[0].index_generation_id
    assert generations.get_active(document_id).id == new_generation_id
    assert next(iter(qdrant.points.values())).payload["is_active"] is True


def test_document_wide_reconciliation_deactivates_unobserved_concurrent_generation():
    document_id, old_generation_id = uuid4(), uuid4()
    service, _generations, _chunks, qdrant = _index_service(
        document_id, old_generation_id
    )
    rogue_generation_id, rogue_point_id = uuid4(), uuid4()
    qdrant.points[str(rogue_point_id)] = SimpleNamespace(
        id=str(rogue_point_id),
        payload={
            "document_id": str(document_id),
            "index_generation": str(rogue_generation_id),
            "is_active": True,
        },
        vector=[0.0] * 8,
    )

    persisted = service.index_document(
        document=SimpleNamespace(id=document_id, conversation_id=uuid4(), user_id=uuid4()),
        built_chunks=[_built_chunk()],
        parse_artifact_id=None,
    )

    authoritative = str(persisted[0].index_generation_id)
    assert qdrant.points[str(rogue_point_id)].payload["is_active"] is False
    assert all(
        point.payload["is_active"]
        == (point.payload["index_generation"] == authoritative)
        for point in qdrant.points.values()
    )


def test_unknown_sql_activation_outcome_does_not_demote_generation_or_chunks():
    document_id, old_generation_id = uuid4(), uuid4()
    service, generations, chunks, qdrant = _index_service(document_id, old_generation_id)
    generations.activate = MagicMock(side_effect=RuntimeError("commit response lost"))
    generations.get_active = MagicMock(side_effect=RuntimeError("database unavailable"))

    with pytest.raises(RuntimeError, match="outcome is unknown"):
        service.index_document(
            document=SimpleNamespace(id=document_id, conversation_id=uuid4(), user_id=uuid4()),
            built_chunks=[_built_chunk()],
            parse_artifact_id=None,
        )

    assert generations.get_latest_failed(document_id) is None
    assert chunks.mark_index_failed.call_count == 0
    assert next(iter(qdrant.points.values())).payload["is_active"] is True


def test_reconciliation_retries_when_active_generation_changes_mid_cleanup():
    document_id, generation_a_id = uuid4(), uuid4()
    service, generations, _chunks, qdrant = _index_service(
        document_id, generation_a_id
    )
    generation_b_id = uuid4()
    generations.rows[generation_b_id] = SimpleNamespace(
        id=generation_b_id,
        document_id=document_id,
        status="ready",
        created_at=datetime.now(timezone.utc),
    )
    for generation_id in (generation_a_id, generation_b_id):
        point_id = uuid4()
        qdrant.points[str(point_id)] = SimpleNamespace(
            id=str(point_id),
            payload={
                "document_id": str(document_id),
                "index_generation": str(generation_id),
                "is_active": generation_id == generation_a_id,
            },
            vector=[0.0] * 8,
        )

    def activate_b_mid_reconcile():
        generations.rows[generation_a_id].status = "retired"
        generations.rows[generation_b_id].status = "active"

    qdrant.on_inactive_payload = activate_b_mid_reconcile

    reconciled = service.reconcile_active_payloads(document_id)

    assert reconciled == generation_b_id
    assert all(
        point.payload["is_active"]
        == (point.payload["index_generation"] == str(generation_b_id))
        for point in qdrant.points.values()
    )


def test_empty_replacement_is_rejected_before_generation_creation():
    document_id, old_generation_id = uuid4(), uuid4()
    service, generations, chunks, qdrant = _index_service(document_id, old_generation_id)

    with pytest.raises(ValueError, match="at least one chunk"):
        service.index_document(
            document=SimpleNamespace(id=document_id, conversation_id=uuid4(), user_id=uuid4()),
            built_chunks=[],
            parse_artifact_id=None,
        )

    assert generations.get_active(document_id).id == old_generation_id
    assert chunks.create_generation_chunks.call_count == 0
    assert qdrant.events == []


def test_purge_removes_failed_generation_artifacts_without_touching_active():
    from app.services.document_index_service import DocumentIndexService

    document_id, active_id, failed_id = uuid4(), uuid4(), uuid4()
    generation_repository = MagicMock()
    generation_repository.purgeable_before.return_value = [
        SimpleNamespace(id=failed_id, document_id=document_id, status="failed")
    ]
    chunk_repository = MagicMock()
    qdrant = MagicMock()
    service = DocumentIndexService(
        chunk_repository=chunk_repository,
        generation_repository=generation_repository,
        qdrant_client=qdrant,
        embedding_service=SimpleNamespace(
            provider="gemini", model_name="gemini", dimension=8
        ),
        collection_name="documents",
        embedding_dimension=8,
    )

    purged = service.purge_retired_generations(
        document_id, datetime.now(timezone.utc) - timedelta(days=7)
    )

    assert purged == [failed_id]
    selector = qdrant.delete.call_args.kwargs["points_selector"].filter
    matches = {condition.key: condition.match.value for condition in selector.must}
    assert matches == {
        "document_id": str(document_id),
        "index_generation": str(failed_id),
    }
    assert str(active_id) not in str(qdrant.delete.call_args)
    chunk_repository.delete_generation.assert_called_once_with(failed_id)
    generation_repository.delete.assert_called_once_with(failed_id)
