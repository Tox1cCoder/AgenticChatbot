"""Deleting a document or an index generation leaves the chunk cascade to the database.

``document_chunks`` cascades from both ``documents`` and
``document_index_generations`` (ON DELETE CASCADE), and each chunk's images are
detached by their own ON DELETE SET NULL. Without ``passive_deletes`` the ORM
loaded every chunk, and then each chunk's images, into the session before
deleting anything: a purge of a large generation read its whole content back.

SQLite enforces none of this unless ``PRAGMA foreign_keys=ON``, and a passive
relationship relies on it entirely, so these tests turn it on. Without it a
passive delete would leave orphaned chunks and still look like it worked.
"""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.base import Base
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_image import DocumentImage
from app.models.document_index_generation import DocumentIndexGeneration
from app.models.document_parse_artifact import DocumentParseArtifact
from app.models.project import Project
from app.models.user import User
from app.repositories.document import DocumentRepository
from app.repositories.document_index_generation import DocumentIndexGenerationRepository


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(_type, _compiler, **_kwargs):
    return "JSON"


@pytest.fixture()
def engine() -> Iterator:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enforce_foreign_keys(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(
        engine,
        tables=[
            User.__table__,
            Project.__table__,
            Conversation.__table__,
            Document.__table__,
            DocumentParseArtifact.__table__,
            DocumentIndexGeneration.__table__,
            DocumentChunk.__table__,
            DocumentImage.__table__,
        ],
    )
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture()
def seeded(engine):
    """One document with one retired generation of two chunks, one image on a chunk."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    user_id, conversation_id, document_id = uuid4(), uuid4(), uuid4()
    generation_id, artifact_id, image_id = uuid4(), uuid4(), uuid4()
    chunk_ids = [uuid4(), uuid4()]
    with factory.begin() as db:
        db.add(User(id=user_id, username="u", email="u@example.test", password_hash="x"))
        db.flush()
        db.add(Conversation(id=conversation_id, owner_id=user_id, title="docs"))
        db.flush()
        db.add(
            Document(
                id=document_id,
                conversation_id=conversation_id,
                filename="a.pdf",
                filename_key="a.pdf",
                file_type="application/pdf",
                status=2,
            )
        )
        db.flush()
        db.add(
            DocumentParseArtifact(
                id=artifact_id,
                document_id=document_id,
                artifact_type="markdown",
                storage_path="a.md",
            )
        )
        db.add(
            DocumentIndexGeneration(
                id=generation_id,
                document_id=document_id,
                status="retired",
                embedding_provider="test",
                embedding_model="test-model",
                embedding_dimension=3,
                chunking_version="v1",
            )
        )
        db.flush()
        for index, chunk_id in enumerate(chunk_ids):
            db.add(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document_id,
                    index_generation_id=generation_id,
                    parse_artifact_id=artifact_id,
                    chunk_index=index,
                    content="text",
                    content_sha256="0" * 64,
                    char_count=4,
                    token_count=1,
                )
            )
        db.flush()
        db.add(
            DocumentImage(
                id=image_id,
                document_id=document_id,
                chunk_id=chunk_ids[0],
                image_path="images/a.png",
                mime_type="image/png",
            )
        )
    return factory, document_id, generation_id, image_id


def _chunk_reads(engine) -> list[str]:
    """Every SELECT that reads document_chunks, recorded from now on."""
    reads: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _record(_conn, _cursor, statement, _params, _context, _many):
        normalized = " ".join(statement.split()).upper()
        if normalized.startswith("SELECT") and "FROM DOCUMENT_CHUNKS" in normalized:
            reads.append(statement)

    return reads


def _count(factory, model) -> int:
    with factory() as db:
        return db.execute(select(func.count()).select_from(model)).scalar_one()


def test_purging_a_generation_never_loads_its_chunks(engine, seeded):
    factory, _document_id, generation_id, image_id = seeded
    reads = _chunk_reads(engine)

    assert DocumentIndexGenerationRepository(factory).delete(generation_id) is True

    assert reads == [], "the ORM loaded the generation's chunks instead of letting SQL cascade"
    assert _count(factory, DocumentChunk) == 0
    with factory() as db:
        image = db.get(DocumentImage, image_id)
        assert image is not None
        assert image.chunk_id is None


def test_deleting_a_document_never_loads_its_chunks(engine, seeded):
    factory, document_id, _generation_id, _image_id = seeded
    reads = _chunk_reads(engine)

    assert DocumentRepository(factory).delete(document_id) is True

    assert reads == [], "the ORM loaded the document's chunks instead of letting SQL cascade"
    # The database cascade removed these ...
    assert _count(factory, DocumentChunk) == 0
    assert _count(factory, DocumentIndexGeneration) == 0
    # ... and the ORM cascade, which images and artifacts still need, these.
    assert _count(factory, DocumentImage) == 0
    assert _count(factory, DocumentParseArtifact) == 0
