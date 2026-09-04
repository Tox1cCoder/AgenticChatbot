"""Real SQL behavior tests for bounded, tenant-scoped RAG exploration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.agents import rag_agent as rag_agent_module
from app.ai.agents.rag_agent import RAGAgent
from app.ai.rag_tool_actions import _resolve_document_reference
from app.models.base import Base
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_image import DocumentImage
from app.models.document_index_generation import DocumentIndexGeneration
from app.models.document_parse_artifact import DocumentParseArtifact
from app.models.user import User
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.document_image import DocumentImageRepository


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(_type, _compiler, **_kwargs):
    return "JSON"


@pytest.fixture()
def rag_scope_db(monkeypatch):
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
            DocumentImage.__table__,
        ],
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(rag_agent_module, "SessionLocal", factory)
    try:
        yield SimpleNamespace(factory=factory)
    finally:
        engine.dispose()


def _user(user_id: UUID) -> User:
    return User(
        id=user_id,
        username=f"rag-{user_id}",
        email=f"{user_id}@example.test",
        password_hash="test",
    )


def _document(
    *,
    document_id: UUID,
    conversation_id: UUID,
    filename: str,
    filename_key: str,
    upload_time: datetime,
) -> Document:
    return Document(
        id=document_id,
        conversation_id=conversation_id,
        filename=filename,
        filename_key=filename_key,
        file_type="application/pdf",
        status=2,
        upload_time=upload_time,
    )


def _chunk(document_id: UUID, chunk_index: int) -> DocumentChunk:
    content = f"chunk {chunk_index}"
    return DocumentChunk(
        id=uuid4(),
        document_id=document_id,
        index_generation_id=document_id,
        chunk_index=chunk_index,
        content=content,
        content_sha256=f"{chunk_index:064x}",
        char_count=len(content),
        token_count=2,
        section_path=[],
        block_provenance=[],
        chunk_metadata={},
    )


def _active_generation(document_id: UUID) -> DocumentIndexGeneration:
    return DocumentIndexGeneration(
        id=document_id,
        document_id=document_id,
        status="active",
        embedding_provider="test",
        embedding_model="test",
        embedding_dimension=8,
        chunking_version="test",
    )


def test_chunk_hydration_rejects_retired_generations(rag_scope_db):
    owner_id, conversation_id, document_id = uuid4(), uuid4(), uuid4()
    active_generation_id, retired_generation_id = uuid4(), uuid4()
    active_chunk = _chunk(document_id, 0)
    active_chunk.index_generation_id = active_generation_id
    active_chunk.qdrant_point_id = str(active_chunk.id)
    retired_chunk = _chunk(document_id, 0)
    retired_chunk.index_generation_id = retired_generation_id
    retired_chunk.qdrant_point_id = str(retired_chunk.id)

    with rag_scope_db.factory.begin() as session:
        session.add(_user(owner_id))
        session.add(Conversation(id=conversation_id, owner_id=owner_id, title="owner"))
        session.add(
            _document(
                document_id=document_id,
                conversation_id=conversation_id,
                filename="versioned.pdf",
                filename_key="versioned.pdf",
                upload_time=datetime.now(timezone.utc),
            )
        )
        session.add_all(
            [
                DocumentIndexGeneration(
                    id=active_generation_id,
                    document_id=document_id,
                    status="active",
                    embedding_provider="test",
                    embedding_model="test",
                    embedding_dimension=8,
                    chunking_version="test",
                ),
                DocumentIndexGeneration(
                    id=retired_generation_id,
                    document_id=document_id,
                    status="retired",
                    embedding_provider="test",
                    embedding_model="test",
                    embedding_dimension=8,
                    chunking_version="test",
                ),
            ]
        )
        session.add_all([active_chunk, retired_chunk])

    repository = DocumentChunkRepository(rag_scope_db.factory)
    assert [row.id for row in repository.get_by_document_ordered(document_id)] == [active_chunk.id]
    assert repository.get_by_ids([retired_chunk.id]) == []
    assert repository.get_by_qdrant_point_ids([str(retired_chunk.id)]) == []


def _minimal_agent() -> RAGAgent:
    return object.__new__(RAGAgent)


def test_chunk_windows_exclude_wrong_owner_and_wrong_conversation(rag_scope_db):
    owner_id, other_id = uuid4(), uuid4()
    conversation_id, other_conversation_id = uuid4(), uuid4()
    document_id = uuid4()
    with rag_scope_db.factory.begin() as session:
        session.add_all([_user(owner_id), _user(other_id)])
        session.add_all(
            [
                Conversation(id=conversation_id, owner_id=owner_id, title="owner"),
                Conversation(id=other_conversation_id, owner_id=other_id, title="other"),
            ]
        )
        session.add(
            _document(
                document_id=document_id,
                conversation_id=conversation_id,
                filename="owned.pdf",
                filename_key="owned.pdf",
                upload_time=datetime.now(timezone.utc),
            )
        )
        session.add(_active_generation(document_id))
        session.add(_chunk(document_id, 0))

    repository = DocumentChunkRepository(rag_scope_db.factory)

    assert len(repository.get_window_for_scope(document_id, owner_id, conversation_id, 0, 8)) == 1
    assert repository.get_window_for_scope(document_id, other_id, conversation_id, 0, 8) == []
    assert (
        repository.get_window_for_scope(
            document_id,
            owner_id,
            other_conversation_id,
            0,
            8,
        )
        == []
    )


def test_search_and_image_repository_paths_exclude_mixed_tenant_rows(rag_scope_db):
    owner_id, other_id = uuid4(), uuid4()
    conversation_id, other_conversation_id = uuid4(), uuid4()
    owned_document_id, foreign_document_id = uuid4(), uuid4()
    owned_chunk = _chunk(owned_document_id, 0)
    foreign_chunk = _chunk(foreign_document_id, 0)
    owned_image_id, foreign_image_id = uuid4(), uuid4()
    now = datetime.now(timezone.utc)
    with rag_scope_db.factory.begin() as session:
        session.add_all([_user(owner_id), _user(other_id)])
        session.add_all(
            [
                Conversation(id=conversation_id, owner_id=owner_id, title="owner"),
                Conversation(id=other_conversation_id, owner_id=other_id, title="other"),
            ]
        )
        session.add_all(
            [
                _document(
                    document_id=owned_document_id,
                    conversation_id=conversation_id,
                    filename="owned.pdf",
                    filename_key="owned.pdf",
                    upload_time=now,
                ),
                _document(
                    document_id=foreign_document_id,
                    conversation_id=other_conversation_id,
                    filename="foreign.pdf",
                    filename_key="foreign.pdf",
                    upload_time=now,
                ),
            ]
        )
        session.add_all(
            [_active_generation(owned_document_id), _active_generation(foreign_document_id)]
        )
        session.add_all([owned_chunk, foreign_chunk])
        session.add_all(
            [
                DocumentImage(
                    id=owned_image_id,
                    document_id=owned_document_id,
                    chunk_id=owned_chunk.id,
                    image_path="owned.png",
                    image_caption="owned",
                    page_number=1,
                    mime_type="image/png",
                ),
                DocumentImage(
                    id=foreign_image_id,
                    document_id=foreign_document_id,
                    chunk_id=foreign_chunk.id,
                    image_path="foreign.png",
                    image_caption="foreign",
                    page_number=1,
                    mime_type="image/png",
                ),
            ]
        )

    chunk_repository = DocumentChunkRepository(rag_scope_db.factory)
    image_repository = DocumentImageRepository(rag_scope_db.factory)

    scoped_chunks = chunk_repository.get_by_ids_for_scope(
        [owned_chunk.id, foreign_chunk.id],
        user_id=owner_id,
        conversation_id=conversation_id,
    )
    assert [chunk.id for chunk in scoped_chunks] == [owned_chunk.id]
    assert (
        chunk_repository.get_by_ids_for_scope(
            [owned_chunk.id],
            user_id=other_id,
            conversation_id=conversation_id,
        )
        == []
    )
    assert (
        chunk_repository.get_by_ids_for_scope(
            [owned_chunk.id],
            user_id=owner_id,
            conversation_id=other_conversation_id,
        )
        == []
    )

    scoped_document_images = image_repository.get_by_document_for_scope(
        owned_document_id,
        user_id=owner_id,
        conversation_id=conversation_id,
    )
    assert [image.id for image in scoped_document_images] == [owned_image_id]
    assert (
        image_repository.get_by_document_for_scope(
            owned_document_id,
            user_id=other_id,
            conversation_id=conversation_id,
        )
        == []
    )
    assert (
        image_repository.get_by_document_for_scope(
            owned_document_id,
            user_id=owner_id,
            conversation_id=other_conversation_id,
        )
        == []
    )

    assert [
        image.id
        for image in image_repository.get_by_chunk_id_for_scope(
            owned_chunk.id,
            user_id=owner_id,
            conversation_id=conversation_id,
        )
    ] == [owned_image_id]
    assert (
        image_repository.get_by_chunk_id_for_scope(
            foreign_chunk.id,
            user_id=owner_id,
            conversation_id=conversation_id,
        )
        == []
    )
    assert (
        image_repository.get_by_id_for_scope(
            owned_image_id,
            user_id=owner_id,
            conversation_id=conversation_id,
        ).id
        == owned_image_id
    )
    assert (
        image_repository.get_by_id_for_scope(
            foreign_image_id,
            user_id=owner_id,
            conversation_id=conversation_id,
        )
        is None
    )


def test_duplicate_filename_resolution_is_rejected(rag_scope_db):
    owner_id, conversation_id = uuid4(), uuid4()
    now = datetime.now(timezone.utc)
    with rag_scope_db.factory.begin() as session:
        session.add(_user(owner_id))
        session.add(Conversation(id=conversation_id, owner_id=owner_id, title="duplicates"))
        session.add_all(
            [
                _document(
                    document_id=uuid4(),
                    conversation_id=conversation_id,
                    filename="duplicate.pdf",
                    filename_key="legacy-duplicate-a",
                    upload_time=now,
                ),
                _document(
                    document_id=uuid4(),
                    conversation_id=conversation_id,
                    filename="duplicate.pdf",
                    filename_key="legacy-duplicate-b",
                    upload_time=now,
                ),
            ]
        )

    resolved = asyncio.run(
        _minimal_agent().resolve_document_filename(
            "duplicate.pdf",
            conversation_id=str(conversation_id),
            user_id=owner_id,
        )
    )

    assert resolved is None


def test_filename_resolution_finds_document_on_later_listing_page(rag_scope_db):
    owner_id, conversation_id = uuid4(), uuid4()
    now = datetime.now(timezone.utc)
    target_id = uuid4()
    documents = [
        _document(
            document_id=uuid4(),
            conversation_id=conversation_id,
            filename=f"recent-{index}.pdf",
            filename_key=f"recent-{index}.pdf",
            upload_time=now + timedelta(minutes=index),
        )
        for index in range(10)
    ]
    documents.append(
        _document(
            document_id=target_id,
            conversation_id=conversation_id,
            filename="later-page.pdf",
            filename_key="later-page.pdf",
            upload_time=now - timedelta(minutes=1),
        )
    )
    with rag_scope_db.factory.begin() as session:
        session.add(_user(owner_id))
        session.add(Conversation(id=conversation_id, owner_id=owner_id, title="pagination"))
        session.add_all(documents)

    agent = _minimal_agent()
    page_two = asyncio.run(
        agent.list_conversation_documents(
            str(conversation_id),
            user_id=owner_id,
            page=2,
            page_size=10,
        )
    )
    resolved = asyncio.run(
        agent.resolve_document_filename(
            "later-page.pdf",
            conversation_id=str(conversation_id),
            user_id=owner_id,
        )
    )

    assert [doc["document_id"] for doc in page_two["documents"]] == [str(target_id)]
    assert resolved == str(target_id)


def test_document_listing_has_deterministic_id_order_when_upload_times_tie(rag_scope_db):
    owner_id, conversation_id = uuid4(), uuid4()
    tied_time = datetime.now(timezone.utc)
    document_ids = [uuid4(), uuid4(), uuid4()]
    with rag_scope_db.factory.begin() as session:
        session.add(_user(owner_id))
        session.add(Conversation(id=conversation_id, owner_id=owner_id, title="ties"))
        session.add_all(
            [
                _document(
                    document_id=document_id,
                    conversation_id=conversation_id,
                    filename=f"{document_id}.pdf",
                    filename_key=f"{document_id}.pdf",
                    upload_time=tied_time,
                )
                for document_id in reversed(document_ids)
            ]
        )

    listing = asyncio.run(
        _minimal_agent().list_conversation_documents(
            str(conversation_id),
            user_id=owner_id,
            page=1,
            page_size=10,
        )
    )

    assert [doc["document_id"] for doc in listing["documents"]] == sorted(
        str(document_id) for document_id in document_ids
    )


def test_numeric_ordinal_document_reference_is_rejected(rag_scope_db):
    resolved = asyncio.run(
        _resolve_document_reference(
            rag_agent=_minimal_agent(),
            document_ref="[1]",
            conversation_id=str(uuid4()),
            user_id=uuid4(),
        )
    )

    assert resolved is None


def test_scoped_cursor_is_true_only_for_another_authorized_chunk(rag_scope_db):
    owner_id, other_id = uuid4(), uuid4()
    conversation_id, other_conversation_id = uuid4(), uuid4()
    document_id = uuid4()
    with rag_scope_db.factory.begin() as session:
        session.add_all([_user(owner_id), _user(other_id)])
        session.add_all(
            [
                Conversation(id=conversation_id, owner_id=owner_id, title="cursor"),
                Conversation(id=other_conversation_id, owner_id=other_id, title="other"),
            ]
        )
        session.add(
            _document(
                document_id=document_id,
                conversation_id=conversation_id,
                filename="cursor.pdf",
                filename_key="cursor.pdf",
                upload_time=datetime.now(timezone.utc),
            )
        )
        session.add(_active_generation(document_id))
        session.add_all([_chunk(document_id, 0), _chunk(document_id, 1)])

    repository = DocumentChunkRepository(rag_scope_db.factory)

    assert repository.has_chunk_after_for_scope(document_id, owner_id, conversation_id, 1)
    assert not repository.has_chunk_after_for_scope(document_id, owner_id, conversation_id, 2)
    assert not repository.has_chunk_after_for_scope(document_id, other_id, conversation_id, 1)
    assert not repository.has_chunk_after_for_scope(
        document_id,
        owner_id,
        other_conversation_id,
        1,
    )


def test_evidence_expansion_stays_in_active_seed_generation_and_scope(rag_scope_db):
    owner_id, other_id = uuid4(), uuid4()
    conversation_id, other_conversation_id = uuid4(), uuid4()
    document_id, foreign_document_id = uuid4(), uuid4()
    active_generation_id, retired_generation_id = uuid4(), uuid4()
    now = datetime.now(timezone.utc)
    active_chunks = [_chunk(document_id, index) for index in range(4)]
    for chunk in active_chunks:
        chunk.index_generation_id = active_generation_id
    active_chunks[1].chunk_metadata = {"parent_chunk_id": str(active_chunks[3].id)}
    retired_neighbor = _chunk(document_id, 2)
    retired_neighbor.index_generation_id = retired_generation_id
    foreign_neighbor = _chunk(foreign_document_id, 2)
    foreign_neighbor.index_generation_id = foreign_document_id

    with rag_scope_db.factory.begin() as session:
        session.add_all([_user(owner_id), _user(other_id)])
        session.add_all(
            [
                Conversation(id=conversation_id, owner_id=owner_id, title="owner"),
                Conversation(
                    id=other_conversation_id,
                    owner_id=other_id,
                    title="other",
                ),
            ]
        )
        session.add_all(
            [
                _document(
                    document_id=document_id,
                    conversation_id=conversation_id,
                    filename="owned.pdf",
                    filename_key="owned.pdf",
                    upload_time=now,
                ),
                _document(
                    document_id=foreign_document_id,
                    conversation_id=other_conversation_id,
                    filename="foreign.pdf",
                    filename_key="foreign.pdf",
                    upload_time=now,
                ),
            ]
        )
        session.add_all(
            [
                DocumentIndexGeneration(
                    id=active_generation_id,
                    document_id=document_id,
                    status="active",
                    embedding_provider="test",
                    embedding_model="test",
                    embedding_dimension=8,
                    chunking_version="test",
                ),
                DocumentIndexGeneration(
                    id=retired_generation_id,
                    document_id=document_id,
                    status="retired",
                    embedding_provider="test",
                    embedding_model="test",
                    embedding_dimension=8,
                    chunking_version="test",
                ),
                _active_generation(foreign_document_id),
            ]
        )
        session.add_all([*active_chunks, retired_neighbor, foreign_neighbor])

    repository = DocumentChunkRepository(rag_scope_db.factory)
    rows = repository.get_context_expansion_for_scope(
        active_chunks[1].id,
        document_id=document_id,
        user_id=owner_id,
        conversation_id=conversation_id,
        max_neighbors=2,
    )

    assert [row.id for row in rows] == [active_chunks[3].id, active_chunks[0].id]
    assert retired_neighbor.id not in {row.id for row in rows}
    assert foreign_neighbor.id not in {row.id for row in rows}
    assert (
        repository.get_context_expansion_for_scope(
            active_chunks[1].id,
            document_id=document_id,
            user_id=other_id,
            conversation_id=conversation_id,
            max_neighbors=2,
        )
        == []
    )
    assert (
        repository.get_context_expansion_for_scope(
            active_chunks[1].id,
            document_id=document_id,
            user_id=owner_id,
            conversation_id=other_conversation_id,
            max_neighbors=2,
        )
        == []
    )
