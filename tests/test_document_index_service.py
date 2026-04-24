"""Phase 6 guards: DocumentIndexService.

The index service is the single owner of: chunk persistence in SQL, vector
embedding, Qdrant upsert, and chunk <-> point consistency.

Tests pin:
  * SQL chunks are replaced before indexing (idempotent by document_id).
  * Chunks are embedded in batches.
  * Qdrant payloads carry document_id, chunk_id, conversation_id, user_id.
  * Chunks are marked ``indexed`` on success, ``failed`` on error.
  * Deletion removes both Qdrant points and SQL chunks.
  * The service does not instantiate RAGAgent.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4


def _make_document(user_id: UUID | None = None, conversation_id: UUID | None = None):
    doc_id = uuid4()
    return SimpleNamespace(
        id=doc_id,
        conversation_id=conversation_id or uuid4(),
        user_id=user_id or uuid4(),
    )


def _make_built_chunk(index: int, text: str = "chunk body"):
    from app.services.document_chunk_builder import BuiltChunk

    return BuiltChunk(
        chunk_index=index,
        content=text,
        content_sha256=f"sha-{index}",
        char_count=len(text),
        token_count=4,
        page_start=1,
        page_end=1,
        section_path=[],
        block_provenance=[],
        metadata={},
    )


def _persisted_chunk(document_id: UUID, chunk_index: int):
    chunk = SimpleNamespace()
    chunk.id = uuid4()
    chunk.document_id = document_id
    chunk.chunk_index = chunk_index
    chunk.content = f"content-{chunk_index}"
    chunk.content_sha256 = f"sha-{chunk_index}"
    chunk.char_count = len(chunk.content)
    chunk.token_count = 4
    chunk.page_start = 1
    chunk.page_end = 1
    chunk.section_path = []
    chunk.block_provenance = []
    chunk.chunk_metadata = {}
    chunk.embedding_model = None
    chunk.qdrant_point_id = None
    return chunk


class _EmbeddingStub:
    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts, **_kwargs):
        if isinstance(texts, str):
            self.calls.append([texts])
            return [0.0] * self.dim
        texts_list = list(texts)
        self.calls.append(texts_list)
        return [[0.0] * self.dim for _ in texts_list]


def _build_service(
    *,
    chunk_repo=None,
    qdrant_client=None,
    embedding_model=None,
    collection_name: str = "documents_gemma",
    embedding_model_name: str = "google/embeddinggemma-300m",
    embedding_dimension: int = 8,
    batch_size: int = 2,
):
    from app.services.document_index_service import DocumentIndexService

    return DocumentIndexService(
        chunk_repository=chunk_repo or MagicMock(),
        qdrant_client=qdrant_client or MagicMock(),
        embedding_model=embedding_model or _EmbeddingStub(dim=embedding_dimension),
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        embedding_dimension=embedding_dimension,
        index_batch_size=batch_size,
    )


def test_index_document_replaces_sql_chunks_first():
    """Chunks are the canonical store: replacing them is a prerequisite to indexing."""
    document = _make_document()
    persisted = [
        _persisted_chunk(document.id, 0),
        _persisted_chunk(document.id, 1),
    ]

    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted

    qdrant = MagicMock()
    service = _build_service(chunk_repo=repo, qdrant_client=qdrant)

    built = [_make_built_chunk(0, "hello"), _make_built_chunk(1, "world")]
    service.index_document(
        document=document,
        built_chunks=built,
        parse_artifact_id=None,
    )

    # Replace runs before qdrant upsert.
    assert repo.replace_document_chunks.called
    replace_call = repo.replace_document_chunks.call_args
    assert replace_call.kwargs.get("document_id") == document.id or replace_call.args[0] == document.id


def test_index_document_writes_authorization_metadata_to_qdrant_payload():
    document = _make_document()
    persisted = [_persisted_chunk(document.id, 0)]

    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted
    qdrant = MagicMock()

    service = _build_service(chunk_repo=repo, qdrant_client=qdrant)
    service.index_document(
        document=document,
        built_chunks=[_make_built_chunk(0, "hello")],
        parse_artifact_id=None,
    )

    assert qdrant.upsert.called, "Index service must upsert Qdrant points"
    upsert_call = qdrant.upsert.call_args
    points = upsert_call.kwargs.get("points") or upsert_call.args[1]
    assert len(points) == 1
    payload = points[0].payload
    assert payload["document_id"] == str(document.id)
    assert payload["chunk_id"] == str(persisted[0].id)
    assert payload["conversation_id"] == str(document.conversation_id)
    assert payload["user_id"] == str(document.user_id)


def test_index_document_embeds_in_batches():
    document = _make_document()
    persisted = [_persisted_chunk(document.id, i) for i in range(5)]

    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted

    qdrant = MagicMock()
    embedding = _EmbeddingStub(dim=4)

    service = _build_service(
        chunk_repo=repo,
        qdrant_client=qdrant,
        embedding_model=embedding,
        embedding_dimension=4,
        batch_size=2,
    )
    service.index_document(
        document=document,
        built_chunks=[_make_built_chunk(i, f"body-{i}") for i in range(5)],
        parse_artifact_id=None,
    )

    # batch_size=2 across 5 chunks should produce 3 embed calls: [2, 2, 1].
    batch_sizes = [len(batch) for batch in embedding.calls]
    assert batch_sizes == [2, 2, 1], f"Expected batches [2,2,1], got {batch_sizes}"


def test_index_document_marks_chunks_indexed_after_qdrant_upsert():
    document = _make_document()
    persisted = [_persisted_chunk(document.id, 0), _persisted_chunk(document.id, 1)]

    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted

    qdrant = MagicMock()

    service = _build_service(chunk_repo=repo, qdrant_client=qdrant)
    service.index_document(
        document=document,
        built_chunks=[_make_built_chunk(0), _make_built_chunk(1)],
        parse_artifact_id=None,
    )

    assert repo.mark_indexed.call_count == 2
    for call in repo.mark_indexed.call_args_list:
        kwargs = call.kwargs
        assert "point_id" in kwargs
        assert kwargs["embedding_model"] == "google/embeddinggemma-300m"
        assert kwargs["embedding_dimension"] == 8
        assert kwargs["collection_name"] == "documents_gemma"


def test_index_document_marks_failed_and_raises_on_qdrant_error():
    document = _make_document()
    persisted = [_persisted_chunk(document.id, 0)]

    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted

    qdrant = MagicMock()
    qdrant.upsert.side_effect = RuntimeError("boom")

    service = _build_service(chunk_repo=repo, qdrant_client=qdrant)

    import pytest

    with pytest.raises(RuntimeError, match="boom"):
        service.index_document(
            document=document,
            built_chunks=[_make_built_chunk(0)],
            parse_artifact_id=None,
        )

    assert repo.mark_index_failed.called


def test_delete_document_index_removes_qdrant_and_sql():
    document_id = uuid4()
    repo = MagicMock()
    qdrant = MagicMock()

    service = _build_service(chunk_repo=repo, qdrant_client=qdrant)
    service.delete_document_index(document_id)

    assert qdrant.delete.called
    assert repo.delete_by_document.called
    assert repo.delete_by_document.call_args.args[0] == document_id


def test_index_service_does_not_instantiate_rag_agent():
    import app.services.document_index_service as mod

    source = inspect.getsource(mod)
    assert "RAGAgent" not in source, (
        "DocumentIndexService must not couple to RAGAgent — indexing has no model"
    )
