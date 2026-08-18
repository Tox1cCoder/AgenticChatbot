"""Phase 6 + Phase 11 guards: DocumentIndexService.

The index service is the single owner of: chunk persistence in SQL, vector
embedding, Qdrant upsert, and chunk <-> point consistency.

Tests pin:
  * SQL chunks are replaced before indexing (idempotent by document_id).
  * Chunks are embedded in batches via the embedding service adapter.
  * Document titles are passed to ``embed_documents`` for the Gemini doc-format prompt.
  * Qdrant payloads carry document_id, chunk_id, conversation_id, user_id,
    embedding_provider, and modality (Phase 11).
  * Chunks are marked ``indexed`` on success, ``failed`` on error.
  * Deletion removes both Qdrant points and SQL chunks.
  * The service does not instantiate RAGAgent.
  * ``ensure_collection`` is the single owner of Qdrant collection bootstrap.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from sqlalchemy.orm.exc import DetachedInstanceError


def _make_document(
    user_id: UUID | None = None,
    conversation_id: UUID | None = None,
    filename: str = "test.pdf",
):
    doc_id = uuid4()
    return SimpleNamespace(
        id=doc_id,
        conversation_id=conversation_id or uuid4(),
        user_id=user_id or uuid4(),
        filename=filename,
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


class _DetachedDocumentRelationshipChunk:
    def __init__(self, document_id: UUID, chunk_index: int):
        self.id = uuid4()
        self.document_id = document_id
        self.chunk_index = chunk_index
        self.content = f"content-{chunk_index}"
        self.content_sha256 = f"sha-{chunk_index}"
        self.char_count = len(self.content)
        self.token_count = 4
        self.page_start = 1
        self.page_end = 1
        self.section_path = []
        self.block_provenance = []
        self.chunk_metadata = {}
        self.embedding_model = None
        self.qdrant_point_id = None

    @property
    def document(self):
        raise DetachedInstanceError(
            "Parent instance is not bound to a Session; lazy load cannot proceed"
        )


class _EmbeddingStub:
    """Stand-in for ``GeminiRAGEmbeddingService`` that records calls."""

    provider = "gemini"

    def __init__(self, dim: int = 8, model_name: str = "gemini-embedding-2"):
        self.dim = dim
        self.model_name = model_name
        self.dimension = dim
        self.doc_calls: list[tuple[list[str], list[str | None]]] = []
        self.query_calls: list[str] = []
        self.image_calls: list[tuple[bytes, str]] = []

    def embed_documents(self, texts, *, titles=None, usage_context=None):
        texts_list = list(texts)
        title_list = list(titles) if titles is not None else [None] * len(texts_list)
        self.doc_calls.append((texts_list, title_list))
        self.last_usage_context = usage_context
        return [[0.0] * self.dim for _ in texts_list]

    def embed_query(self, query: str, *, usage_context=None):
        self.query_calls.append(query)
        return [0.0] * self.dim

    def embed_image(self, image_bytes, *, mime_type, usage_context=None):
        self.image_calls.append((image_bytes, mime_type))
        return [0.0] * self.dim


class _GenerationRepoStub:
    def __init__(self):
        self.rows = {}

    def get_active(self, document_id):
        return next(
            (
                row
                for row in self.rows.values()
                if row.document_id == document_id and row.status == "active"
            ),
            None,
        )

    def create(self, *, document_id, **kwargs):
        row = SimpleNamespace(id=uuid4(), document_id=document_id, status="building", **kwargs)
        self.rows[row.id] = row
        return row

    def mark_ready(self, generation_id):
        self.rows[generation_id].status = "ready"
        return self.rows[generation_id]

    def activate(self, generation_id):
        target = self.rows[generation_id]
        for row in self.rows.values():
            if row.document_id == target.document_id and row.status == "active":
                row.status = "retired"
        target.status = "active"
        return target

    def mark_failed(self, generation_id, failure_code):
        self.rows[generation_id].status = "failed"
        return self.rows[generation_id]


def _condition_matches(payload: dict, condition) -> bool:
    match = getattr(condition, "match", None)
    value = getattr(match, "value", None)
    if value is None:
        # Only MatchValue conditions are used by the verify-path filters this
        # fake supports; anything else is treated conservatively as no match.
        return False
    return payload.get(condition.key) == value


def _generation_aware_qdrant(qdrant):
    captured = []

    def capture_upsert(*args, **kwargs):
        captured.extend(kwargs.get("points") or args[1])

    def _filtered(count_filter):
        if count_filter is None:
            return list(captured)
        conditions = getattr(count_filter, "must", None) or []
        return [
            point
            for point in captured
            if all(_condition_matches(point.payload, condition) for condition in conditions)
        ]

    def _retrieve(**kwargs):
        wanted = {str(value) for value in (kwargs.get("ids") or [])}
        return [
            SimpleNamespace(id=point.id, payload=point.payload, vector=point.vector)
            for point in captured
            if str(point.id) in wanted
        ]

    if qdrant.upsert.side_effect is None:
        qdrant.upsert.side_effect = capture_upsert
    if qdrant.count.side_effect is None:
        qdrant.count.side_effect = lambda **kwargs: SimpleNamespace(
            count=len(_filtered(kwargs.get("count_filter")))
        )
    if qdrant.retrieve.side_effect is None:
        qdrant.retrieve.side_effect = _retrieve
    return qdrant


def _build_service(
    *,
    chunk_repo=None,
    qdrant_client=None,
    embedding_service=None,
    collection_name: str = "documents_gemini_embedding_2_3072",
    embedding_model_name: str = "gemini-embedding-2",
    embedding_dimension: int = 8,
    embedding_provider: str = "gemini",
    document_image_repository=None,
    multimodal_image_embeddings_enabled: bool = False,
):
    from app.services.document_index_service import DocumentIndexService

    repository = chunk_repo or MagicMock()
    repository.create_generation_chunks.side_effect = (
        lambda _document_id, generation_id, _rows: [
            setattr(chunk, "index_generation_id", generation_id) or chunk
            for chunk in repository.replace_document_chunks.return_value
        ]
    )
    qdrant = _generation_aware_qdrant(qdrant_client or MagicMock())
    return DocumentIndexService(
        chunk_repository=repository,
        generation_repository=_GenerationRepoStub(),
        qdrant_client=qdrant,
        embedding_service=embedding_service or _EmbeddingStub(dim=embedding_dimension),
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        embedding_dimension=embedding_dimension,
        embedding_provider=embedding_provider,
        document_image_repository=document_image_repository,
        multimodal_image_embeddings_enabled=multimodal_image_embeddings_enabled,
    )


def document():
    return _make_document()


def document_image(document_id: UUID | None = None, **overrides):
    defaults = dict(
        id=uuid4(),
        document_id=document_id or uuid4(),
        chunk_id=None,
        page_number=1,
        mime_type="image/png",
        image_path="images/chart.png",
        content_sha256="a" * 64,
        section_path=["Q1 Results"],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class _ImageRepoStub:
    def __init__(self):
        self.updates: list[tuple[UUID, Any]] = []

    def update(self, image_id, update_data):
        self.updates.append((image_id, update_data))
        return SimpleNamespace(id=image_id, chunk_id=update_data.chunk_id)


def test_constructor_does_not_expose_retired_index_batch_size():
    from app.services.document_index_service import DocumentIndexService

    assert "index_batch_size" not in inspect.signature(DocumentIndexService).parameters


def test_index_document_creates_generation_chunks_first():
    """Inactive generation chunks are persisted before indexing."""
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

    assert repo.create_generation_chunks.called
    replace_call = repo.create_generation_chunks.call_args
    assert (
        replace_call.kwargs.get("document_id") == document.id or replace_call.args[0] == document.id
    )


def test_index_document_threads_usage_context_to_embeddings():
    """Owner attribution flows to the embedding service (Task 10)."""
    from uuid import uuid4

    from app.usage.types import UsageContext

    document = _make_document()
    persisted = [_persisted_chunk(document.id, 0)]
    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted
    stub = _EmbeddingStub(dim=8)

    service = _build_service(chunk_repo=repo, qdrant_client=MagicMock(), embedding_service=stub)
    owner_id = uuid4()
    service.index_document(
        document=document,
        built_chunks=[_make_built_chunk(0, "hello")],
        parse_artifact_id=None,
        usage_context=UsageContext(user_id=owner_id, operation="document_index"),
    )

    assert stub.last_usage_context is not None
    assert stub.last_usage_context.user_id == owner_id


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


def test_index_document_payload_includes_phase11_provider_and_modality():
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

    points = qdrant.upsert.call_args.kwargs.get("points")
    payload = points[0].payload
    assert payload["embedding_provider"] == "gemini"
    assert payload["modality"] == "text"
    assert payload["embedding_model"] == "gemini-embedding-2"


def test_index_document_passes_titles_to_embedding_service():
    document = _make_document(filename="quarterly.pdf")
    persisted = [_persisted_chunk(document.id, 0)]
    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted

    embedding = _EmbeddingStub(dim=4)
    service = _build_service(
        chunk_repo=repo,
        embedding_service=embedding,
        embedding_dimension=4,
    )

    service.index_document(
        document=document,
        built_chunks=[_make_built_chunk(0, "body")],
        parse_artifact_id=None,
    )

    assert embedding.doc_calls, "embed_documents must be called for indexing"
    texts, titles = embedding.doc_calls[0]
    assert titles == ["quarterly.pdf"], (
        f"document title must be threaded into titles list: {titles}"
    )


def test_index_document_does_not_lazy_load_document_from_detached_chunks():
    document = SimpleNamespace(id=uuid4(), conversation_id=uuid4(), user_id=uuid4())
    persisted = [_DetachedDocumentRelationshipChunk(document.id, 0)]
    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted

    service = _build_service(chunk_repo=repo)

    service.index_document(
        document=document,
        built_chunks=[_make_built_chunk(0, "body")],
        parse_artifact_id=None,
    )

    # T004: bulk mark is used now, not per-chunk mark_indexed.
    assert repo.mark_indexed_bulk.called


def test_index_document_embeds_in_single_call():
    """After T004, _embed_and_upsert makes a single embed_documents call for
    all chunks. Internal batching is the embedding service's responsibility."""
    document = _make_document()
    persisted = [_persisted_chunk(document.id, i) for i in range(5)]

    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted

    qdrant = MagicMock()
    embedding = _EmbeddingStub(dim=4)

    service = _build_service(
        chunk_repo=repo,
        qdrant_client=qdrant,
        embedding_service=embedding,
        embedding_dimension=4,
    )
    service.index_document(
        document=document,
        built_chunks=[_make_built_chunk(i, f"body-{i}") for i in range(5)],
        parse_artifact_id=None,
    )

    # DocumentIndexService now makes exactly one embed_documents call for all
    # chunks. The embedding service handles internal batching.
    assert len(embedding.doc_calls) == 1, (
        f"Expected 1 embed_documents call, got {len(embedding.doc_calls)}"
    )
    texts, _ = embedding.doc_calls[0]
    assert len(texts) == 5, f"All 5 chunk texts must be passed; got {len(texts)}"


def test_index_document_marks_chunks_indexed_after_qdrant_upsert():
    """After T004, mark_indexed_bulk is called once (not per-chunk mark_indexed)."""
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

    assert repo.mark_indexed_bulk.call_count == 1
    bulk_call = repo.mark_indexed_bulk.call_args
    kwargs = bulk_call.kwargs
    assert kwargs["embedding_model"] == "gemini-embedding-2"
    assert kwargs["embedding_dimension"] == 8
    assert kwargs["collection_name"] == "documents_gemini_embedding_2_3072"
    # Verify both chunk IDs are included in the bulk call.
    chunk_ids = bulk_call.args[0] if bulk_call.args else kwargs.get("chunk_ids", [])
    assert len(chunk_ids) == 2


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


def test_ensure_collection_creates_if_absent():
    repo = MagicMock()
    qdrant = MagicMock()
    qdrant.get_collections.return_value = SimpleNamespace(collections=[])

    service = _build_service(chunk_repo=repo, qdrant_client=qdrant)
    service.ensure_collection()

    assert qdrant.create_collection.called, (
        "Missing collection must be created by ensure_collection"
    )
    create_kwargs = qdrant.create_collection.call_args.kwargs
    assert create_kwargs["collection_name"] == "documents_gemini_embedding_2_3072"
    vectors = create_kwargs["vectors_config"]
    assert getattr(vectors, "size", None) == 8


def test_collection_bootstrap_creates_filter_indexes_before_first_upsert():
    document = _make_document()
    persisted = [_persisted_chunk(document.id, 0)]
    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted
    qdrant = MagicMock()
    qdrant.get_collections.return_value = SimpleNamespace(collections=[])
    service = _build_service(chunk_repo=repo, qdrant_client=qdrant)

    service.index_document(
        document=document,
        built_chunks=[_make_built_chunk(0)],
        parse_artifact_id=None,
    )

    fields = [
        call.kwargs["field_name"]
        for call in qdrant.create_payload_index.call_args_list
    ]
    assert fields == [
        "user_id",
        "conversation_id",
        "document_id",
        "modality",
        "index_generation",
        "is_active",
    ]
    first_upsert = next(
        index
        for index, call in enumerate(qdrant.mock_calls)
        if call[0] == "upsert"
    )
    last_index = max(
        index
        for index, call in enumerate(qdrant.mock_calls)
        if call[0] == "create_payload_index"
    )
    assert last_index < first_upsert


def test_collection_bootstrap_is_idempotent_per_service_instance():
    qdrant = MagicMock()
    qdrant.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="documents_gemini_embedding_2_3072")]
    )
    qdrant.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=8)))
    )
    service = _build_service(qdrant_client=qdrant)

    service.ensure_collection()
    service.ensure_collection()

    assert qdrant.create_payload_index.call_count == 6


def test_tenant_payload_index_falls_back_only_for_explicit_unsupported_version():
    from httpx import Headers
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.models import KeywordIndexParams, PayloadSchemaType

    qdrant = MagicMock()
    qdrant.get_collections.return_value = SimpleNamespace(collections=[])
    unsupported = UnexpectedResponse(
        status_code=400,
        reason_phrase="Bad Request",
        content=b'Unknown field "is_tenant" for KeywordIndexParams',
        headers=Headers(),
    )
    qdrant.create_payload_index.side_effect = [unsupported, None, None, None, None, None, None]
    service = _build_service(qdrant_client=qdrant)

    service.ensure_collection()

    user_calls = [
        call for call in qdrant.create_payload_index.call_args_list
        if call.kwargs["field_name"] == "user_id"
    ]
    assert len(user_calls) == 2
    assert isinstance(user_calls[0].kwargs["field_schema"], KeywordIndexParams)
    assert user_calls[0].kwargs["field_schema"].is_tenant is True
    assert user_calls[1].kwargs["field_schema"] == PayloadSchemaType.KEYWORD


def test_payload_index_does_not_hide_unrelated_qdrant_errors():
    qdrant = MagicMock()
    qdrant.get_collections.return_value = SimpleNamespace(collections=[])
    qdrant.create_payload_index.side_effect = RuntimeError("network down")
    service = _build_service(qdrant_client=qdrant)

    import pytest

    with pytest.raises(RuntimeError, match="network down"):
        service.ensure_collection()


def test_indexing_aborts_before_embedding_and_upsert_when_bootstrap_is_unavailable():
    document = _make_document()
    repository = MagicMock()
    qdrant = MagicMock()
    qdrant.get_collections.side_effect = RuntimeError("unavailable")
    embedding = _EmbeddingStub(dim=8)
    service = _build_service(
        chunk_repo=repository,
        qdrant_client=qdrant,
        embedding_service=embedding,
    )

    import pytest

    with pytest.raises(RuntimeError, match="payload indexes"):
        service.index_document(
            document=document,
            built_chunks=[_make_built_chunk(0)],
            parse_artifact_id=None,
        )

    assert embedding.doc_calls == []
    qdrant.upsert.assert_not_called()
    repository.create_generation_chunks.assert_not_called()


def test_ensure_collection_validates_dimension_match():
    import pytest

    repo = MagicMock()
    qdrant = MagicMock()
    qdrant.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="documents_gemini_embedding_2_3072")]
    )
    info = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=512)))
    )
    qdrant.get_collection.return_value = info

    service = _build_service(chunk_repo=repo, qdrant_client=qdrant, embedding_dimension=8)

    with pytest.raises(ValueError, match="vector size"):
        service.ensure_collection()


def test_index_document_calls_bulk_mark_indexed(monkeypatch=None):
    """T005 — T004: mark_indexed_bulk is called once; mark_indexed is NOT called.

    After T004, DocumentIndexService.index_document calls
    chunk_repository.mark_indexed_bulk once with all chunk IDs instead of
    calling mark_indexed once per chunk.
    """
    import pytest  # noqa: F401 — needed for assertions in this scope

    document = _make_document()
    persisted = [
        _persisted_chunk(document.id, 0),
        _persisted_chunk(document.id, 1),
        _persisted_chunk(document.id, 2),
    ]

    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted

    qdrant = MagicMock()

    service = _build_service(chunk_repo=repo, qdrant_client=qdrant)
    service.index_document(
        document=document,
        built_chunks=[_make_built_chunk(i) for i in range(3)],
        parse_artifact_id=None,
    )

    # Bulk mark is called exactly once.
    assert repo.mark_indexed_bulk.call_count == 1, (
        f"mark_indexed_bulk must be called once; got {repo.mark_indexed_bulk.call_count}"
    )
    # Per-chunk mark_indexed must NOT be called.
    assert repo.mark_indexed.call_count == 0, (
        f"mark_indexed (per-chunk) must not be called; got {repo.mark_indexed.call_count}"
    )
    # The IDs passed to bulk mark match the persisted chunk IDs.
    bulk_call = repo.mark_indexed_bulk.call_args
    chunk_ids_passed = bulk_call.args[0]
    expected_ids = {chunk.id for chunk in persisted}
    assert set(chunk_ids_passed) == expected_ids, (
        f"Bulk mark must receive all persisted chunk IDs; "
        f"expected {expected_ids}, got {set(chunk_ids_passed)}"
    )


# ---------------------------------------------------------------------------
# Task 11: image points, chunk-linking, and native multimodal embeddings
# ---------------------------------------------------------------------------


def test_image_point_links_image_without_chunk_id():
    index_service = _build_service()
    point = index_service._point_for_image(document(), document_image())

    assert point.payload["modality"] == "image"
    assert point.payload["image_id"]
    assert "chunk_id" not in point.payload


def test_index_document_links_image_rows_to_matching_chunk_by_page():
    document_row = _make_document()
    persisted = [_persisted_chunk(document_row.id, 0)]  # page_start = page_end = 1
    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted
    image_repo = _ImageRepoStub()
    image = document_image(document_row.id, page_number=1)

    service = _build_service(chunk_repo=repo, document_image_repository=image_repo)
    service.index_document(
        document=document_row,
        built_chunks=[_make_built_chunk(0)],
        parse_artifact_id=None,
        image_rows=[image],
    )

    assert image_repo.updates, "image chunk_id must be linked to the matching persisted chunk"
    linked_image_id, update_data = image_repo.updates[0]
    assert linked_image_id == image.id
    assert update_data.chunk_id == persisted[0].id


def test_index_document_requires_image_repository_when_image_rows_given():
    import pytest

    document_row = _make_document()
    persisted = [_persisted_chunk(document_row.id, 0)]
    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted
    image = document_image(document_row.id)

    service = _build_service(chunk_repo=repo)  # no document_image_repository

    with pytest.raises(ValueError, match="document_image_repository"):
        service.index_document(
            document=document_row,
            built_chunks=[_make_built_chunk(0)],
            parse_artifact_id=None,
            image_rows=[image],
        )


def test_index_document_skips_native_image_embedding_when_flag_disabled():
    document_row = _make_document()
    persisted = [_persisted_chunk(document_row.id, 0)]
    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted
    image_repo = _ImageRepoStub()
    embedding = _EmbeddingStub(dim=4)
    image = document_image(document_row.id, page_number=1)

    service = _build_service(
        chunk_repo=repo,
        embedding_service=embedding,
        embedding_dimension=4,
        document_image_repository=image_repo,
        multimodal_image_embeddings_enabled=False,
    )
    service.index_document(
        document=document_row,
        built_chunks=[_make_built_chunk(0)],
        parse_artifact_id=None,
        image_rows=[image],
    )

    assert embedding.image_calls == [], "flag off must never call embed_image"
    assert image_repo.updates, "linking must still happen when the flag is off"


def test_index_document_embeds_and_upserts_image_points_when_flag_enabled(tmp_path):
    document_row = _make_document()
    persisted = [_persisted_chunk(document_row.id, 0)]
    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted
    qdrant = MagicMock()
    image_repo = _ImageRepoStub()
    embedding = _EmbeddingStub(dim=4)

    image_file = tmp_path / "chart.png"
    image_file.write_bytes(b"fake-image-bytes")
    image = document_image(document_row.id, page_number=1, image_path=str(image_file))

    service = _build_service(
        chunk_repo=repo,
        qdrant_client=qdrant,
        embedding_service=embedding,
        embedding_dimension=4,
        document_image_repository=image_repo,
        multimodal_image_embeddings_enabled=True,
    )
    service.index_document(
        document=document_row,
        built_chunks=[_make_built_chunk(0)],
        parse_artifact_id=None,
        image_rows=[image],
    )

    assert embedding.image_calls, "embed_image must be called once per canonical image"
    assert embedding.image_calls[0][1] == "image/png"

    image_points = [
        point
        for call in qdrant.upsert.call_args_list
        for point in (call.kwargs.get("points") or call.args[1])
        if point.payload.get("modality") == "image"
    ]
    assert len(image_points) == 1
    assert image_points[0].payload["image_id"] == str(image.id)
    assert image_points[0].payload["document_id"] == str(document_row.id)


def test_index_document_raises_before_activation_when_image_points_go_missing(tmp_path):
    """Both text and image point counts must verify before a generation activates."""
    import pytest

    document_row = _make_document()
    persisted = [_persisted_chunk(document_row.id, 0)]
    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted
    image_repo = _ImageRepoStub()
    embedding = _EmbeddingStub(dim=4)

    image_file = tmp_path / "chart.png"
    image_file.write_bytes(b"fake-image-bytes")
    image = document_image(document_row.id, page_number=1, image_path=str(image_file))

    captured = []

    def _drop_image_points_on_upsert(*args, **kwargs):
        points = kwargs.get("points") or args[1]
        # Simulate an upstream failure that silently drops image points while
        # text points still land — activation must still be blocked.
        captured.extend(point for point in points if point.payload.get("modality") != "image")

    qdrant = MagicMock()
    qdrant.upsert.side_effect = _drop_image_points_on_upsert
    qdrant.count.side_effect = lambda **kwargs: SimpleNamespace(
        count=len(
            [
                point
                for point in captured
                if all(
                    _condition_matches(point.payload, condition)
                    for condition in (getattr(kwargs.get("count_filter"), "must", None) or [])
                )
            ]
        )
    )
    qdrant.retrieve.side_effect = lambda **kwargs: [
        SimpleNamespace(id=point.id, payload=point.payload, vector=point.vector)
        for point in captured
        if str(point.id) in {str(value) for value in (kwargs.get("ids") or [])}
    ]

    service = _build_service(
        chunk_repo=repo,
        qdrant_client=qdrant,
        embedding_service=embedding,
        embedding_dimension=4,
        document_image_repository=image_repo,
        multimodal_image_embeddings_enabled=True,
    )

    with pytest.raises(ValueError, match="image generation point"):
        service.index_document(
            document=document_row,
            built_chunks=[_make_built_chunk(0)],
            parse_artifact_id=None,
            image_rows=[image],
        )

    assert repo.mark_index_failed.called


# ---------------------------------------------------------------------------
# Review round 1 findings
# ---------------------------------------------------------------------------


def test_point_id_for_image_is_scoped_per_generation():
    """Finding 2: the same image must get a different Qdrant point id per
    generation, so reindexing upserts a new point rather than overwriting
    (and later, on purge, deleting) the previously-active generation's
    image point."""
    service = _build_service()
    image = document_image()
    generation_a = uuid4()
    generation_b = uuid4()

    point_a = service._point_for_image(document(), image, index_generation_id=generation_a)
    point_b = service._point_for_image(document(), image, index_generation_id=generation_b)

    assert point_a.id != point_b.id


def test_index_document_does_not_link_images_when_verification_fails():
    """Finding 3: images must not be relinked to a generation that never
    passes verification — that would repoint (and later orphan, via purge)
    an older, still-active generation's image links."""
    document_row = _make_document()
    persisted = [_persisted_chunk(document_row.id, 0)]
    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted
    image_repo = _ImageRepoStub()
    image = document_image(document_row.id, page_number=1)

    qdrant = MagicMock()
    # Force the text-point verification check to fail regardless of what
    # was actually upserted — isolates the linking-order guarantee from
    # unrelated upsert/count bookkeeping.
    qdrant.count.side_effect = lambda **_kwargs: SimpleNamespace(count=0)

    service = _build_service(
        chunk_repo=repo,
        qdrant_client=qdrant,
        document_image_repository=image_repo,
    )

    import pytest

    with pytest.raises(ValueError, match="generation point count mismatch"):
        service.index_document(
            document=document_row,
            built_chunks=[_make_built_chunk(0)],
            parse_artifact_id=None,
            image_rows=[image],
        )

    assert image_repo.updates == [], "images must not be linked when verification fails"


def test_index_document_skips_unreadable_image_without_failing_generation(tmp_path):
    """Finding 7: an unreadable image file must not abort the whole
    generation — RAGImageSelector already tolerates the same condition."""
    document_row = _make_document()
    persisted = [_persisted_chunk(document_row.id, 0)]
    repo = MagicMock()
    repo.replace_document_chunks.return_value = persisted
    image_repo = _ImageRepoStub()
    embedding = _EmbeddingStub(dim=4)
    missing_path = tmp_path / "does-not-exist.png"
    image = document_image(document_row.id, page_number=1, image_path=str(missing_path))

    service = _build_service(
        chunk_repo=repo,
        embedding_service=embedding,
        embedding_dimension=4,
        document_image_repository=image_repo,
        multimodal_image_embeddings_enabled=True,
    )

    persisted_chunks = service.index_document(
        document=document_row,
        built_chunks=[_make_built_chunk(0)],
        parse_artifact_id=None,
        image_rows=[image],
    )

    assert persisted_chunks, "the generation must still succeed"
    assert embedding.image_calls == [], "the unreadable image must be skipped, not embedded"
    assert image_repo.updates, "chunk linking still happens for the unembedded image"
