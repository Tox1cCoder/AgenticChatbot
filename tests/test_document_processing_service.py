"""Phase 1 guards: server-side document processing isolation.

Assertions:
  * Processing uses a per-document working directory so concurrent uploads
    cannot share temp paths.
  * Chunks are persisted with the server-owned ``document_id`` /
    ``conversation_id`` / ``user_id`` so retrieval can filter by them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from PIL import Image

from app.schemas.document_image import DocumentImageCreate
from app.services.document_processing_service import DocumentProcessingService


def _build_service(tmp_path: Path, captured_payloads: list[dict] | None = None) -> DocumentProcessingService:
    service = object.__new__(DocumentProcessingService)
    service.settings = MagicMock()
    service.settings.temp_storage_path = str(tmp_path)
    service.settings.qdrant_collection_name = "documents_gemma"
    service.settings.qdrant_upsert_batch_size = 100
    service.settings.rag_embedding_dimension = 768
    service.settings.max_file_size_mb = 10
    service.settings.rag_chunk_target_tokens = 400
    service.settings.rag_chunk_overlap_tokens = 40
    service.settings.rag_chunk_max_tokens = 800
    service.settings.document_images_storage_path = str(tmp_path / "document_images")

    service.collection_name = "documents_gemma"
    service.embedding_dimension = 768

    def _upsert(collection_name: str, points):
        if captured_payloads is not None:
            for point in points:
                captured_payloads.append(point.payload)

    qdrant = MagicMock()
    qdrant.upsert.side_effect = _upsert
    service.qdrant_client = qdrant

    embedding = MagicMock()
    embedding.embed_documents.return_value = [[0.0] * 768]
    embedding.embed_query.return_value = [0.0] * 768
    embedding.dimension = 768
    embedding.model_name = "gemini-embedding-2"
    embedding.provider = "gemini"
    service.embedding_service = embedding

    service.document_image_repository = MagicMock()
    service.document_index_service = None
    service.celery_app = MagicMock()
    service._event_bus = MagicMock()
    service.gemini_client = None
    service._mineru_output_path = None
    return service


def test_store_chunks_includes_document_and_conversation_ids(tmp_path):
    captured: list[dict] = []
    service = _build_service(tmp_path, captured)

    chunks = [{"text": "chunk body"}]
    asyncio.run(
        service._store_chunks(
            chunks_with_metadata=chunks,
            filename="sample.txt",
            document_id="doc-1",
            conversation_id="conv-1",
            user_id="user-1",
        )
    )

    assert captured, "Expected at least one chunk to be upserted to Qdrant"
    payload = captured[0]
    assert payload["document_id"] == "doc-1"
    assert payload["conversation_id"] == "conv-1"


def test_store_chunks_writes_user_id_to_payload(tmp_path):
    """Server-owned user_id must be written to Qdrant payloads so retrieval can filter by it."""
    captured: list[dict] = []
    service = _build_service(tmp_path, captured)

    asyncio.run(
        service._store_chunks(
            chunks_with_metadata=[{"text": "chunk"}],
            filename="a.txt",
            document_id="doc-1",
            conversation_id="conv-1",
            user_id="user-42",
        )
    )

    assert captured, "Expected chunk upsert"
    assert captured[0]["user_id"] == "user-42"


def test_process_document_signature_accepts_user_id():
    """process_document must accept the server-owned user_id so the write path is scoped."""
    import inspect

    sig = inspect.signature(DocumentProcessingService.process_document)
    assert "user_id" in sig.parameters, (
        f"process_document must accept user_id; current params: {list(sig.parameters)}"
    )


def test_mineru_processing_uses_per_document_output_directory(tmp_path):
    """MinerU output paths must embed document_id so concurrent uploads don't collide."""
    import inspect

    source = inspect.getsource(DocumentProcessingService._process_with_mineru)
    # We only inspect the naming convention; running MinerU is out of scope for unit tests.
    assert "mineru_output_" in source
    assert "document_id" in source


def test_process_document_indexes_generated_image_caption_and_links_sql_chunk(tmp_path):
    """Generated visual captions must be indexed and image rows must point at SQL chunks."""
    service = _build_service(tmp_path)

    source_image = tmp_path / "chart.png"
    Image.new("RGB", (4, 4), color="red").save(source_image)

    image_data = {
        "path": str(source_image),
        "page_number": 0,
        "mime_type": "image/png",
    }
    chunks_with_metadata = [
        {
            "text": "Quarterly results\n[Image]",
            "page_start": 0,
            "page_end": 0,
            "has_images": True,
            "image_count": 1,
            "images": [image_data],
        }
    ]

    async def _fake_process_with_mineru(*_args, **_kwargs):
        service._extracted_images = [image_data]
        return chunks_with_metadata

    sql_chunk_id = uuid4()
    service._process_with_mineru = _fake_process_with_mineru
    service.gemini_client = object()
    service._generate_image_caption_with_retry = AsyncMock(
        return_value="A red bar chart showing revenue increasing each quarter."
    )
    service.document_index_service = MagicMock()
    service.document_index_service.index_document.return_value = [
        SimpleNamespace(id=sql_chunk_id, chunk_index=0, page_start=0, page_end=0)
    ]

    document_id = uuid4()
    conversation_id = uuid4()
    user_id = uuid4()

    result = asyncio.run(
        service.process_document(
            file_path=str(tmp_path / "report.pdf"),
            filename="report.pdf",
            document_id=str(document_id),
            conversation_id=str(conversation_id),
            user_id=str(user_id),
        )
    )

    service.document_index_service.index_document.assert_called_once()
    indexed_chunks = service.document_index_service.index_document.call_args.kwargs[
        "built_chunks"
    ]
    assert indexed_chunks
    assert "red bar chart" in indexed_chunks[0].content

    create_call = service.document_image_repository.create.call_args
    assert create_call is not None, "Expected image metadata to be persisted"
    image_record = create_call.args[0]
    assert isinstance(image_record, DocumentImageCreate)
    assert image_record.chunk_id == sql_chunk_id
    assert image_record.image_caption == "A red bar chart showing revenue increasing each quarter."
    assert result["chunks_stored"] == 1
    assert result["images_stored"] == 1


def test_process_document_passes_filename_to_index_document_reference(tmp_path):
    """The indexer needs the upload filename for Gemini document-title formatting."""
    service = _build_service(tmp_path)

    async def _fake_process_with_mineru(*_args, **_kwargs):
        return [{"text": "blue whale facts", "page_start": 0, "page_end": 0}]

    service._process_with_mineru = _fake_process_with_mineru
    service.document_index_service = MagicMock()
    service.document_index_service.index_document.return_value = []

    document_id = uuid4()
    conversation_id = uuid4()
    user_id = uuid4()

    asyncio.run(
        service.process_document(
            file_path=str(tmp_path / "blue-whale.pdf"),
            filename="Blue-whale-A4-fact-sheet.pdf",
            document_id=str(document_id),
            conversation_id=str(conversation_id),
            user_id=str(user_id),
        )
    )

    document_ref = service.document_index_service.index_document.call_args.kwargs[
        "document"
    ]
    assert document_ref.filename == "Blue-whale-A4-fact-sheet.pdf"


def test_process_document_parses_xlsx_without_mineru(tmp_path):
    """Excel uploads need a server-side parser because MinerU may emit no markdown."""
    from openpyxl import Workbook

    workbook_path = tmp_path / "Book1.xlsx"
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Sheet1"
    worksheet.append(["Thang", "Thu nhap", "Thue", "Nhan"])
    worksheet.append([1, 1000, 0.08, "=B2*(1-C2)"])
    worksheet.append([2, 1500, 0.10, "=B3*(1-C3)"])
    workbook.save(workbook_path)

    service = _build_service(tmp_path)
    service._process_with_mineru = AsyncMock(
        side_effect=AssertionError(".xlsx should not be routed through MinerU")
    )
    service.document_index_service = MagicMock()
    service.document_index_service.index_document.return_value = []

    document_id = uuid4()
    conversation_id = uuid4()
    user_id = uuid4()

    asyncio.run(
        service.process_document(
            file_path=str(workbook_path),
            filename="Book1.xlsx",
            document_id=str(document_id),
            conversation_id=str(conversation_id),
            user_id=str(user_id),
        )
    )

    built_chunks = service.document_index_service.index_document.call_args.kwargs[
        "built_chunks"
    ]
    content = "\n".join(chunk.content for chunk in built_chunks)
    assert "Sheet: Sheet1" in content
    assert "Thu nhap" in content
    assert "1000" in content
    assert "=B2*(1-C2)" in content
