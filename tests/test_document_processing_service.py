"""Phase 1 guards: server-side document processing isolation.

Assertions:
  * Processing uses a per-document working directory so concurrent uploads
    cannot share temp paths.
  * Chunks are persisted with the server-owned ``document_id`` /
    ``conversation_id`` / ``user_id`` so retrieval can filter by them.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from PIL import Image

from app.schemas.document_image import DocumentImageCreate
from app.services.document_blocks import NormalizedBlock
from app.services.document_parse_service import DocumentParseService
from app.services.document_processing_service import DocumentProcessingService


def _build_service(tmp_path: Path) -> DocumentProcessingService:
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
    service.settings.image_caption_max_concurrency = 4

    service.collection_name = "documents_gemma"
    service.embedding_dimension = 768

    service.document_image_repository = MagicMock()
    service.document_index_service = None
    service.celery_app = MagicMock()
    service._event_bus = MagicMock()
    service.gemini_client = None
    service._mineru_output_path = None
    return service


def test_document_processing_service_requires_index_service_in_constructor():
    """Processing must not accept dependencies used only by the retired direct-Qdrant path."""
    params = inspect.signature(DocumentProcessingService.__init__).parameters
    assert "document_index_service" in params
    assert "qdrant_client" not in params
    assert "embedding_service" not in params


def test_document_processing_service_has_no_direct_qdrant_fallback():
    """DocumentIndexService is the only owner of chunk persistence and Qdrant writes."""
    source = inspect.getsource(DocumentProcessingService)
    forbidden = [
        "def _store_chunks(",
        "def _store_images(",
        "def _update_chunks_with_images(",
        "PointStruct",
        "set_payload(",
        ".upsert(",
    ]
    for token in forbidden:
        assert token not in source


def test_process_document_fails_fast_without_index_service(tmp_path):
    """A missing index service is configuration error, not permission to use legacy indexing."""
    service = _build_service(tmp_path)

    async def _fake_process_with_mineru(*_args, **_kwargs):
        return [{"text": "body", "page_start": 0, "page_end": 0}]

    service._process_with_mineru = _fake_process_with_mineru

    try:
        asyncio.run(
            service.process_document(
                file_path=str(tmp_path / "report.pdf"),
                filename="report.pdf",
                document_id=str(uuid4()),
                conversation_id=str(uuid4()),
                user_id=str(uuid4()),
            )
        )
    except RuntimeError as exc:
        assert "DocumentIndexService" in str(exc)
    else:
        raise AssertionError("process_document must fail when document_index_service is missing")


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
    indexed_chunks = service.document_index_service.index_document.call_args.kwargs["built_chunks"]
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

    document_ref = service.document_index_service.index_document.call_args.kwargs["document"]
    assert document_ref.filename == "Blue-whale-A4-fact-sheet.pdf"


def test_build_chunks_for_indexing_preserves_table_metadata(tmp_path):
    """Parsed table flags must survive the token-aware builder into SQL chunk metadata."""
    service = _build_service(tmp_path)

    built_chunks = service._build_chunks_for_indexing(
        [
            {
                "text": "| Metric | Value |\n|---|---|\n| Accuracy | 91% |",
                "page_start": 0,
                "page_end": 0,
                "has_tables": True,
                "table_count": 2,
            }
        ]
    )

    assert len(built_chunks) == 1
    assert built_chunks[0].metadata["has_tables"] is True
    assert built_chunks[0].metadata["table_count"] == 2


def test_prepared_images_attach_only_to_their_structural_owner(tmp_path):
    service = _build_service(tmp_path)
    blocks = [
        NormalizedBlock("p", "paragraph", "Page summary", page_start=0, page_end=0),
        NormalizedBlock(
            "i1",
            "image",
            "[Image]",
            page_start=0,
            page_end=0,
            metadata={"img_path": "images/a.png"},
        ),
        NormalizedBlock(
            "i2",
            "image",
            "[Image]",
            page_start=0,
            page_end=0,
            metadata={"img_path": "images/b.png"},
        ),
    ]
    prepared = [
        {
            "path": str(tmp_path / "a.png"),
            "relative_path": "images/a.png",
            "page_number": 0,
            "caption": "Alpha chart",
        },
        {
            "path": str(tmp_path / "b.png"),
            "relative_path": "images/b.png",
            "page_number": 0,
            "caption": "Beta chart",
        },
    ]

    updated = service._attach_prepared_images_to_blocks(blocks, prepared)

    assert updated[0].text == "Page summary"
    assert "Alpha chart" in updated[1].text
    assert "Beta chart" not in updated[1].text
    assert "Beta chart" in updated[2].text
    assert "Alpha chart" not in updated[2].text


def test_mineru_content_list_table_body_is_indexed_as_searchable_text(tmp_path):
    """MinerU table_body values must reach DocumentChunk.content for RAG search."""
    service = _build_service(tmp_path)
    parse_service = DocumentParseService(settings=service.settings)

    chunks_with_metadata = parse_service._create_chunks_with_page_metadata(
        [
            {"type": "text", "text": "Quarterly financial report", "page_idx": 0},
            {
                "type": "table",
                "page_idx": 0,
                "table_caption": ["Revenue by segment"],
                "table_body": (
                    "| Segment | Revenue |\n|---|---|\n| Cloud | 12345 |\n| Devices | 67890 |"
                ),
                "table_footnote": [],
            },
        ],
        page_to_images={},
    )

    built_chunks = service._build_chunks_for_indexing(chunks_with_metadata)
    indexed_text = "\n".join(chunk.content for chunk in built_chunks)

    assert "Revenue by segment" in indexed_text
    assert "Cloud" in indexed_text
    assert "12345" in indexed_text
    assert built_chunks[0].metadata["has_tables"] is True


def test_build_chunks_for_indexing_recovers_table_body_from_parse_metadata(tmp_path):
    """Existing parse artifacts may carry table bodies only in tables metadata."""
    service = _build_service(tmp_path)

    built_chunks = service._build_chunks_for_indexing(
        [
            {
                "text": "Quarterly financial report\n[Table: Revenue by segment]",
                "page_start": 0,
                "page_end": 0,
                "has_tables": True,
                "table_count": 1,
                "tables": [
                    {
                        "caption": ["Revenue by segment"],
                        "body": (
                            "| Segment | Revenue |\n"
                            "|---|---|\n"
                            "| Cloud | 12345 |\n"
                            "| Devices | 67890 |"
                        ),
                    }
                ],
            }
        ]
    )

    indexed_text = "\n".join(chunk.content for chunk in built_chunks)

    assert "Cloud" in indexed_text
    assert "12345" in indexed_text
    assert indexed_text.count("Cloud") == 1


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

    built_chunks = service.document_index_service.index_document.call_args.kwargs["built_chunks"]
    content = "\n".join(chunk.content for chunk in built_chunks)
    assert "Sheet: Sheet1" in content
    assert "Thu nhap" in content
    assert "1000" in content
    assert "=B2*(1-C2)" in content


def test_captioning_concurrency_bounded_by_semaphore(tmp_path):
    """All images are captioned even when the semaphore limit is smaller than the image count."""
    service = _build_service(tmp_path)
    # Set concurrency limit below number of images to exercise the semaphore gate.
    service.settings.image_caption_max_concurrency = 2

    # Create 5 distinct source images.
    num_images = 5
    images_data = []
    for i in range(num_images):
        img_path = tmp_path / f"img_{i}.png"
        Image.new("RGB", (4, 4), color=(i * 40, 0, 0)).save(img_path)
        images_data.append({"path": str(img_path), "page_number": i, "mime_type": "image/png"})

    call_count = 0

    async def _fake_caption(*, image_bytes, image_name):
        nonlocal call_count
        call_count += 1
        return f"caption for {image_name}"

    service.gemini_client = object()
    service._generate_image_caption_with_retry = _fake_caption

    results = asyncio.run(
        service._prepare_images_for_indexing(
            images_data=images_data,
            document_id="doc-semaphore-test",
        )
    )

    assert len(results) == num_images, f"Expected {num_images} results, got {len(results)}"
    assert call_count == num_images, f"Expected {num_images} caption calls, got {call_count}"
    for result in results:
        assert result["caption"].startswith("caption for")


def test_captioning_failure_degrades_to_metadata_caption(tmp_path):
    """A captioning exception must not drop the image — it falls back to metadata caption."""
    service = _build_service(tmp_path)
    service.settings.image_caption_max_concurrency = 4

    img_path = tmp_path / "chart.png"
    Image.new("RGB", (4, 4), color="blue").save(img_path)
    images_data = [
        {
            "path": str(img_path),
            "page_number": 1,
            "mime_type": "image/png",
            "caption": "metadata caption from pdf",
        }
    ]

    async def _failing_caption(*, image_bytes, image_name):
        raise RuntimeError("Simulated Gemini API failure")

    service.gemini_client = object()
    service._generate_image_caption_with_retry = _failing_caption

    results = asyncio.run(
        service._prepare_images_for_indexing(
            images_data=images_data,
            document_id="doc-failure-test",
        )
    )

    assert len(results) == 1, "Image must still be returned despite captioning failure"
    # The fallback caption comes from _caption_from_image_metadata; for images with a
    # "caption" key in the input dict that method returns it directly.
    assert results[0]["caption"] == "metadata caption from pdf"
    assert "stored_path" in results[0]
