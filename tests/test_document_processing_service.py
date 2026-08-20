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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from PIL import Image

from app.schemas.document_image import DocumentImageCreate
from app.services.document_blocks import NormalizedBlock
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
    service.recorder = None
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
    """Generated visual captions must be indexed; the image row persists ahead
    of the index call with a nullable chunk_id for DocumentIndexService to
    resolve (Task 11: persist-before-vector-write ordering)."""
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
    persisted_image_stub = SimpleNamespace(id=uuid4(), chunk_id=None)
    service._process_with_mineru = _fake_process_with_mineru
    service.gemini_client = object()
    service._generate_image_caption_with_retry = AsyncMock(
        return_value="A red bar chart showing revenue increasing each quarter."
    )
    service.document_image_repository.create.return_value = persisted_image_stub
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
    assert image_record.chunk_id is None, (
        "chunk_id must stay nullable at persist time; DocumentIndexService links it"
    )
    assert image_record.image_caption == "A red bar chart showing revenue increasing each quarter."
    assert image_record.content_sha256, "content hash must be computed for provenance"

    index_call = service.document_index_service.index_document.call_args
    assert list(index_call.kwargs["image_rows"]) == [persisted_image_stub]

    assert result["chunks_stored"] == 1
    assert result["images_stored"] == 1


def test_process_document_persists_images_before_calling_index_document(tmp_path):
    """Images must be persisted (nullable chunk_id) before the index service's
    vector writes, never after."""
    service = _build_service(tmp_path)

    source_image = tmp_path / "chart.png"
    Image.new("RGB", (4, 4), color="blue").save(source_image)
    image_data = {
        "path": str(source_image),
        "page_number": 0,
        "mime_type": "image/png",
    }
    chunks_with_metadata = [
        {
            "text": "Body text\n[Image]",
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

    call_order: list[str] = []
    service._process_with_mineru = _fake_process_with_mineru
    service.document_image_repository.create.side_effect = lambda _data: (
        call_order.append("create_image") or SimpleNamespace(id=uuid4(), chunk_id=None)
    )
    service.document_index_service = MagicMock()

    def _record_index_document(**_kwargs):
        call_order.append("index_document")
        return []

    service.document_index_service.index_document.side_effect = _record_index_document

    asyncio.run(
        service.process_document(
            file_path=str(tmp_path / "report.pdf"),
            filename="report.pdf",
            document_id=str(uuid4()),
            conversation_id=str(uuid4()),
            user_id=str(uuid4()),
        )
    )

    assert call_order == ["create_image", "index_document"]


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
    """MinerU table_body values must reach DocumentChunk.content for RAG search.

    Exercises the current production boundary end to end: MinerU
    content_list.json-shaped entries go through DocumentNormalizer (the
    only supported normalizer for MinerU output), and the resulting
    NormalizedBlocks go through the token-aware DocumentChunkBuilder (via
    _build_chunks_for_indexing) — not the retired character-count chunker.
    """
    from app.services.document_normalizer import DocumentNormalizer

    service = _build_service(tmp_path)

    blocks = DocumentNormalizer().normalize_mineru(
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
        images_data=[],
    )

    built_chunks = service._build_chunks_for_indexing(blocks)
    indexed_text = "\n".join(chunk.content for chunk in built_chunks)

    assert "Revenue by segment" in indexed_text
    assert "Cloud" in indexed_text
    assert "12345" in indexed_text
    table_chunk = next(c for c in built_chunks if "Revenue by segment" in c.content)
    assert table_chunk.metadata["has_tables"] is True


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


# ---------------------------------------------------------------------------
# Task 11: structured captions and image provenance (bbox/section_path)
# ---------------------------------------------------------------------------


def test_request_image_caption_renders_structured_sections_from_parsed_response(tmp_path):
    """A validated structured caption is flattened into the searchable text."""
    from app.schemas.document_image import ImageCaptionSections

    service = _build_service(tmp_path)
    service.gemini_client = MagicMock()
    service.gemini_client.models.generate_content.return_value = SimpleNamespace(
        parsed=ImageCaptionSections(
            chart_title="Quarterly Revenue",
            axes="X: Quarter, Y: Revenue (USD)",
            values="Q1 $10M, Q2 $12M",
            trends="Revenue increased each quarter",
        ),
        text="{}",
    )

    caption = service._request_image_caption(b"fake-jpeg-bytes", "chart.png")

    assert caption == (
        "Title: Quarterly Revenue\n"
        "Axes: X: Quarter, Y: Revenue (USD)\n"
        "Values: Q1 $10M, Q2 $12M\n"
        "Trends: Revenue increased each quarter"
    )
    call_kwargs = service.gemini_client.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].response_mime_type == "application/json"


def test_request_image_caption_falls_back_to_raw_text_for_non_json_response(tmp_path):
    """Non-JSON model output is preserved as-is rather than discarded."""
    service = _build_service(tmp_path)
    service.gemini_client = MagicMock()
    service.gemini_client.models.generate_content.return_value = SimpleNamespace(
        parsed=None,
        text="a plain-language description, not JSON",
    )

    caption = service._request_image_caption(b"fake-jpeg-bytes", "chart.png")

    assert caption == "a plain-language description, not JSON"


def test_request_image_caption_returns_none_when_response_has_no_text(tmp_path):
    service = _build_service(tmp_path)
    service.gemini_client = MagicMock()
    service.gemini_client.models.generate_content.return_value = SimpleNamespace(
        parsed=None, text=""
    )

    assert service._request_image_caption(b"fake-jpeg-bytes", "chart.png") is None


def test_attach_prepared_images_to_blocks_stamps_bbox_and_section_path_from_owning_block(tmp_path):
    service = _build_service(tmp_path)
    owning_block = NormalizedBlock(
        "i1",
        "image",
        "[Image]",
        page_start=0,
        page_end=0,
        section_path=("Chapter 1", "Figures"),
        metadata={"img_path": "images/a.png", "bbox": [1, 2, 3, 4]},
    )
    prepared = [
        {
            "path": "/tmp/a.png",
            "relative_path": "images/a.png",
            "page_number": 0,
            "caption": "Alpha chart",
        }
    ]

    service._attach_prepared_images_to_blocks([owning_block], prepared)

    assert prepared[0]["bbox"] == [1, 2, 3, 4]
    assert prepared[0]["section_path"] == ["Chapter 1", "Figures"]


def test_persist_prepared_images_writes_bbox_section_path_and_content_hash(tmp_path):
    service = _build_service(tmp_path)
    prepared = [
        {
            "stored_path": "document_images/doc/chart.png",
            "caption": "A chart",
            "content_sha256": "deadbeef" * 8,
            "page_number": 0,
            "mime_type": "image/png",
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "section_path": ["Results"],
        }
    ]

    service._persist_prepared_images(prepared, str(uuid4()))

    create_call = service.document_image_repository.create.call_args
    image_record = create_call.args[0]
    assert image_record.chunk_id is None
    assert image_record.bbox == [0.1, 0.2, 0.3, 0.4]
    assert image_record.section_path == ["Results"]
    assert image_record.content_sha256 == "deadbeef" * 8


@pytest.mark.parametrize(
    "malformed_bbox",
    [
        {"x": 1},
        "a,b",
        [[1, 2], [3, 4]],
        [1, 2, 3],
        [1, 2, 3, 4, 5],
    ],
)
def test_persist_prepared_images_drops_malformed_bbox_instead_of_failing(tmp_path, malformed_bbox):
    """Item 2: unexpected parser bbox shapes must not raise ``ValidationError``.

    ``ValidationError`` subclasses ``ValueError`` and
    ``document_processor.py`` classifies ``ValueError`` as non-retryable, so
    an unvalidated bbox from MinerU's untrusted output would permanently
    fail the whole document. A malformed bbox should just become ``None``.
    """
    service = _build_service(tmp_path)
    prepared = [
        {
            "stored_path": "document_images/doc/chart.png",
            "caption": "A chart",
            "content_sha256": "deadbeef" * 8,
            "page_number": 0,
            "mime_type": "image/png",
            "bbox": malformed_bbox,
            "section_path": ["Results"],
        }
    ]

    service._persist_prepared_images(prepared, str(uuid4()))

    create_call = service.document_image_repository.create.call_args
    image_record = create_call.args[0]
    assert image_record.bbox is None


class _FakeImageRepo:
    """Stateful stand-in tracking rows across repeated persist calls, so a
    retry-accumulation regression is actually observable (a plain MagicMock
    would happily "create" without ever reflecting prior state).

    ``created_at`` increments per row so ``keep_created_at_on_or_before``
    (item 3) can be exercised deterministically, mirroring the real
    repository's per-insert timestamp ordering.
    """

    def __init__(self):
        self.rows: dict[UUID, SimpleNamespace] = {}
        self._next_created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def delete_unlinked_by_document_id(
        self,
        document_id: UUID,
        *,
        keep_created_at_on_or_before: datetime | None = None,
    ) -> int:
        orphans = [
            row_id
            for row_id, row in self.rows.items()
            if row.document_id == document_id
            and row.chunk_id is None
            and (
                keep_created_at_on_or_before is None
                or row.created_at > keep_created_at_on_or_before
            )
        ]
        for row_id in orphans:
            del self.rows[row_id]
        return len(orphans)

    def create(self, image_data):
        row = SimpleNamespace(
            id=uuid4(),
            document_id=image_data.document_id,
            chunk_id=image_data.chunk_id,
            image_path=image_data.image_path,
            created_at=self._next_created_at,
        )
        self._next_created_at += timedelta(seconds=1)
        self.rows[row.id] = row
        return row


def test_active_generation_created_at_reads_from_generation_repository(tmp_path):
    """Item 3: the orphan-delete narrowing needs the active generation's own
    ``created_at`` to distinguish it from a later abandoned attempt."""
    service = _build_service(tmp_path)
    generation_repo = MagicMock()
    active_created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    generation_repo.get_active.return_value = SimpleNamespace(created_at=active_created_at)
    service.document_index_service = SimpleNamespace(generation_repository=generation_repo)
    document_id = str(uuid4())

    result = service._active_generation_created_at(document_id)

    assert result == active_created_at
    generation_repo.get_active.assert_called_once_with(UUID(document_id))


def test_active_generation_created_at_is_none_without_index_service(tmp_path):
    """No wired index service (or no active generation yet) must fall back
    to ``None``, matching the original pre-item-3 delete-everything-orphaned
    behavior for a document's first attempt."""
    service = _build_service(tmp_path)  # document_index_service is None

    assert service._active_generation_created_at(str(uuid4())) is None


def test_persist_prepared_images_clears_orphans_from_failed_attempts(tmp_path):
    """Round 2 finding B: two failed attempts followed by a success must
    leave exactly one row set. Persistence now happens before
    index_document links chunk_id, so a Celery retry that re-enters this
    whole path with no delete or dedup would otherwise leave every failed
    attempt's rows behind — chunk_id stays NULL forever, but VIEW_IMAGES
    lists by document, so a user would see N duplicate copies."""
    service = _build_service(tmp_path)
    service.document_image_repository = _FakeImageRepo()
    document_id = str(uuid4())
    prepared = [
        {
            "stored_path": "document_images/doc/chart.png",
            "caption": "A chart",
            "content_sha256": "a" * 64,
            "page_number": 0,
            "mime_type": "image/png",
        }
    ]

    service._persist_prepared_images(prepared, document_id)  # attempt 1: "fails" after this
    service._persist_prepared_images(prepared, document_id)  # attempt 2: "fails" after this
    final = service._persist_prepared_images(prepared, document_id)  # attempt 3: succeeds

    remaining = [
        row
        for row in service.document_image_repository.rows.values()
        if row.document_id == UUID(document_id)
    ]
    assert len(remaining) == 1, f"expected exactly one row set, got {len(remaining)}"
    assert final and final[0].id == remaining[0].id
