"""Phase 3 pipeline tests — parse → index Celery chain.

Tests exercise the two Celery tasks in eager mode (no broker required):
  * parse_document_task  — parses a document, persists an artifact, returns artifact_id str
  * index_document_task  — loads artifact, embeds, upserts Qdrant, marks document READY

All external I/O (DB, Qdrant, filesystem beyond tmp_path) is mocked.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from PIL import Image

from app.core.events import DocumentEvent
from app.schemas.document import DocumentStatus
from app.services.document_parse_service import DocumentParseService, ParseResult
from app.services.document_processing_service import DocumentProcessingService
from app.workers.celery_app import celery_app
from app.workers.document_processor import _cleanup_parse_artifacts

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def celery_eager(monkeypatch):
    """Run all Celery tasks synchronously in-process — no broker needed."""
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield
    celery_app.conf.task_always_eager = False
    celery_app.conf.task_eager_propagates = False


def _fake_document(document_id: str | None = None, conversation_id=None):
    return SimpleNamespace(
        id=UUID(document_id or str(uuid4())),
        conversation_id=conversation_id or uuid4(),
        filename="test.txt",
        status=DocumentStatus.PROCESSING.value,
    )


def _fake_artifact(document_id: str, artifact_id=None, storage_path: str | None = None):
    aid = artifact_id or uuid4()
    return SimpleNamespace(
        id=aid,
        document_id=UUID(document_id),
        storage_path=storage_path or f"/tmp/test_artifacts/{document_id}/normalized_chunks.json",
    )


def _minimal_parse_result(n_chunks: int = 2, n_images: int = 0) -> ParseResult:
    return ParseResult(
        chunks_with_metadata=[
            {"text": f"chunk {i}", "page_start": i, "page_end": i} for i in range(n_chunks)
        ],
        images_data=[{"file_path": f"/tmp/img_{i}.png", "page": i} for i in range(n_images)],
        parse_elapsed_s=0.5,
        backend_used="pipeline",
    )


def _make_mock_session_local(monkeypatch, owner_id: UUID | None = None):
    """Patch SessionLocal to return a mock context-manager session.

    The session's query(Conversation) chain returns a fake conversation whose
    ``owner_id`` is either None (default) or the given UUID, so that
    ``UUID(owner_id)`` in index_document_task does not blow up.
    """
    fake_conversation = SimpleNamespace(owner_id=owner_id)
    mock_query = MagicMock()
    mock_query.filter.return_value.one_or_none.return_value = fake_conversation

    mock_session = MagicMock()
    mock_session.query.return_value = mock_query

    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_session)
    mock_cm.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr("app.workers.document_processor.SessionLocal", lambda: mock_cm)
    return mock_session, mock_cm


def _make_mock_doc_repo(monkeypatch, document: SimpleNamespace):
    """Patch DocumentRepository so get_by_id returns the fake document."""
    mock_repo = MagicMock()
    mock_repo.get_by_id.return_value = document
    mock_repo.update.return_value = None
    monkeypatch.setattr(
        "app.workers.document_processor.DocumentRepository",
        lambda _session_factory: mock_repo,
    )
    return mock_repo


def _make_mock_event_bus(monkeypatch):
    mock_bus = MagicMock()
    mock_bus.emit = AsyncMock()
    monkeypatch.setattr("app.workers.document_processor.get_event_bus", lambda: mock_bus)
    return mock_bus


# ---------------------------------------------------------------------------
# Test 1 — Artifact write/read roundtrip (real disk I/O)
# ---------------------------------------------------------------------------


def test_artifact_roundtrip_preserves_all_fields(tmp_path):
    """persist_parse_result writes JSON; load_parse_result reads it back faithfully."""
    document_id = str(uuid4())
    artifact_id = uuid4()

    # The service writes to settings.parse_artifacts_storage_path / document_id /
    settings = MagicMock()
    settings.parse_artifacts_storage_path = str(tmp_path)
    settings.rag_chunk_target_tokens = 400
    settings.rag_chunk_overlap_tokens = 40
    settings.rag_chunk_max_tokens = 800

    # Mock artifact repo: replace_for_document returns a real-looking artifact
    # with the actual storage path computed by the service.
    tmp_path / document_id / "normalized_chunks.json"

    def _fake_replace_for_document(*, document_id: UUID, artifacts: list):
        row = artifacts[0]
        return [
            SimpleNamespace(
                id=artifact_id,
                document_id=document_id,
                storage_path=row["storage_path"],
                checksum_sha256=row.get("checksum_sha256", ""),
                size_bytes=row.get("size_bytes", 0),
            )
        ]

    mock_artifact_repo = MagicMock()
    mock_artifact_repo.replace_for_document.side_effect = _fake_replace_for_document

    parse_service = DocumentParseService(
        settings=settings,
        artifact_repo=mock_artifact_repo,
    )

    parse_result = ParseResult(
        chunks_with_metadata=[
            {"text": f"chunk {i}", "page_start": i, "page_end": i + 1} for i in range(3)
        ],
        images_data=[{"file_path": f"/tmp/img_{i}.png", "page": i} for i in range(2)],
        parse_elapsed_s=1.23,
        backend_used="pipeline",
    )

    artifact = parse_service.persist_parse_result(document_id, parse_result)

    # File must exist at the expected path
    expected_path = tmp_path / document_id / "normalized_chunks.json"
    assert expected_path.exists(), "JSON artifact file was not created"

    # Artifact metadata
    assert artifact.checksum_sha256, "sha256 should be non-empty"
    assert artifact.size_bytes > 0, "size_bytes should be > 0"

    # Roundtrip
    loaded = parse_service.load_parse_result(artifact)
    assert len(loaded.chunks_with_metadata) == 3
    assert len(loaded.images_data) == 2
    assert loaded.parse_elapsed_s == pytest.approx(1.23, abs=1e-6)
    assert loaded.backend_used == "pipeline"


def test_mineru_images_survive_parse_cleanup_and_feed_index_preparation(tmp_path, monkeypatch):
    """The parse artifact must not retain image paths under MinerU temp storage."""
    document_id = str(uuid4())
    artifact_id = uuid4()
    temp_root = tmp_path / "temp"
    mineru_images = temp_root / f"mineru_output_{document_id}" / "report" / "images"
    mineru_images.mkdir(parents=True)
    extracted_image = mineru_images / "chart.png"
    Image.new("RGB", (4, 4), color="blue").save(extracted_image)
    staged_upload = temp_root / "staged-report.pdf"
    staged_upload.write_bytes(b"pdf")

    settings = MagicMock()
    settings.temp_storage_path = str(temp_root)
    settings.parse_artifacts_storage_path = str(tmp_path / "parse-artifacts")
    settings.document_images_storage_path = str(tmp_path / "document-images")
    settings.image_caption_max_concurrency = 1
    settings.rag_chunk_target_tokens = 400
    settings.rag_chunk_overlap_tokens = 40
    settings.rag_chunk_max_tokens = 800

    def _replace_artifact(*, document_id: UUID, artifacts: list):
        row = artifacts[0]
        return [
            SimpleNamespace(
                id=artifact_id,
                document_id=document_id,
                storage_path=row["storage_path"],
                checksum_sha256=row["checksum_sha256"],
                size_bytes=row["size_bytes"],
            )
        ]

    artifact_repo = MagicMock()
    artifact_repo.replace_for_document.side_effect = _replace_artifact
    parse_service = DocumentParseService(settings=settings, artifact_repo=artifact_repo)
    artifact = parse_service.persist_parse_result(
        document_id,
        ParseResult(
            chunks_with_metadata=[
                {
                    "text": "Chart summary",
                    "page_start": 0,
                    "page_end": 0,
                    "has_images": True,
                    "image_count": 1,
                }
            ],
            images_data=[
                {
                    "path": str(extracted_image),
                    "page_number": 0,
                    "mime_type": "image/png",
                }
            ],
            parse_elapsed_s=0.25,
            backend_used="mineru/pipeline",
        ),
    )

    loaded_before_cleanup = parse_service.load_parse_result(artifact)
    durable_image = Path(loaded_before_cleanup.images_data[0]["path"])
    assert durable_image.is_file()
    assert durable_image.is_relative_to(
        Path(settings.parse_artifacts_storage_path) / document_id / "images"
    )

    monkeypatch.setattr("app.workers.document_processor.get_settings", lambda: settings)
    _cleanup_parse_artifacts(str(staged_upload), document_id)

    assert not staged_upload.exists()
    assert not extracted_image.exists()
    assert durable_image.is_file()

    loaded_after_cleanup = parse_service.load_parse_result(artifact)
    processing_service = object.__new__(DocumentProcessingService)
    processing_service.settings = settings
    processing_service.gemini_client = None
    prepared = asyncio.run(
        processing_service._prepare_images_for_indexing(
            loaded_after_cleanup.images_data,
            document_id,
        )
    )

    assert len(prepared) == 1
    assert Path(prepared[0]["stored_path"]).is_file()


def test_parse_cleanup_refuses_staged_file_outside_temp_storage(tmp_path, monkeypatch):
    settings = MagicMock()
    settings.temp_storage_path = str(tmp_path / "temp")
    outside_file = tmp_path / "outside-upload.pdf"
    outside_file.write_bytes(b"keep")

    monkeypatch.setattr("app.workers.document_processor.get_settings", lambda: settings)
    _cleanup_parse_artifacts(str(outside_file), str(uuid4()))

    assert outside_file.is_file()


# ---------------------------------------------------------------------------
# Test 2a — parse_document_task returns artifact_id as str
# ---------------------------------------------------------------------------


def test_parse_task_returns_artifact_id_string(tmp_path, monkeypatch):
    """parse_document_task.apply() must return str(artifact_id)."""
    document_id = str(uuid4())
    artifact_id = uuid4()

    # Write a real temp file for the task to validate
    temp_file = tmp_path / "upload.txt"
    temp_file.write_text("Hello world", encoding="utf-8")

    fake_doc = _fake_document(document_id)
    _make_mock_doc_repo(monkeypatch, fake_doc)
    _make_mock_session_local(monkeypatch)
    _make_mock_event_bus(monkeypatch)

    fake_artifact = _fake_artifact(document_id, artifact_id=artifact_id)
    fake_parse_result = _minimal_parse_result(n_chunks=2)

    mock_container = MagicMock()
    mock_container.document_parse_artifact_repository.return_value = MagicMock()
    mock_container.document_chunk_builder.return_value = MagicMock()
    monkeypatch.setattr("app.workers.document_processor.get_container", lambda: mock_container)

    with patch("app.services.document_parse_service.DocumentParseService") as MockParseService:
        instance = MockParseService.return_value
        instance.parse_document = AsyncMock(return_value=fake_parse_result)
        instance.persist_parse_result.return_value = fake_artifact

        result = celery_app.tasks["app.workers.document_processor.parse_document_task"].apply(
            args=[document_id, str(temp_file), "upload.txt"]
        )

    assert not result.failed(), f"Task failed unexpectedly: {result.result}"
    retval = result.get()
    assert isinstance(retval, str), f"Return value must be str, got {type(retval)}"
    assert retval == str(artifact_id)


# ---------------------------------------------------------------------------
# Test 2b — index_document_task accepts artifact_id and marks READY
# ---------------------------------------------------------------------------


def test_index_task_accepts_artifact_id_from_parse(tmp_path, monkeypatch):
    """index_document_task marks the document READY given a valid artifact_id."""
    document_id = str(uuid4())
    artifact_id = uuid4()

    # Write a real artifact JSON so load_parse_result can read it
    artifact_dir = tmp_path / document_id
    artifact_dir.mkdir()
    artifact_file = artifact_dir / "normalized_chunks.json"
    payload = {
        "chunks_with_metadata": [{"text": "a"}, {"text": "b"}],
        "images_data": [],
        "parse_elapsed_s": 0.1,
        "backend_used": "text",
    }
    artifact_file.write_text(json.dumps(payload), encoding="utf-8")

    fake_artifact = _fake_artifact(
        document_id, artifact_id=artifact_id, storage_path=str(artifact_file)
    )
    fake_doc = _fake_document(document_id)

    _make_mock_session_local(monkeypatch)
    mock_doc_repo = _make_mock_doc_repo(monkeypatch, fake_doc)
    _make_mock_event_bus(monkeypatch)

    mock_artifact_repo = MagicMock()
    mock_artifact_repo.get_by_id.return_value = fake_artifact

    mock_container = MagicMock()
    mock_container.document_parse_artifact_repository.return_value = mock_artifact_repo
    mock_container.document_chunk_builder.return_value = MagicMock()

    # Mock processing service
    mock_proc_service = MagicMock()
    mock_proc_service._build_chunks_for_indexing.return_value = [MagicMock(), MagicMock()]
    mock_proc_service.document_index_service.index_document.return_value = [
        MagicMock(),
        MagicMock(),
    ]
    mock_container.document_processing_service.return_value = mock_proc_service

    monkeypatch.setattr("app.workers.document_processor.get_container", lambda: mock_container)

    with patch("app.services.document_parse_service.DocumentParseService") as MockParseService:
        instance = MockParseService.return_value
        instance.load_parse_result.return_value = ParseResult(
            chunks_with_metadata=[{"text": "a"}, {"text": "b"}],
            images_data=[],
            parse_elapsed_s=0.1,
            backend_used="text",
        )

        result = celery_app.tasks["app.workers.document_processor.index_document_task"].apply(
            args=[str(artifact_id)]
        )

    assert not result.failed(), f"Task failed: {result.result}"
    retval = result.get()
    assert retval.get("success") is True

    # Verify _mark_document was called with READY
    update_calls = mock_doc_repo.update.call_args_list
    statuses_set = [call.args[1].status for call in update_calls if call.args]
    assert DocumentStatus.READY.value in statuses_set, (
        f"Expected READY status in update calls, got: {statuses_set}"
    )


class _FakeCaptionMetrics:
    """Deterministic recorder matching ``RAGMetrics.stage``'s signature."""

    def __init__(self) -> None:
        self.stage_calls: list[tuple[str, float, dict]] = []

    def stage(self, stage, *, elapsed_seconds, labels=None):
        self.stage_calls.append((stage, elapsed_seconds, dict(labels or {})))

    def stage_failure(self, stage, failure_code):
        pass


def test_index_task_records_the_caption_stage_metric(tmp_path, monkeypatch):
    """Round-2 fix (finding 3): the caption stage call sits inside a
    blanket try/except Exception -- a wrong stage name, a wrong label, or a
    raising call would all pass silently without a dedicated test.
    """
    document_id = str(uuid4())
    artifact_id = uuid4()

    fake_artifact = _fake_artifact(document_id, artifact_id=artifact_id)
    fake_doc = _fake_document(document_id)

    _make_mock_session_local(monkeypatch)
    _make_mock_doc_repo(monkeypatch, fake_doc)
    _make_mock_event_bus(monkeypatch)

    mock_artifact_repo = MagicMock()
    mock_artifact_repo.get_by_id.return_value = fake_artifact

    mock_container = MagicMock()
    mock_container.document_parse_artifact_repository.return_value = mock_artifact_repo
    mock_container.document_chunk_builder.return_value = MagicMock()

    mock_proc_service = MagicMock()
    mock_proc_service._build_chunks_for_indexing.return_value = [MagicMock(), MagicMock()]
    mock_proc_service.document_index_service.index_document.return_value = [
        MagicMock(),
        MagicMock(),
    ]
    mock_container.document_processing_service.return_value = mock_proc_service

    monkeypatch.setattr("app.workers.document_processor.get_container", lambda: mock_container)

    fake_metrics = _FakeCaptionMetrics()
    monkeypatch.setattr("app.workers.document_processor.rag_metrics", fake_metrics)

    with patch("app.services.document_parse_service.DocumentParseService") as MockParseService:
        instance = MockParseService.return_value
        instance.load_parse_result.return_value = ParseResult(
            chunks_with_metadata=[{"text": "a"}, {"text": "b"}],
            images_data=[],
            parse_elapsed_s=0.1,
            backend_used="text",
        )

        result = celery_app.tasks["app.workers.document_processor.index_document_task"].apply(
            args=[str(artifact_id)]
        )

    assert not result.failed(), f"Task failed: {result.result}"

    caption_calls = [call for call in fake_metrics.stage_calls if call[0] == "caption"]
    assert len(caption_calls) == 1
    _, elapsed, labels = caption_calls[0]
    assert elapsed >= 0.0
    assert labels["modality"] == "image"


# ---------------------------------------------------------------------------
# Test 2c — Task 11 review finding 1 (Critical): the production worker must
# persist images before index_document's vector writes and pass image_rows=
# ---------------------------------------------------------------------------


def test_index_task_persists_images_before_calling_index_document(tmp_path, monkeypatch):
    """DocumentProcessingService.process_document has zero non-test callers —
    index_document_task is the only path that actually ingests documents in
    production, so it must persist DocumentImage rows (nullable chunk_id)
    before index_document's vector writes and pass image_rows= through,
    mirroring the ordering already pinned for process_document."""
    document_id = str(uuid4())
    artifact_id = uuid4()

    artifact_dir = tmp_path / document_id
    artifact_dir.mkdir()
    artifact_file = artifact_dir / "normalized_chunks.json"
    payload = {
        "chunks_with_metadata": [{"text": "a"}],
        "images_data": [{"path": "/tmp/chart.png", "page_number": 0, "mime_type": "image/png"}],
        "parse_elapsed_s": 0.1,
        "backend_used": "text",
    }
    artifact_file.write_text(json.dumps(payload), encoding="utf-8")

    fake_artifact = _fake_artifact(
        document_id, artifact_id=artifact_id, storage_path=str(artifact_file)
    )
    fake_doc = _fake_document(document_id)

    _make_mock_session_local(monkeypatch)
    _make_mock_doc_repo(monkeypatch, fake_doc)
    _make_mock_event_bus(monkeypatch)

    mock_artifact_repo = MagicMock()
    mock_artifact_repo.get_by_id.return_value = fake_artifact

    mock_container = MagicMock()
    mock_container.document_parse_artifact_repository.return_value = mock_artifact_repo
    mock_container.document_chunk_builder.return_value = MagicMock()

    call_order: list[str] = []
    persisted_image_stub = SimpleNamespace(id=uuid4(), chunk_id=None)

    mock_proc_service = MagicMock()
    mock_proc_service._build_chunks_for_indexing.return_value = [MagicMock()]
    mock_proc_service._prepare_images_for_indexing = AsyncMock(
        return_value=[
            {"stored_path": "images/chart.png", "page_number": 0, "mime_type": "image/png"}
        ]
    )
    mock_proc_service._attach_prepared_images_to_blocks.side_effect = (
        lambda blocks, _images: blocks
    )

    def _record_persist(_prepared_images, _document_id):
        call_order.append("persist_images")
        return [persisted_image_stub]

    mock_proc_service._persist_prepared_images.side_effect = _record_persist

    def _record_index_document(**kwargs):
        call_order.append("index_document")
        assert list(kwargs["image_rows"]) == [persisted_image_stub], (
            "index_document must receive the persisted image rows"
        )
        return [MagicMock()]

    mock_proc_service.document_index_service.index_document.side_effect = _record_index_document
    mock_container.document_processing_service.return_value = mock_proc_service

    monkeypatch.setattr("app.workers.document_processor.get_container", lambda: mock_container)

    with patch("app.services.document_parse_service.DocumentParseService") as MockParseService:
        instance = MockParseService.return_value
        instance.load_parse_result.return_value = ParseResult(
            chunks_with_metadata=[{"text": "a"}],
            images_data=[{"path": "/tmp/chart.png", "page_number": 0, "mime_type": "image/png"}],
            parse_elapsed_s=0.1,
            backend_used="text",
        )

        result = celery_app.tasks["app.workers.document_processor.index_document_task"].apply(
            args=[str(artifact_id)]
        )

    assert not result.failed(), f"Task failed: {result.result}"
    retval = result.get()
    assert retval.get("success") is True
    assert retval.get("images_stored") == 1

    assert call_order == ["persist_images", "index_document"], (
        "images must be persisted before index_document is called"
    )


# ---------------------------------------------------------------------------
# Test 3 — Index retry never calls parse_document
# ---------------------------------------------------------------------------


def test_index_retry_does_not_call_parse_document(tmp_path, monkeypatch):
    """index_document_task must never call parse_document (it loads from artifact)."""
    document_id = str(uuid4())
    artifact_id = uuid4()

    artifact_dir = tmp_path / document_id
    artifact_dir.mkdir()
    artifact_file = artifact_dir / "normalized_chunks.json"
    payload = {
        "chunks_with_metadata": [{"text": "x"}],
        "images_data": [],
        "parse_elapsed_s": 0.1,
        "backend_used": "text",
    }
    artifact_file.write_text(json.dumps(payload), encoding="utf-8")

    fake_artifact = _fake_artifact(
        document_id, artifact_id=artifact_id, storage_path=str(artifact_file)
    )
    fake_doc = _fake_document(document_id)

    _make_mock_session_local(monkeypatch)
    _make_mock_doc_repo(monkeypatch, fake_doc)
    _make_mock_event_bus(monkeypatch)

    mock_artifact_repo = MagicMock()
    mock_artifact_repo.get_by_id.return_value = fake_artifact

    mock_container = MagicMock()
    mock_container.document_parse_artifact_repository.return_value = mock_artifact_repo
    mock_container.document_chunk_builder.return_value = MagicMock()

    mock_proc_service = MagicMock()
    mock_proc_service._build_chunks_for_indexing.return_value = [MagicMock()]
    mock_proc_service.document_index_service.index_document.return_value = [MagicMock()]
    mock_container.document_processing_service.return_value = mock_proc_service

    monkeypatch.setattr("app.workers.document_processor.get_container", lambda: mock_container)

    parse_document_spy = AsyncMock(side_effect=AssertionError("parse_document called!"))

    with patch("app.services.document_parse_service.DocumentParseService") as MockParseService:
        instance = MockParseService.return_value
        # Spy — if ever called, the test will explode
        instance.parse_document = parse_document_spy
        instance.load_parse_result.return_value = ParseResult(
            chunks_with_metadata=[{"text": "x"}],
            images_data=[],
            parse_elapsed_s=0.1,
            backend_used="text",
        )

        result = celery_app.tasks["app.workers.document_processor.index_document_task"].apply(
            args=[str(artifact_id)]
        )

    # parse_document must never have been called
    assert parse_document_spy.call_count == 0, (
        "index_document_task called parse_document — it must not"
    )
    assert not result.failed(), f"Task unexpectedly failed: {result.result}"


# ---------------------------------------------------------------------------
# Test 4a — parse_document_task marks FAILED on terminal error
# ---------------------------------------------------------------------------


def test_parse_task_marks_document_failed_on_value_error(tmp_path, monkeypatch):
    """A non-retryable error (FileNotFoundError) must set document status to FAILED."""
    document_id = str(uuid4())
    fake_doc = _fake_document(document_id)

    _make_mock_session_local(monkeypatch)
    mock_doc_repo = _make_mock_doc_repo(monkeypatch, fake_doc)
    _make_mock_event_bus(monkeypatch)

    mock_container = MagicMock()
    mock_container.document_parse_artifact_repository.return_value = MagicMock()
    mock_container.document_chunk_builder.return_value = MagicMock()
    monkeypatch.setattr("app.workers.document_processor.get_container", lambda: mock_container)

    # Use a nonexistent file — triggers FileNotFoundError inside the task
    nonexistent = str(tmp_path / "nonexistent.txt")

    # task_eager_propagates=True means Celery re-raises the exception directly
    task_raised = False
    try:
        celery_app.tasks["app.workers.document_processor.parse_document_task"].apply(
            args=[document_id, nonexistent, "nonexistent.txt"]
        )
    except Exception:
        task_raised = True

    assert task_raised, "Task should have raised an exception for nonexistent file"

    # Verify FAILED status was set via DocumentRepository.update
    update_calls = mock_doc_repo.update.call_args_list
    assert update_calls, "DocumentRepository.update was never called"
    statuses_set = [call.args[1].status for call in update_calls if call.args]
    assert DocumentStatus.FAILED.value in statuses_set, (
        f"Expected FAILED in update calls, got statuses: {statuses_set}"
    )


# ---------------------------------------------------------------------------
# Test 4b — index_document_task raises when artifact not found
# ---------------------------------------------------------------------------


def test_index_task_marks_document_failed_when_artifact_not_found(monkeypatch):
    """index_document_task raises ValueError when artifact_id is unknown."""
    _make_mock_session_local(monkeypatch)
    _make_mock_event_bus(monkeypatch)

    # doc_repo.get_by_id is not expected to be called (document_id unknown at that point)
    mock_doc_repo = MagicMock()
    monkeypatch.setattr(
        "app.workers.document_processor.DocumentRepository",
        lambda _: mock_doc_repo,
    )

    mock_artifact_repo = MagicMock()
    mock_artifact_repo.get_by_id.return_value = None  # artifact not found

    mock_container = MagicMock()
    mock_container.document_parse_artifact_repository.return_value = mock_artifact_repo
    monkeypatch.setattr("app.workers.document_processor.get_container", lambda: mock_container)

    phantom_id = str(uuid4())
    task_raised = False
    raised_exc = None
    try:
        celery_app.tasks["app.workers.document_processor.index_document_task"].apply(
            args=[phantom_id]
        )
    except Exception as exc:
        task_raised = True
        raised_exc = exc

    assert task_raised, "Task should have raised when artifact is not found"
    assert isinstance(raised_exc, ValueError)


# ---------------------------------------------------------------------------
# Test 4c — parse_document_task emits PROCESSING_FAILED event on terminal error
# ---------------------------------------------------------------------------


def test_parse_task_emits_failed_event_on_terminal_error(tmp_path, monkeypatch):
    """Terminal parse failure must emit DocumentEvent.PROCESSING_FAILED."""
    document_id = str(uuid4())
    fake_doc = _fake_document(document_id)

    _make_mock_session_local(monkeypatch)
    _make_mock_doc_repo(monkeypatch, fake_doc)
    mock_bus = _make_mock_event_bus(monkeypatch)

    mock_container = MagicMock()
    mock_container.document_parse_artifact_repository.return_value = MagicMock()
    mock_container.document_chunk_builder.return_value = MagicMock()
    monkeypatch.setattr("app.workers.document_processor.get_container", lambda: mock_container)

    nonexistent = str(tmp_path / "nope.txt")

    with contextlib.suppress(Exception):
        celery_app.tasks["app.workers.document_processor.parse_document_task"].apply(
            args=[document_id, nonexistent, "nope.txt"]
        )

    # _emit_failed calls get_event_bus().emit(DocumentEvent.PROCESSING_FAILED, ...)
    # It runs in a _run_async() block so the AsyncMock is awaited.
    assert mock_bus.emit.called, "event_bus.emit was never called"
    first_call_args = mock_bus.emit.call_args_list[0]
    emitted_event = first_call_args.args[0]
    assert emitted_event == DocumentEvent.PROCESSING_FAILED, (
        f"Expected PROCESSING_FAILED, got: {emitted_event}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Small files complete independently of a large file
# ---------------------------------------------------------------------------


def test_small_documents_complete_independently_of_large(tmp_path, monkeypatch):
    """Each document is processed with its own artifact_id — no shared state."""
    doc_ids = [str(uuid4()) for _ in range(3)]  # small docs
    large_doc_id = str(uuid4())
    all_doc_ids = doc_ids + [large_doc_id]

    # Track READY calls per document
    ready_docs: set[str] = set()

    def _make_per_doc_context(doc_id: str):
        """Return a tuple of (parse_result, artifact) for one document."""
        artifact_id = uuid4()
        artifact_dir = tmp_path / doc_id
        artifact_dir.mkdir(exist_ok=True)
        artifact_file = artifact_dir / "normalized_chunks.json"
        payload = {
            "chunks_with_metadata": [{"text": "body"}],
            "images_data": [],
            "parse_elapsed_s": 0.1,
            "backend_used": "text",
        }
        artifact_file.write_text(json.dumps(payload), encoding="utf-8")

        artifact = _fake_artifact(doc_id, artifact_id=artifact_id, storage_path=str(artifact_file))
        parse_result = _minimal_parse_result(n_chunks=1)
        return artifact, parse_result

    _make_mock_event_bus(monkeypatch)

    for doc_id in all_doc_ids:
        artifact, parse_result = _make_per_doc_context(doc_id)
        fake_doc = _fake_document(doc_id)

        # Fresh session mock per iteration — conversation query must return owner_id=None
        fake_conversation = SimpleNamespace(owner_id=None)
        mock_query = MagicMock()
        mock_query.filter.return_value.one_or_none.return_value = fake_conversation
        mock_session = MagicMock()
        mock_session.query.return_value = mock_query
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_session)
        mock_cm.__exit__ = MagicMock(return_value=False)

        mock_doc_repo = MagicMock()
        mock_doc_repo.get_by_id.return_value = fake_doc
        mock_doc_repo.update.return_value = None

        mock_artifact_repo = MagicMock()
        mock_artifact_repo.get_by_id.return_value = artifact

        mock_container = MagicMock()
        mock_container.document_parse_artifact_repository.return_value = mock_artifact_repo
        mock_container.document_chunk_builder.return_value = MagicMock()

        mock_proc_service = MagicMock()
        mock_proc_service._build_chunks_for_indexing.return_value = [MagicMock()]
        mock_proc_service.document_index_service.index_document.return_value = [MagicMock()]
        mock_container.document_processing_service.return_value = mock_proc_service

        # Write a real temp file for parse step
        temp_file = tmp_path / f"{doc_id}_upload.txt"
        temp_file.write_text("content", encoding="utf-8")

        with (
            patch(
                "app.workers.document_processor.SessionLocal",
                lambda mock_cm=mock_cm: mock_cm,
            ),
            patch(
                "app.workers.document_processor.DocumentRepository",
                lambda _, mock_doc_repo=mock_doc_repo: mock_doc_repo,
            ),
            patch(
                "app.workers.document_processor.get_container",
                lambda mock_container=mock_container: mock_container,
            ),
            patch("app.services.document_parse_service.DocumentParseService") as MockParseService,
        ):
            svc_instance = MockParseService.return_value
            svc_instance.parse_document = AsyncMock(return_value=parse_result)
            svc_instance.persist_parse_result.return_value = artifact
            svc_instance.load_parse_result.return_value = parse_result

            # Parse stage
            parse_result_task = celery_app.tasks[
                "app.workers.document_processor.parse_document_task"
            ].apply(args=[doc_id, str(temp_file), f"{doc_id}.txt"])
            assert not parse_result_task.failed(), (
                f"parse failed for {doc_id}: {parse_result_task.result}"
            )
            returned_artifact_id = parse_result_task.get()
            assert returned_artifact_id == str(artifact.id)

            # Index stage
            index_result = celery_app.tasks[
                "app.workers.document_processor.index_document_task"
            ].apply(args=[returned_artifact_id])
            assert not index_result.failed(), f"index failed for {doc_id}: {index_result.result}"

            # Confirm READY was set
            update_calls = mock_doc_repo.update.call_args_list
            statuses = [c.args[1].status for c in update_calls if c.args]
            assert DocumentStatus.READY.value in statuses, (
                f"Document {doc_id} was not marked READY; got: {statuses}"
            )
            ready_docs.add(doc_id)

    assert len(ready_docs) == 4, (
        f"Expected all 4 documents to reach READY; only {len(ready_docs)} did: {ready_docs}"
    )
