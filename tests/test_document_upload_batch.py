"""Contract tests for the canonical batch document upload helper.

Covers ``_upload_documents_batch`` and the small status-code derivation
helpers in ``app.api.documents``. These tests intentionally avoid the
FastAPI auto-injection harness to stay fast and focused on the batch
control flow.

Acceptance criteria mirrored from rag_overhaul_2.md::Task 1.2:

* batch upload accepts multiple files and returns ordered per-file
  accepted results
* an existing same-conversation filename returns DUPLICATE_FILENAME for
  that file and still enqueues non-duplicate siblings
* duplicate names inside the same incoming batch accept the first
  candidate and reject later duplicates
* all-duplicate batches resolve to HTTP 409
* mixed accepted/rejected batches resolve to HTTP 207
* accepted-only batches resolve to HTTP 201
* all-validation-failed batches resolve to HTTP 400
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4


class _UploadFileStub:
    def __init__(self, filename: str, content: bytes, content_type: str = "text/plain"):
        self.filename = filename
        self._content = content
        self.content_type = content_type
        self._read = False

    async def read(self, size: int = -1) -> bytes:
        if self._read:
            return b""
        self._read = True
        return self._content

    async def seek(self, offset: int) -> None:
        self._read = False


def _build_dependencies(
    *,
    existing_filename_keys: set[str] | None = None,
):
    """Build minimal mocks for the batch upload helper."""
    existing_filename_keys = set(existing_filename_keys or [])
    seen_persisted: list[str] = []

    document_service = MagicMock()

    async def _validate_and_create(filename, file_size, content_type, conversation_id):
        from app.core.exceptions.validation import DuplicateDocumentFilenameError
        from app.schemas.document import DocumentResponse, DocumentStatus
        from app.services.document_service import normalize_document_filename

        key = normalize_document_filename(filename)
        if key in existing_filename_keys or key in seen_persisted:
            raise DuplicateDocumentFilenameError(
                detail=f"A document named '{filename}' already exists in this conversation."
            )
        seen_persisted.append(key)
        return DocumentResponse(
            id=uuid4(),
            conversation_id=conversation_id,
            filename=filename,
            file_type=content_type,
            status=DocumentStatus.PROCESSING.value,
            upload_time=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )

    async def _set_processing_task_id(document_id, task_id):
        return None

    document_service.validate_and_create_document = AsyncMock(side_effect=_validate_and_create)
    document_service.set_processing_task_id = AsyncMock(side_effect=_set_processing_task_id)
    document_service.repository = MagicMock()
    document_service.repository.session_factory = MagicMock()

    processing_service = MagicMock()
    staged_counter = {"i": 0}

    async def _stage(file, filename):
        staged_counter["i"] += 1
        return {
            "temp_file_path": f"/tmp/staged-{staged_counter['i']}",
            "file_size": len(file._content),
            "file_info": {"valid": True},
        }

    async def _start_task(document_id, temp_file_path, filename, file_size):
        return {
            "task_id": f"task-{document_id}",
            "success": True,
            "document_id": document_id,
        }

    processing_service.stage_upload_file = AsyncMock(side_effect=_stage)
    processing_service.start_processing_task = AsyncMock(side_effect=_start_task)

    return document_service, processing_service


def test_normalize_document_filename_casefolds_and_strips_paths():
    from app.services.document_service import normalize_document_filename

    assert normalize_document_filename("Report.PDF") == "report.pdf"
    assert normalize_document_filename("  Spaced.txt  ") == "spaced.txt"
    assert normalize_document_filename("C:/Users/Tox/notes.md") == "notes.md"
    assert normalize_document_filename("relative/dir/file.txt") == "file.txt"


def test_batch_upload_all_accepted_returns_201():
    from app.api.documents import _status_code_for_batch_result, _upload_documents_batch

    conv = uuid4()
    user = uuid4()
    document_service, processing_service = _build_dependencies()

    files = [
        _UploadFileStub("alpha.pdf", b"a", "application/pdf"),
        _UploadFileStub("beta.pdf", b"b", "application/pdf"),
    ]

    result = asyncio.run(
        _upload_documents_batch(
            document_service=document_service,
            document_processing_service=processing_service,
            current_user_id=user,
            files=files,
            conversation_id=conv,
        )
    )

    assert result.total_count == 2
    assert result.accepted_count == 2
    assert result.rejected_count == 0
    assert [item.filename for item in result.files] == ["alpha.pdf", "beta.pdf"]
    assert all(item.status == "accepted" for item in result.files)
    assert _status_code_for_batch_result(result) == 201


def test_batch_upload_rejects_existing_duplicate_and_accepts_sibling():
    from app.api.documents import _status_code_for_batch_result, _upload_documents_batch

    conv = uuid4()
    user = uuid4()
    document_service, processing_service = _build_dependencies(
        existing_filename_keys={"existing.pdf"}
    )

    files = [
        _UploadFileStub("existing.pdf", b"old", "application/pdf"),
        _UploadFileStub("new.pdf", b"new", "application/pdf"),
    ]

    result = asyncio.run(
        _upload_documents_batch(
            document_service=document_service,
            document_processing_service=processing_service,
            current_user_id=user,
            files=files,
            conversation_id=conv,
        )
    )

    assert result.accepted_count == 1
    assert result.rejected_count == 1
    assert result.files[0].status == "rejected"
    assert result.files[0].error_code == "DUPLICATE_FILENAME"
    assert result.files[1].status == "accepted"
    assert _status_code_for_batch_result(result) == 207


def test_batch_upload_rejects_later_in_batch_duplicates_of_same_filename():
    from app.api.documents import _status_code_for_batch_result, _upload_documents_batch

    conv = uuid4()
    user = uuid4()
    document_service, processing_service = _build_dependencies()

    files = [
        _UploadFileStub("alpha.pdf", b"a1", "application/pdf"),
        _UploadFileStub("Alpha.PDF", b"a2", "application/pdf"),  # casefold dupe
    ]

    result = asyncio.run(
        _upload_documents_batch(
            document_service=document_service,
            document_processing_service=processing_service,
            current_user_id=user,
            files=files,
            conversation_id=conv,
        )
    )

    assert result.accepted_count == 1
    assert result.rejected_count == 1
    assert result.files[0].status == "accepted"
    assert result.files[1].status == "rejected"
    assert result.files[1].error_code == "DUPLICATE_FILENAME"
    assert _status_code_for_batch_result(result) == 207


def test_batch_upload_all_duplicates_returns_409():
    from app.api.documents import _status_code_for_batch_result, _upload_documents_batch

    conv = uuid4()
    user = uuid4()
    document_service, processing_service = _build_dependencies(
        existing_filename_keys={"a.pdf", "b.pdf"}
    )

    files = [
        _UploadFileStub("a.pdf", b"1", "application/pdf"),
        _UploadFileStub("b.pdf", b"2", "application/pdf"),
    ]

    result = asyncio.run(
        _upload_documents_batch(
            document_service=document_service,
            document_processing_service=processing_service,
            current_user_id=user,
            files=files,
            conversation_id=conv,
        )
    )

    assert result.accepted_count == 0
    assert result.rejected_count == 2
    assert all(item.error_code == "DUPLICATE_FILENAME" for item in result.files)
    assert _status_code_for_batch_result(result) == 409


def test_batch_upload_all_validation_failures_returns_400():
    """When everything fails for non-duplicate reasons, status code is 400."""
    from app.api.documents import _status_code_for_batch_result, _upload_documents_batch
    from app.core.exceptions.validation import FileValidationError

    conv = uuid4()
    user = uuid4()
    document_service, processing_service = _build_dependencies()

    # Force a non-duplicate validation failure from staging.
    async def _stage_fail(file, filename):
        raise FileValidationError(detail="bad extension")

    processing_service.stage_upload_file = AsyncMock(side_effect=_stage_fail)

    files = [
        _UploadFileStub("a.weird", b"1", "application/octet-stream"),
        _UploadFileStub("b.weird", b"2", "application/octet-stream"),
    ]

    result = asyncio.run(
        _upload_documents_batch(
            document_service=document_service,
            document_processing_service=processing_service,
            current_user_id=user,
            files=files,
            conversation_id=conv,
        )
    )

    assert result.accepted_count == 0
    assert result.rejected_count == 2
    assert all(item.status == "rejected" for item in result.files)
    assert all(item.error_code != "DUPLICATE_FILENAME" for item in result.files)
    assert _status_code_for_batch_result(result) == 400


def test_batch_upload_preserves_input_order():
    from app.api.documents import _upload_documents_batch

    conv = uuid4()
    user = uuid4()
    document_service, processing_service = _build_dependencies()

    filenames = ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
    files = [_UploadFileStub(name, name.encode(), "application/pdf") for name in filenames]

    result = asyncio.run(
        _upload_documents_batch(
            document_service=document_service,
            document_processing_service=processing_service,
            current_user_id=user,
            files=files,
            conversation_id=conv,
        )
    )

    assert [item.filename for item in result.files] == filenames
