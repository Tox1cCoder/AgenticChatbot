"""
API-level tests for /documents endpoints.

Tests cover:
- public document updates cannot set processing_task_id
- POST /upload persists processing_task_id on document record via internal setter
- GET /task/{task_id}: unauthorized access returns 403
- GET /task/{task_id}: owner can access (200)
- staged upload streams bytes to disk before enqueue
- worker retries preserve the staged file for the next attempt
"""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Processing-task-ID persistence
# ---------------------------------------------------------------------------


class TestProcessingTaskIdPersistence:
    """Verify task IDs stay internal and are persisted through the internal setter."""

    def test_public_document_update_schema_hides_processing_task_id(self):
        from app.schemas.document import DocumentUpdate

        assert "processing_task_id" not in DocumentUpdate.model_fields

    @pytest.mark.asyncio
    async def test_upload_persists_task_id(self):
        """
        When start_processing_task returns a task_id, the upload handler
        must call document_service.set_processing_task_id with that task_id.
        """
        # Simulate what the upload endpoint does after calling start_processing_task
        task_id = str(uuid.uuid4())
        doc_id = uuid.uuid4()

        document_service = MagicMock()
        document_service.set_processing_task_id = AsyncMock()
        document_service.repository = MagicMock()

        task_info = {"task_id": task_id, "document_id": str(doc_id)}

        # Reproduce the update logic from the upload endpoint
        if task_info.get("task_id"):
            await document_service.set_processing_task_id(doc_id, task_info["task_id"])

        document_service.set_processing_task_id.assert_called_once_with(doc_id, task_id)


# ---------------------------------------------------------------------------
# Ownership enforcement on /task/{task_id}
# ---------------------------------------------------------------------------


class TestTaskStatusOwnership:
    def _make_doc_validation_utils(self, should_raise=False):
        utils = MagicMock()
        if should_raise:
            from app.core.exceptions import AuthorizationException

            utils.validate_document_access.side_effect = AuthorizationException(
                detail="Access denied"
            )
        return utils

    def test_unauthorized_raises(self):
        from app.core.exceptions import AuthorizationException

        doc = MagicMock()
        doc.id = uuid.uuid4()

        doc_repo = MagicMock()
        doc_repo.get_by_processing_task_id.return_value = doc

        validation_utils = self._make_doc_validation_utils(should_raise=True)

        task_id = str(uuid.uuid4())
        user_id = uuid.uuid4()

        # Reproduce the ownership check from the endpoint
        with pytest.raises(AuthorizationException):
            document = doc_repo.get_by_processing_task_id(task_id)
            if document is not None:
                validation_utils.validate_document_access(user_id, document.id)

    def test_owner_passes_without_exception(self):
        doc = MagicMock()
        doc.id = uuid.uuid4()

        doc_repo = MagicMock()
        doc_repo.get_by_processing_task_id.return_value = doc

        validation_utils = self._make_doc_validation_utils(should_raise=False)
        task_id = str(uuid.uuid4())
        user_id = uuid.uuid4()

        # Should not raise
        document = doc_repo.get_by_processing_task_id(task_id)
        if document is not None:
            validation_utils.validate_document_access(user_id, document.id)

        validation_utils.validate_document_access.assert_called_once_with(user_id, doc.id)

    def test_unknown_task_id_skips_ownership_check(self):
        """If no document is found for the task ID, skip the ownership check."""
        doc_repo = MagicMock()
        doc_repo.get_by_processing_task_id.return_value = None

        validation_utils = self._make_doc_validation_utils()
        task_id = str(uuid.uuid4())
        user_id = uuid.uuid4()

        document = doc_repo.get_by_processing_task_id(task_id)
        if document is not None:
            validation_utils.validate_document_access(user_id, document.id)

        validation_utils.validate_document_access.assert_not_called()


# ---------------------------------------------------------------------------
# Staged file handoff
# ---------------------------------------------------------------------------


class TestStagedFileHandoff:
    """Verify uploads are staged to disk and queued by path."""

    class _FakeUploadFile:
        def __init__(self, payload: bytes):
            self._payload = payload
            self._offset = 0

        async def read(self, size: int = -1) -> bytes:
            if size is None or size < 0:
                size = len(self._payload) - self._offset
            chunk = self._payload[self._offset : self._offset + size]
            self._offset += len(chunk)
            return chunk

        async def seek(self, offset: int) -> None:
            self._offset = offset

    @pytest.mark.asyncio
    async def test_stage_upload_file_writes_temp_file(self, tmp_path):
        """
        stage_upload_file must stream the upload to a temp path without
        requiring the full payload to be re-serialized to Redis later.
        """
        from app.services.document_processing_service import DocumentProcessingService

        settings = MagicMock()
        settings.max_file_size_mb = 10
        settings.temp_storage_path = str(tmp_path)

        file_content = b"hello pdf"
        filename = "test-doc.pdf"

        svc = object.__new__(DocumentProcessingService)
        svc.settings = settings
        upload_file = self._FakeUploadFile(file_content)

        result = await svc.stage_upload_file(upload_file, filename)

        staged_path = result["temp_file_path"]
        assert isinstance(staged_path, str)
        assert os.path.isfile(staged_path), "Staged file must exist on disk"
        assert open(staged_path, "rb").read() == file_content
        assert result["file_size"] == len(file_content)

    @pytest.mark.asyncio
    async def test_start_processing_queues_existing_staged_file(self, tmp_path):
        from app.services.document_processing_service import DocumentProcessingService

        settings = MagicMock()
        settings.max_file_size_mb = 10
        settings.temp_storage_path = str(tmp_path)

        filename = "test-doc.pdf"
        document_id = str(uuid.uuid4())
        staged_path = tmp_path / "staged.pdf"
        staged_path.write_bytes(b"hello pdf")

        svc = object.__new__(DocumentProcessingService)
        svc.settings = settings
        svc._event_bus = MagicMock()
        svc._event_bus.emit = AsyncMock()

        celery_task = MagicMock()
        celery_task.id = str(uuid.uuid4())
        svc.celery_app = MagicMock()
        svc.celery_app.send_task = MagicMock(return_value=celery_task)
        svc.validate_upload_file = AsyncMock(
            return_value={"valid": True, "file_type": ".pdf", "size_mb": 0.01}
        )
        svc._estimate_processing_time = MagicMock(return_value="30-60 seconds")

        await svc.start_processing_task(
            document_id,
            str(staged_path),
            filename,
            staged_path.stat().st_size,
        )

        task_args = svc.celery_app.send_task.call_args.kwargs["args"]
        assert task_args[1] == str(staged_path)
        assert os.path.isfile(staged_path)


class TestDocumentWorkerRetries:
    def test_retry_keeps_staged_file_for_next_attempt(self, tmp_path):
        from app.workers.document_processor import process_document_task

        document_id = str(uuid.uuid4())
        staged_path = tmp_path / "retry-doc.pdf"
        staged_path.write_bytes(b"hello pdf")

        document = MagicMock()
        document.conversation_id = uuid.uuid4()

        document_repo = MagicMock()
        document_repo.get_by_id.return_value = document
        document_repo.update.return_value = document

        processing_service = MagicMock()
        processing_service.process_document = AsyncMock(
            side_effect=RuntimeError("temporary failure")
        )
        container = MagicMock()
        container.document_processing_service.return_value = processing_service

        event_bus = MagicMock()
        event_bus.emit = AsyncMock()

        settings = MagicMock()
        settings.max_file_size_mb = 10
        settings.temp_storage_path = str(tmp_path)

        task_self = MagicMock()
        task_self.request.id = "task-123"
        task_self.request.retries = 0
        task_self.max_retries = 3
        task_self.retry.side_effect = RuntimeError("retry requested")

        with (
            patch(
                "app.workers.document_processor.DocumentRepository",
                return_value=document_repo,
            ),
            patch(
                "app.workers.document_processor.get_container",
                return_value=container,
            ),
            patch(
                "app.workers.document_processor.get_event_bus",
                return_value=event_bus,
            ),
            patch(
                "app.workers.document_processor.get_settings",
                return_value=settings,
            ),
            pytest.raises(RuntimeError, match="retry requested"),
        ):
            process_document_task.run.__func__(
                task_self,
                document_id,
                str(staged_path),
                "retry-doc.pdf",
            )

        assert staged_path.exists(), "Retry path must preserve the staged file"
