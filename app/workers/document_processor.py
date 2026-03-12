import asyncio
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from celery import Task

from app.core.config import get_settings
from app.core.container import get_container
from app.core.events import DocumentEvent, DocumentEventData, get_event_bus
from app.database.session import SessionLocal
from app.models.document import Document
from app.repositories.document import DocumentRepository
from app.schemas.document import DocumentStatus, DocumentUpdate
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {task_id} completed successfully")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")


@celery_app.task(
    bind=True,
    base=CallbackTask,
    max_retries=3,
    default_retry_delay=60,
    name="app.workers.document_processor.process_document_task",
)
def process_document_task(
    self, document_id: str, temp_file_path: str, filename: str
) -> dict[str, Any]:
    """Process a previously staged document file.

    Args:
        document_id: UUID string of the Document record.
        temp_file_path: Filesystem path where the upload was staged.
            The API layer writes the file before enqueuing so that raw
            bytes are not serialised through the Celery broker.
        filename: Original filename (used for loader selection).
    """
    task_id = self.request.id
    cleanup_staged_file = True

    document_repo = DocumentRepository(SessionLocal)

    try:
        # Get document record to retrieve conversation_id
        document = document_repo.get_by_id(UUID(document_id))
        if not document:
            raise ValueError(f"Document {document_id} not found")

        update_data = DocumentUpdate(status=DocumentStatus.PROCESSING.value)
        document_repo.update(UUID(document_id), update_data)

        settings = get_settings()
        if not os.path.isfile(temp_file_path):
            raise FileNotFoundError(f"Staged upload file not found: {temp_file_path}")

        file_size = os.path.getsize(temp_file_path)
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValueError(
                f"File size exceeds maximum allowed size of {settings.max_file_size_mb}MB"
            )

        # Get DocumentProcessingService from container
        container = get_container()
        processing_service = container.document_processing_service()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        processing_result = loop.run_until_complete(
            processing_service.process_document(
                file_path=temp_file_path,
                filename=filename,
                document_id=str(document_id),
                conversation_id=str(document.conversation_id),
            )
        )

        update_data = DocumentUpdate(status=DocumentStatus.READY.value)
        document = document_repo.update(UUID(document_id), update_data)

        event_bus = get_event_bus()
        loop.run_until_complete(
            event_bus.emit(
                DocumentEvent.PROCESSING_COMPLETED,
                DocumentEventData(
                    document_id=UUID(document_id),
                    conversation_id=(document.conversation_id if document else None),
                    filename=filename,
                    status="READY",
                    metadata={
                        "chunks_created": processing_result.get("chunks_created", 0),
                        "chunks_stored": processing_result.get("chunks_stored", 0),
                        "processing_time": processing_result.get("processing_time", 0),
                        "task_id": task_id,
                    },
                ),
            )
        )

        return {
            "success": True,
            "document_id": document_id,
            "chunks_created": processing_result.get("chunks_created", 0),
            "chunks_stored": processing_result.get("chunks_stored", 0),
            "processing_time": processing_result.get("processing_time", 0),
            "message": f"Document '{filename}' processed successfully",
        }

    except Exception as exc:
        logger.error(
            f"Error processing document {document_id} ('{filename}'): {exc}",
            exc_info=True,
        )
        logger.error(f"Document {document_id} processing failed: {str(exc)}", exc_info=True)

        try:
            update_data = DocumentUpdate(status=DocumentStatus.FAILED.value)
            document_repo.update(UUID(document_id), update_data)
        except Exception as db_exc:
            logger.error(f"Failed to update document status: {db_exc}")

        # Emit PROCESSING_FAILED event
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    get_event_bus().emit(
                        DocumentEvent.PROCESSING_FAILED,
                        DocumentEventData(
                            document_id=UUID(document_id),
                            filename=filename,
                            status="FAILED",
                            error=str(exc),
                            metadata={"task_id": task_id},
                        ),
                    )
                )
            finally:
                asyncio.set_event_loop(None)
                loop.close()
        except Exception:
            pass

        retryable = self.request.retries < self.max_retries and not isinstance(
            exc, (FileNotFoundError, ValueError)
        )
        if retryable:
            cleanup_staged_file = False
            retry_delay = min(300, 60 * (2**self.request.retries))
            logger.info(
                f"Retrying task {task_id} for document {document_id} "
                f"(attempt {self.request.retries + 1}/{self.max_retries}) in {retry_delay}s"
            )
            raise self.retry(exc=exc, countdown=retry_delay)

        logger.error(
            f"Document {document_id} ('{filename}') failed after {self.max_retries} attempts"
        )
        return {
            "success": False,
            "document_id": document_id,
            "error": str(exc),
            "message": f"Failed to process document '{filename}' after {self.max_retries} attempts",
        }

    finally:
        # Clean up the staged temp file after processing (success or failure).
        try:
            if cleanup_staged_file and os.path.isfile(temp_file_path):
                os.unlink(temp_file_path)
        except Exception:
            pass

        try:
            settings_ = get_settings()
            mineru_output_path = Path(settings_.temp_storage_path) / f"mineru_output_{document_id}"
            if mineru_output_path.exists():
                shutil.rmtree(mineru_output_path)
        except Exception:
            pass

        try:
            if "loop" in locals() and loop is not None and not loop.is_closed():
                asyncio.set_event_loop(None)
                loop.close()
        except Exception:
            pass


@celery_app.task(name="app.workers.document_processor.cleanup_failed_documents")
def cleanup_failed_documents() -> dict[str, Any]:
    db = SessionLocal()
    document_repo = DocumentRepository(SessionLocal)

    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)

        stuck_documents = (
            db.query(Document)
            .filter(
                Document.status == DocumentStatus.PROCESSING.value,
                Document.upload_time < cutoff_time,
            )
            .all()
        )

        failed_count = 0
        for doc in stuck_documents:
            update_data = DocumentUpdate(status=DocumentStatus.FAILED.value)
            document_repo.update(doc.id, update_data)
            failed_count += 1
            logger.warning(f"Marked document {doc.id} as failed due to timeout")

        return {
            "success": True,
            "failed_documents": failed_count,
            "message": f"Cleaned up {failed_count} stuck documents",
        }

    except Exception as exc:
        logger.error(f"Error in cleanup task: {exc}")
        return {"success": False, "error": str(exc), "message": "Cleanup task failed"}

    finally:
        db.close()
