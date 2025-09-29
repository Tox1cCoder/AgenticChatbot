"""
Document Processing Tasks

This module contains Celery tasks for processing documents in the background.
"""

import os
import tempfile
import traceback
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
from uuid import UUID

from celery import Task
from sqlalchemy.orm import Session

from app.workers.celery_app import celery_app
from app.database.session import get_db
from app.repositories.document import DocumentRepository
from app.schemas.document import DocumentUpdate, DocumentStatus
from app.ai.agents.rag_agent import RAGAgent
from app.models.document import Document

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Base task class with callback support for status updates."""

    def on_success(self, retval, task_id, args, kwargs):
        """Called upon successful task completion."""
        logger.info(f"Task {task_id} completed successfully")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called upon task failure."""
        logger.error(f"Task {task_id} failed: {exc}")


@celery_app.task(
    bind=True,
    base=CallbackTask,
    max_retries=3,
    default_retry_delay=60,
    name="app.workers.document_processor.process_document_task",
)
def process_document_task(
    self, document_id: str, file_content: bytes, filename: str
) -> Dict[str, Any]:
    """
    Process uploaded document: extract text, create chunks, and embed in Qdrant.

    Args:
        document_id: UUID of the document record
        file_content: Binary content of the uploaded file
        filename: Original filename

    Returns:
        Dict with processing results
    """
    task_id = self.request.id
    logger.info(
        f"Starting document processing task {task_id} for document {document_id}"
    )

    db: Session = next(get_db())
    document_repo = DocumentRepository(db)

    try:
        # Get document record to retrieve conversation_id
        document = document_repo.get_by_id(UUID(document_id))
        if not document:
            raise ValueError(f"Document {document_id} not found")

        # Update document status to processing
        update_data = DocumentUpdate(status=DocumentStatus.PROCESSING.value)
        document_repo.update(UUID(document_id), update_data)

        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(filename)[1]
        ) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            # Initialize RAG agent for document processing
            rag_agent = RAGAgent()
            # Use sync initialization since we're in a sync Celery task
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                loop.run_until_complete(rag_agent.initialize())

                # Process document
                processing_result = loop.run_until_complete(
                    rag_agent.process_document(
                        file_path=temp_file_path,
                        filename=filename,
                        document_id=str(document_id),
                        conversation_id=str(document.conversation_id),
                    )
                )
            finally:
                loop.close()

            # Update document status to ready
            update_data = DocumentUpdate(status=DocumentStatus.READY.value)
            document = document_repo.update(UUID(document_id), update_data)

            logger.info(
                f"Document {document_id} processed successfully: {processing_result}"
            )

            return {
                "success": True,
                "document_id": document_id,
                "chunks_created": processing_result.get("chunks_created", 0),
                "chunks_stored": processing_result.get("chunks_stored", 0),
                "processing_time": processing_result.get("processing_time", 0),
                "message": f"Document '{filename}' processed successfully",
            }

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

            # Clean up RAG agent
            if "rag_agent" in locals():
                # Use sync cleanup since we're in a sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(rag_agent.cleanup())
                finally:
                    loop.close()

    except Exception as exc:
        logger.error(f"Error processing document {document_id}: {exc}")
        logger.error(traceback.format_exc())

        try:
            # Update document status to failed
            update_data = DocumentUpdate(status=DocumentStatus.FAILED.value)
            document_repo.update(UUID(document_id), update_data)
        except Exception as db_exc:
            logger.error(f"Failed to update document status: {db_exc}")

        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(
                f"Retrying task {task_id} (attempt {self.request.retries + 1}/{self.max_retries})"
            )
            raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))

        return {
            "success": False,
            "document_id": document_id,
            "error": str(exc),
            "message": f"Failed to process document '{filename}'",
        }

    finally:
        db.close()


@celery_app.task(name="app.workers.document_processor.cleanup_failed_documents")
def cleanup_failed_documents() -> Dict[str, Any]:
    """
    Cleanup task to handle documents stuck in processing state.

    This task should be run periodically to find documents that have been
    in 'processing' state for too long and mark them as 'failed'.
    """
    db: Session = next(get_db())
    document_repo = DocumentRepository(db)

    try:
        # Find documents that have been processing for more than 1 hour
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
