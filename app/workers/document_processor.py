import asyncio
import contextlib
import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

from celery import Task

from app.core.config import get_settings
from app.core.container import get_container
from app.core.events import DocumentEvent, DocumentEventData, get_event_bus
from app.database.session import SessionLocal
from app.models.conversation import Conversation
from app.models.document import Document
from app.repositories.document import DocumentRepository
from app.schemas.document import DocumentStatus, DocumentUpdate
from app.usage import bind_usage_context
from app.usage.types import UsageContext
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

_settings = get_settings()

# ---------------------------------------------------------------------------
# Base task class
# ---------------------------------------------------------------------------


class CallbackTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {task_id} completed successfully")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine in a fresh event loop and close it afterward."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _mark_document(document_repo: DocumentRepository, document_id: str, status: DocumentStatus):
    try:
        update_data = DocumentUpdate(status=status.value)
        document_repo.update(UUID(document_id), update_data)
    except Exception as exc:
        logger.error(f"Failed to update document {document_id} status to {status}: {exc}")


def _emit_failed(document_id: str, filename: str, task_id: str, exc: Exception):
    try:
        _run_async(
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
    except Exception:
        logger.warning(
            "Failed to emit PROCESSING_FAILED event for document %s", document_id, exc_info=True
        )


def _cleanup_parse_artifacts(temp_file_path: str, document_id: str):
    """Delete staged files only when they resolve inside configured temp storage."""
    settings_ = get_settings()
    temp_root = Path(settings_.temp_storage_path)
    if not temp_root.is_absolute():
        temp_root = Path.cwd() / temp_root
    temp_root = temp_root.resolve()

    def _contained(candidate: Path) -> Path | None:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(temp_root)
            return resolved
        except (OSError, ValueError):
            return None

    try:
        staged_file = _contained(Path(temp_file_path))
        if staged_file is None:
            logger.warning("Refusing to clean staged file outside temp storage: %s", temp_file_path)
        elif staged_file.is_file():
            staged_file.unlink()
    except Exception:
        logger.warning("Cleanup error for %s", temp_file_path, exc_info=True)

    try:
        mineru_candidate = temp_root / f"mineru_output_{document_id}"
        mineru_output_path = _contained(mineru_candidate)
        if mineru_output_path is None:
            logger.warning(
                "Refusing to clean MinerU output outside temp storage: %s", mineru_candidate
            )
        elif mineru_output_path.is_dir():
            shutil.rmtree(mineru_output_path)
    except Exception:
        logger.warning(
            "Cleanup error for MinerU output dir for document %s", document_id, exc_info=True
        )


# ---------------------------------------------------------------------------
# parse_document_task
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    base=CallbackTask,
    max_retries=3,
    name="app.workers.document_processor.parse_document_task",
    queue="parse",
    time_limit=_settings.mineru_timeout + 60,
    soft_time_limit=_settings.mineru_timeout + 30,
)
def parse_document_task(self, document_id: str, temp_file_path: str, filename: str) -> str:
    """Parse document, persist artifact, return artifact_id.

    Returns:
        artifact_id (str) — passed as the first positional arg to index_document_task
        by Celery chain mechanics.
    """
    task_id = self.request.id
    cleanup_staged_file = True

    document_repo = DocumentRepository(SessionLocal)

    try:
        # Validate staged file
        if not os.path.isfile(temp_file_path):
            raise FileNotFoundError(f"Staged upload file not found: {temp_file_path}")

        settings = get_settings()
        file_size = os.path.getsize(temp_file_path)
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValueError(
                f"File size exceeds maximum allowed size of {settings.max_file_size_mb}MB"
            )

        # Verify document exists
        document = document_repo.get_by_id(UUID(document_id))
        if not document:
            raise ValueError(f"Document {document_id} not found")

        # Mark document PROCESSING (only if not already)
        if document.status != DocumentStatus.PROCESSING.value:
            _mark_document(document_repo, document_id, DocumentStatus.PROCESSING)

        # Get services from container
        container = get_container()
        artifact_repo = container.document_parse_artifact_repository()

        # Construct DocumentParseService with artifact_repo injected
        from app.services.document_parse_service import DocumentParseService

        chunk_builder = container.document_chunk_builder()
        parse_service = DocumentParseService(
            settings=settings,
            chunk_builder=chunk_builder,
            artifact_repo=artifact_repo,
        )

        # Parse document (async)
        result = _run_async(
            parse_service.parse_document(
                file_path=temp_file_path,
                filename=filename,
                document_id=document_id,
            )
        )

        # Persist parse result to disk + DB
        artifact = parse_service.persist_parse_result(document_id, result)

        logger.info(
            "parse_document_task: document %s parsed and artifact %s persisted "
            "(backend=%s, chunks=%d, images=%d, elapsed=%.1fs)",
            document_id,
            artifact.id,
            result.backend_used,
            len(result.chunks_with_metadata),
            len(result.images_data),
            result.parse_elapsed_s,
        )

        # T014: stage completion log for pipeline observability
        logger.info(
            "Document %s parse stage complete: artifact_id=%s, backend=%s, parse_elapsed_s=%.2f",
            document_id,
            str(artifact.id),
            result.backend_used,
            result.parse_elapsed_s,
        )

        return str(artifact.id)

    except Exception as exc:
        logger.error(
            f"Error in parse_document_task for document {document_id} ('{filename}'): {exc}",
            exc_info=True,
        )

        retryable = self.request.retries < self.max_retries and not isinstance(
            exc, (FileNotFoundError, ValueError)
        )
        if retryable:
            cleanup_staged_file = False
            retry_delay = min(300, 60 * (2**self.request.retries))
            logger.info(
                f"Retrying parse_document_task {task_id} for document {document_id} "
                f"(attempt {self.request.retries + 1}/{self.max_retries}) in {retry_delay}s"
            )
            raise self.retry(exc=exc, countdown=retry_delay) from exc

        # Terminal failure — only mark FAILED and emit event when not retrying
        _mark_document(document_repo, document_id, DocumentStatus.FAILED)
        _emit_failed(document_id, filename, task_id, exc)
        logger.error(
            f"Document {document_id} ('{filename}') parse failed after {self.max_retries} attempts"
        )
        raise

    finally:
        if cleanup_staged_file:
            _cleanup_parse_artifacts(temp_file_path, document_id)


# ---------------------------------------------------------------------------
# index_document_task
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    base=CallbackTask,
    max_retries=3,
    name="app.workers.document_processor.index_document_task",
    queue="index",
    time_limit=get_settings().celery_index_time_limit,
    soft_time_limit=max(1, get_settings().celery_index_time_limit - 30),
)
def index_document_task(self, artifact_id: str) -> dict[str, Any]:
    """Load artifact, caption images, embed, upsert Qdrant, mark READY.

    Receives artifact_id as the first positional argument from the preceding
    parse_document_task via Celery chain pass-through.
    On retry, reloads the artifact from disk — never re-parses.
    """
    task_id = self.request.id
    document_id: str | None = None
    filename: str | None = None
    index_start = None

    document_repo = DocumentRepository(SessionLocal)

    try:
        # Load artifact
        container = get_container()
        artifact_repo = container.document_parse_artifact_repository()
        artifact = artifact_repo.get_by_id(UUID(artifact_id))
        if not artifact:
            raise ValueError(f"ParseArtifact {artifact_id} not found")

        document_id = str(artifact.document_id)

        # Verify document exists
        document = document_repo.get_by_id(UUID(document_id))
        if not document:
            raise ValueError(f"Document {document_id} not found")

        filename = document.filename if hasattr(document, "filename") else document_id

        # Load ParseResult from disk
        from app.services.document_parse_service import DocumentParseService

        settings = get_settings()
        chunk_builder = container.document_chunk_builder()
        parse_service = DocumentParseService(
            settings=settings,
            chunk_builder=chunk_builder,
        )
        parse_result = parse_service.load_parse_result(artifact)

        # Resolve conversation owner for chunk scoping
        owner_id: str | None = None
        with SessionLocal() as session:
            conversation = (
                session.query(Conversation)
                .filter(Conversation.id == document.conversation_id)
                .one_or_none()
            )
            if conversation and conversation.owner_id is not None:
                owner_id = str(conversation.owner_id)

        # Get DocumentProcessingService for indexing helpers
        processing_service = container.document_processing_service()

        index_timings: dict[str, float] = {}
        index_start = time.monotonic()

        # Build document reference
        doc_ref = SimpleNamespace(
            id=UUID(document_id),
            conversation_id=document.conversation_id,
            user_id=(UUID(owner_id) if owner_id else None),
            filename=filename,
        )

        # Ownership for usage attribution is derived here from the verified
        # conversation owner — never trusted from the Celery payload. Bound for
        # in-thread captioning; passed explicitly to index_document because the
        # embedding batches run in a ThreadPoolExecutor that would not inherit
        # a bound ContextVar.
        usage_context = UsageContext(
            user_id=(UUID(owner_id) if owner_id else None),
            conversation_id=document.conversation_id,
            document_id=UUID(document_id),
            operation="document_index",
        )

        # Caption images + attach to chunks (async)
        caption_t0 = time.monotonic()
        if parse_result.images_data:
            with bind_usage_context(usage_context):
                prepared_images = _run_async(
                    processing_service._prepare_images_for_indexing(
                        parse_result.images_data,
                        document_id,
                    )
                )
            parse_result.blocks = processing_service._attach_prepared_images_to_blocks(
                parse_result.blocks,
                prepared_images,
            )
        else:
            prepared_images = []
        index_timings["caption_s"] = time.monotonic() - caption_t0

        # Build BuiltChunks for indexing (sync)
        built_chunks = processing_service._build_chunks_for_indexing(parse_result.blocks)

        # Index document in Qdrant (sync)
        persisted_chunks = processing_service.document_index_service.index_document(
            document=doc_ref,
            built_chunks=built_chunks,
            parse_artifact_id=artifact.id,
            timing_sink=index_timings,
            usage_context=usage_context,
        )

        # Store image records (async)
        images_stored = 0
        if prepared_images:
            images_stored = _run_async(
                processing_service._store_prepared_images(
                    prepared_images,
                    document_id,
                    persisted_chunks,
                )
            )

        # Mark document READY
        _mark_document(document_repo, document_id, DocumentStatus.READY)
        updated_document = document_repo.get_by_id(UUID(document_id))

        total_s = time.monotonic() - index_start
        index_timings["total_s"] = total_s

        # Build timings dict with all stages
        timings = {
            "parse_s": round(parse_result.parse_elapsed_s, 3),
            "caption_s": round(index_timings.get("caption_s", 0.0), 3),
            "embed_s": round(index_timings.get("embed_s", 0.0), 3),
            "upsert_s": round(index_timings.get("upsert_s", 0.0), 3),
            "total_s": round(total_s, 3),
        }

        # Structured log line
        logger.info(
            "document=%s indexed: parse_s=%.3f caption_s=%.3f embed_s=%.3f "
            "upsert_s=%.3f total_s=%.3f",
            document_id,
            timings["parse_s"],
            timings["caption_s"],
            timings["embed_s"],
            timings["upsert_s"],
            timings["total_s"],
        )

        # Emit PROCESSING_COMPLETED event
        with contextlib.suppress(Exception):
            _run_async(
                get_event_bus().emit(
                    DocumentEvent.PROCESSING_COMPLETED,
                    DocumentEventData(
                        document_id=UUID(document_id),
                        conversation_id=(
                            updated_document.conversation_id if updated_document else None
                        ),
                        filename=filename,
                        status="READY",
                        metadata={
                            "chunks_created": len(built_chunks),
                            "chunks_stored": len(persisted_chunks),
                            "images_stored": images_stored,
                            "processing_time": timings["total_s"],
                            "task_id": task_id,
                            "artifact_id": artifact_id,
                            "parse_artifact_id": artifact_id,
                            "stage": "index",
                            "timings": timings,
                        },
                    ),
                )
            )

        return {
            "success": True,
            "document_id": document_id,
            "artifact_id": artifact_id,
            "chunks_created": len(built_chunks),
            "chunks_stored": len(persisted_chunks),
            "images_stored": images_stored,
            "processing_time": timings["total_s"],
            "timings": timings,
            "message": f"Document '{filename}' indexed successfully",
        }

    except Exception as exc:
        logger.error(
            f"Error in index_document_task for artifact {artifact_id} "
            f"(document {document_id}): {exc}",
            exc_info=True,
        )

        retryable = self.request.retries < self.max_retries and not isinstance(
            exc, (FileNotFoundError, ValueError)
        )
        if retryable:
            retry_delay = min(300, 60 * (2**self.request.retries))
            logger.info(
                f"Retrying index_document_task {task_id} for artifact {artifact_id} "
                f"(attempt {self.request.retries + 1}/{self.max_retries}) in {retry_delay}s"
            )
            # Reload artifact from disk on retry — never re-parse
            raise self.retry(exc=exc, countdown=retry_delay) from exc

        # Terminal failure — only mark FAILED and emit event when not retrying
        if document_id:
            _mark_document(document_repo, document_id, DocumentStatus.FAILED)
            _emit_failed(document_id, filename or artifact_id, task_id, exc)
        else:
            logger.error(
                "index_document_task: cannot mark document FAILED — document_id unknown "
                "(artifact_id=%s, error=%s)",
                artifact_id,
                exc,
            )
        logger.error(f"Artifact {artifact_id} indexing failed after {self.max_retries} attempts")
        raise exc  # Let Celery see this as a failure


# ---------------------------------------------------------------------------
# Backward compatibility shim
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    base=CallbackTask,
    max_retries=3,
    default_retry_delay=60,
    name="app.workers.document_processor.process_document_task",
)
def process_document_task(self, document_id: str, temp_file_path: str, filename: str) -> str:
    """Compatibility shim: enqueues the parse→index chain.

    Handles any tasks already in the queue at deploy time that reference
    the old monolithic task name.
    """
    from celery import chain as celery_chain

    parse_sig = celery_app.signature(
        "app.workers.document_processor.parse_document_task",
        args=[document_id, temp_file_path, filename],
    )
    index_sig = celery_app.signature(
        "app.workers.document_processor.index_document_task",
    )
    chain_result = celery_chain(parse_sig, index_sig).apply_async()
    return chain_result.id


# ---------------------------------------------------------------------------
# Periodic cleanup task
# ---------------------------------------------------------------------------


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
