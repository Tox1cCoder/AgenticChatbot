import contextlib
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import (
    APIRouter,
    File,
    Form,
    Response,
    UploadFile,
    status,
)

from app.core.dependency_injection import AppAutoInjector
from app.core.events import DocumentEvent, DocumentEventData, get_event_bus
from app.core.exceptions.resource import ResourceNotFoundException
from app.core.exceptions.validation import (
    DuplicateDocumentFilenameError,
    FileValidationError,
)
from app.interfaces.document_service_interface import IDocumentService
from app.repositories.document import DocumentRepository
from app.schemas.document import (
    DocumentBatchUploadResponse,
    DocumentUpdate,
    DocumentUploadFileResult,
    DocumentUploadFileStatus,
)
from app.schemas.responses.api_response import ApiResponse
from app.services.document_processing_service import DocumentProcessingService
from app.services.document_service import normalize_document_filename
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.utils.validation.document_validation import DocumentValidationUtils

router = APIRouter(prefix="/documents", tags=["documents"])

logger = logging.getLogger(__name__)

# Single source of truth for accepted document types. The sidecar may
# optionally short-circuit obviously invalid requests for UX, but this set
# is the authoritative server-owned gate.
SUPPORTED_UPLOAD_EXTENSIONS: frozenset[str] = frozenset(
    {".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md"}
)


async def _stage_create_and_enqueue_document(
    *,
    document_service: IDocumentService,
    document_processing_service: DocumentProcessingService,
    current_user_id: UUID,
    file: UploadFile,
    conversation_id: UUID,
) -> DocumentUploadFileResult:
    """Stage one upload, create the Document row, and enqueue processing.

    Returns a single ``accepted`` per-file result. Translates known
    validation errors (duplicate filename, file validation) into a
    ``rejected`` result rather than propagating them, so a batch can
    succeed even if individual files fail.
    """

    display_name = file.filename or "unknown"

    try:
        staged_upload = await document_processing_service.stage_upload_file(
            file, file.filename or "unknown"
        )
    except DuplicateDocumentFilenameError as exc:
        return _rejection_from_duplicate(display_name, exc)
    except FileValidationError as exc:
        return _rejection_from_validation(display_name, exc)
    except ValueError as exc:
        return DocumentUploadFileResult(
            filename=display_name,
            status=DocumentUploadFileStatus.REJECTED,
            error_code="FILE_VALIDATION_ERROR",
            message=str(exc),
        )

    staged_file_path = Path(staged_upload["temp_file_path"])

    try:
        document = await document_service.validate_and_create_document(
            filename=file.filename or "",
            file_size=staged_upload["file_size"],
            content_type=file.content_type or "unknown",
            conversation_id=conversation_id,
        )

        task_info = await document_processing_service.start_processing_task(
            str(document.id),
            str(staged_file_path),
            file.filename or "unknown",
            staged_upload["file_size"],
        )

        task_id = task_info.get("task_id")
        if task_id:
            updated = await document_service.set_processing_task_id(document.id, task_id)
            if updated is not None:
                document = updated

        with contextlib.suppress(Exception):
            await get_event_bus().emit(
                DocumentEvent.UPLOAD_STARTED,
                DocumentEventData(
                    document_id=document.id,
                    conversation_id=conversation_id,
                    user_id=current_user_id,
                    filename=file.filename,
                    status="PROCESSING",
                    metadata={"task_id": task_info.get("task_id")},
                ),
            )

        return DocumentUploadFileResult(
            filename=display_name,
            status=DocumentUploadFileStatus.ACCEPTED,
            document=document,
            processing=task_info,
        )
    except DuplicateDocumentFilenameError as exc:
        _safe_unlink(staged_file_path)
        return _rejection_from_duplicate(display_name, exc)
    except FileValidationError as exc:
        _safe_unlink(staged_file_path)
        return _rejection_from_validation(display_name, exc)
    except Exception:
        _safe_unlink(staged_file_path)
        raise


def _rejection_from_duplicate(
    filename: str, exc: DuplicateDocumentFilenameError
) -> DocumentUploadFileResult:
    return DocumentUploadFileResult(
        filename=filename,
        status=DocumentUploadFileStatus.REJECTED,
        error_code=exc.error_code or "DUPLICATE_FILENAME",
        message=str(exc.detail),
    )


def _rejection_from_validation(filename: str, exc: FileValidationError) -> DocumentUploadFileResult:
    return DocumentUploadFileResult(
        filename=filename,
        status=DocumentUploadFileStatus.REJECTED,
        error_code=exc.error_code or "FILE_VALIDATION_ERROR",
        message=str(exc.detail),
    )


def _safe_unlink(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
    except Exception:
        pass


async def _upload_documents_batch(
    *,
    document_service: IDocumentService,
    document_processing_service: DocumentProcessingService,
    current_user_id: UUID,
    files: Iterable[UploadFile],
    conversation_id: UUID,
) -> DocumentBatchUploadResponse:
    """Stage, create, and enqueue every file in the incoming batch.

    Order of the returned ``files`` list mirrors the input order. Rejects
    same-batch duplicates without staging the later candidate and rejects
    files whose normalized filename already exists in the conversation
    before staging when possible.
    """
    file_list = list(files)
    results: list[DocumentUploadFileResult] = []
    seen_batch_keys: set[str] = set()

    for upload in file_list:
        display_name = upload.filename or "unknown"

        try:
            filename_key = normalize_document_filename(upload.filename or "")
        except FileValidationError as exc:
            results.append(_rejection_from_validation(display_name, exc))
            continue

        if filename_key in seen_batch_keys:
            results.append(
                DocumentUploadFileResult(
                    filename=display_name,
                    status=DocumentUploadFileStatus.REJECTED,
                    error_code="DUPLICATE_FILENAME",
                    message=(
                        f"A document named '{display_name}' was already submitted "
                        "earlier in this batch."
                    ),
                )
            )
            continue

        result = await _stage_create_and_enqueue_document(
            document_service=document_service,
            document_processing_service=document_processing_service,
            current_user_id=current_user_id,
            file=upload,
            conversation_id=conversation_id,
        )

        if result.status == DocumentUploadFileStatus.ACCEPTED:
            seen_batch_keys.add(filename_key)

        results.append(result)

    accepted_count = sum(1 for item in results if item.status == DocumentUploadFileStatus.ACCEPTED)
    rejected_count = len(results) - accepted_count

    return DocumentBatchUploadResponse(
        conversation_id=conversation_id,
        total_count=len(results),
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        files=results,
    )


def _status_code_for_batch_result(result: DocumentBatchUploadResponse) -> int:
    """Map a batch outcome to an HTTP status code per the plan's status policy."""
    if result.total_count == 0:
        return status.HTTP_400_BAD_REQUEST
    if result.accepted_count > 0 and result.rejected_count == 0:
        return status.HTTP_201_CREATED
    if result.accepted_count > 0 and result.rejected_count > 0:
        return status.HTTP_207_MULTI_STATUS
    # All rejected — choose 409 if everything was a duplicate, otherwise 400.
    if all(item.error_code == "DUPLICATE_FILENAME" for item in result.files):
        return status.HTTP_409_CONFLICT
    return status.HTTP_400_BAD_REQUEST


def _message_for_batch_result(result: DocumentBatchUploadResponse) -> str:
    if result.accepted_count == 0:
        if all(item.error_code == "DUPLICATE_FILENAME" for item in result.files):
            return "All files were rejected because of duplicate filenames."
        return "All files were rejected."
    if result.rejected_count == 0:
        return (
            f"{result.accepted_count} document(s) uploaded successfully and queued for processing."
        )
    return (
        f"{result.accepted_count} document(s) queued; "
        f"{result.rejected_count} file(s) were rejected."
    )


@router.post(
    "/upload",
    response_model=ApiResponse[dict[str, Any]],
    status_code=status.HTTP_201_CREATED,
)
@AppAutoInjector.auto_inject()
async def upload_document(
    response: Response,
    document_service: IDocumentService,
    document_processing_service: DocumentProcessingService,
    current_user_id: UUID,
    file: UploadFile = File(...),  # noqa: B008
    conversation_id: UUID = Form(...),  # noqa: B008
) -> ApiResponse[dict[str, Any]]:
    """Legacy single-file upload route, kept as a thin compatibility wrapper.

    Delegates to ``_upload_documents_batch`` so single- and multi-file
    paths share staging, duplicate detection, and enqueue behavior.
    Emits the same ``DocumentEvent.UPLOAD_STARTED`` event as before.
    """

    ConversationValidationUtils(
        document_service.repository.session_factory
    ).validate_conversation_access(current_user_id, conversation_id)

    upload_result = await _upload_documents_batch(
        document_service=document_service,
        document_processing_service=document_processing_service,
        current_user_id=current_user_id,
        files=[file],
        conversation_id=conversation_id,
    )

    response.status_code = _status_code_for_batch_result(upload_result)

    if upload_result.accepted_count == 1:
        accepted = upload_result.files[0]
        return ApiResponse(
            success=True,
            message=(
                f"Document '{accepted.filename}' uploaded successfully and is being "
                "processed in the background."
            ),
            data={
                "document": accepted.document.model_dump() if accepted.document else None,
                "processing": accepted.processing,
            },
        )

    rejected = upload_result.files[0]
    return ApiResponse(
        success=False,
        message=rejected.message or "Upload was rejected.",
        data={
            "error_code": rejected.error_code,
            "filename": rejected.filename,
        },
    )


@router.post(
    "/uploads",
    response_model=ApiResponse[dict[str, Any]],
)
@AppAutoInjector.auto_inject()
async def upload_documents(
    response: Response,
    document_service: IDocumentService,
    document_processing_service: DocumentProcessingService,
    current_user_id: UUID,
    files: list[UploadFile] = File(...),  # noqa: B008
    conversation_id: UUID = Form(...),  # noqa: B008
) -> ApiResponse[dict[str, Any]]:
    """Canonical batch upload route.

    Accepts multiple files in one multipart request. Each file is staged,
    created, and enqueued independently so a single invalid or duplicate
    file does not prevent its siblings from being processed.
    """

    if not files:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return ApiResponse(
            success=False,
            message="No files were submitted.",
            data={"error_code": "EMPTY_BATCH"},
        )

    ConversationValidationUtils(
        document_service.repository.session_factory
    ).validate_conversation_access(current_user_id, conversation_id)

    upload_result = await _upload_documents_batch(
        document_service=document_service,
        document_processing_service=document_processing_service,
        current_user_id=current_user_id,
        files=files,
        conversation_id=conversation_id,
    )
    response.status_code = _status_code_for_batch_result(upload_result)
    return ApiResponse(
        success=upload_result.accepted_count > 0,
        message=_message_for_batch_result(upload_result),
        data=upload_result.model_dump(),
    )


@router.get("/task/{task_id}", response_model=ApiResponse[dict[str, Any]])
@AppAutoInjector.auto_inject()
async def get_task_status(
    document_service: IDocumentService,
    document_processing_service: DocumentProcessingService,
    current_user_id: UUID,
    task_id: str,
) -> ApiResponse[dict[str, Any]]:
    """Get sanitized background task status by task ID.

    Enforces document ownership: only the user who uploaded the document
    associated with this task may query its status.
    """
    # Ownership check: look up the document by Celery task ID.
    doc_repo = DocumentRepository(document_service.repository.session_factory)
    document = doc_repo.get_by_processing_task_id(task_id)
    if document is not None:
        # Verify that the requesting user owns the conversation this document
        # belongs to.  Raises AuthorizationException on mismatch.
        DocumentValidationUtils(
            document_service.repository.session_factory
        ).validate_document_access(current_user_id, document.id)

    task_status = await document_processing_service.get_processing_status(task_id)

    return ApiResponse(
        success=True,
        message="Task status retrieved successfully",
        data=task_status,
    )


@router.get("/{document_id}", response_model=ApiResponse[dict[str, Any]])
@AppAutoInjector.auto_inject()
async def get_document(
    document_service: IDocumentService,
    current_user_id: UUID,
    document_id: UUID,
) -> ApiResponse[dict[str, Any]]:
    """Get document by ID"""

    # Validate document access
    DocumentValidationUtils(document_service.repository.session_factory).validate_document_access(
        current_user_id, document_id
    )

    document = await document_service.get_document(document_id)

    if not document:
        raise ResourceNotFoundException(detail="Document not found")

    return ApiResponse(
        success=True,
        message="Document retrieved successfully",
        data=document.model_dump(),
    )


@router.get("/conversation/{conversation_id}", response_model=ApiResponse[dict[str, Any]])
@AppAutoInjector.auto_inject()
async def get_conversation_documents(
    document_service: IDocumentService,
    current_user_id: UUID,
    conversation_id: UUID,
    page: int = 1,
    page_size: int = 20,
) -> ApiResponse[dict[str, Any]]:
    """Get documents for a conversation with pagination"""
    # Validate conversation access
    ConversationValidationUtils(
        document_service.repository.session_factory
    ).validate_conversation_access(current_user_id, conversation_id)
    document_list = await document_service.get_documents_by_conversation(
        conversation_id, page, page_size
    )

    return ApiResponse(
        success=True,
        message="Documents retrieved successfully",
        data=document_list.model_dump(),
    )


@router.put("/{document_id}", response_model=ApiResponse[dict[str, Any]])
@AppAutoInjector.auto_inject()
async def update_document(
    document_service: IDocumentService,
    current_user_id: UUID,
    document_id: UUID,
    update_data: DocumentUpdate,
) -> ApiResponse[dict[str, Any]]:
    """Update document"""

    DocumentValidationUtils(document_service.repository.session_factory).validate_document_access(
        current_user_id, document_id
    )

    document = await document_service.update_document(document_id, update_data)

    return ApiResponse(
        success=True,
        message="Document updated successfully",
        data=document.model_dump(),
    )


@router.delete("/{document_id}", response_model=ApiResponse[dict[str, Any]])
@AppAutoInjector.auto_inject()
async def delete_document(
    document_service: IDocumentService,
    current_user_id: UUID,
    document_id: UUID,
) -> ApiResponse[dict[str, Any]]:
    """Delete document"""
    DocumentValidationUtils(document_service.repository.session_factory).validate_document_access(
        current_user_id, document_id
    )
    success = await document_service.delete_document(document_id)

    if not success:
        raise ResourceNotFoundException(detail="Document not found or access denied")

    return ApiResponse(
        success=True,
        message="Document deleted successfully",
        data={"deleted_document_id": document_id},
    )
