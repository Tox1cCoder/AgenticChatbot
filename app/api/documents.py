import contextlib
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import (
    APIRouter,
    File,
    Form,
    UploadFile,
    status,
)

from app.core.dependency_injection import AppAutoInjector
from app.core.events import DocumentEvent, DocumentEventData, get_event_bus
from app.core.exceptions.resource import ResourceNotFoundException
from app.interfaces.document_service_interface import IDocumentService
from app.repositories.document import DocumentRepository
from app.schemas.document import (
    DocumentUpdate,
)
from app.schemas.responses.api_response import ApiResponse
from app.services.document_processing_service import DocumentProcessingService
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.utils.validation.document_validation import DocumentValidationUtils

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=ApiResponse[dict[str, Any]],
    status_code=status.HTTP_201_CREATED,
)
@AppAutoInjector.auto_inject()
async def upload_document(
    document_service: IDocumentService,
    document_processing_service: DocumentProcessingService,
    current_user_id: UUID,
    file: UploadFile = File(...),  # noqa: B008
    conversation_id: UUID = Form(...),  # noqa: B008
) -> ApiResponse[dict[str, Any]]:
    """Upload a document and start background processing.

    Emits: DocumentEvent.UPLOAD_STARTED after document record creation.
    """

    # Validate conversation ownership
    ConversationValidationUtils(
        document_service.repository.session_factory
    ).validate_conversation_access(current_user_id, conversation_id)

    staged_upload = await document_processing_service.stage_upload_file(
        file, file.filename or "unknown"
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
    except Exception:
        try:
            if staged_file_path.is_file():
                staged_file_path.unlink()
        except Exception:
            pass
        raise

    # Persist the Celery task ID so ownership can be verified on status lookups.
    task_id = task_info.get("task_id")
    if task_id:
        await document_service.set_processing_task_id(document.id, task_id)

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

    return ApiResponse(
        success=True,
        message=f"Document '{file.filename}' uploaded successfully and is being processed in the background.",
        data={"document": document.model_dump(), "processing": task_info},
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
