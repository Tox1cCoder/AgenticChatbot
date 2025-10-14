from uuid import UUID

from fastapi import (
    APIRouter,
    File,
    Form,
    UploadFile,
    status,
)

from app.core.dependency_injection import AppAutoInjector
from app.core.exceptions.validation import FileValidationError
from app.core.exceptions.resource import ResourceNotFoundException
from app.interfaces.document_service_interface import IDocumentService
from app.schemas.document import (
    DocumentUpdate,
)
from app.schemas.responses.api_response import ApiResponse
from app.services.document_processing_service import DocumentProcessingService
from app.utils.validation.document_validation import DocumentValidationUtils
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.core.events import get_event_bus, DocumentEvent, DocumentEventData

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
@AppAutoInjector.auto_inject()
async def upload_document(
    document_service: IDocumentService,
    document_processing_service: DocumentProcessingService,
    current_user_id: UUID,
    file: UploadFile = File(...),
    conversation_id: UUID = Form(...),
) -> ApiResponse:
    """Upload a document and start background processing.

    Emits: DocumentEvent.UPLOAD_STARTED after document record creation.
    """

    file_content = await file.read()

    # Validate conversation ownership
    ConversationValidationUtils(
        document_service.repository.session_factory
    ).validate_conversation_access(current_user_id, conversation_id)

    document = await document_service.validate_and_create_document(
        filename=file.filename or "",
        file_content=file_content,
        content_type=file.content_type or "unknown",
        conversation_id=conversation_id,
    )

    task_info = await document_processing_service.start_processing_task(
        str(document.id), file_content, file.filename or "unknown"
    )

    # Emit UPLOAD_STARTED event
    try:
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
    except Exception:
        # Non-blocking on event failure
        pass

    return ApiResponse(
        success=True,
        message=f"Document '{file.filename}' uploaded successfully and is being processed in the background.",
        data={"document": document.model_dump(), "processing": task_info},
    )


@router.get("/task/{task_id}", response_model=ApiResponse)
@AppAutoInjector.auto_inject()
async def get_task_status(
    document_processing_service: DocumentProcessingService,
    task_id: str,
) -> ApiResponse:
    """Get Celery task status by task ID"""
    task_status = await document_processing_service.get_processing_status(task_id)

    return ApiResponse(
        success=True,
        message="Task status retrieved successfully",
        data=task_status,
    )


@router.get("/{document_id}", response_model=ApiResponse)
@AppAutoInjector.auto_inject()
async def get_document(
    document_service: IDocumentService,
    current_user_id: UUID,
    document_id: UUID,
) -> ApiResponse:
    """Get document by ID"""

    # Validate document access
    DocumentValidationUtils(
        document_service.repository.session_factory
    ).validate_document_access(current_user_id, document_id)

    document = await document_service.get_document(document_id)

    if not document:
        raise ResourceNotFoundException(detail="Document not found")

    return ApiResponse(
        success=True,
        message="Document retrieved successfully",
        data=document.model_dump(),
    )


@router.get("/conversation/{conversation_id}", response_model=ApiResponse)
@AppAutoInjector.auto_inject()
async def get_conversation_documents(
    document_service: IDocumentService,
    current_user_id: UUID,
    conversation_id: UUID,
    page: int = 1,
    page_size: int = 20,
) -> ApiResponse:
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


@router.put("/{document_id}", response_model=ApiResponse)
@AppAutoInjector.auto_inject()
async def update_document(
    document_service: IDocumentService,
    current_user_id: UUID,
    document_id: UUID,
    update_data: DocumentUpdate,
) -> ApiResponse:
    """Update document"""

    DocumentValidationUtils(
        document_service.repository.session_factory
    ).validate_document_access(current_user_id, document_id)

    document = await document_service.update_document(document_id, update_data)

    return ApiResponse(
        success=True,
        message="Document updated successfully",
        data=document.model_dump(),
    )


@router.delete("/{document_id}", response_model=ApiResponse)
@AppAutoInjector.auto_inject()
async def delete_document(
    document_service: IDocumentService,
    current_user_id: UUID,
    document_id: UUID,
) -> ApiResponse:
    """Delete document"""
    DocumentValidationUtils(
        document_service.repository.session_factory
    ).validate_document_access(current_user_id, document_id)
    success = await document_service.delete_document(document_id)

    if not success:
        raise ResourceNotFoundException(detail="Document not found or access denied")

    return ApiResponse(
        success=True,
        message="Document deleted successfully",
        data={"deleted_document_id": document_id},
    )
