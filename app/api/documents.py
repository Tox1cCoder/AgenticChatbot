import logging
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
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
    DocumentResponse,
    DocumentListResponse,
    DocumentUpdate,
    DocumentCreate,
    DocumentStatus,
)
from app.schemas.responses.api_response import ApiResponse
from app.services.document_processing_service import DocumentProcessingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
@AppAutoInjector.auto_inject()
async def upload_document(
    document_service: IDocumentService,
    file: UploadFile = File(...),
    conversation_id: UUID = Form(...),
) -> ApiResponse:
    """Upload a document and start background processing"""
    file_content = await file.read()

    # Validate and create document
    document = await document_service.validate_and_create_document(
        filename=file.filename or "",
        file_content=file_content,
        content_type=file.content_type or "unknown",
        conversation_id=conversation_id,
    )

    # Start background processing
    processing_service = DocumentProcessingService()
    task_info = await processing_service.start_processing_task(
        str(document.id), file_content, file.filename or "unknown"
    )

    return ApiResponse(
        success=True,
        message=f"Document '{file.filename}' uploaded successfully and is being processed in the background.",
        data={"document": document.model_dump(), "processing": task_info},
    )


@router.get("/{document_id}", response_model=ApiResponse)
@AppAutoInjector.auto_inject()
async def get_document(
    document_service: IDocumentService, current_user_id: UUID, document_id: UUID
) -> ApiResponse:
    """Get document by ID"""
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
    document = await document_service.update_document(document_id, update_data)

    if not document:
        raise ResourceNotFoundException(detail="Document not found")

    return ApiResponse(
        success=True,
        message="Document updated successfully",
        data=document.model_dump(),
    )


@router.delete("/{document_id}", response_model=ApiResponse)
@AppAutoInjector.auto_inject()
async def delete_document(
    document_service: IDocumentService, current_user_id: UUID, document_id: UUID
) -> ApiResponse:
    """Delete document"""
    success = await document_service.delete_document(document_id)

    if not success:
        raise ResourceNotFoundException(detail="Document not found or access denied")

    return ApiResponse(
        success=True,
        message="Document deleted successfully",
        data={"deleted_document_id": document_id},
    )
