from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    UploadFile,
    HTTPException,
    status,
)
from typing import List, Optional, Dict, Any, Annotated
from uuid import UUID

from app.schemas.document import (
    DocumentResponse,
    DocumentListResponse,
    DocumentUpdate,
)
from app.core.dependency_injection import AppAutoInjector
from app.interfaces.document_service_interface import IDocumentService
from app.schemas.responses.api_response import ApiResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=ApiResponse)
@AppAutoInjector.auto_inject()
async def upload_document(
    document_service: IDocumentService,
    file: UploadFile = File(...),
    conversation_id: UUID = Form(...),
) -> ApiResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided"
        )

    # Create basic document record for now
    from app.schemas.document import DocumentCreate, DocumentStatus

    document_data = DocumentCreate(
        conversation_id=conversation_id,
        filename=file.filename,
        file_type=file.content_type or "unknown",
        status=DocumentStatus.PROCESSING,
    )

    document = await document_service.create_document(document_data)

    return ApiResponse(
        success=True,
        message=f"Document '{file.filename}' uploaded successfully",
        data=document.model_dump(),
    )


@router.get("/{document_id}", response_model=ApiResponse)
@AppAutoInjector.auto_inject()
async def get_document(
    document_service: IDocumentService, current_user_id: UUID, document_id: UUID
) -> ApiResponse:
    document = await document_service.get_document(document_id)

    if not document:
        return ApiResponse(success=False, message="Document not found", data=None)

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
    document = await document_service.update_document(document_id, update_data)

    if not document:
        return ApiResponse(success=False, message="Document not found", data=None)

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
    success = await document_service.delete_document(document_id)

    if not success:
        return ApiResponse(
            success=False, message="Document not found or access denied", data=None
        )

    return ApiResponse(
        success=True,
        message="Document deleted successfully",
        data={"deleted_document_id": document_id},
    )
