from fastapi import APIRouter, UploadFile, File
from typing import Dict, Any, Annotated
from uuid import UUID

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.document_service_interface import IDocumentService
from app.schemas.responses.api_response import ApiResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/", response_model=ApiResponse)
@AppAutoInjector.auto_inject()
async def upload_document(
    document_service: IDocumentService,
    current_user_id: UUID,
    file: UploadFile = File(...),
) -> ApiResponse:
    """
    Upload and process a document for RAG functionality.

    Args:
        file: The uploaded file (PDF, TXT, or DOCX)
        current_user_id: The authenticated user ID
        document_service: Document service instance

    Returns:
        ApiResponse: Success response with processing details
    """

    file_content = await file.read()

    result = await document_service.upload_and_process_document(
        file_content=file_content,
        filename=file.filename,
        file_size=file.size,
        user_id=str(current_user_id),
    )

    return ApiResponse(
        success=True,
        message=f"Document '{file.filename}' processed successfully",
        data=result,
    )
