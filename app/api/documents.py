from fastapi import APIRouter, UploadFile, File, Depends
from typing import Dict, Any, Annotated

from dependency_injector.wiring import Provide, inject

from app.core.auth import get_current_user_id
from app.core.container import Container
from app.interfaces.document_service_interface import IDocumentService
from app.schemas.responses.api_response import ApiResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/", response_model=ApiResponse)
@inject
async def upload_document(
    file: UploadFile = File(...),
    document_service: Annotated[
        IDocumentService, Depends(Provide[Container.document_service])
    ] = None,
    current_user_id=Depends(get_current_user_id),
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
