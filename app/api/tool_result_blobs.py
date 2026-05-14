"""API for reading offloaded tool result blobs."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse

from app.core.auth import get_current_user_id
from app.core.container import Container
from app.repositories.tool_result_blob import ToolResultBlobRepository
from app.services.tool_result_blob_service import ToolResultBlobService

router = APIRouter(prefix="/tool-results", tags=["tool-results"])


def _get_repository() -> ToolResultBlobRepository:
    return Container().tool_result_blob_repository()


def _get_service() -> ToolResultBlobService:
    return Container().tool_result_blob_service()


@router.get("/{blob_id}", response_class=PlainTextResponse)
async def read_tool_result_blob(
    blob_id: UUID,
    current_user_id: UUID = Depends(get_current_user_id),
    repository: ToolResultBlobRepository = Depends(_get_repository),
    service: ToolResultBlobService = Depends(_get_service),
) -> PlainTextResponse:
    record = repository.get_for_user(blob_id, current_user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool result not found")
    return PlainTextResponse(
        service.read_text(record), media_type=record.content_type or "text/plain"
    )
