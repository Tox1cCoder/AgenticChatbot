"""API for reading offloaded tool result blobs."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse

from app.core.auth import get_current_user_id
from app.core.container import Container
from app.repositories.tool_result_blob import ToolResultBlobRepository
from app.services.tool_result_blob_service import ToolResultBlobService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tool-results", tags=["tool-results"])

_NOT_FOUND_DETAIL = "Tool result not found"


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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND_DETAIL)
    try:
        text = service.read_text(record)
    except (ValueError, OSError) as exc:
        # A corrupt record (neither content nor storage_path) or a vanished
        # legacy on-disk blob is an operational problem, not a client error.
        # Log it for an operator and tell the client only that it is gone —
        # the storage internals are not the client's concern.
        logger.warning("Tool result blob %s is unreadable: %s", blob_id, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND_DETAIL
        ) from exc
    return PlainTextResponse(text, media_type=record.content_type or "text/plain")
