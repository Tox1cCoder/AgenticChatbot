"""
Document proxy endpoints for the local client backend.
"""

from typing import Any

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile, status

from client_backend.api.common import raise_server_error
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload
from client_backend.services.server_api import get_server_client

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    response: Response,
    file: UploadFile = File(...),  # noqa: B008
    conversation_id: str = Form(...),  # noqa: B008
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """Relay document upload to the canonical server."""
    try:
        response.status_code = status.HTTP_201_CREATED
        content = await file.read()
        return await get_server_client().upload_document_bytes(
            conversation_id=conversation_id,
            filename=file.filename or "upload.bin",
            content=content,
            content_type=file.content_type or "application/octet-stream",
        )
    except Exception as exc:
        raise_server_error(exc)


@router.get("/task/{task_id}")
async def get_document_task_status(
    task_id: str,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """Fetch background processing task status."""
    try:
        return await get_server_client().get_document_task_status(task_id)
    except Exception as exc:
        raise_server_error(exc)


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """Fetch a document by ID."""
    try:
        return await get_server_client().get_document_status(document_id)
    except Exception as exc:
        raise_server_error(exc)


@router.get("/conversation/{conversation_id}")
async def list_conversation_documents(
    conversation_id: str,
    page: int = 1,
    page_size: int = 20,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """List documents for a conversation."""
    try:
        return await get_server_client().get(
            f"/documents/conversation/{conversation_id}",
            params={"page": page, "page_size": page_size},
        )
    except Exception as exc:
        raise_server_error(exc)


@router.put("/{document_id}")
async def update_document(
    document_id: str,
    payload: dict[str, Any],
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """Update document metadata."""
    try:
        return await get_server_client().update_document(document_id, payload)
    except Exception as exc:
        raise_server_error(exc)


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """Delete a document."""
    try:
        return await get_server_client().delete_document(document_id)
    except Exception as exc:
        raise_server_error(exc)
