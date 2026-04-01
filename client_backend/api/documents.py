"""
Document proxy endpoints for the local client backend.
"""

from typing import Any

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile, status

from client_backend.api.common import proxy_server_request, raise_server_error
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
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    """Fetch background processing task status."""
    return await proxy_server_request(request, upstream_path=f"/documents/task/{task_id}")


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    """Fetch a document by ID."""
    return await proxy_server_request(request, upstream_path=f"/documents/{document_id}")


@router.get("/conversation/{conversation_id}")
async def list_conversation_documents(
    conversation_id: str,
    request: Request,
    page: int = 1,
    page_size: int = 20,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    """List documents for a conversation."""
    return await proxy_server_request(
        request,
        upstream_path=f"/documents/conversation/{conversation_id}",
        params_override={"page": page, "page_size": page_size},
    )


@router.put("/{document_id}")
async def update_document(
    document_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    """Update document metadata."""
    return await proxy_server_request(request, upstream_path=f"/documents/{document_id}")


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    """Delete a document."""
    return await proxy_server_request(request, upstream_path=f"/documents/{document_id}")
