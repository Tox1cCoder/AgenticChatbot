"""
Conversation proxy endpoints for the local client backend.
"""

from typing import Any

from fastapi import APIRouter, Depends, Query, Response, status

from client_backend.api.common import raise_server_error
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload
from client_backend.services.server_api import get_server_client

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("/")
async def list_conversations(
    page: int = 1,
    limit: int = 20,
    include: list[str] = Query(default=[]),  # noqa: B008
    latest_messages: int = Query(default=3, alias="latestMessages"),
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """Proxy conversation listing to the canonical server."""
    try:
        return await get_server_client().get(
            "/conversations/",
            params={
                "page": page,
                "limit": limit,
                "include": include,
                "latestMessages": latest_messages,
            },
        )
    except Exception as exc:
        raise_server_error(exc)


@router.post("/generate-title")
async def generate_title(
    payload: dict[str, Any],
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """Generate a conversation title using the upstream server."""
    try:
        return await get_server_client().post("/conversations/generate-title", json=payload)
    except Exception as exc:
        raise_server_error(exc)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: dict[str, Any],
    response: Response,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """Create a conversation."""
    try:
        response.status_code = status.HTTP_201_CREATED
        return await get_server_client().post("/conversations/", json=payload)
    except Exception as exc:
        raise_server_error(exc)


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """Fetch a conversation by ID."""
    try:
        return await get_server_client().get(f"/conversations/{conversation_id}")
    except Exception as exc:
        raise_server_error(exc)


@router.get("/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    page: int = 1,
    limit: int = 50,
    include: list[str] = Query(default=[]),  # noqa: B008
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """Fetch conversation messages."""
    try:
        return await get_server_client().get(
            f"/conversations/{conversation_id}/messages",
            params={"page": page, "limit": limit, "include": include},
        )
    except Exception as exc:
        raise_server_error(exc)


@router.patch("/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    payload: dict[str, Any],
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """Update a conversation."""
    try:
        return await get_server_client().update_conversation(conversation_id, payload)
    except Exception as exc:
        raise_server_error(exc)


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> dict[str, Any]:
    """Delete a conversation."""
    try:
        return await get_server_client().delete(f"/conversations/{conversation_id}")
    except Exception as exc:
        raise_server_error(exc)
