"""
Conversation proxy endpoints for the local client backend.
"""

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import Response

from client_backend.api.common import proxy_server_request
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("/")
async def list_conversations(
    request: Request,
    page: int = 1,
    limit: int = 20,
    order_by: str = Query(default="updatedAt", alias="orderBy"),
    order_direction: str = Query(default="desc", alias="orderDirection"),
    include: list[str] = Query(default=[]),  # noqa: B008
    latest_messages: int = Query(default=3, alias="latestMessages"),
    search: str | None = Query(default=None, max_length=200),
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Proxy conversation listing to the canonical server."""
    params = {
        "page": page,
        "limit": limit,
        "include": include,
        "latestMessages": latest_messages,
    }
    if "orderBy" in request.query_params:
        params["orderBy"] = order_by
    if "orderDirection" in request.query_params:
        params["orderDirection"] = order_direction
    if search is not None:
        params["search"] = search

    return await proxy_server_request(
        request,
        upstream_path="/conversations/",
        params_override=params,
    )


@router.post("/generate-title")
async def generate_title(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Generate a conversation title using the upstream server."""
    return await proxy_server_request(request, upstream_path="/conversations/generate-title")


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Create a conversation."""
    return await proxy_server_request(request, upstream_path="/conversations/")


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Fetch a conversation by ID."""
    return await proxy_server_request(request, upstream_path=f"/conversations/{conversation_id}")


@router.get("/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    request: Request,
    page: int = 1,
    limit: int = 50,
    include: list[str] = Query(default=[]),  # noqa: B008
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Fetch conversation messages."""
    return await proxy_server_request(
        request,
        upstream_path=f"/conversations/{conversation_id}/messages",
        params_override={"page": page, "limit": limit, "include": include},
    )


@router.patch("/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Update a conversation."""
    return await proxy_server_request(request, upstream_path=f"/conversations/{conversation_id}")


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Delete a conversation."""
    return await proxy_server_request(request, upstream_path=f"/conversations/{conversation_id}")
