"""
Project proxy endpoints for the local client backend.
"""

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import Response

from client_backend.api.common import proxy_server_request
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("")
async def list_projects(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """List the signed-in user's projects."""
    return await proxy_server_request(request, upstream_path="/projects")


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Create a project."""
    return await proxy_server_request(request, upstream_path="/projects")


@router.get("/{project_id}")
async def get_project(
    project_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Fetch a project by ID."""
    return await proxy_server_request(request, upstream_path=f"/projects/{project_id}")


@router.patch("/{project_id}")
async def update_project(
    project_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Update a project."""
    return await proxy_server_request(request, upstream_path=f"/projects/{project_id}")


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Delete a project."""
    return await proxy_server_request(request, upstream_path=f"/projects/{project_id}")


@router.get("/{project_id}/custom-agents")
async def list_project_custom_agents(
    project_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """The project's ordered default agents."""
    return await proxy_server_request(
        request, upstream_path=f"/projects/{project_id}/custom-agents"
    )


@router.put("/{project_id}/custom-agents")
async def set_project_custom_agents(
    project_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Replace the project's default agent set."""
    return await proxy_server_request(
        request, upstream_path=f"/projects/{project_id}/custom-agents"
    )


@router.put("/{project_id}/conversations/{conversation_id}")
async def attach_conversation(
    project_id: str,
    conversation_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Move a conversation into the project."""
    return await proxy_server_request(
        request,
        upstream_path=f"/projects/{project_id}/conversations/{conversation_id}",
    )


@router.delete("/{project_id}/conversations/{conversation_id}")
async def detach_conversation(
    project_id: str,
    conversation_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Release a conversation from the project."""
    return await proxy_server_request(
        request,
        upstream_path=f"/projects/{project_id}/conversations/{conversation_id}",
    )
