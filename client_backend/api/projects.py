"""
Project proxy endpoints for the local client backend.
"""

from typing import Any

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import Response

from app.schemas.custom_agent import CustomAgentRead
from app.schemas.project import (
    ProjectCreate,
    ProjectCustomAgentsUpdate,
    ProjectRead,
    ProjectUpdate,
)
from app.schemas.responses import ApiResponse
from app.schemas.responses.paginated_response import PaginatedApiResponse
from client_backend.api.common import proxy_server_request
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload

router = APIRouter(prefix="/projects", tags=["projects"])


def _request_body(schema: type) -> dict[str, Any]:
    """Publish the upstream body schema without validating twice in the proxy."""
    return {
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": schema.model_json_schema(by_alias=True)}},
        }
    }


LIST_PROJECTS_PARAMETERS = {
    "parameters": [
        {
            "name": "page",
            "in": "query",
            "schema": {"type": "integer", "minimum": 1, "default": 1},
            "description": "Page number (1-based)",
        },
        {
            "name": "limit",
            "in": "query",
            "schema": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
            "description": "Projects per page",
        },
        {
            "name": "search",
            "in": "query",
            "schema": {"type": "string", "maxLength": 200},
            "description": "Case-insensitive project name or description search",
        },
    ]
}


@router.get(
    "", response_model=PaginatedApiResponse[ProjectRead], openapi_extra=LIST_PROJECTS_PARAMETERS
)
async def list_projects(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """List the signed-in user's projects."""
    return await proxy_server_request(request, upstream_path="/projects")


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=ApiResponse[ProjectRead],
    openapi_extra=_request_body(ProjectCreate),
)
async def create_project(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Create a project."""
    return await proxy_server_request(request, upstream_path="/projects")


@router.get("/{project_id}", response_model=ApiResponse[ProjectRead])
async def get_project(
    project_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Fetch a project by ID."""
    return await proxy_server_request(request, upstream_path=f"/projects/{project_id}")


@router.patch(
    "/{project_id}",
    response_model=ApiResponse[ProjectRead],
    openapi_extra=_request_body(ProjectUpdate),
)
async def update_project(
    project_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Update a project."""
    return await proxy_server_request(request, upstream_path=f"/projects/{project_id}")


@router.delete("/{project_id}", response_model=ApiResponse[Any])
async def delete_project(
    project_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Delete a project."""
    return await proxy_server_request(request, upstream_path=f"/projects/{project_id}")


@router.get("/{project_id}/custom-agents", response_model=ApiResponse[list[CustomAgentRead]])
async def list_project_custom_agents(
    project_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """The project's ordered default agents."""
    return await proxy_server_request(
        request, upstream_path=f"/projects/{project_id}/custom-agents"
    )


@router.put(
    "/{project_id}/custom-agents",
    response_model=ApiResponse[list[CustomAgentRead]],
    openapi_extra=_request_body(ProjectCustomAgentsUpdate),
)
async def set_project_custom_agents(
    project_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),  # noqa: B008
) -> Response:
    """Replace the project's default agent set."""
    return await proxy_server_request(
        request, upstream_path=f"/projects/{project_id}/custom-agents"
    )


@router.put("/{project_id}/conversations/{conversation_id}", response_model=ApiResponse[Any])
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


@router.delete("/{project_id}/conversations/{conversation_id}", response_model=ApiResponse[Any])
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
