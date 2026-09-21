"""Project CRUD, default agents, and conversation membership routes."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, status

from app.core.dependency_injection import AppAutoInjector
from app.schemas.custom_agent import CustomAgentRead
from app.schemas.project import (
    ProjectCreate,
    ProjectCustomAgentsUpdate,
    ProjectRead,
    ProjectUpdate,
)
from app.schemas.responses import ApiResponse
from app.services.project_service import ProjectService

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=ApiResponse[list[ProjectRead]])
@AppAutoInjector.auto_inject()
async def list_projects(
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[list[ProjectRead]]:
    """List the authenticated user's projects."""
    result = project_service.list_projects(user_id)
    return ApiResponse(success=True, message="Projects retrieved", data=result)


@router.post("", response_model=ApiResponse[ProjectRead], status_code=status.HTTP_201_CREATED)
@AppAutoInjector.auto_inject()
async def create_project(
    payload: ProjectCreate,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[ProjectRead]:
    """Create a project."""
    result = project_service.create_project(user_id, payload)
    return ApiResponse(success=True, message="Project created", data=result)


@router.get("/{project_id}", response_model=ApiResponse[ProjectRead])
@AppAutoInjector.auto_inject()
async def get_project(
    project_id: UUID,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[ProjectRead]:
    """Read one project, including its default agent ids."""
    result = project_service.get_project(user_id, project_id, include_agents=True)
    return ApiResponse(success=True, message="Project retrieved", data=result)


@router.patch("/{project_id}", response_model=ApiResponse[ProjectRead])
@AppAutoInjector.auto_inject()
async def update_project(
    project_id: UUID,
    payload: ProjectUpdate,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[ProjectRead]:
    """Update a project's name, description, or instructions."""
    result = project_service.update_project(user_id, project_id, payload)
    return ApiResponse(success=True, message="Project updated", data=result)


@router.delete("/{project_id}", response_model=ApiResponse[Any])
@AppAutoInjector.auto_inject()
async def delete_project(
    project_id: UUID,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[Any]:
    """Soft-delete a project; its conversations are detached, not deleted."""
    project_service.delete_project(user_id, project_id)
    return ApiResponse(success=True, message="Project deleted", data=None)


@router.get("/{project_id}/custom-agents", response_model=ApiResponse[list[CustomAgentRead]])
@AppAutoInjector.auto_inject()
async def list_project_custom_agents(
    project_id: UUID,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[list[CustomAgentRead]]:
    """The project's ordered default agents."""
    result = project_service.list_agents(user_id, project_id)
    return ApiResponse(success=True, message="Project agents retrieved", data=result)


@router.put("/{project_id}/custom-agents", response_model=ApiResponse[list[CustomAgentRead]])
@AppAutoInjector.auto_inject()
async def set_project_custom_agents(
    project_id: UUID,
    payload: ProjectCustomAgentsUpdate,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[list[CustomAgentRead]]:
    """Replace the default set. Conversations already in the project are untouched."""
    result = project_service.set_agents(user_id, project_id, payload.custom_agent_ids)
    return ApiResponse(success=True, message="Project agents updated", data=result)


@router.put("/{project_id}/conversations/{conversation_id}", response_model=ApiResponse[Any])
@AppAutoInjector.auto_inject()
async def attach_conversation(
    project_id: UUID,
    conversation_id: UUID,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[Any]:
    """Move a conversation into the project and seed the project's agents."""
    project_service.attach_conversation(user_id, project_id, conversation_id)
    return ApiResponse(success=True, message="Conversation attached", data=None)


@router.delete("/{project_id}/conversations/{conversation_id}", response_model=ApiResponse[Any])
@AppAutoInjector.auto_inject()
async def detach_conversation(
    project_id: UUID,
    conversation_id: UUID,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[Any]:
    """Release a conversation from the project, keeping its seeded agents."""
    project_service.detach_conversation(user_id, project_id, conversation_id)
    return ApiResponse(success=True, message="Conversation detached", data=None)
