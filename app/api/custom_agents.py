"""Custom-agent CRUD, options, and per-conversation attachment routes.

Both routers are registered twice in ``app.main`` — once at the canonical
prefix and once under ``/ai`` — so the AI SDK surface gets the same contract
without duplicating handlers.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Query, status

from app.core.dependency_injection import AppAutoInjector
from app.schemas.custom_agent import (
    ConversationCustomAgentsUpdate,
    CustomAgentCreate,
    CustomAgentOptions,
    CustomAgentRead,
    CustomAgentUpdate,
)
from app.schemas.responses import ApiResponse
from app.services.custom_agent_service import CustomAgentService

router = APIRouter(prefix="/custom-agents", tags=["custom-agents"])
conversation_router = APIRouter(prefix="/conversations", tags=["custom-agents"])

_DEVICE_QUERY = Query(default=None, alias="deviceId", description="Active client device id")


@router.get("", response_model=ApiResponse[list[CustomAgentRead]])
@AppAutoInjector.auto_inject()
async def list_custom_agents(
    custom_agent_service: CustomAgentService,
    user_id: UUID,
) -> ApiResponse[list[CustomAgentRead]]:
    """List the authenticated user's custom agents."""
    result = custom_agent_service.list_agents(user_id)
    return ApiResponse(success=True, message="Custom agents retrieved", data=result)


@router.post("", response_model=ApiResponse[CustomAgentRead], status_code=status.HTTP_201_CREATED)
@AppAutoInjector.auto_inject()
async def create_custom_agent(
    payload: CustomAgentCreate,
    custom_agent_service: CustomAgentService,
    user_id: UUID,
    device_id: str | None = _DEVICE_QUERY,
) -> ApiResponse[CustomAgentRead]:
    """Create a new custom agent."""
    await custom_agent_service.refresh_server_tool_catalog()
    result = custom_agent_service.create_agent(user_id, payload, device_id=device_id)
    return ApiResponse(success=True, message="Custom agent created", data=result)


@router.get("/options", response_model=ApiResponse[CustomAgentOptions])
@AppAutoInjector.auto_inject()
async def get_custom_agent_options(
    custom_agent_service: CustomAgentService,
    user_id: UUID,
    device_id: str | None = _DEVICE_QUERY,
) -> ApiResponse[CustomAgentOptions]:
    """Selectable providers, server-default tools, client tools, and skills."""
    result = await custom_agent_service.get_options(user_id, device_id=device_id)
    return ApiResponse(success=True, message="Custom agent options retrieved", data=result)


@router.get("/{custom_agent_id}", response_model=ApiResponse[CustomAgentRead])
@AppAutoInjector.auto_inject()
async def get_custom_agent(
    custom_agent_id: UUID,
    custom_agent_service: CustomAgentService,
    user_id: UUID,
) -> ApiResponse[CustomAgentRead]:
    """Get a single custom agent owned by the authenticated user."""
    result = custom_agent_service.get_agent(user_id, custom_agent_id)
    return ApiResponse(success=True, message="Custom agent retrieved", data=result)


@router.patch("/{custom_agent_id}", response_model=ApiResponse[CustomAgentRead])
@AppAutoInjector.auto_inject()
async def update_custom_agent(
    custom_agent_id: UUID,
    payload: CustomAgentUpdate,
    custom_agent_service: CustomAgentService,
    user_id: UUID,
    device_id: str | None = _DEVICE_QUERY,
) -> ApiResponse[CustomAgentRead]:
    """Update a custom agent (blocked while it is active or paused)."""
    await custom_agent_service.refresh_server_tool_catalog()
    result = custom_agent_service.update_agent(
        user_id, custom_agent_id, payload, device_id=device_id
    )
    return ApiResponse(success=True, message="Custom agent updated", data=result)


@router.delete("/{custom_agent_id}", response_model=ApiResponse[Any])
@AppAutoInjector.auto_inject()
async def delete_custom_agent(
    custom_agent_id: UUID,
    custom_agent_service: CustomAgentService,
    user_id: UUID,
) -> ApiResponse[Any]:
    """Soft-delete a custom agent and detach it from all conversations."""
    custom_agent_service.delete_agent(user_id, custom_agent_id)
    return ApiResponse(success=True, message="Custom agent deleted")


@conversation_router.get(
    "/{conversation_id}/custom-agents",
    response_model=ApiResponse[list[CustomAgentRead]],
)
@AppAutoInjector.auto_inject()
async def list_conversation_custom_agents(
    conversation_id: UUID,
    custom_agent_service: CustomAgentService,
    user_id: UUID,
) -> ApiResponse[list[CustomAgentRead]]:
    """List custom agents attached to a conversation, in attachment order."""
    result = custom_agent_service.list_conversation_agents(user_id, conversation_id)
    return ApiResponse(success=True, message="Conversation custom agents retrieved", data=result)


@conversation_router.put(
    "/{conversation_id}/custom-agents",
    response_model=ApiResponse[list[CustomAgentRead]],
)
@AppAutoInjector.auto_inject()
async def set_conversation_custom_agents(
    conversation_id: UUID,
    payload: ConversationCustomAgentsUpdate,
    custom_agent_service: CustomAgentService,
    user_id: UUID,
) -> ApiResponse[list[CustomAgentRead]]:
    """Replace the ordered set of custom agents attached to a conversation."""
    result = custom_agent_service.set_conversation_agents(user_id, conversation_id, payload)
    return ApiResponse(success=True, message="Conversation custom agents updated", data=result)
