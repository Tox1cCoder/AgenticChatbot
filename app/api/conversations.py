from typing import Any, List
from uuid import UUID
from fastapi import APIRouter, status, Query

from langchain_core.messages import ToolMessage

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.conversation_service_interface import IConversationService
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationRead,
)
from app.interfaces.message_service_interface import IMessageService
from app.schemas.message import MessageRead, MessageResumeRequest
from app.schemas.responses import ApiResponse
from app.schemas.responses.paginated_response import PaginatedApiResponse
from app.schemas.pagination import ConversationPaginationParams, MessagePaginationParams

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post(
    "/",
    response_model=ApiResponse[ConversationRead],
    status_code=status.HTTP_201_CREATED,
)
@AppAutoInjector.auto_inject()
async def create_conversation(
    conversation_data: ConversationCreate,
    conversation_service: IConversationService,
    user_id: UUID,
) -> ApiResponse[ConversationRead]:
    """Create a new conversation for authenticated user"""
    result = conversation_service.create_conversation(conversation_data, user_id)
    return ApiResponse(
        success=True, message="Conversation created successfully", data=result
    )


@router.get("/{conversation_id}", response_model=ApiResponse[ConversationRead])
@AppAutoInjector.auto_inject()
async def get_conversation(
    conversation_id: UUID,
    conversation_service: IConversationService,
) -> ApiResponse[ConversationRead]:
    """Get conversation by ID"""
    result = conversation_service.get_by_id(conversation_id)
    return ApiResponse(
        success=True, message="Conversation retrieved successfully", data=result
    )


@router.get("/", response_model=PaginatedApiResponse[ConversationRead])
@AppAutoInjector.auto_inject()
async def get_conversations(
    conversation_service: IConversationService,
    user_id: UUID,
    pagination: ConversationPaginationParams,
    include: List[str] = Query(
        default=[], description="Array of includes e.g. ['messages', 'feedback']"
    ),
    latest_messages: int = Query(
        3,
        alias="latestMessages",
        description="Number of latest messages to include",
    ),
) -> PaginatedApiResponse[ConversationRead]:
    """Get all conversations for authenticated user"""
    paginated_result = conversation_service.get_by_user_id(
        user_id,
        page=pagination.page,
        limit=pagination.limit,
        order_by=pagination.order_by.to_snake_case(),
        order_direction=pagination.order_direction.value,
        include=include,
        latest_messages=latest_messages,
    )
    return PaginatedApiResponse.from_paginator(
        paginated_result, "Conversations retrieved successfully"
    )


@router.get(
    "/{conversation_id}/messages",
    response_model=PaginatedApiResponse[MessageRead],
)
@AppAutoInjector.auto_inject()
async def get_conversation_messages(
    conversation_id: UUID,
    message_service: IMessageService,
    user_id: UUID,
    pagination: MessagePaginationParams,
    include: List[str] = Query(
        default=[], description="Array of includes e.g. ['feedback']"
    ),
) -> PaginatedApiResponse[MessageRead]:
    """Get conversation's messages (requires user ownership)"""
    include_feedback = "feedback" in include
    paginated_result = message_service.get_conversation_messages(
        conversation_id,
        user_id,
        page=pagination.page,
        limit=pagination.limit,
        order_by=pagination.order_by.to_snake_case(),
        order_direction=pagination.order_direction.value,
        include_feedback=include_feedback,
    )
    return PaginatedApiResponse.from_paginator(
        paginated_result, "Conversation messages retrieved successfully"
    )


@router.patch("/{conversation_id}", response_model=ApiResponse[ConversationRead])
@AppAutoInjector.auto_inject()
async def update_conversation(
    conversation_id: UUID,
    conversation_data: ConversationUpdate,
    conversation_service: IConversationService,
    user_id: UUID,
) -> ApiResponse[ConversationRead]:
    """Update conversation (requires user ownership)"""
    result = conversation_service.update_conversation(
        conversation_id, user_id, conversation_data
    )
    return ApiResponse(
        success=True, message="Conversation updated successfully", data=result
    )


@router.delete("/{conversation_id}", response_model=ApiResponse[Any])
@AppAutoInjector.auto_inject()
async def delete_conversation(
    conversation_id: UUID,
    conversation_service: IConversationService,
    user_id: UUID,
) -> ApiResponse[Any]:
    """Delete conversation (requires user ownership)"""
    conversation_service.delete_conversation(conversation_id, user_id)
    return ApiResponse(success=True, message="Conversation deleted successfully")


@router.post("/{conversation_id}/resume", response_model=ApiResponse[MessageRead])
@AppAutoInjector.auto_inject()
async def resume_conversation_workflow(
    conversation_id: UUID,
    resume_data: MessageResumeRequest,
    message_service: IMessageService,
    user_id: UUID,
) -> ApiResponse[MessageRead]:
    """
    Resume a workflow that was interrupted for human approval.

    This endpoint is called after the workflow has been paused (e.g., for tool execution approval).
    It resumes the workflow and returns the final bot response.
    """
    # Handle rejection
    if not resume_data.approved:
        # Get the current workflow state to extract pending tool calls
        thread_id = str(conversation_id)
        state_info = await message_service.ai_service.workflow.get_state(thread_id)
        pending_tool_calls = state_info.get("pending_tool_calls", [])
        
        # Create rejection ToolMessages for each pending tool call
        rejection_messages = []
        rejection_reason = resume_data.rejection_reason or "User declined tool execution"
        
        for tool_call in pending_tool_calls:
            tool_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
            tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", "unknown")
            
            # Create a ToolMessage indicating rejection
            rejection_messages.append(
                ToolMessage(
                    content=f"Tool execution declined by user. Reason: {rejection_reason}",
                    tool_call_id=tool_id,
                    name=tool_name,
                    status="error",
                )
            )
        
        # Resume workflow with rejection messages so LLM can generate appropriate response
        bot_message = await message_service.resume_workflow(
            conversation_id=conversation_id,
            user_id=user_id,
            user_input=None,
            rejection_messages=rejection_messages,
        )
        
        return ApiResponse(
            success=True,
            message="Tool execution rejected, LLM generated response",
            data=bot_message
        )
    
    # Handle approval - resume the workflow
    user_input = resume_data.user_input if resume_data.user_input else None

    # Resume the workflow through message service
    bot_message = await message_service.resume_workflow(
        conversation_id=conversation_id, user_id=user_id, user_input=user_input
    )

    return ApiResponse(
        success=True, message="Workflow resumed successfully", data=bot_message
    )
