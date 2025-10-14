from typing import List
from uuid import UUID
from fastapi import APIRouter, status, Query

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.conversation_service_interface import IConversationService
from app.schemas.conversation import (
    ConversationCreate,
    ConversationRead,
)
from app.interfaces.message_service_interface import IMessageService
from app.schemas.message import MessageRead
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
) -> PaginatedApiResponse[MessageRead]:
    """Get conversation's messages (requires user ownership)"""
    paginated_result = message_service.get_conversation_messages(
        conversation_id,
        user_id,
        page=pagination.page,
        limit=pagination.limit,
        order_by=pagination.order_by.to_snake_case(),
        order_direction=pagination.order_direction.value,
    )
    return PaginatedApiResponse.from_paginator(
        paginated_result, "Conversation messages retrieved successfully"
    )


@router.delete("/{conversation_id}", response_model=ApiResponse)
@AppAutoInjector.auto_inject()
async def delete_conversation(
    conversation_id: UUID,
    conversation_service: IConversationService,
    user_id: UUID,
) -> ApiResponse:
    """Delete conversation (requires user ownership)"""
    conversation_service.delete_conversation(conversation_id, user_id)
    return ApiResponse(success=True, message="Conversation deleted successfully")
