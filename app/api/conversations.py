from typing import List
from uuid import UUID
from typing import Annotated
from fastapi import APIRouter, Depends, status, HTTPException

from dependency_injector.wiring import Provide, inject

from app.core.container import Container
from app.core.auth import get_current_user_id
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
from app.utils.validation.pagination_validation import PaginationError

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post(
    "/",
    response_model=ApiResponse[ConversationRead],
    status_code=status.HTTP_201_CREATED,
)
@inject
async def create_conversation(
    conversation_data: ConversationCreate,
    user_id: UUID = Depends(get_current_user_id),
    conversation_service: Annotated[
        IConversationService, Depends(Provide[Container.conversation_service])
    ] = None,
) -> ApiResponse[ConversationRead]:
    """Create a new conversation for authenticated user"""
    result = conversation_service.create_conversation(conversation_data, user_id)
    return ApiResponse(
        success=True, message="Conversation created successfully", data=result
    )


@router.get("/{conversation_id}", response_model=ApiResponse[ConversationRead])
@inject
async def get_conversation(
    conversation_id: UUID,
    conversation_service: Annotated[
        IConversationService, Depends(Provide[Container.conversation_service])
    ],
) -> ApiResponse[ConversationRead]:
    """Get conversation by ID"""
    result = conversation_service.get_by_id(conversation_id)
    return ApiResponse(
        success=True, message="Conversation retrieved successfully", data=result
    )


@router.get("/", response_model=PaginatedApiResponse[ConversationRead])
@inject
async def get_conversations(
    conversation_service: Annotated[
        IConversationService, Depends(Provide[Container.conversation_service])
    ],
    user_id: UUID = Depends(get_current_user_id),
    pagination: ConversationPaginationParams = Depends(),
) -> PaginatedApiResponse[ConversationRead]:
    """Get all conversations for authenticated user"""
    paginated_result = conversation_service.get_user_conversations(
        user_id,
        page=pagination.page,
        limit=pagination.limit,
        order_by=pagination.order_by.value,
        order_direction=pagination.order_direction.value,
    )
    return PaginatedApiResponse.from_paginator(
        paginated_result, "Conversations retrieved successfully"
    )


@router.get(
    "/{conversation_id}/messages",
    response_model=PaginatedApiResponse[MessageRead],
)
@inject
async def get_conversation_messages(
    conversation_id: UUID,
    message_service: Annotated[
        IMessageService, Depends(Provide[Container.message_service])
    ],
    user_id: UUID = Depends(get_current_user_id),
    pagination: MessagePaginationParams = Depends(),
) -> PaginatedApiResponse[MessageRead]:
    """Get conversation's messages (requires user ownership)"""
    paginated_result = message_service.get_conversation_messages(
        conversation_id,
        user_id,
        page=pagination.page,
        limit=pagination.limit,
        order_by=pagination.order_by.value,
        order_direction=pagination.order_direction.value,
    )
    return PaginatedApiResponse.from_paginator(
        paginated_result, "Conversation messages retrieved successfully"
    )


@router.delete("/{conversation_id}", response_model=ApiResponse)
@inject
async def delete_conversation(
    conversation_id: UUID,
    conversation_service: Annotated[
        IConversationService, Depends(Provide[Container.conversation_service])
    ],
    user_id: UUID = Depends(get_current_user_id),
) -> ApiResponse:
    """Delete conversation (requires user ownership)"""
    conversation_service.delete_conversation(conversation_id, user_id)
    return ApiResponse(success=True, message="Conversation deleted successfully")
