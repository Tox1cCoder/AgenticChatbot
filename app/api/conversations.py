from typing import List
from uuid import UUID
from typing import Annotated
from fastapi import APIRouter, Depends, status

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
        success=True,
        code="ok",
        message="Conversation created successfully",
        data=result
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
        success=True,
        code="ok",
        message="Conversation retrieved successfully",
        data=result
    )


@router.get("/", response_model=ApiResponse[List[ConversationRead]])
@inject
async def get_conversations(
    conversation_service: Annotated[
        IConversationService, Depends(Provide[Container.conversation_service])
    ],
    user_id: UUID = Depends(get_current_user_id),
    page: int = 1,
    limit: int = 100,
) -> ApiResponse[List[ConversationRead]]:
    """Get all conversations for authenticated user"""
    skip = (page - 1) * limit
    result = conversation_service.get_user_conversations(
        user_id, skip=skip, limit=limit
    )
    return ApiResponse(
        success=True,
        code="ok",
        message="Conversations retrieved successfully",
        data=result
    )


@router.get(
    "/{conversation_id}/messages",
    response_model=ApiResponse[List[MessageRead]],
)
@inject
async def get_conversation_thread(
    conversation_id: UUID,
    message_service: Annotated[
        IMessageService, Depends(Provide[Container.message_service])
    ],
    user_id: UUID = Depends(get_current_user_id),
) -> ApiResponse[List[MessageRead]]:
    """Get conversation's messages by timestamp (requires user ownership)"""
    result = message_service.get_conversation_thread(conversation_id, user_id)
    return ApiResponse(
        success=True,
        code="ok",
        message="Conversation thread retrieved successfully",
        data=result
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
    return ApiResponse(
        success=True,
        code="ok",
        message="Conversation deleted successfully"
    )
