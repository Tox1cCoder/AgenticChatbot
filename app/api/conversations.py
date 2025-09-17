from typing import List
from uuid import UUID
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status

from dependency_injector.wiring import Provide, inject

from app.core.container import Container
from app.core.auth import get_current_user_id
from app.interfaces.conversation_service_interface import IConversationService
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationRead,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post(
    "/",
    response_model=ConversationRead,
    status_code=status.HTTP_201_CREATED,
)
@inject
async def create_conversation(
    conversation_data: ConversationCreate,
    user_id: UUID = Depends(get_current_user_id),
    conversation_service: Annotated[
        IConversationService, Depends(Provide[Container.conversation_service])
    ] = None,
) -> ConversationRead:
    """Create a new conversation for authenticated user"""
    return conversation_service.create_conversation(conversation_data, user_id)


@router.get("/{conversation_id}", response_model=ConversationRead)
@inject
async def get_conversation(
    conversation_id: UUID,
    conversation_service: Annotated[
        IConversationService, Depends(Provide[Container.conversation_service])
    ],
) -> ConversationRead:
    """Get conversation by ID"""
    return conversation_service.get_by_id(conversation_id)


@router.get("/", response_model=List[ConversationRead])
@inject
async def get_conversations(
    conversation_service: Annotated[
        IConversationService, Depends(Provide[Container.conversation_service])
    ],
    user_id: UUID = Depends(get_current_user_id),
    page: int = 1,
    limit: int = 100,
) -> List[ConversationRead]:
    """Get all conversations for authenticated user"""
    skip = (page - 1) * limit
    return conversation_service.get_user_conversations(user_id, skip=skip, limit=limit)


# @router.put("/{conversation_id}", response_model=ConversationRead)
# @inject
# async def update_conversation(
#     conversation_id: UUID,
#     conversation_data: ConversationUpdate,
#     conversation_service: Annotated[
#         ConversationService, Depends(Provide[Container.conversation_service])
#     ],
#     user_id: UUID = Depends(get_current_user_id),
# ) -> ConversationRead:
#     """Update conversation (requires user ownership)"""
#     return conversation_service.update_conversation(
#         conversation_id, user_id, conversation_data
#     )


@router.delete("/{conversation_id}")
@inject
async def delete_conversation(
    conversation_id: UUID,
    conversation_service: Annotated[
        IConversationService, Depends(Provide[Container.conversation_service])
    ],
    user_id: UUID = Depends(get_current_user_id),
) -> dict:
    """Delete conversation (requires user ownership)"""
    success = conversation_service.delete_conversation(conversation_id, user_id)
    if success:
        return {"message": "Conversation deleted successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
