from typing import List
from uuid import UUID
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status

from dependency_injector.wiring import Provide, inject

from app.core.container import Container
from app.core.auth import get_current_user_id
from app.services.message_service import MessageService
from app.schemas.message import MessageCreate, MessageUpdate, MessageRead

router = APIRouter(prefix="/messages", tags=["messages"])


@router.post("/", response_model=MessageRead, status_code=status.HTTP_201_CREATED)
@inject
async def create_message(
    message_data: MessageCreate,
    message_service: Annotated[
        MessageService, Depends(Provide[Container.message_service])
    ],
) -> MessageRead:
    """Create a new message"""
    return message_service.create_message(message_data)


@router.get("/{message_id}", response_model=MessageRead)
@inject
async def get_message(
    message_id: UUID,
    message_service: Annotated[
        MessageService, Depends(Provide[Container.message_service])
    ],
) -> MessageRead:
    """Get message by ID"""
    return message_service.get_message_by_id(message_id)


# @router.get(
#     "/conversations/{conversation_id}/messages", response_model=List[MessageRead]
# )
# @inject
# async def get_conversation_messages(
#     conversation_id: UUID,
#     message_service: Annotated[
#         MessageService, Depends(Provide[Container.message_service])
#     ],
#     user_id: UUID = Depends(get_current_user_id),
#     page: int = 1,
#     limit: int = 100,
# ) -> List[MessageRead]:
#     """Get messages for a conversation (requires user ownership)"""
#     skip = (page - 1) * limit
#     return message_service.get_conversation_messages(
#         conversation_id, user_id, skip=skip, limit=limit
#     )


@router.get(
    "/conversations/{conversation_id}/messages/thread", response_model=List[MessageRead]
)
@inject
async def get_conversation_thread(
    conversation_id: UUID,
    message_service: Annotated[
        MessageService, Depends(Provide[Container.message_service])
    ],
    user_id: UUID = Depends(get_current_user_id),
) -> List[MessageRead]:
    """Get conversation thread ordered by timestamp (requires user ownership)"""
    return message_service.get_conversation_thread(conversation_id, user_id)
