from typing import List
from uuid import UUID
from typing import Annotated
from fastapi import APIRouter, Depends, status

from dependency_injector.wiring import Provide, inject

from app.core.container import Container
from app.core.auth import get_current_user_id
from app.interfaces.message_service_interface import IMessageService
from app.schemas.message import MessageCreate, MessageRead
from app.schemas.responses import ApiResponse

router = APIRouter(prefix="/messages", tags=["messages"])


@router.post(
    "/", response_model=ApiResponse[MessageRead], status_code=status.HTTP_201_CREATED
)
@inject
async def create_message(
    message_data: MessageCreate,
    message_service: Annotated[
        IMessageService, Depends(Provide[Container.message_service])
    ],
) -> ApiResponse[MessageRead]:
    """Create a new message"""
    result = message_service.create_message(message_data)
    return ApiResponse(
        success=True,
        code="ok",
        message="Message created successfully",
        data=result
    )

@router.get("/{message_id}", response_model=ApiResponse[MessageRead])
@inject
async def get_message(
    message_id: UUID,
    message_service: Annotated[
        IMessageService, Depends(Provide[Container.message_service])
    ],
    user_id: UUID = Depends(get_current_user_id),
) -> ApiResponse[MessageRead]:
    """Get message by ID"""
    result = message_service.get_by_id(message_id, user_id)
    return ApiResponse(
        success=True,
        code="ok",
        message="Message retrieved successfully",
        data=result
    )