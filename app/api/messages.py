from typing import List
from uuid import UUID
from typing import Annotated
from fastapi import APIRouter, status, HTTPException

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.message_service_interface import IMessageService
from app.schemas.message import MessageCreate, MessageRead
from app.schemas.responses import ApiResponse
from app.schemas.responses.paginated_response import PaginatedApiResponse
from app.schemas.pagination import MessagePaginationParams
from app.utils.validation.pagination_validation import PaginationError

router = APIRouter(prefix="/messages", tags=["messages"])


@router.post(
    "/", response_model=ApiResponse[MessageRead], status_code=status.HTTP_201_CREATED
)
@AppAutoInjector.auto_inject()
async def create_message(
    message_data: MessageCreate,
    message_service: IMessageService,
) -> ApiResponse[MessageRead]:
    """Create a new message"""
    result = message_service.create_message(message_data)
    return ApiResponse(
        success=True, message="Message created successfully", data=result
    )


@router.get("/{message_id}", response_model=ApiResponse[MessageRead])
@AppAutoInjector.auto_inject()
async def get_message(
    message_id: UUID,
    message_service: IMessageService,
    user_id: UUID,
) -> ApiResponse[MessageRead]:
    """Get message by ID"""
    result = message_service.get_by_id(message_id, user_id)
    return ApiResponse(
        success=True, message="Message retrieved successfully", data=result
    )


@router.get("/", response_model=PaginatedApiResponse[MessageRead])
@AppAutoInjector.auto_inject()
async def get_user_messages(
    message_service: IMessageService,
    user_id: UUID,
    pagination: MessagePaginationParams,
) -> PaginatedApiResponse[MessageRead]:
    """Get all messages for authenticated user with pagination"""
    paginated_result = message_service.get_user_messages(
        user_id,
        page=pagination.page,
        limit=pagination.limit,
        order_by=pagination.order_by.to_snake_case(),
        order_direction=pagination.order_direction.value,
    )
    return PaginatedApiResponse.from_paginator(
        paginated_result, "User messages retrieved successfully"
    )
