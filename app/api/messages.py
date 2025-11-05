from typing import List
from uuid import UUID
import json
from fastapi import APIRouter, status, Query
from fastapi.responses import StreamingResponse

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.message_service_interface import IMessageService
from app.schemas.message import MessageCreate, MessageRead
from app.schemas.responses import ApiResponse
from app.schemas.responses.paginated_response import PaginatedApiResponse
from app.schemas.pagination import MessagePaginationParams

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
    result = await message_service.create_message(message_data)
    return ApiResponse(
        success=True, message="Message created successfully", data=result
    )


@router.post("/stream", status_code=status.HTTP_200_OK)
@AppAutoInjector.auto_inject()
async def create_message_stream(
    message_data: MessageCreate,
    message_service: IMessageService,
):
    """Create a new message and stream the bot response"""

    async def event_generator():
        """Generate Server-Sent Events (SSE) from the message stream"""
        try:
            async for event in message_service.create_message_stream(message_data):
                event_type = event.get("type")

                # Format as SSE: data: {json}\n\n
                event_json = json.dumps(event)
                yield f"data: {event_json}\n\n"

                if event_type in ["complete", "error"]:
                    break

        except Exception as exc:
            # Send error event
            error_event = {"type": "error", "error": str(exc)}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
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
    include: List[str] = Query(
        default=[], description="Array of includes e.g. ['feedback']"
    ),
) -> PaginatedApiResponse[MessageRead]:
    """Get all messages for authenticated user with pagination"""
    include_feedback = "feedback" in include
    paginated_result = message_service.get_user_messages(
        user_id,
        page=pagination.page,
        limit=pagination.limit,
        order_by=pagination.order_by.to_snake_case(),
        order_direction=pagination.order_direction.value,
        include_feedback=include_feedback,
    )
    return PaginatedApiResponse.from_paginator(
        paginated_result, "User messages retrieved successfully"
    )
