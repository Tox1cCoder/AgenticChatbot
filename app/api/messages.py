from typing import List
from uuid import UUID
import json
from fastapi import APIRouter, status, Query, Response
from fastapi.responses import StreamingResponse

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.message_service_interface import IMessageService
from app.schemas.message import MessageCreate, MessageRead, InterruptResumeRequest
from app.schemas.responses import ApiResponse
from app.schemas.responses.paginated_response import PaginatedApiResponse
from app.schemas.pagination import MessagePaginationParams

router = APIRouter(prefix="/messages", tags=["messages"])


@router.post(
    "/",
    response_model=ApiResponse[MessageRead],
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Message created successfully"},
        202: {"description": "Message created, tool execution requires approval"},
    },
)
@AppAutoInjector.auto_inject()
async def create_message(
    message_data: MessageCreate,
    message_service: IMessageService,
    response: Response,
) -> ApiResponse[MessageRead]:
    """
    Create a new message.

    If tool execution requires human approval, returns HTTP 202 Accepted with interrupt
    details in the response data's interrupt field. Client should then call
    /messages/resume-interrupt with approval decisions.

    Otherwise, returns HTTP 201 Created with the completed message.
    """
    result = await message_service.create_message(message_data)

    # Check if result contains interrupt information
    if result.interrupt:
        response.status_code = status.HTTP_202_ACCEPTED
        return ApiResponse(
            success=True,
            message="Tool execution requires approval",
            data=result,
        )

    return ApiResponse(
        success=True, message="Message created successfully", data=result
    )


@router.post("/stream", status_code=status.HTTP_200_OK)
@AppAutoInjector.auto_inject()
async def create_message_stream(
    message_data: MessageCreate,
    message_service: IMessageService,
):
    """
    Create a new message and stream the bot response.

    Streams Server-Sent Events (SSE) with the following event types:
    - user_message_created: User message was persisted
    - token: Incremental response token
    - tool: Tool execution event (status: start/end)
    - interrupt: Tool execution requires approval - stream will close after this event.
      Client must inspect the 'interrupt' field and call /messages/resume-interrupt.
    - complete: Final response with full message data
    - error: Error occurred during processing

    When an interrupt event is received, the stream closes. Client should handle the
    interrupt by calling /messages/resume-interrupt with approval decisions.
    """

    async def event_generator():
        """Generate Server-Sent Events (SSE) from the message stream"""
        try:
            async for event in message_service.create_message_stream(message_data):
                event_type = event.get("type")

                # Format as SSE: data: {json}\n\n
                event_json = json.dumps(event)
                yield f"data: {event_json}\n\n"

                if event_type in ["complete", "error", "interrupt"]:
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


@router.post(
    "/resume-interrupt",
    response_model=ApiResponse[MessageRead],
    status_code=status.HTTP_200_OK,
)
@AppAutoInjector.auto_inject()
async def resume_interrupt(
    resume_request: InterruptResumeRequest,
    message_service: IMessageService,
) -> ApiResponse[MessageRead]:
    """
    Resume execution after handling tool execution interrupts.

    This endpoint should be called after receiving an interrupt response from
    POST /messages or POST /messages/stream. Provide the thread_id from the
    interrupt response along with approval/rejection/edit decisions for each
    tool that was awaiting approval.
    """
    result = await message_service.resume_message_creation(
        thread_id=resume_request.thread_id,
        conversation_id=resume_request.conversation_id,
        interrupt_id=resume_request.interrupt_id,
        decisions=resume_request.decisions,
    )
    return ApiResponse(
        success=True,
        message="Message creation resumed successfully",
        data=result,
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
