import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncGenerator, Callable
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.message_service_interface import IMessageService
from app.schemas.message import (
    InterruptResumeRequest,
    MessageCreate,
    MessageRead,
    StopGenerationRequest,
)
from app.schemas.pagination import MessagePaginationParams
from app.schemas.responses import ApiResponse
from app.schemas.responses.paginated_response import PaginatedApiResponse

logger = logging.getLogger(__name__)

# Heartbeat interval in seconds (MVP: 1.0s for Streamlit responsiveness)
HEARTBEAT_INTERVAL_SECONDS = 1.0

router = APIRouter(prefix="/messages", tags=["messages"])


def _internal_event_stream_response(
    producer_factory: Callable[[], AsyncGenerator[dict, None]],
    request: Request,
) -> StreamingResponse:
    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()

        async def producer():
            try:
                async for event in producer_factory():
                    await queue.put(event)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                await queue.put({"type": "error", "error": str(exc)})
            finally:
                await queue.put(None)

        task = asyncio.create_task(producer())
        try:
            while True:
                if await request.is_disconnected():
                    logger.debug("Client disconnected during SSE stream")
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_INTERVAL_SECONDS)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    continue

                if event is None:
                    break

                event_type = event.get("type")
                yield f"data: {json.dumps(event)}\n\n"

                if event_type in ("complete", "error", "interrupt"):
                    break

        except asyncio.CancelledError:
            return
        except Exception as exc:
            error_event = {"type": "error", "error": str(exc)}
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
    user_id: UUID,
) -> ApiResponse[MessageRead]:
    """
    Create a new message.
    """
    result = await message_service.create_message(message_data, user_id)

    # Check if result contains interrupt information
    if result.interrupt:
        response.status_code = status.HTTP_202_ACCEPTED
        return ApiResponse(
            success=True,
            message="Tool execution requires approval",
            data=result,
        )

    return ApiResponse(success=True, message="Message created successfully", data=result)


@router.post("/stream", status_code=status.HTTP_200_OK)
@AppAutoInjector.auto_inject()
async def create_message_stream(
    message_data: MessageCreate,
    message_service: IMessageService,
    user_id: UUID,
    request: Request,
):
    """
    Create a new message and stream the bot response.

    Uses an asyncio.Queue + producer-task pattern so that heartbeat events can
    be emitted even while the service-layer generator is blocked (e.g. waiting
    for a long tool call).  This keeps the SSE connection alive and gives
    Streamlit frequent yield-points for a responsive "Stop generating" UX.
    """
    return _internal_event_stream_response(
        lambda: message_service.create_message_stream(message_data, user_id),
        request,
    )


@router.post(
    "/stop",
    response_model=ApiResponse,
    status_code=status.HTTP_200_OK,
)
@AppAutoInjector.auto_inject()
async def stop_message_generation(
    stop_request: StopGenerationRequest,
    message_service: IMessageService,
    user_id: UUID,
) -> ApiResponse:
    """
    Request cancellation of an in-flight streaming generation.

    Idempotent: calling stop multiple times is safe.  If the generation has
    already completed, returns ``status: "not_inflight"`` so the UI can
    refresh messages normally.
    """
    result = await message_service.stop_message_generation(
        conversation_id=stop_request.conversation_id,
        user_id=user_id,
        user_message_id=stop_request.user_message_id,
    )
    return ApiResponse(
        success=True,
        message=(
            "Generation stopped"
            if result.get("status") == "cancelled"
            else "Generation not in flight"
        ),
        data=result,
    )


@router.post(
    "/resume-interrupt",
    status_code=status.HTTP_200_OK,
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Internal SSE stream for resumed approval flow",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
@AppAutoInjector.auto_inject()
async def resume_interrupt(
    resume_request: InterruptResumeRequest,
    message_service: IMessageService,
    user_id: UUID,
    request: Request,
):
    """
    Resume execution after handling tool execution interrupts and stream the result.
    """
    return _internal_event_stream_response(
        lambda: message_service.resume_message_creation_stream(
            thread_id=resume_request.thread_id,
            conversation_id=resume_request.conversation_id,
            user_id=user_id,
            interrupt_id=resume_request.interrupt_id,
            decisions=resume_request.decisions,
        ),
        request,
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
    return ApiResponse(success=True, message="Message retrieved successfully", data=result)


@router.get("/", response_model=PaginatedApiResponse[MessageRead])
@AppAutoInjector.auto_inject()
async def get_user_messages(
    message_service: IMessageService,
    user_id: UUID,
    pagination: MessagePaginationParams,
    include: list[str] = Query(default=[], description="Array of includes e.g. ['feedback']"),  # noqa: B008
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
