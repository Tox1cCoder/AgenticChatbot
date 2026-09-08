import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncGenerator, Callable
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from app.core.dependency_injection import AppAutoInjector
from app.core.exceptions import CustomHTTPException
from app.interfaces.message_service_interface import IMessageService
from app.schemas.message import (
    ContinueGenerationRequest,
    GenerationSnapshotResponse,
    InterruptResumeRequest,
    MessageCreate,
    MessageRead,
    StopGenerationRequest,
)
from app.schemas.pagination import MessagePaginationParams
from app.schemas.responses import ApiResponse
from app.schemas.responses.paginated_response import PaginatedApiResponse
from app.services.event_streaming.events import V3StreamEvent
from app.services.event_streaming.internal_sse import legacy_event_from_v3

logger = logging.getLogger(__name__)

# Heartbeat interval in seconds (MVP: 1.0s for Streamlit responsiveness)
HEARTBEAT_INTERVAL_SECONDS = 1.0

router = APIRouter(prefix="/messages", tags=["messages"])


def _to_internal_sse_event(event: dict | V3StreamEvent) -> dict | None:
    """Project a service-layer event to the legacy Streamlit JSON SSE shape.

    Canonical ``V3StreamEvent`` is mapped via ``legacy_event_from_v3``; legacy
    dicts (pre-migration) pass through unchanged.
    """
    if isinstance(event, V3StreamEvent):
        return legacy_event_from_v3(event)
    return event


def _stream_error_event(exc: Exception) -> dict:
    """Project known stream exceptions to the internal SSE error shape."""
    error_event: dict = {"type": "error", "error": str(exc)}
    if isinstance(exc, CustomHTTPException):
        error_event["error"] = str(exc.detail)
        error_event["status_code"] = exc.status_code
        if exc.error_code is not None:
            error_event["error_code"] = exc.error_code
    return error_event


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
                await queue.put(_stream_error_event(exc))
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

                public_event = _to_internal_sse_event(event)
                if public_event is None:
                    continue

                event_type = public_event.get("type")
                yield f"data: {json.dumps(public_event)}\n\n"

                if event_type in ("complete", "error", "interrupt"):
                    break

        except asyncio.CancelledError:
            return
        except Exception as exc:
            yield f"data: {json.dumps(_stream_error_event(exc))}\n\n"
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
    responses={202: {"description": "Stop accepted; the worker has not confirmed yet"}},
)
@AppAutoInjector.auto_inject()
async def stop_message_generation(
    stop_request: StopGenerationRequest,
    message_service: IMessageService,
    user_id: UUID,
    response: Response,
) -> ApiResponse:
    """
    Request cancellation of an in-flight streaming generation.

    Idempotent: calling stop multiple times is safe. ``status`` distinguishes
    three outcomes -- ``cancelled`` (the producer confirmed and its partial is
    persisted), ``stop_requested`` (asked, not yet confirmed; poll
    ``GET /messages/generations/{id}``), and ``not_inflight`` (already finished,
    or never held here) so the UI can refresh messages normally.

    ``202`` accompanies ``stop_requested`` and ``200`` everything else, so a
    client can tell "settled" from "accepted" without parsing the body. The
    worker may be mid-provider-call in another process; reporting a stop it has
    not confirmed is the one thing this endpoint must never do.
    """
    result = await _stop_generation_result(message_service, stop_request, user_id)

    # Each status gets its own sentence. Reporting a pending stop as "stopped"
    # is the claim this endpoint must not make.
    messages = {
        "cancelled": "Generation stopped",
        "stop_requested": "Stop requested; the generation has not confirmed yet",
        "not_inflight": "Generation not in flight",
    }
    status_name = str(result.get("status"))
    if status_name == "stop_requested":
        # 202: accepted, not yet settled. A client polls the snapshot endpoint
        # rather than assuming the turn is over.
        response.status_code = status.HTTP_202_ACCEPTED
    return ApiResponse(
        success=True,
        message=messages.get(status_name, "Generation not in flight"),
        data=result,
    )


async def _stop_generation_result(
    message_service: IMessageService,
    stop_request: StopGenerationRequest,
    user_id: UUID,
) -> dict:
    """Dispatch a Stop by whichever identity the client supplied.

    A client that sends ``generationId`` gets the fenced command directly. One
    that sends only ``userMessageId`` predates ``run_start``, so the turn-scoped
    entry point resolves it through the logical turn and derives the fence from
    the row — the endpoint cannot invent a fence the client never held.
    """
    if stop_request.generation_id is None:
        return await message_service.stop_message_generation(
            conversation_id=stop_request.conversation_id,
            user_id=user_id,
            user_message_id=stop_request.user_message_id,
        )

    expected_version = stop_request.expected_version
    if expected_version is None:
        snapshot = await message_service.aget_generation(
            generation_id=stop_request.generation_id,
            conversation_id=stop_request.conversation_id,
            user_id=user_id,
        )
        if snapshot is None:
            return {"status": "not_inflight", "message": None}
        expected_version = snapshot.version

    settled = await message_service.stop_generation(
        generation_id=stop_request.generation_id,
        conversation_id=stop_request.conversation_id,
        user_id=user_id,
        idempotency_key=(
            stop_request.idempotency_key or f"stop-{stop_request.generation_id}-{expected_version}"
        ),
        expected_version=expected_version,
    )
    return message_service._legacy_stop_result(settled, user_id=user_id)


@router.get(
    "/generations/{generation_id}",
    response_model=ApiResponse[GenerationSnapshotResponse],
    status_code=status.HTTP_200_OK,
)
@AppAutoInjector.auto_inject()
async def get_generation(
    generation_id: UUID,
    conversation_id: UUID,
    message_service: IMessageService,
    user_id: UUID,
) -> ApiResponse:
    """The authoritative lifecycle state of one generation.

    What a client polls after a ``202`` Stop, and what a reconnecting client
    reads to find out whether the turn it lost is running, finished, or waiting
    to be continued. A closed socket is not evidence about any of that.

    A generation that is not this owner's is a 404, identical to one that does
    not exist: "it exists but is not yours" is information about another user's
    conversation.
    """
    snapshot = await message_service.aget_generation(
        generation_id=generation_id,
        conversation_id=conversation_id,
        user_id=user_id,
    )
    if snapshot is None:
        raise CustomHTTPException(
            status_code=404,
            detail="Generation not found.",
            error_code="GENERATION_NOT_FOUND",
        )
    return ApiResponse(
        success=True,
        message="Generation status",
        data=GenerationSnapshotResponse(**snapshot.model_dump()),
    )


@router.post(
    "/continue",
    status_code=status.HTTP_200_OK,
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Internal SSE stream for a continued generation",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
@AppAutoInjector.auto_inject()
async def continue_generation(
    continue_request: ContinueGenerationRequest,
    message_service: IMessageService,
    user_id: UUID,
    request: Request,
):
    """Continue a paused turn, streaming the next epoch.

    Not a new turn: no user message is appended and the router is not consulted,
    so the specialist the turn already chose is the one that resumes with the
    evidence it already gathered. ``continuationId`` is single-use, so a
    replayed Continue is refused rather than opening a second epoch.
    """
    return _internal_event_stream_response(
        lambda: message_service.continue_message_generation_stream(
            generation_id=continue_request.generation_id,
            continuation_id=continue_request.continuation_id,
            conversation_id=continue_request.conversation_id,
            user_id=user_id,
            idempotency_key=continue_request.idempotency_key,
            expected_version=continue_request.expected_version,
            inline_rich_response_v1=continue_request.inline_rich_response_v1,
        ),
        request,
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
            device_id=resume_request.device_id,
            decisions=resume_request.decisions,
            inline_rich_response_v1=resume_request.inline_rich_response_v1,
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
