import base64
from collections.abc import AsyncGenerator, Callable
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.ai.image_context import normalize_image_attachment
from app.core.config import settings
from app.core.dependency_injection import AppAutoInjector
from app.interfaces.conversation_service_interface import IConversationService
from app.interfaces.message_service_interface import IMessageService
from app.models.enums import MessageRole
from app.repositories.utils.pagination import PaginationMeta
from app.schemas.conversation import (
    ConversationCreate,
    ConversationRead,
    ConversationUpdate,
)
from app.schemas.message import (
    ContinueGenerationRequest,
    InterruptResumeRequest,
    MessageCreate,
)
from app.schemas.pagination import ConversationPaginationParams, MessagePaginationParams
from app.schemas.responses import ApiResponse
from app.schemas.responses.paginated_response import PaginatedApiResponse
from app.services.event_streaming.ai_sdk_projection import (
    attach_image_parts_to_message,
    ensure_leading_text_part,
    is_v1_rich_items_message,
    project_ai_sdk_message_for_capability,
    scrub_legacy_metadata,
    visible_image_file_parts,
)
from app.services.event_streaming.ai_sdk_v6 import (
    AISDKV6StreamAdapter,
)
from app.services.event_streaming.ai_sdk_v6 import (
    AISDKV6StreamState as StreamState,
)

router = APIRouter(tags=["ai-sdk"])


class AISDKChatRequest(BaseModel):
    """
    - messages: the UI message history
    - message: optional latest UI message for custom AI SDK transports
    - userId: optional (enables server-side memory features)
    - inlineRichResponseV1: optional client capability flag for rich response v1
    """

    messages: list[dict[str, Any]] = Field(default_factory=list)
    message: Any | None = None
    content: Any | None = None
    user_id: UUID | None = Field(default=None, alias="userId")
    inline_rich_response_v1: bool = Field(
        default=False,
        alias="inlineRichResponseV1",
        description=(
            "When true, the client declares it can render the inline rich-response v1 "
            "contract: HTML-comment rich markers in assistant content, transient "
            "`data-rich-items` stream parts, and persisted `rich_items` metadata."
        ),
    )

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class AISDKUIMessage(BaseModel):
    """
    Vercel AI SDK ``UIMessage`` format.

    Compatible with the ``initialMessages`` prop of ``useChat()`` and
    ``useAssistant()``.  Each item in the ``messages`` array returned by
    ``GET /ai/conversations/{conversationId}/messages`` is one of these.

    Reference: https://sdk.vercel.ai/docs/reference/ai-sdk-ui/use-chat
    """

    id: str = Field(..., description="Unique message ID (UUID string)")
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Plain-text content of the message")
    parts: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Structured message parts (text / file / reasoning). Always present; "
            "a `text` part carrying the message content is guaranteed. Mirrors the "
            "``parts`` field of the AI SDK UIMessage spec."
        ),
    )
    metadata: dict[str, Any] | None = Field(
        None,
        description=(
            "Backend message metadata: provider/model info, RAG citations, rich items, "
            "suggested questions, and additive `context_window` model-limit and usage "
            "fields. Unknown metadata remains forward-compatible; only documented legacy "
            "renderer and database-only fields are scrubbed."
        ),
    )
    created_at: str | None = Field(
        None, alias="createdAt", description="ISO-8601 creation timestamp"
    )

    model_config = ConfigDict(populate_by_name=True)


class AISDKMessagesData(BaseModel):
    """Paginated AI SDK messages payload (returned by the messages listing endpoint)."""

    messages: list[AISDKUIMessage] = Field(
        ..., description="Messages in Vercel AI SDK UIMessage format"
    )
    meta: PaginationMeta = Field(..., description="Pagination metadata (includes `total`)")


def _extract_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        role = msg.get("role")
        if role != "user":
            continue

        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content

        # Some UIs serialize message parts into `content` as an array.
        parts = msg.get("parts")
        if not isinstance(parts, list) and isinstance(content, list):
            parts = content
        if isinstance(parts, list):
            texts: list[str] = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
            joined = "".join(texts).strip()
            if joined:
                return joined

        # Fallbacks for non-standard payloads.
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()

        text = msg.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

    return ""


def _normalize_ai_sdk_chat_messages(payload: AISDKChatRequest) -> list[dict[str, Any]]:
    messages = [message for message in (payload.messages or []) if isinstance(message, dict)]

    extra = getattr(payload, "model_extra", {}) or {}
    latest_message = payload.message or extra.get("message")
    if isinstance(latest_message, dict):
        latest_id = latest_message.get("id")
        last_id = messages[-1].get("id") if messages else None
        if latest_id is not None:
            should_append = latest_id != last_id
        else:
            should_append = not messages or latest_message != messages[-1]
        if should_append:
            messages.append(latest_message)
    elif isinstance(latest_message, str) and latest_message.strip() and not messages:
        messages.append({"role": "user", "content": latest_message})

    if not messages:
        text = payload.content or extra.get("content") or extra.get("text")
        if isinstance(text, str) and text.strip():
            messages.append({"role": "user", "content": text})

    return messages


def _latest_user_attachment_candidates(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue

        candidates: list[dict[str, Any]] = []

        parts = msg.get("parts")
        content = msg.get("content")
        if not isinstance(parts, list) and isinstance(content, list):
            parts = content
        if isinstance(parts, list):
            for item in parts:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "").lower()
                if item_type in {"image", "file"}:
                    candidates.append(item)

        for key in ("attachments", "experimental_attachments", "files"):
            value = msg.get(key)
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "").lower()
                if not item_type or item_type in {"image", "file"}:
                    candidates.append(item)

        return candidates

    return []


def _has_user_attachment_candidates(messages: list[dict[str, Any]]) -> bool:
    return bool(_latest_user_attachment_candidates(messages))


def _attachment_source_value(item: dict[str, Any]) -> str | None:
    for candidate in (
        item.get("data"),
        item.get("base64"),
        item.get("url"),
        item.get("path"),
        item.get("image"),
        item.get("source"),
    ):
        if isinstance(candidate, dict):
            candidate = candidate.get("url") or candidate.get("data") or candidate.get("base64")
        if isinstance(candidate, str):
            cleaned = candidate.strip()
            if cleaned:
                return cleaned
    return None


def _is_valid_raw_base64_source(value: str) -> bool:
    if value.startswith(("data:", "http://", "https://")):
        return True
    try:
        base64.b64decode(value, validate=True)
    except Exception:
        return False
    return True


def _extract_user_attachments(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Extract usable image attachments from the most recent user message in AI SDK payload.
    Supports common UI payload variants:
    - parts/content list entries with `type=image|file`
    - message-level `attachments` / `experimental_attachments`
    - image data URLs, raw base64 payloads, and http(s) image URLs
    """
    attachments: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in _latest_user_attachment_candidates(messages):
        source_value = _attachment_source_value(item)
        if source_value is None or not _is_valid_raw_base64_source(source_value):
            continue

        normalized = normalize_image_attachment(item)
        if normalized is None:
            continue

        raw_data = normalized["url"]
        mime = normalized["mime"]
        name = normalized["name"]
        key = (mime, raw_data)
        if key in seen:
            continue
        seen.add(key)

        attachments.append({"name": name, "mime": mime, "data": raw_data})

    return attachments


def _build_ui_message_stream_response(
    event_source_factory: Callable[[], AsyncGenerator[dict[str, Any], None]],
    state: StreamState,
) -> StreamingResponse:
    adapter = AISDKV6StreamAdapter(
        event_source_factory,
        state,
    )
    return StreamingResponse(
        adapter.iter_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "x-vercel-ai-ui-message-stream": "v1",
        },
    )


@router.post(
    "/ai/conversations",
    response_model=ApiResponse[ConversationRead],
    status_code=status.HTTP_201_CREATED,
    summary="Create conversation (AI SDK)",
    description=(
        "Create a new conversation. Functionally identical to `POST /conversations/` "
        "but under the `/ai/` namespace for AI SDK clients."
    ),
)
@AppAutoInjector.auto_inject()
async def create_conversation_ai_sdk(
    conversation_data: ConversationCreate,
    conversation_service: IConversationService,
    current_user_id: UUID,
) -> ApiResponse[ConversationRead]:
    """Create a new conversation for AI SDK client."""
    result = conversation_service.create_conversation(conversation_data, current_user_id)
    return ApiResponse(success=True, message="Conversation created successfully", data=result)


@router.get(
    "/ai/conversations",
    response_model=PaginatedApiResponse[ConversationRead],
    summary="List conversations (AI SDK)",
    description=(
        "List conversations with pagination. Functionally identical to "
        "`GET /conversations/` but under the `/ai/` namespace for AI SDK clients."
    ),
)
@AppAutoInjector.auto_inject()
async def get_conversations_ai_sdk(
    conversation_service: IConversationService,
    current_user_id: UUID,
    pagination: ConversationPaginationParams,
) -> PaginatedApiResponse[ConversationRead]:
    """Get all conversations for AI SDK client."""
    paginated_result = conversation_service.get_by_user_id(
        current_user_id,
        page=pagination.page,
        limit=pagination.limit,
        order_by=pagination.order_by.to_snake_case(),
        order_direction=pagination.order_direction.value,
        include=[],
        latest_messages=0,
    )
    return PaginatedApiResponse.from_paginator(
        paginated_result, "Conversations retrieved successfully"
    )


@router.get(
    "/ai/conversations/{conversation_id}",
    response_model=ApiResponse[ConversationRead],
    summary="Get conversation (AI SDK)",
    description=(
        "Get a conversation by ID. Functionally identical to `GET /conversations/{conversationId}`."
    ),
)
@AppAutoInjector.auto_inject()
async def get_conversation_ai_sdk(
    conversation_id: UUID,
    conversation_service: IConversationService,
    current_user_id: UUID,
) -> ApiResponse[ConversationRead]:
    """Get conversation by ID for AI SDK client."""
    result = conversation_service.get_by_id_for_user(conversation_id, current_user_id)
    return ApiResponse(success=True, message="Conversation retrieved successfully", data=result)


@router.patch(
    "/ai/conversations/{conversation_id}",
    response_model=ApiResponse[ConversationRead],
    summary="Update conversation (AI SDK)",
    description=(
        "Update a conversation's title, persona prompt, or planning mode. "
        "Functionally identical to `PATCH /conversations/{conversationId}`. "
        "Accepts the same `ConversationUpdate` body (camelCase fields)."
    ),
)
@AppAutoInjector.auto_inject()
async def update_conversation_ai_sdk(
    conversation_id: UUID,
    conversation_data: ConversationUpdate,
    conversation_service: IConversationService,
    current_user_id: UUID,
) -> ApiResponse[ConversationRead]:
    """Update conversation for AI SDK client."""
    result = conversation_service.update_conversation(
        conversation_id, current_user_id, conversation_data
    )
    return ApiResponse(success=True, message="Conversation updated successfully", data=result)


@router.delete(
    "/ai/conversations/{conversation_id}",
    response_model=ApiResponse[Any],
    summary="Delete conversation (AI SDK)",
    description=(
        "Delete a conversation. Functionally identical to `DELETE /conversations/{conversationId}`."
    ),
)
@AppAutoInjector.auto_inject()
async def delete_conversation_ai_sdk(
    conversation_id: UUID,
    conversation_service: IConversationService,
    current_user_id: UUID,
) -> ApiResponse[Any]:
    """Delete conversation for AI SDK client."""
    conversation_service.delete_conversation(conversation_id, current_user_id)
    return ApiResponse(success=True, message="Conversation deleted successfully")


@router.get(
    "/ai/conversations/{conversation_id}/messages",
    response_model=ApiResponse[AISDKMessagesData],
    summary="Get messages (AI SDK UIMessage format)",
    description=(
        "Returns conversation messages formatted as Vercel AI SDK `UIMessage` objects. "
        "Pass `response.data.messages` directly to the `initialMessages` prop of "
        "`useChat()`. Supports page-based pagination via `page`, `limit`, `orderBy`, "
        "and `orderDirection`. Image attachments are embedded as `file` parts inside "
        "each message."
    ),
)
@AppAutoInjector.auto_inject()
async def get_conversation_messages_ai_sdk(
    conversation_id: UUID,
    message_service: IMessageService,
    current_user_id: UUID,
    pagination: MessagePaginationParams,
    inline_rich_response_v1: Annotated[bool, Query(alias="inlineRichResponseV1")] = False,
    inline_rich_response_v1_snake: Annotated[bool, Query(alias="inline_rich_response_v1")] = False,
) -> ApiResponse[AISDKMessagesData]:
    """Get conversation messages in Vercel AI SDK UIMessage format."""
    paginated_result = message_service.get_conversation_messages(
        conversation_id,
        current_user_id,
        page=pagination.page,
        limit=pagination.limit,
        order_by=pagination.order_by.to_snake_case(),
        order_direction=pagination.order_direction.value,
        include_feedback=False,
    )
    rich_response_capable = bool(
        (inline_rich_response_v1 or inline_rich_response_v1_snake)
        and getattr(settings, "inline_rich_response_enabled", False)
    )

    messages: list[AISDKUIMessage] = []
    for msg in paginated_result.items:
        role = "user" if msg.sender == 1 else "assistant"
        message_payload: dict[str, Any] = {
            "id": str(msg.id),
            "role": role,
            "content": msg.content,
            "createdAt": msg.created_at.isoformat() if msg.created_at else None,
        }

        if isinstance(msg.message_metadata, dict):
            message_payload["metadata"] = msg.message_metadata

        if role == "assistant":
            # v1-ness is decided on the persisted metadata so hidden image
            # candidates cannot fall back onto legacy `images` after the
            # capability projection strips the rich keys.
            is_v1 = is_v1_rich_items_message(msg.message_metadata)
            file_parts = visible_image_file_parts(
                message_payload,
                is_v1=is_v1,
                inline_rich_response_v1=rich_response_capable,
            )
            message_payload = project_ai_sdk_message_for_capability(
                message_payload,
                inline_rich_response_v1=rich_response_capable,
            )
            message_payload = attach_image_parts_to_message(
                message_payload,
                image_parts=file_parts,
                is_v1=is_v1,
            )

        ensure_leading_text_part(message_payload)
        metadata = message_payload.get("metadata")
        if isinstance(metadata, dict):
            message_payload["metadata"] = scrub_legacy_metadata(metadata)

        messages.append(AISDKUIMessage.model_validate(message_payload))

    return ApiResponse(
        success=True,
        message="Messages retrieved successfully",
        data=AISDKMessagesData(
            messages=messages,
            meta=paginated_result.meta,
        ),
    )


@router.post(
    "/api/chat/{conversation_id}",
    summary="Chat stream (AI SDK)",
    description=(
        "Send a user message and receive a streaming assistant response using the "
        "**Vercel AI SDK UI Message Stream** protocol. "
        "Configure `useChat({ api: '/api/chat/<conversationId>' })` in your Next.js app. "
        "The `messages` array must contain the full conversation history per the AI SDK spec. "
        "Streams `text/event-stream` SSE events: `text-delta`, `tool-input-start`, "
        "`tool-output-available`, `data-interrupt`, `finish`, `[DONE]`, etc."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Vercel AI SDK UI Message Stream (SSE)",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
@router.post(
    "/ai/chat/{conversation_id}",
    summary="Chat stream (AI SDK — /ai/ alias)",
    description=(
        "Alias of `POST /api/chat/{conversationId}`. "
        "Streams a response using the Vercel AI SDK UI Message Stream protocol. "
        "Prefer `POST /api/chat/{conversationId}` for new integrations."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Vercel AI SDK UI Message Stream (SSE)",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
    include_in_schema=False,
)
@AppAutoInjector.auto_inject()
async def chat_ui_message_stream(
    conversation_id: UUID,
    payload: AISDKChatRequest,
    message_service: IMessageService,
    current_user_id: UUID,
):
    """
    Vercel AI SDK UI Message Stream protocol (SSE).
    """
    request_messages = _normalize_ai_sdk_chat_messages(payload)
    user_text = _extract_user_text(request_messages)
    has_attachment_candidates = _has_user_attachment_candidates(request_messages)
    user_attachments = _extract_user_attachments(request_messages)
    if not user_text and not user_attachments:
        if has_attachment_candidates:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No supported image attachments found. Send image data URLs, raw "
                    "base64, or http(s) image URLs; upload documents via /documents/uploads."
                ),
            )
        raise HTTPException(status_code=400, detail="No user message found")

    # Pre-generate the bot message UUID so the stream's messageId matches the
    # DB record.  This prevents the Next.js client from showing a duplicate
    # message (one from the stream, one from the API) after re-fetching.
    bot_message_id = uuid4()

    extra = getattr(payload, "model_extra", {}) or {}
    inline_rich_response_v1 = bool(
        getattr(payload, "inline_rich_response_v1", False)
        or extra.get("inline_rich_response_v1")
        or extra.get("inlineRichResponseV1")
    )
    state = StreamState(
        message_id=str(bot_message_id),
        text_id=str(uuid4()),
        reasoning_id=str(uuid4()),
        inline_rich_response_v1=(
            inline_rich_response_v1 and getattr(settings, "inline_rich_response_enabled", False)
        ),
    )

    def event_source():
        message_create = MessageCreate(
            conversation_id=conversation_id,
            content=user_text or "Please analyze the attached image.",
            role=MessageRole.user,
            attachments=user_attachments or None,
            device_id=extra.get("device_id") or extra.get("deviceId"),
            inline_rich_response_v1=inline_rich_response_v1,
        )
        return message_service.create_message_stream(
            message_create, current_user_id, bot_message_id=bot_message_id
        )

    return _build_ui_message_stream_response(event_source, state)


@router.post(
    "/ai/resume-interrupt",
    summary="Resume interrupt (AI SDK)",
    description=(
        "Resume execution after the user approves or rejects a tool-call interrupt and "
        "stream the assistant continuation using the Vercel AI SDK UI Message Stream "
        "protocol. Accepts the same `InterruptResumeRequest` body (camelCase fields: "
        "`threadId`, `conversationId`, `interruptId`, `decisions`)."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Vercel AI SDK UI Message Stream (SSE)",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
@AppAutoInjector.auto_inject()
async def resume_interrupt_ai_sdk(
    resume_request: InterruptResumeRequest,
    message_service: IMessageService,
    current_user_id: UUID,
) -> StreamingResponse:
    """Resume execution after handling tool execution interrupts for AI SDK client."""
    bot_message_id = uuid4()
    inline_rich_response_v1 = bool(getattr(resume_request, "inline_rich_response_v1", False))
    state = StreamState(
        message_id=str(bot_message_id),
        text_id=str(uuid4()),
        reasoning_id=str(uuid4()),
        inline_rich_response_v1=(
            inline_rich_response_v1 and getattr(settings, "inline_rich_response_enabled", False)
        ),
    )

    def event_source():
        return message_service.resume_message_creation_stream(
            thread_id=resume_request.thread_id,
            conversation_id=resume_request.conversation_id,
            user_id=current_user_id,
            interrupt_id=resume_request.interrupt_id,
            device_id=resume_request.device_id,
            decisions=resume_request.decisions,
            bot_message_id=bot_message_id,
            inline_rich_response_v1=inline_rich_response_v1,
        )

    return _build_ui_message_stream_response(event_source, state)


@router.post(
    "/ai/continue",
    summary="Continue a paused generation (AI SDK)",
    description=(
        "Continue a turn that paused at its execution budget, streaming the next epoch "
        "as a **Vercel AI SDK UI Message Stream**. The `generationId`, `continuationId` "
        "and `expectedVersion` come from the `data-generation` part whose `phase` is "
        "`continuation_available`.\n\n"
        "**This is not `resumeStream`.** Keep `resume: false` configured: semantic "
        "continuation is an explicit user decision, while `resumeStream` is transport "
        "recovery for a dropped socket. Do not call `regenerate` or `sendMessage` "
        "either — both would append a turn and re-route it, which can land the "
        "continuation on a different agent than the one holding the evidence.\n\n"
        "```ts\n"
        "async function continueGeneration(snapshot: GenerationSnapshot) {\n"
        "  const response = await fetch('/ai/continue', {\n"
        "    method: 'POST',\n"
        "    headers: { 'content-type': 'application/json' },\n"
        "    body: JSON.stringify({\n"
        "      conversationId: snapshot.conversationId,\n"
        "      generationId: snapshot.generationId,\n"
        "      continuationId: snapshot.continuationId,\n"
        "      expectedVersion: snapshot.version,\n"
        "      idempotencyKey: crypto.randomUUID(),\n"
        "    }),\n"
        "  })\n"
        "  // Consume as a UI Message Stream, the same as /api/chat.\n"
        "  return response\n"
        "}\n"
        "```\n\n"
        "**Stop** pairs the local abort with the explicit server command, in that "
        "order — `stop()` ends this client's read immediately, and the POST is what "
        "reaches the worker, which may be in another process:\n\n"
        "```ts\n"
        "async function stopGeneration(snapshot: GenerationSnapshot) {\n"
        "  stop() // abort useChat's fetch immediately\n"
        "  return fetch('/messages/stop', {\n"
        "    method: 'POST',\n"
        "    headers: { 'content-type': 'application/json' },\n"
        "    body: JSON.stringify({\n"
        "      generationId: snapshot.generationId,\n"
        "      conversationId: snapshot.conversationId,\n"
        "      expectedVersion: snapshot.version,\n"
        "      idempotencyKey: crypto.randomUUID(),\n"
        "    }),\n"
        "  })\n"
        "}\n"
        "```\n\n"
        "A `202` from that POST means accepted but unconfirmed: poll "
        "`GET /messages/generations/{generationId}` rather than assuming the turn "
        "ended. `stop()` alone changes no server state, and a closed socket is never "
        "treated as a lifecycle transition."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Vercel AI SDK UI Message Stream (SSE)",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
@AppAutoInjector.auto_inject()
async def continue_generation_ai_sdk(
    continue_request: ContinueGenerationRequest,
    message_service: IMessageService,
    current_user_id: UUID,
) -> StreamingResponse:
    """Continue a paused generation for an AI SDK client.

    Differs from the internal SSE route only in the adapter: both call the same
    service method, so the lifecycle result cannot diverge between transports.
    """
    bot_message_id = uuid4()
    inline_rich_response_v1 = bool(continue_request.inline_rich_response_v1)
    state = StreamState(
        message_id=str(bot_message_id),
        text_id=str(uuid4()),
        reasoning_id=str(uuid4()),
        inline_rich_response_v1=(
            inline_rich_response_v1 and getattr(settings, "inline_rich_response_enabled", False)
        ),
    )

    def event_source():
        return message_service.continue_message_generation_stream(
            generation_id=continue_request.generation_id,
            continuation_id=continue_request.continuation_id,
            conversation_id=continue_request.conversation_id,
            user_id=current_user_id,
            idempotency_key=continue_request.idempotency_key,
            expected_version=continue_request.expected_version,
            bot_message_id=bot_message_id,
            inline_rich_response_v1=inline_rich_response_v1,
        )

    return _build_ui_message_stream_response(event_source, state)
