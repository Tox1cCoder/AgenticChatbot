import json
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.message_service_interface import IMessageService
from app.interfaces.conversation_service_interface import IConversationService
from app.models.enums import MessageRole
from app.schemas.message import MessageCreate
from app.schemas.conversation import ConversationCreate, ConversationUpdate


router = APIRouter(tags=["ai-sdk"])


class AISDKChatRequest(BaseModel):
    """
    - messages: the UI message history
    - userId: optional (enables server-side memory features)
    """

    messages: List[Dict[str, Any]] = Field(default_factory=list)
    user_id: Optional[UUID] = Field(default=None, alias="userId")

    model_config = ConfigDict(populate_by_name=True, extra="allow")


def _extract_user_text(messages: List[Dict[str, Any]]) -> str:
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
            texts: List[str] = []
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


def _sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _clean_tool_output(value: Any) -> Any:
    """
    Clean tool output by extracting actual data from LangChain Content objects.
    """
    if value is None:
        return None

    # Handle lists - recursively clean each item
    if isinstance(value, list):
        cleaned = []
        for item in value:
            if isinstance(item, dict):
                # Extract text from LangChain Content objects
                if "type" in item and item.get("type") == "text" and "text" in item:
                    cleaned.append(item["text"])
                else:
                    # Recursively clean nested dicts
                    cleaned.append(_clean_tool_output(item))
            else:
                cleaned.append(_clean_tool_output(item))

        # Unwrap single-item lists
        if len(cleaned) == 1:
            return cleaned[0]
        return cleaned

    # Handle dicts - recursively clean nested structures
    if isinstance(value, dict):
        # If it's a LangChain Content object, extract the text
        if "type" in value and value.get("type") == "text" and "text" in value:
            return value["text"]
        # Otherwise, clean nested values
        return {k: _clean_tool_output(v) for k, v in value.items()}

    return value


def _coerce_json_object(value: Any) -> Any:
    """
    Vercel AI SDK UI message stream expects tool `input`/`output` to be JSON-serializable.
    Returns the exact tool result without any wrapping.
    """
    if value is None:
        return None

    # Return dicts and lists as-is
    if isinstance(value, (dict, list)):
        return value

    # Return primitives as-is (except strings that look like JSON)
    if isinstance(value, (int, float, bool)):
        return value

    # Try to parse strings as JSON if they look like JSON
    if isinstance(value, str):
        s = value.strip()
        if (s.startswith("{") and s.endswith("}")) or (
            s.startswith("[") and s.endswith("]")
        ):
            try:
                # Parse the JSON string to return the actual object
                # This prevents double-encoding and escaped newlines
                return json.loads(s)
            except Exception:
                pass
        return value

    # Fallback for other types - convert to string
    return str(value)


class StreamState:
    """Maintains the streaming state across event handlers."""

    def __init__(self, message_id: str, text_id: str, reasoning_id: str):
        self.message_id = message_id
        self.text_id = text_id
        self.reasoning_id = reasoning_id
        self.text_started = False
        self.reasoning_started = False
        self.any_text_delta = False
        self.tool_seq = 0
        self.pending_tool_call_ids: List[str] = []


class EventHandler(ABC):
    """Base class for event handlers."""

    @abstractmethod
    async def handle(
        self, event: Dict[str, Any], state: StreamState
    ) -> AsyncGenerator[str, None]:
        """Handle an event and yield SSE messages."""
        pass


class TokenEventHandler(EventHandler):
    """Handles token streaming events."""

    async def handle(
        self, event: Dict[str, Any], state: StreamState
    ) -> AsyncGenerator[str, None]:
        delta = event.get("content") or ""
        if delta:
            state.any_text_delta = True
            yield _sse({"type": "text-delta", "id": state.text_id, "delta": delta})


class ThinkingEventHandler(EventHandler):
    """Handles thinking/reasoning events."""

    async def handle(
        self, event: Dict[str, Any], state: StreamState
    ) -> AsyncGenerator[str, None]:
        delta = event.get("content") or ""
        if not delta:
            return

        if not state.reasoning_started:
            state.reasoning_started = True
            yield _sse({"type": "reasoning-start", "id": state.reasoning_id})

        yield _sse(
            {"type": "reasoning-delta", "id": state.reasoning_id, "delta": delta}
        )


class AgentSelectedEventHandler(EventHandler):
    """Handles agent selection events."""

    async def handle(
        self, event: Dict[str, Any], state: StreamState
    ) -> AsyncGenerator[str, None]:
        agent = event.get("agent")
        yield _sse(
            {
                "type": "data-agent-selected",
                "data": {"agent": agent},
                "transient": True,
            }
        )


class UserMessageCreatedEventHandler(EventHandler):
    """Handles user message creation events."""

    async def handle(
        self, event: Dict[str, Any], state: StreamState
    ) -> AsyncGenerator[str, None]:
        yield _sse(
            {
                "type": "data-user-message",
                "data": {"message": event.get("message")},
                "transient": True,
            }
        )


class ToolEventHandler(EventHandler):
    """Handles tool execution events."""

    async def handle(
        self, event: Dict[str, Any], state: StreamState
    ) -> AsyncGenerator[str, None]:
        tool_name = event.get("name") or "unknown"
        status = event.get("status")
        tool_call_id = self._get_tool_call_id(event, status, state)

        if status == "start":
            async for msg in self._handle_tool_start(
                event, tool_call_id, tool_name, state
            ):
                yield msg
        elif status == "end":
            async for msg in self._handle_tool_end(event, tool_call_id, state):
                yield msg

    def _get_tool_call_id(
        self, event: Dict[str, Any], status: str, state: StreamState
    ) -> str:
        """Get or generate tool call ID."""
        tool_call_id = event.get("tool_call_id")
        if tool_call_id:
            return str(tool_call_id)

        if status == "end" and state.pending_tool_call_ids:
            return state.pending_tool_call_ids.pop(0)

        state.tool_seq += 1
        return f"tool_{state.tool_seq}"

    async def _handle_tool_start(
        self,
        event: Dict[str, Any],
        tool_call_id: str,
        tool_name: str,
        state: StreamState,
    ) -> AsyncGenerator[str, None]:
        """Handle tool start event."""
        state.pending_tool_call_ids.append(tool_call_id)
        yield _sse(
            {
                "type": "tool-input-start",
                "toolCallId": tool_call_id,
                "toolName": tool_name,
            }
        )

        tool_input = event.get("args")
        yield _sse(
            {
                "type": "tool-input-available",
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "input": _coerce_json_object(_clean_tool_output(tool_input)),
            }
        )

    async def _handle_tool_end(
        self, event: Dict[str, Any], tool_call_id: str, state: StreamState
    ) -> AsyncGenerator[str, None]:
        """Handle tool end event."""
        if tool_call_id in state.pending_tool_call_ids:
            try:
                state.pending_tool_call_ids.remove(tool_call_id)
            except ValueError:
                pass

        output = event.get("result")
        yield _sse(
            {
                "type": "tool-output-available",
                "toolCallId": tool_call_id,
                "output": _coerce_json_object(_clean_tool_output(output)),
            }
        )


class InterruptEventHandler(EventHandler):
    """Handles interrupt events."""

    async def handle(
        self, event: Dict[str, Any], state: StreamState
    ) -> AsyncGenerator[str, None]:
        yield _sse(
            {
                "type": "text-delta",
                "id": state.text_id,
                "delta": "Tool execution requires approval.",
            }
        )

        if state.text_started:
            yield _sse({"type": "text-end", "id": state.text_id})
        if state.reasoning_started:
            yield _sse({"type": "reasoning-end", "id": state.reasoning_id})

        yield _sse(
            {
                "type": "data-interrupt",
                "data": {
                    "threadId": event.get("thread_id"),
                    "next": event.get("next"),
                    "pendingToolCalls": event.get("pending_tool_calls"),
                    "interrupt": event.get("interrupt"),
                    "message": event.get("message"),
                },
            }
        )

        yield _sse({"type": "finish-step"})
        yield _sse({"type": "finish"})
        yield "data: [DONE]\n\n"


class ErrorEventHandler(EventHandler):
    """Handles error events."""

    async def handle(
        self, event: Dict[str, Any], state: StreamState
    ) -> AsyncGenerator[str, None]:
        yield _sse({"type": "error", "errorText": event.get("error") or ""})

        if state.text_started:
            yield _sse({"type": "text-end", "id": state.text_id})
        if state.reasoning_started:
            yield _sse({"type": "reasoning-end", "id": state.reasoning_id})

        if event.get("message"):
            yield _sse(
                {
                    "type": "data-error-message",
                    "data": {"message": event.get("message")},
                    "transient": True,
                }
            )

        yield _sse({"type": "finish-step"})
        yield _sse({"type": "finish"})
        yield "data: [DONE]\n\n"


class CompleteEventHandler(EventHandler):
    """Handles completion events."""

    async def handle(
        self, event: Dict[str, Any], state: StreamState
    ) -> AsyncGenerator[str, None]:
        message = event.get("message") or {}

        if not state.any_text_delta:
            content = ""
            if isinstance(message, dict):
                content = message.get("content") or ""
            if isinstance(content, str) and content.strip():
                state.any_text_delta = True
                yield _sse(
                    {
                        "type": "text-delta",
                        "id": state.text_id,
                        "delta": content,
                    }
                )

        if message:
            yield _sse(
                {
                    "type": "data-assistant-message",
                    "data": {"message": message},
                    "transient": True,
                }
            )


class EventHandlerFactory:
    """Factory for creating event handlers."""

    _handlers = {
        "token": TokenEventHandler(),
        "thinking": ThinkingEventHandler(),
        "agent_selected": AgentSelectedEventHandler(),
        "user_message_created": UserMessageCreatedEventHandler(),
        "tool": ToolEventHandler(),
        "interrupt": InterruptEventHandler(),
        "error": ErrorEventHandler(),
        "complete": CompleteEventHandler(),
    }

    @classmethod
    def get_handler(cls, event_type: str) -> Optional[EventHandler]:
        """Get handler for the given event type."""
        return cls._handlers.get(event_type)


@router.post(
    "/ai/conversations",
    response_model=Dict[str, Any],
    status_code=status.HTTP_201_CREATED,
)
@AppAutoInjector.auto_inject()
async def create_conversation_ai_sdk(
    conversation_service: IConversationService,
    current_user_id: UUID,
) -> Dict[str, Any]:
    """Create a new conversation for AI SDK client"""
    from app.schemas.conversation import ConversationCreate
    from app.interfaces.conversation_service_interface import IConversationService

    conversation_data = ConversationCreate(title="New Conversation")
    result = conversation_service.create_conversation(
        conversation_data, current_user_id
    )
    return {
        "id": str(result.id),
        "title": result.title,
        "created_at": result.created_at.isoformat() if result.created_at else None,
        "updated_at": result.updated_at.isoformat() if result.updated_at else None,
    }


@router.get("/ai/conversations", response_model=Dict[str, Any])
@AppAutoInjector.auto_inject()
async def get_conversations_ai_sdk(
    conversation_service: IConversationService,
    current_user_id: UUID,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=100, ge=1, le=100),
    order_by: str = Query(default="updatedAt", alias="orderBy"),
    order_direction: str = Query(default="desc", alias="orderDirection"),
) -> Dict[str, Any]:
    """Get all conversations for AI SDK client"""

    # Convert camelCase to snake_case for order_by
    order_by_mapping = {"createdAt": "created_at", "updatedAt": "updated_at"}
    order_by_snake = order_by_mapping.get(order_by, "updated_at")

    paginated_result = conversation_service.get_by_user_id(
        current_user_id,
        page=page,
        limit=limit,
        order_by=order_by_snake,
        order_direction=order_direction,
        include=[],
        latest_messages=0,
    )

    conversations = [
        {
            "id": str(conv.id),
            "title": conv.title,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
        }
        for conv in paginated_result.items
    ]

    return {
        "conversations": conversations,
        "total": paginated_result.meta.total,
    }


@router.get("/ai/conversations/{conversation_id}", response_model=Dict[str, Any])
@AppAutoInjector.auto_inject()
async def get_conversation_ai_sdk(
    conversation_id: UUID,
    conversation_service: IConversationService,
    current_user_id: UUID,
) -> Dict[str, Any]:
    """Get conversation by ID for AI SDK client"""
    from app.interfaces.conversation_service_interface import IConversationService

    result = conversation_service.get_by_id_for_user(conversation_id, current_user_id)
    return {
        "id": str(result.id),
        "title": result.title,
        "created_at": result.created_at.isoformat() if result.created_at else None,
        "updated_at": result.updated_at.isoformat() if result.updated_at else None,
    }


@router.patch("/ai/conversations/{conversation_id}", response_model=Dict[str, Any])
@AppAutoInjector.auto_inject()
async def update_conversation_ai_sdk(
    conversation_id: UUID,
    payload: Dict[str, Any],
    conversation_service: IConversationService,
    current_user_id: UUID,
) -> Dict[str, Any]:
    """Update conversation for AI SDK client"""
    from app.schemas.conversation import ConversationUpdate
    from app.interfaces.conversation_service_interface import IConversationService

    conversation_data = ConversationUpdate(title=payload.get("title"))
    result = conversation_service.update_conversation(
        conversation_id, current_user_id, conversation_data
    )
    return {
        "id": str(result.id),
        "title": result.title,
        "created_at": result.created_at.isoformat() if result.created_at else None,
        "updated_at": result.updated_at.isoformat() if result.updated_at else None,
    }


@router.delete(
    "/ai/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT
)
@AppAutoInjector.auto_inject()
async def delete_conversation_ai_sdk(
    conversation_id: UUID,
    conversation_service: IConversationService,
    current_user_id: UUID,
):
    """Delete conversation for AI SDK client"""
    from app.interfaces.conversation_service_interface import IConversationService

    conversation_service.delete_conversation(conversation_id, current_user_id)


@router.get(
    "/ai/conversations/{conversation_id}/messages", response_model=Dict[str, Any]
)
@AppAutoInjector.auto_inject()
async def get_conversation_messages_ai_sdk(
    conversation_id: UUID,
    message_service: IMessageService,
    current_user_id: UUID,
) -> Dict[str, Any]:
    """Get conversation messages for AI SDK client"""
    paginated_result = message_service.get_conversation_messages(
        conversation_id,
        current_user_id,
        page=1,
        limit=100,
        order_by="created_at",
        order_direction="asc",
        include_feedback=False,
    )

    messages = [
        {
            "id": str(msg.id),
            "role": "user" if msg.sender == 1 else "assistant",
            "content": msg.content,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
        }
        for msg in paginated_result.items
    ]

    return {
        "messages": messages,
        "total": paginated_result.meta.total,
    }


@router.post("/ai/chat/{conversation_id}")
@router.post("/api/chat/{conversation_id}")
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
    user_text = _extract_user_text(payload.messages)
    if not user_text:
        raise HTTPException(status_code=400, detail="No user message found")

    state = StreamState(
        message_id=str(uuid4()), text_id=str(uuid4()), reasoning_id=str(uuid4())
    )

    async def event_generator():
        try:
            yield _sse({"type": "start-step"})

            # Start assistant message + first text block.
            yield _sse({"type": "start", "messageId": state.message_id})
            yield _sse({"type": "text-start", "id": state.text_id})
            state.text_started = True

            message_create = MessageCreate(
                conversation_id=conversation_id,
                content=user_text,
                role=MessageRole.user,
            )

            async for event in message_service.create_message_stream(
                message_create, current_user_id
            ):
                event_type = event.get("type")
                handler = EventHandlerFactory.get_handler(event_type)

                if handler:
                    async for msg in handler.handle(event, state):
                        yield msg

                    # Early return for interrupt and error events
                    if event_type in ("interrupt", "error"):
                        return

                    # Break loop for complete event
                    if event_type == "complete":
                        break

            # Close blocks and finish.
            if state.text_started:
                yield _sse({"type": "text-end", "id": state.text_id})
            if state.reasoning_started:
                yield _sse({"type": "reasoning-end", "id": state.reasoning_id})
            yield _sse({"type": "finish-step"})
            yield _sse({"type": "finish"})
            yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            return
        except Exception as exc:
            yield _sse({"type": "error", "errorText": str(exc)})
            if state.text_started:
                yield _sse({"type": "text-end", "id": state.text_id})
            if state.reasoning_started:
                yield _sse({"type": "reasoning-end", "id": state.reasoning_id})
            yield _sse({"type": "finish-step"})
            yield _sse({"type": "finish"})
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "x-vercel-ai-ui-message-stream": "v1",
        },
    )


@router.post("/ai/resume-interrupt", response_model=Dict[str, Any])
@AppAutoInjector.auto_inject()
async def resume_interrupt_ai_sdk(
    payload: Dict[str, Any],
    message_service: IMessageService,
    current_user_id: UUID,
) -> Dict[str, Any]:
    """Resume execution after handling tool execution interrupts for AI SDK client"""
    from app.schemas.message import InterruptResumeRequest

    resume_request = InterruptResumeRequest(
        thread_id=payload.get("threadId"),
        conversation_id=UUID(payload.get("conversationId")),
        interrupt_id=payload.get("interruptId"),
        decisions=payload.get("decisions", {}),
    )

    result = await message_service.resume_message_creation(
        thread_id=resume_request.thread_id,
        conversation_id=resume_request.conversation_id,
        user_id=current_user_id,
        interrupt_id=resume_request.interrupt_id,
        decisions=resume_request.decisions,
    )

    return {
        "id": str(result.id),
        "role": result.role.value if hasattr(result.role, "value") else result.role,
        "content": result.content,
        "created_at": result.created_at.isoformat() if result.created_at else None,
    }
