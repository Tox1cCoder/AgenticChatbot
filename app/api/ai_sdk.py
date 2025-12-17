import json
import asyncio
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.message_service_interface import IMessageService
from app.models.enums import MessageRole
from app.schemas.message import MessageCreate


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
    
    # Return dicts and lists as-is - they'll be properly serialized by json.dumps
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

    message_id = str(uuid4())
    text_id = str(uuid4())
    reasoning_id = str(uuid4())
    reasoning_started = False

    tool_seq = 0
    pending_tool_call_ids: List[str] = []

    async def event_generator():
        nonlocal reasoning_started, tool_seq, pending_tool_call_ids
        text_started = False
        any_text_delta = False
        try:
            # One backend call == one step for AI SDK UI step tracking.
            yield _sse({"type": "start-step"})

            # Start assistant message + first text block.
            yield _sse({"type": "start", "messageId": message_id})
            yield _sse({"type": "text-start", "id": text_id})
            text_started = True

            message_create = MessageCreate(
                conversation_id=conversation_id,
                content=user_text,
                role=MessageRole.user,
            )

            async for event in message_service.create_message_stream(
                message_create, current_user_id
            ):
                event_type = event.get("type")

                if event_type == "token":
                    delta = event.get("content") or ""
                    if delta:
                        any_text_delta = True
                        yield _sse(
                            {"type": "text-delta", "id": text_id, "delta": delta}
                        )

                elif event_type == "thinking":
                    delta = event.get("content") or ""
                    if not delta:
                        continue
                    if not reasoning_started:
                        reasoning_started = True
                        yield _sse({"type": "reasoning-start", "id": reasoning_id})
                    yield _sse(
                        {"type": "reasoning-delta", "id": reasoning_id, "delta": delta}
                    )

                elif event_type == "agent_selected":
                    agent = event.get("agent")
                    yield _sse(
                        {
                            "type": "data-agent-selected",
                            "data": {"agent": agent},
                            "transient": True,
                        }
                    )
                elif event_type == "user_message_created":
                    yield _sse(
                        {
                            "type": "data-user-message",
                            "data": {"message": event.get("message")},
                            "transient": True,
                        }
                    )

                elif event_type == "tool":
                    tool_name = event.get("name") or "unknown"
                    status = event.get("status")
                    tool_call_id = event.get("tool_call_id")
                    if tool_call_id:
                        tool_call_id = str(tool_call_id)
                    else:
                        if status == "end" and pending_tool_call_ids:
                            tool_call_id = pending_tool_call_ids.pop(0)
                        else:
                            tool_seq += 1
                            tool_call_id = f"tool_{tool_seq}"

                    if status == "start":
                        pending_tool_call_ids.append(tool_call_id)
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
                                "input": _coerce_json_object(
                                    _clean_tool_output(tool_input)
                                ),
                            }
                        )
                    elif status == "end":
                        if tool_call_id in pending_tool_call_ids:
                            try:
                                pending_tool_call_ids.remove(tool_call_id)
                            except ValueError:
                                pass
                        output = event.get("result")
                        yield _sse(
                            {
                                "type": "tool-output-available",
                                "toolCallId": tool_call_id,
                                "output": _coerce_json_object(
                                    _clean_tool_output(output)
                                ),
                            }
                        )

                elif event_type == "interrupt":
                    yield _sse(
                        {
                            "type": "text-delta",
                            "id": text_id,
                            "delta": "Tool execution requires approval.",
                        }
                    )
                    if text_started:
                        yield _sse({"type": "text-end", "id": text_id})
                    if reasoning_started:
                        yield _sse({"type": "reasoning-end", "id": reasoning_id})
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
                    return

                elif event_type == "error":
                    yield _sse({"type": "error", "errorText": event.get("error") or ""})
                    if text_started:
                        yield _sse({"type": "text-end", "id": text_id})
                    if reasoning_started:
                        yield _sse({"type": "reasoning-end", "id": reasoning_id})
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
                    return

                elif event_type == "complete":
                    message = event.get("message") or {}
                    if not any_text_delta:
                        content = ""
                        if isinstance(message, dict):
                            content = message.get("content") or ""
                        if isinstance(content, str) and content.strip():
                            any_text_delta = True
                            yield _sse(
                                {
                                    "type": "text-delta",
                                    "id": text_id,
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
                    break

            # Close blocks and finish.
            if text_started:
                yield _sse({"type": "text-end", "id": text_id})
            if reasoning_started:
                yield _sse({"type": "reasoning-end", "id": reasoning_id})
            yield _sse({"type": "finish-step"})
            yield _sse({"type": "finish"})
            yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            return
        except Exception as exc:
            yield _sse({"type": "error", "errorText": str(exc)})
            if text_started:
                yield _sse({"type": "text-end", "id": text_id})
            if reasoning_started:
                yield _sse({"type": "reasoning-end", "id": reasoning_id})
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
