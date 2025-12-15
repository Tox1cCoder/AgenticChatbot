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


router = APIRouter(prefix="/ai", tags=["ai-sdk"])


class AISDKChatRequest(BaseModel):
    """
    - messages: the UI message history
    - conversationId: required (used for DB history + HITL resume)
    - userId: optional (enables server-side memory features)
    """

    messages: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_id: Optional[UUID] = Field(default=None, alias="conversationId")
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

        parts = msg.get("parts")
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

    return ""


def _sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
@AppAutoInjector.auto_inject()
async def chat_ui_message_stream(
    payload: AISDKChatRequest,
    message_service: IMessageService,
):
    """
    Vercel AI SDK UI Message Stream protocol (SSE).
    """
    user_text = _extract_user_text(payload.messages)
    if not user_text:
        raise HTTPException(status_code=400, detail="No user message found")

    if not payload.conversation_id:
        raise HTTPException(
            status_code=400, detail="conversationId is required for /ai/chat"
        )

    message_id = str(uuid4())
    text_id = str(uuid4())
    reasoning_id = str(uuid4())
    reasoning_started = False

    tool_seq = 0

    async def event_generator():
        nonlocal reasoning_started, tool_seq
        text_started = False
        try:
            # One backend call == one step for AI SDK UI step tracking.
            yield _sse({"type": "start-step"})

            # Start assistant message + first text block.
            yield _sse({"type": "start", "messageId": message_id})
            yield _sse({"type": "text-start", "id": text_id})
            text_started = True

            message_create = MessageCreate(
                conversation_id=payload.conversation_id,
                content=user_text,
                role=MessageRole.user,
            )

            async for event in message_service.create_message_stream(message_create):
                event_type = event.get("type")

                if event_type == "token":
                    delta = event.get("content") or ""
                    if delta:
                        yield _sse({"type": "text-delta", "id": text_id, "delta": delta})

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
                        tool_seq += 1
                        tool_call_id = f"tool_{tool_seq}"

                    if status == "start":
                        yield _sse(
                            {
                                "type": "tool-input-start",
                                "toolCallId": tool_call_id,
                                "toolName": tool_name,
                            }
                        )
                        tool_input = event.get("args")
                        if tool_input is not None:
                            yield _sse(
                                {
                                    "type": "tool-input-available",
                                    "toolCallId": tool_call_id,
                                    "toolName": tool_name,
                                    "input": tool_input,
                                }
                            )
                    elif status == "end":
                        output = event.get("result")
                        yield _sse(
                            {
                                "type": "tool-output-available",
                                "toolCallId": tool_call_id,
                                "output": output,
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
                    if event.get("message"):
                        yield _sse(
                            {
                                "type": "data-assistant-message",
                                "data": {"message": event.get("message")},
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
