import asyncio
import base64
import contextlib
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.core.dependency_injection import AppAutoInjector
from app.interfaces.conversation_service_interface import IConversationService
from app.interfaces.message_service_interface import IMessageService
from app.models.enums import MessageRole
from app.schemas.conversation import (
    ConversationCreate,
    ConversationRead,
    ConversationUpdate,
)
from app.schemas.message import InterruptResumeRequest, MessageCreate
from app.schemas.pagination import ConversationPaginationParams
from app.schemas.responses import ApiResponse
from app.schemas.responses.paginated_response import PaginatedApiResponse
from app.services.stream_events import normalize_tool_phase

router = APIRouter(tags=["ai-sdk"])

_AI_SDK_HEARTBEAT_INTERVAL_SECONDS = 15.0


class AISDKChatRequest(BaseModel):
    """
    - messages: the UI message history
    - userId: optional (enables server-side memory features)
    """

    messages: list[dict[str, Any]] = Field(default_factory=list)
    user_id: UUID | None = Field(default=None, alias="userId")

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class AISDKMessagePart(BaseModel):
    """
    A part within a Vercel AI SDK UIMessage.

    Matches the ``UIPart`` discriminated union used by ``useChat()``:
    - ``type='text'``      → plain text delta
    - ``type='file'``      → image / binary attachment (data URL or remote URL)
    - ``type='reasoning'`` → chain-of-thought / thinking text
    """

    type: str = Field(..., description="Part type: 'text', 'file', or 'reasoning'")
    text: str | None = Field(None, description="Text content (when type='text')")
    url: str | None = Field(None, description="File or data URL (when type='file')")
    media_type: str | None = Field(
        None, alias="mediaType", description="MIME type (when type='file')"
    )
    reasoning: str | None = Field(None, description="Reasoning text (when type='reasoning')")

    model_config = ConfigDict(populate_by_name=True)


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
    parts: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "Structured message parts (text / file / reasoning) for multimodal messages. "
            "Mirrors the ``parts`` field of the AI SDK UIMessage spec."
        ),
    )
    metadata: dict[str, Any] | None = Field(None, description="AI SDK client-side metadata")
    message_metadata: dict[str, Any] | None = Field(
        None,
        alias="messageMetadata",
        description=(
            "Backend metadata: RAG citations, images, canvas artifacts, suggested questions, etc."
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
    total: int = Field(..., description="Total number of messages in the conversation")


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


def _extract_data_from_candidate(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    payload = value.strip()
    if not payload:
        return None

    if payload.startswith("data:"):
        return payload

    if payload.startswith(("http://", "https://", "blob:")):
        return payload

    return payload


def _extract_user_attachments(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Extract image attachments from the most recent user message in AI SDK payload.
    Supports common UI payload variants:
    - parts/content list entries with `type=image|file`
    - message-level `attachments` / `experimental_attachments`
    - data URLs and raw base64 payloads
    """
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue

        candidates: list[Any] = []

        # Parts-based message formats
        parts = msg.get("parts")
        content = msg.get("content")
        if not isinstance(parts, list) and isinstance(content, list):
            parts = content
        if isinstance(parts, list):
            candidates.extend(parts)

        # Attachment-based message formats
        for key in ("attachments", "experimental_attachments", "files"):
            value = msg.get(key)
            if isinstance(value, list):
                candidates.extend(value)

        attachments: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for item in candidates:
            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type") or "").lower()
            if item_type and item_type not in {"image", "file"}:
                continue

            mime = (
                item.get("mime")
                or item.get("mimeType")
                or item.get("mediaType")
                or item.get("contentType")
                or "image/jpeg"
            )
            mime = str(mime).strip() if mime else "image/jpeg"

            name = item.get("name") or item.get("filename") or "attachment"
            name = str(name)

            raw_data: str | None = None
            data_candidates: list[Any] = [
                item.get("data"),
                item.get("base64"),
                item.get("url"),
                item.get("path"),
                item.get("image"),
                item.get("source"),
            ]
            for candidate in data_candidates:
                if isinstance(candidate, dict):
                    candidate = (
                        candidate.get("data") or candidate.get("base64") or candidate.get("url")
                    )
                extracted = _extract_data_from_candidate(candidate)
                if extracted:
                    raw_data = extracted
                    break

            if not raw_data:
                continue

            if not raw_data.startswith(("data:", "http://", "https://", "blob:")):
                # Skip obvious local path values (e.g. C:\fakepath\image.png).
                if ":\\" in raw_data or raw_data.startswith(("/", "./", "../")):
                    continue

                # Best-effort sanity check that payload is decodable base64.
                try:
                    base64.b64decode(raw_data, validate=True)
                except Exception:
                    continue

            key = (mime, raw_data)
            if key in seen:
                continue
            seen.add(key)

            attachments.append({"name": name, "mime": mime, "data": raw_data})

        return attachments

    return []


def _extract_mime_from_data_url(value: str) -> str | None:
    if not isinstance(value, str):
        return None

    payload = value.strip()
    if not payload.startswith("data:"):
        return None

    header, _, _ = payload.partition(",")
    mime = header[5:].split(";")[0].strip()
    return mime if "/" in mime else None


def _normalize_image_item_to_file_part(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None

    mime = (
        item.get("mime")
        or item.get("mimeType")
        or item.get("mediaType")
        or item.get("contentType")
        or "image/png"
    )
    mime = str(mime).strip() if mime else "image/png"

    candidate_values: list[Any] = [
        item.get("url"),
        item.get("data"),
        item.get("base64"),
        item.get("image"),
        item.get("source"),
    ]

    for candidate in candidate_values:
        if isinstance(candidate, dict):
            candidate = candidate.get("url") or candidate.get("data") or candidate.get("base64")
        if not isinstance(candidate, str):
            continue

        raw_value = candidate.strip()
        if not raw_value:
            continue

        if raw_value.startswith("data:"):
            detected_mime = _extract_mime_from_data_url(raw_value)
            if detected_mime:
                mime = detected_mime
            return {"url": raw_value, "mediaType": mime}

        if raw_value.startswith(("http://", "https://", "blob:")):
            return {"url": raw_value, "mediaType": mime}

        try:
            base64.b64decode(raw_value, validate=False)
        except Exception:
            continue

        return {
            "url": f"data:{mime};base64,{raw_value}",
            "mediaType": mime,
        }

    return None


def _extract_image_file_parts_from_metadata(
    metadata: dict[str, Any] | None,
) -> list[dict[str, str]]:
    if not isinstance(metadata, dict):
        return []

    images = metadata.get("images")
    if not isinstance(images, list):
        return []

    file_parts: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in images:
        file_part = _normalize_image_item_to_file_part(item)
        if not file_part:
            continue

        key = (file_part["url"], file_part["mediaType"])
        if key in seen:
            continue
        seen.add(key)
        file_parts.append(file_part)

    return file_parts


def _is_v1_rich_items_message(metadata: dict[str, Any] | None) -> bool:
    if not isinstance(metadata, dict):
        return False
    return metadata.get("rich_items_version") == 1


def _scrub_v1_legacy_image_fields(payload: dict[str, Any]) -> None:
    """Mutate ``payload`` to remove unselected legacy image data for v1 messages.

    For v1 rich-items messages we never want hidden image candidates to surface
    through the legacy ``metadata["images"]`` channel or via the AI SDK
    ``parts`` array. The selected images are already exposed by
    ``_selected_image_file_parts_from_rich_items()``.
    """
    for meta_key in ("message_metadata", "messageMetadata", "metadata"):
        meta = payload.get(meta_key)
        if isinstance(meta, dict) and "images" in meta:
            scrubbed = dict(meta)
            scrubbed.pop("images", None)
            payload[meta_key] = scrubbed
    parts = payload.get("parts")
    if isinstance(parts, list):
        rich_items = None
        for meta_key in ("message_metadata", "messageMetadata", "metadata"):
            meta = payload.get(meta_key)
            if isinstance(meta, dict):
                rich_items = meta.get("rich_items")
                break
        allowed_urls: set[str] = set()
        if isinstance(rich_items, list):
            for item in rich_items:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                pl = item.get("payload") or {}
                if pl.get("url"):
                    allowed_urls.add(str(pl["url"]))
                elif pl.get("data"):
                    mime = pl.get("mime_type") or "image/png"
                    allowed_urls.add(f"data:{mime};base64,{pl['data']}")
        filtered_parts = []
        for part in parts:
            if not isinstance(part, dict):
                filtered_parts.append(part)
                continue
            if part.get("type") == "file" and part.get("url") not in allowed_urls:
                continue
            filtered_parts.append(part)
        payload["parts"] = filtered_parts


def _selected_image_file_parts_from_rich_items(
    metadata: dict[str, Any] | None,
) -> list[dict[str, str]]:
    if not isinstance(metadata, dict):
        return []
    rich_items = metadata.get("rich_items")
    if not isinstance(rich_items, list):
        return []
    file_parts: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in rich_items:
        if not isinstance(item, dict) or item.get("type") != "image":
            continue
        payload = item.get("payload") or {}
        url = payload.get("url")
        data = payload.get("data")
        mime_type = payload.get("mime_type") or "image/png"
        if url:
            file_part = {"url": str(url), "mediaType": str(mime_type)}
        elif data:
            file_part = {"url": f"data:{mime_type};base64,{data}", "mediaType": str(mime_type)}
        else:
            continue
        key = (file_part["url"], file_part["mediaType"])
        if key in seen:
            continue
        seen.add(key)
        file_parts.append(file_part)
    return file_parts


_RICH_MARKER_LINE_PATTERN = None  # Compiled lazily inside the projection helper.


def project_ai_sdk_message_for_capability(
    message: dict[str, Any],
    *,
    inline_rich_response_v1: bool,
) -> dict[str, Any]:
    """Project a persisted assistant message for an AI SDK consumer.

    When ``inline_rich_response_v1`` is True, the message passes through
    unchanged (markers preserved, `rich_items` available). When False, the
    standalone HTML-comment marker lines are stripped from ``content`` so
    non-capable clients do not render them verbatim, and `rich_items` /
    `rich_items_version` keys are removed from any metadata field present.
    """
    if not isinstance(message, dict):
        return message
    if inline_rich_response_v1:
        return message
    projected = dict(message)
    content = projected.get("content")
    if isinstance(content, str) and "<!--rich:" in content:
        import re

        global _RICH_MARKER_LINE_PATTERN
        if _RICH_MARKER_LINE_PATTERN is None:
            _RICH_MARKER_LINE_PATTERN = re.compile(
                r"^[ ]{0,3}<!--rich:[A-Za-z0-9_\-.:]+-->[ \t]*$",
                re.MULTILINE,
            )
        projected["content"] = _RICH_MARKER_LINE_PATTERN.sub("", content)
    for meta_key in ("message_metadata", "messageMetadata", "metadata"):
        meta = projected.get(meta_key)
        if isinstance(meta, dict):
            scrubbed = {
                k: v
                for k, v in meta.items()
                if k not in {"rich_items", "rich_items_version", "rich_reference_warnings"}
            }
            projected[meta_key] = scrubbed
    return projected


def _extract_image_file_parts_from_message(
    message: dict[str, Any],
) -> list[dict[str, str]]:
    if not isinstance(message, dict):
        return []

    metadata = None
    for key in ("message_metadata", "messageMetadata", "metadata"):
        value = message.get(key)
        if isinstance(value, dict):
            metadata = value
            break

    return _extract_image_file_parts_from_metadata(metadata)


def _attach_image_parts_to_message(message: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(message, dict):
        return message

    payload = dict(message)

    # Mirror backend metadata key to a generic `metadata` field for UI compatibility.
    if "metadata" not in payload:
        if isinstance(payload.get("message_metadata"), dict):
            payload["metadata"] = payload["message_metadata"]
        elif isinstance(payload.get("messageMetadata"), dict):
            payload["metadata"] = payload["messageMetadata"]

    # For v1 rich-items messages, file parts come only from finalized rich_items
    # (selected images). Skip the legacy metadata["images"] attachment entirely
    # so hidden candidates cannot leak via `parts`.
    metadata = None
    for key in ("message_metadata", "messageMetadata", "metadata"):
        value = payload.get(key)
        if isinstance(value, dict):
            metadata = value
            break
    if _is_v1_rich_items_message(metadata):
        image_parts = _selected_image_file_parts_from_rich_items(metadata)
    else:
        image_parts = _extract_image_file_parts_from_message(payload)
    if not image_parts:
        return payload

    existing_parts = payload.get("parts")
    parts: list[dict[str, Any]] = (
        [p for p in existing_parts if isinstance(p, dict)]
        if isinstance(existing_parts, list)
        else []
    )

    if not isinstance(existing_parts, list):
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            parts.append({"type": "text", "text": content})

    existing_urls = {
        p.get("url") for p in parts if p.get("type") == "file" and isinstance(p.get("url"), str)
    }

    for file_part in image_parts:
        url = file_part["url"]
        if url in existing_urls:
            continue
        parts.append(
            {
                "type": "file",
                "url": url,
                "mediaType": file_part["mediaType"],
            }
        )
        existing_urls.add(url)

    payload["parts"] = parts
    return payload


def _sse(data: dict[str, Any]) -> str:
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
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
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

    def __init__(
        self,
        message_id: str,
        text_id: str,
        reasoning_id: str,
        *,
        inline_rich_response_v1: bool = False,
    ):
        self.message_id = message_id
        self.text_id = text_id
        self.reasoning_id = reasoning_id
        self.text_started = False
        self.reasoning_started = False
        self.any_text_delta = False
        self.tool_seq = 0
        self.pending_tool_call_ids: list[str] = []
        self.inline_rich_response_v1 = bool(inline_rich_response_v1)


class EventHandler(ABC):
    """Base class for event handlers."""

    @abstractmethod
    async def handle(self, event: dict[str, Any], state: StreamState) -> AsyncGenerator[str, None]:
        """Handle an event and yield SSE messages."""
        pass


class TokenEventHandler(EventHandler):
    """Handles token streaming events."""

    async def handle(self, event: dict[str, Any], state: StreamState) -> AsyncGenerator[str, None]:
        delta = event.get("content") or ""
        if delta:
            state.any_text_delta = True
            yield _sse({"type": "text-delta", "id": state.text_id, "delta": delta})


class ThinkingEventHandler(EventHandler):
    """Handles thinking/reasoning events."""

    async def handle(self, event: dict[str, Any], state: StreamState) -> AsyncGenerator[str, None]:
        delta = event.get("content") or ""
        if not delta:
            return

        if not state.reasoning_started:
            state.reasoning_started = True
            yield _sse({"type": "reasoning-start", "id": state.reasoning_id})

        yield _sse({"type": "reasoning-delta", "id": state.reasoning_id, "delta": delta})


class AgentSelectedEventHandler(EventHandler):
    """Handles agent selection events."""

    async def handle(self, event: dict[str, Any], state: StreamState) -> AsyncGenerator[str, None]:
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

    async def handle(self, event: dict[str, Any], state: StreamState) -> AsyncGenerator[str, None]:
        yield _sse(
            {
                "type": "data-user-message",
                "data": {"message": event.get("message")},
                "transient": True,
            }
        )


class ToolEventHandler(EventHandler):
    """Handles tool execution events."""

    async def handle(self, event: dict[str, Any], state: StreamState) -> AsyncGenerator[str, None]:
        tool_name = event.get("name") or "unknown"
        phase = normalize_tool_phase(event.get("phase") or event.get("status"))
        tool_call_id = self._get_tool_call_id(event, phase, state)

        if phase == "start":
            async for msg in self._handle_tool_start(event, tool_call_id, tool_name, state):
                yield msg
        elif phase == "end":
            async for msg in self._handle_tool_end(event, tool_call_id, state):
                yield msg

    def _get_tool_call_id(
        self, event: dict[str, Any], phase: str | None, state: StreamState
    ) -> str:
        """Get or generate tool call ID."""
        tool_call_id = event.get("tool_call_id")
        if tool_call_id:
            return str(tool_call_id)

        if phase == "end" and state.pending_tool_call_ids:
            return state.pending_tool_call_ids.pop(0)

        state.tool_seq += 1
        return f"tool_{state.tool_seq}"

    async def _handle_tool_start(
        self,
        event: dict[str, Any],
        tool_call_id: str,
        tool_name: str,
        state: StreamState,
    ) -> AsyncGenerator[str, None]:
        """Handle tool start event."""
        if tool_call_id not in state.pending_tool_call_ids:
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
        self, event: dict[str, Any], tool_call_id: str, state: StreamState
    ) -> AsyncGenerator[str, None]:
        """Handle tool end event."""
        if tool_call_id in state.pending_tool_call_ids:
            with contextlib.suppress(ValueError):
                state.pending_tool_call_ids.remove(tool_call_id)

        output = event.get("result")
        payload = {
            "type": "tool-output-available",
            "toolCallId": tool_call_id,
            "output": _coerce_json_object(_clean_tool_output(output)),
        }
        render = event.get("render")
        if isinstance(render, dict):
            payload["render"] = render
        yield _sse(payload)


class InterruptEventHandler(EventHandler):
    """Handles interrupt events."""

    async def handle(self, event: dict[str, Any], state: StreamState) -> AsyncGenerator[str, None]:
        interrupt_message = event.get("message")
        if isinstance(interrupt_message, str) and interrupt_message.strip():
            yield _sse(
                {
                    "type": "text-delta",
                    "id": state.text_id,
                    "delta": interrupt_message.strip(),
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

    async def handle(self, event: dict[str, Any], state: StreamState) -> AsyncGenerator[str, None]:
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

    async def handle(self, event: dict[str, Any], state: StreamState) -> AsyncGenerator[str, None]:
        message = event.get("message") or {}
        if isinstance(message, dict):
            message = _attach_image_parts_to_message(message)
            message = project_ai_sdk_message_for_capability(
                message,
                inline_rich_response_v1=getattr(state, "inline_rich_response_v1", False),
            )

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

        if isinstance(message, dict):
            metadata = None
            for key in ("message_metadata", "messageMetadata", "metadata"):
                value = message.get(key)
                if isinstance(value, dict):
                    metadata = value
                    break
            # For v1 messages, file parts come only from finalized rich_items
            # (the selected images). For legacy messages, fall back to the
            # existing extraction over metadata["images"].
            if _is_v1_rich_items_message(metadata):
                file_parts = _selected_image_file_parts_from_rich_items(metadata)
            else:
                file_parts = _extract_image_file_parts_from_message(message)
            for file_part in file_parts:
                yield _sse(
                    {
                        "type": "file",
                        "url": file_part["url"],
                        "mediaType": file_part["mediaType"],
                    }
                )

        if message:
            # Strip `content` from the data event — the content has already
            # been conveyed char-by-char via `text-delta` events.  Re-sending
            # it here causes the response to appear twice if the client renders
            # from both the streamed text and this data payload.  We only send
            # side-channel metadata that is not available in the stream itself
            # (backend message ID, citations, suggested questions, etc.).
            STRIP_KEYS = {"content"}
            message_meta = {k: v for k, v in message.items() if k not in STRIP_KEYS}
            # For v1 messages, the legacy `images` field cannot leak unselected
            # candidates and `parts` must be limited to selected images.
            if _is_v1_rich_items_message(metadata):
                _scrub_v1_legacy_image_fields(message_meta)
            if message_meta:
                yield _sse(
                    {
                        "type": "data-assistant-message",
                        "data": {"message": message_meta},
                    }
                )


class ContinuationEventHandler(EventHandler):
    """Handles auto-continue round marker events."""

    async def handle(self, event: dict[str, Any], state: StreamState) -> AsyncGenerator[str, None]:
        yield _sse(
            {
                "type": "data-continuation",
                "data": {
                    "round": event.get("round"),
                    "max_rounds": event.get("max_rounds"),
                    "reason": event.get("reason"),
                },
                "transient": True,
            }
        )


class NodeCompleteEventHandler(EventHandler):
    """Handles node completion events."""

    async def handle(self, event: dict[str, Any], state: StreamState) -> AsyncGenerator[str, None]:
        yield _sse(
            {
                "type": "data-node-complete",
                "data": {
                    "node": event.get("node"),
                },
                "transient": True,
            }
        )


class HeartbeatEventHandler(EventHandler):
    """Keeps long-running SSE streams alive while upstream work is quiet."""

    async def handle(self, event: dict[str, Any], state: StreamState) -> AsyncGenerator[str, None]:
        yield _sse({"type": "heartbeat"})


class RichItemsEventHandler(EventHandler):
    """Maps canonical `rich_items` upserts to AI SDK transient data parts.

    Image items are excluded by `select_transient_upsert_items()`; only safe
    created non-image records (widgets, tool renders, citations, resource
    links) reach the wire. Clients without the v1 capability never receive
    these events.
    """

    async def handle(self, event: dict[str, Any], state: StreamState) -> AsyncGenerator[str, None]:
        if not getattr(state, "inline_rich_response_v1", False):
            return
        from app.core.rich_response import select_transient_upsert_items

        items = event.get("items") or []
        safe_items = select_transient_upsert_items(items)
        if not safe_items:
            return
        operation = event.get("operation") or "upsert"
        yield _sse(
            {
                "type": "data-rich-items",
                "data": {"operation": operation, "items": safe_items},
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
        "continuation_start": ContinuationEventHandler(),
        "node_complete": NodeCompleteEventHandler(),
        "heartbeat": HeartbeatEventHandler(),
        "rich_items": RichItemsEventHandler(),
    }

    @classmethod
    def get_handler(cls, event_type: str) -> EventHandler | None:
        """Get handler for the given event type."""
        return cls._handlers.get(event_type)


def _build_ui_message_stream_response(
    event_source_factory: Callable[[], AsyncGenerator[dict[str, Any], None]],
    state: StreamState,
) -> StreamingResponse:
    async def events_with_heartbeats() -> AsyncGenerator[dict[str, Any], None]:
        source = event_source_factory()
        pending_next: asyncio.Task | None = None
        try:
            while True:
                if pending_next is None:
                    pending_next = asyncio.create_task(anext(source))

                done, _ = await asyncio.wait(
                    {pending_next},
                    timeout=_AI_SDK_HEARTBEAT_INTERVAL_SECONDS,
                )
                if not done:
                    yield {"type": "heartbeat"}
                    continue

                try:
                    event = pending_next.result()
                except StopAsyncIteration:
                    break
                pending_next = None
                yield event
        finally:
            if pending_next is not None and not pending_next.done():
                pending_next.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending_next
            aclose = getattr(source, "aclose", None)
            if callable(aclose):
                with contextlib.suppress(Exception):
                    await aclose()

    async def event_generator():
        try:
            yield _sse({"type": "start", "messageId": state.message_id})
            yield _sse({"type": "start-step"})
            yield _sse({"type": "text-start", "id": state.text_id})
            state.text_started = True

            async for event in events_with_heartbeats():
                event_type = event.get("type")
                handler = EventHandlerFactory.get_handler(event_type)

                if handler:
                    async for msg in handler.handle(event, state):
                        yield msg

                    if event_type in ("interrupt", "error"):
                        return

                    if event_type == "complete":
                        break

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
        "`useChat()`. Image attachments are embedded as `file` parts inside each message."
    ),
)
@AppAutoInjector.auto_inject()
async def get_conversation_messages_ai_sdk(
    conversation_id: UUID,
    message_service: IMessageService,
    current_user_id: UUID,
) -> ApiResponse[AISDKMessagesData]:
    """Get conversation messages in Vercel AI SDK UIMessage format."""
    paginated_result = message_service.get_conversation_messages(
        conversation_id,
        current_user_id,
        page=1,
        limit=100,
        order_by="created_at",
        order_direction="asc",
        include_feedback=False,
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
            message_payload["messageMetadata"] = msg.message_metadata
            message_payload["metadata"] = msg.message_metadata

        if role == "assistant":
            message_payload = _attach_image_parts_to_message(message_payload)

        messages.append(AISDKUIMessage.model_validate(message_payload))

    return ApiResponse(
        success=True,
        message="Messages retrieved successfully",
        data=AISDKMessagesData(messages=messages, total=paginated_result.meta.total),
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
    user_text = _extract_user_text(payload.messages)
    user_attachments = _extract_user_attachments(payload.messages)
    if not user_text and not user_attachments:
        raise HTTPException(status_code=400, detail="No user message found")

    # Pre-generate the bot message UUID so the stream's messageId matches the
    # DB record.  This prevents the Next.js client from showing a duplicate
    # message (one from the stream, one from the API) after re-fetching.
    bot_message_id = uuid4()

    extra = getattr(payload, "model_extra", {}) or {}
    inline_rich_response_v1 = bool(
        extra.get("inline_rich_response_v1") or extra.get("inlineRichResponseV1")
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
