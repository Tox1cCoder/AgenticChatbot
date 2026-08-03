"""Canonical v3 → Vercel AI SDK UI Message Stream adapter.

Consumes canonical :class:`V3StreamEvent` events and emits the AI SDK
UI Message Stream SSE chunks assistant-ui expects. The response header
``x-vercel-ai-ui-message-stream: v1`` is set by the endpoint that wraps this.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, Callable
from typing import Any

from app.core.config import settings
from app.core.exceptions import CustomHTTPException
from app.core.rich_response import select_transient_upsert_items

from .ai_sdk_projection import (
    attach_image_parts_to_message,
    clean_tool_output,
    coerce_json_object,
    find_message_metadata,
    is_v1_rich_items_message,
    project_ai_sdk_message_event,
    project_ai_sdk_message_for_capability,
    visible_image_file_parts,
)
from .events import (
    IMAGE_PREVIEW_STATUS_SKIPPED,
    SUBAGENT_PHASE_BY_EVENT,
    V3StreamEvent,
    apply_inline_preview_wire_budget,
    make_event,
    resolve_image_preview_delivery,
)

_AI_SDK_HEARTBEAT_INTERVAL_SECONDS = 15.0


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _ai_sdk_error_from_exception(exc: Exception) -> dict[str, Any]:
    """Project a known server exception to the AI SDK error event shape."""
    event: dict[str, Any] = {"type": "error", "errorText": str(exc)}
    if isinstance(exc, CustomHTTPException):
        event["errorText"] = str(exc.detail)
        event["statusCode"] = exc.status_code
        if exc.error_code is not None:
            event["errorCode"] = exc.error_code
    return event


def _ai_sdk_error_from_data(data: dict[str, Any]) -> dict[str, Any]:
    """Project known V3 error metadata to the AI SDK error event shape."""
    message = data.get("error") or data.get("message") or ""
    event: dict[str, Any] = {"type": "error", "errorText": message}
    status_code = data.get("status_code")
    if status_code is not None:
        event["statusCode"] = status_code
    error_code = data.get("error_code")
    if error_code is not None:
        event["errorCode"] = error_code
    return event


class AISDKV6StreamState:
    """Streaming state carried across an AI SDK UI message stream response."""

    def __init__(
        self,
        message_id: str,
        text_id: str,
        reasoning_id: str,
        *,
        inline_rich_response_v1: bool = False,
    ) -> None:
        self.message_id = message_id
        self.text_id = text_id
        self.reasoning_id = reasoning_id
        self.text_started = False
        self.reasoning_started = False
        self.any_text_delta = False
        self.tool_seq = 0
        self.pending_tool_call_ids: list[str] = []
        self.inline_rich_response_v1 = bool(inline_rich_response_v1)


class AISDKV6StreamAdapter:
    def __init__(
        self,
        event_source_factory: Callable[[], AsyncGenerator[V3StreamEvent, None]],
        state: AISDKV6StreamState,
        *,
        heartbeat_interval_seconds: float | None = None,
    ) -> None:
        self._event_source_factory = event_source_factory
        self._state = state
        self._heartbeat_interval = (
            heartbeat_interval_seconds
            if heartbeat_interval_seconds is not None
            else _AI_SDK_HEARTBEAT_INTERVAL_SECONDS
        )

    # -- public API --------------------------------------------------------

    async def iter_sse(self) -> AsyncGenerator[str, None]:
        state = self._state
        try:
            yield _sse({"type": "start", "messageId": state.message_id})
            yield _sse({"type": "start-step"})
            yield _sse({"type": "text-start", "id": state.text_id})
            state.text_started = True

            async for event in self._events_with_heartbeats():
                async for chunk in self._map_event(event):
                    yield chunk
                if event.type in {"interrupt", "error"}:
                    return  # terminal chunks already emitted by the handler
                if event.type == "complete":
                    break

            async for chunk in self._terminate():
                yield chunk
        except asyncio.CancelledError:
            return
        except Exception as exc:  # pragma: no cover - defensive
            yield _sse(_ai_sdk_error_from_exception(exc))
            async for chunk in self._terminate():
                yield chunk

    # -- terminal chunks ---------------------------------------------------

    async def _terminate(self) -> AsyncGenerator[str, None]:
        state = self._state
        if state.text_started:
            yield _sse({"type": "text-end", "id": state.text_id})
        if state.reasoning_started:
            yield _sse({"type": "reasoning-end", "id": state.reasoning_id})
        yield _sse({"type": "finish-step"})
        yield _sse({"type": "finish"})
        yield "data: [DONE]\n\n"

    # -- heartbeat wrapper -------------------------------------------------

    async def _events_with_heartbeats(
        self,
    ) -> AsyncGenerator[V3StreamEvent, None]:
        queue: asyncio.Queue[V3StreamEvent | object] = asyncio.Queue()
        source_complete = object()

        async def produce_events() -> None:
            source = self._event_source_factory()
            try:
                async for event in source:
                    await queue.put(event)
            finally:
                aclose = getattr(source, "aclose", None)
                if callable(aclose):
                    with contextlib.suppress(Exception):
                        await aclose()
                await queue.put(source_complete)

        producer = asyncio.create_task(produce_events())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=self._heartbeat_interval,
                    )
                except asyncio.TimeoutError:
                    yield make_event("heartbeat", sequence=0)
                    continue

                if item is source_complete:
                    await producer
                    break
                yield item
        finally:
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await producer

    # -- canonical event mapping ------------------------------------------

    async def _map_event(self, event: V3StreamEvent) -> AsyncGenerator[str, None]:
        state = self._state
        etype = event.type
        data = event.data or {}

        if etype == "message_delta":
            delta = data.get("text") or ""
            if delta:
                state.any_text_delta = True
                yield _sse({"type": "text-delta", "id": state.text_id, "delta": delta})
            return

        if etype == "reasoning_delta":
            delta = data.get("text") or ""
            if not delta:
                return
            if not state.reasoning_started:
                state.reasoning_started = True
                yield _sse({"type": "reasoning-start", "id": state.reasoning_id})
            yield _sse({"type": "reasoning-delta", "id": state.reasoning_id, "delta": delta})
            return

        if etype == "tool_call_available":
            async for chunk in self._tool_input(event):
                yield chunk
            return

        if etype == "tool_execution_end":
            async for chunk in self._tool_output(event):
                yield chunk
            return

        if etype == "rich_items":
            async for chunk in self._rich_items(data):
                yield chunk
            return

        if etype == "image_preview":
            async for chunk in self._image_preview(data):
                yield chunk
            return

        if etype == "agent_selected":
            yield _sse(
                {
                    "type": "data-agent-selected",
                    "data": {"agent": event.agent or data.get("agent")},
                    "transient": True,
                }
            )
            return

        if etype == "user_message_created":
            message = data.get("message")
            if isinstance(message, dict):
                message = project_ai_sdk_message_event(message, include_content=True)
            yield _sse(
                {
                    "type": "data-user-message",
                    "data": {"message": message},
                    "transient": True,
                }
            )
            return

        if etype == "state_snapshot":
            legacy_type = data.get("legacy_type")
            if legacy_type == "continuation_start":
                yield _sse(
                    {
                        "type": "data-continuation",
                        "data": {
                            "round": data.get("round"),
                            "max_rounds": data.get("max_rounds"),
                            "reason": data.get("reason"),
                        },
                        "transient": True,
                    }
                )
            elif legacy_type == "node_complete":
                yield _sse(
                    {
                        "type": "data-node-complete",
                        "data": {"node": data.get("node") or event.node},
                        "transient": True,
                    }
                )
            return

        if etype == "heartbeat":
            yield _sse({"type": "heartbeat"})
            return

        if etype == "interrupt":
            async for chunk in self._interrupt(data):
                yield chunk
            return

        if etype == "error":
            async for chunk in self._error(data):
                yield chunk
            return

        if etype == "complete":
            async for chunk in self._complete(data):
                yield chunk
            return

        if etype in SUBAGENT_PHASE_BY_EVENT:
            async for chunk in self._subagent(event):
                yield chunk
            return

        # message_start/end, reasoning_start/end, tool_call_delta,
        # title_updated carry no assistant-ui projection without a capability flag.

    # -- tool handlers -----------------------------------------------------

    def _resolve_tool_call_id(self, event: V3StreamEvent, *, is_end: bool) -> str:
        state = self._state
        if event.tool_call_id:
            return str(event.tool_call_id)
        if is_end and state.pending_tool_call_ids:
            return state.pending_tool_call_ids.pop(0)
        state.tool_seq += 1
        return f"tool_{state.tool_seq}"

    async def _tool_input(self, event: V3StreamEvent) -> AsyncGenerator[str, None]:
        state = self._state
        tool_call_id = self._resolve_tool_call_id(event, is_end=False)
        tool_name = event.tool_name or "unknown"
        if tool_call_id not in state.pending_tool_call_ids:
            state.pending_tool_call_ids.append(tool_call_id)
        yield _sse({"type": "tool-input-start", "toolCallId": tool_call_id, "toolName": tool_name})
        yield _sse(
            {
                "type": "tool-input-available",
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "input": coerce_json_object(clean_tool_output((event.data or {}).get("args"))),
            }
        )

    async def _tool_output(self, event: V3StreamEvent) -> AsyncGenerator[str, None]:
        state = self._state
        tool_call_id = self._resolve_tool_call_id(event, is_end=True)
        if tool_call_id in state.pending_tool_call_ids:
            with contextlib.suppress(ValueError):
                state.pending_tool_call_ids.remove(tool_call_id)
        data = event.data or {}
        payload: dict[str, Any] = {
            "type": "tool-output-available",
            "toolCallId": tool_call_id,
            "output": coerce_json_object(clean_tool_output(data.get("output"))),
        }
        render = data.get("render")
        if isinstance(render, dict):
            payload["render"] = render
        yield _sse(payload)

    async def _rich_items(self, data: dict[str, Any]) -> AsyncGenerator[str, None]:
        if not self._state.inline_rich_response_v1:
            return
        safe_items = select_transient_upsert_items(data.get("items") or [])
        if not safe_items:
            return
        yield _sse(
            {
                "type": "data-rich-items",
                "data": {"operation": data.get("operation") or "upsert", "items": safe_items},
                "transient": True,
            }
        )

    async def _image_preview(self, data: dict[str, Any]) -> AsyncGenerator[str, None]:
        """Project an early-delivery image preview as a transient data part.

        The stable part ``id`` (one per image index) lets AI SDK clients
        replace a partial/preview with the next partial/final in place. Reads
        both schema-v1 (top-level ``data_b64``) and schema-v2 (``delivery``)
        payloads. Inline deliveries carry an assembled ``data:`` URL; a final
        reference carries the protected relative ``/chat-images/...`` URL
        verbatim so a credentialed client can fetch it. The authoritative image
        still arrives as a ``file`` part on ``complete`` for clients that ignore
        preview events.
        """
        item_id = data.get("item_id")
        if not item_id:
            return

        # Second-defense inline budget at serialization time (FR-IMG-007).
        budget = getattr(settings, "image_stream_preview_max_b64_chars", 0)
        data = apply_inline_preview_wire_budget(data, budget=budget)

        projection = resolve_image_preview_delivery(data)

        if projection["status"] == IMAGE_PREVIEW_STATUS_SKIPPED:
            # Structured status — never a silent drop.
            yield _sse(
                {
                    "type": "data-image-preview",
                    "id": str(item_id),
                    "data": {
                        "imageIndex": projection["image_index"],
                        "status": IMAGE_PREVIEW_STATUS_SKIPPED,
                        "seq": projection["seq"],
                        "reason": projection["reason"],
                    },
                    "transient": True,
                }
            )
            return

        url = projection["url"]
        if not url:
            return
        yield _sse(
            {
                "type": "data-image-preview",
                "id": str(item_id),
                "data": {
                    "imageIndex": projection["image_index"],
                    "status": projection["status"],
                    "mediaType": projection["media_type"],
                    "url": url,
                    "seq": projection["seq"],
                },
                "transient": True,
            }
        )

    async def _subagent(self, event: V3StreamEvent) -> AsyncGenerator[str, None]:
        data = event.data or {}
        payload_data: dict[str, Any] = {
            "phase": SUBAGENT_PHASE_BY_EVENT.get(event.type, "update"),
            "subagent": event.subagent.model_dump(mode="json") if event.subagent else None,
        }
        if event.tool_call_id:
            payload_data["toolCallId"] = event.tool_call_id
        if event.tool_name:
            payload_data["toolName"] = event.tool_name
        for src_key, out_key in (
            ("task", "task"),
            ("output", "output"),
            ("summary", "summary"),
            ("thinking", "thinking"),
            ("status", "status"),
            ("error", "error"),
            ("render", "render"),
            ("text", "text"),
            ("channel", "channel"),
            ("elapsed_ms", "elapsedMs"),
            ("requested_model", "requestedModel"),
            ("resolved_model", "resolvedModel"),
        ):
            value = data.get(src_key)
            if value is not None:
                payload_data[out_key] = value
        yield _sse({"type": "data-subagent", "data": payload_data, "transient": True})

    # -- terminal-bearing handlers ----------------------------------------

    async def _interrupt(self, data: dict[str, Any]) -> AsyncGenerator[str, None]:
        state = self._state
        message = data.get("message")
        if isinstance(message, str) and message.strip():
            yield _sse({"type": "text-delta", "id": state.text_id, "delta": message.strip()})
        projected_message = message
        if isinstance(message, dict):
            projected_message = project_ai_sdk_message_event(message, include_content=True)
        if state.text_started:
            yield _sse({"type": "text-end", "id": state.text_id})
        if state.reasoning_started:
            yield _sse({"type": "reasoning-end", "id": state.reasoning_id})
        yield _sse(
            {
                "type": "data-interrupt",
                "data": {
                    "threadId": data.get("thread_id"),
                    "next": data.get("next"),
                    "interrupt": data.get("interrupt"),
                    "message": projected_message,
                },
            }
        )
        yield _sse({"type": "finish-step"})
        yield _sse({"type": "finish"})
        yield "data: [DONE]\n\n"

    async def _error(self, data: dict[str, Any]) -> AsyncGenerator[str, None]:
        state = self._state
        yield _sse(_ai_sdk_error_from_data(data))
        if state.text_started:
            yield _sse({"type": "text-end", "id": state.text_id})
        if state.reasoning_started:
            yield _sse({"type": "reasoning-end", "id": state.reasoning_id})
        message = data.get("message")
        if isinstance(message, dict):
            message = project_ai_sdk_message_event(message, include_content=True)
        if message:
            yield _sse(
                {"type": "data-error-message", "data": {"message": message}, "transient": True}
            )
        yield _sse({"type": "finish-step"})
        yield _sse({"type": "finish"})
        yield "data: [DONE]\n\n"

    async def _complete(self, data: dict[str, Any]) -> AsyncGenerator[str, None]:
        state = self._state
        original_message = data.get("message") or {}
        message = original_message
        is_v1 = False
        file_parts: list[dict[str, str]] = []
        if isinstance(original_message, dict):
            # v1-ness is decided on the original metadata so hidden image
            # candidates cannot fall back onto legacy `images` after the
            # capability projection strips the rich keys.
            is_v1 = is_v1_rich_items_message(find_message_metadata(original_message))
            file_parts = visible_image_file_parts(
                original_message,
                is_v1=is_v1,
                inline_rich_response_v1=state.inline_rich_response_v1,
            )
            message = project_ai_sdk_message_for_capability(
                original_message,
                inline_rich_response_v1=state.inline_rich_response_v1,
            )
            message = attach_image_parts_to_message(
                message,
                image_parts=file_parts,
                is_v1=is_v1,
            )

        if not state.any_text_delta:
            content = message.get("content") or "" if isinstance(message, dict) else ""
            if isinstance(content, str) and content.strip():
                state.any_text_delta = True
                yield _sse({"type": "text-delta", "id": state.text_id, "delta": content})

        if isinstance(message, dict):
            for file_part in file_parts:
                yield _sse(
                    {
                        "type": "file",
                        "url": file_part["url"],
                        "mediaType": file_part["mediaType"],
                    }
                )

        if isinstance(message, dict) and message:
            message_meta = project_ai_sdk_message_event(message)
            # A message carrying only an id has no side-channel metadata worth a
            # dedicated data part (the streamed text already conveyed the body).
            if message_meta and any(key != "id" for key in message_meta):
                yield _sse(
                    {
                        "type": "data-assistant-message",
                        "data": {"message": message_meta},
                    }
                )
