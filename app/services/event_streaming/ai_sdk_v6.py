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

from .events import SUBAGENT_PHASE_BY_EVENT, V3StreamEvent, make_event

_AI_SDK_HEARTBEAT_INTERVAL_SECONDS = float(
    getattr(settings, "ai_sdk_heartbeat_interval_seconds", 15.0) or 15.0
)


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


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
            yield _sse({"type": "error", "errorText": str(exc)})
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
        source = self._event_source_factory()
        pending_next: asyncio.Task | None = None
        try:
            while True:
                if pending_next is None:
                    pending_next = asyncio.create_task(anext(source))
                done, _ = await asyncio.wait({pending_next}, timeout=self._heartbeat_interval)
                if not done:
                    yield make_event("heartbeat", sequence=0)
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
            yield _sse(
                {
                    "type": "data-user-message",
                    "data": {"message": data.get("message")},
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
        from app.api.ai_sdk import _clean_tool_output, _coerce_json_object

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
                "input": _coerce_json_object(_clean_tool_output((event.data or {}).get("args"))),
            }
        )

    async def _tool_output(self, event: V3StreamEvent) -> AsyncGenerator[str, None]:
        from app.api.ai_sdk import _clean_tool_output, _coerce_json_object

        state = self._state
        tool_call_id = self._resolve_tool_call_id(event, is_end=True)
        if tool_call_id in state.pending_tool_call_ids:
            with contextlib.suppress(ValueError):
                state.pending_tool_call_ids.remove(tool_call_id)
        data = event.data or {}
        payload: dict[str, Any] = {
            "type": "tool-output-available",
            "toolCallId": tool_call_id,
            "output": _coerce_json_object(_clean_tool_output(data.get("output"))),
        }
        render = data.get("render")
        if isinstance(render, dict):
            payload["render"] = render
        yield _sse(payload)

    async def _rich_items(self, data: dict[str, Any]) -> AsyncGenerator[str, None]:
        if not self._state.inline_rich_response_v1:
            return
        from app.core.rich_response import select_transient_upsert_items

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
            ("status", "status"),
            ("error", "error"),
            ("render", "render"),
            ("text", "text"),
            ("elapsed_ms", "elapsedMs"),
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
            from app.api.ai_sdk import project_ai_sdk_assistant_message_event

            projected_message = project_ai_sdk_assistant_message_event(
                message,
                include_content=True,
            )
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
                    "pendingToolCalls": data.get("pending_tool_calls"),
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
        yield _sse({"type": "error", "errorText": data.get("error") or data.get("message") or ""})
        if state.text_started:
            yield _sse({"type": "text-end", "id": state.text_id})
        if state.reasoning_started:
            yield _sse({"type": "reasoning-end", "id": state.reasoning_id})
        message = data.get("message")
        if message:
            yield _sse(
                {"type": "data-error-message", "data": {"message": message}, "transient": True}
            )
        yield _sse({"type": "finish-step"})
        yield _sse({"type": "finish"})
        yield "data: [DONE]\n\n"

    async def _complete(self, data: dict[str, Any]) -> AsyncGenerator[str, None]:
        from app.api.ai_sdk import (
            _attach_image_parts_to_message,
            _extract_image_file_parts_from_message,
            _is_v1_rich_items_message,
            _scrub_v1_legacy_image_fields,
            _selected_image_file_parts_from_rich_items,
            project_ai_sdk_assistant_message_event,
            project_ai_sdk_message_for_capability,
        )

        state = self._state
        message = data.get("message") or {}
        if isinstance(message, dict):
            message = project_ai_sdk_message_for_capability(
                message,
                inline_rich_response_v1=state.inline_rich_response_v1,
            )
            message = _attach_image_parts_to_message(message)

        if not state.any_text_delta:
            content = message.get("content") or "" if isinstance(message, dict) else ""
            if isinstance(content, str) and content.strip():
                state.any_text_delta = True
                yield _sse({"type": "text-delta", "id": state.text_id, "delta": content})

        metadata = None
        if isinstance(message, dict):
            for key in ("message_metadata", "messageMetadata", "metadata"):
                value = message.get(key)
                if isinstance(value, dict):
                    metadata = value
                    break
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

        if isinstance(message, dict) and message:
            message_meta = project_ai_sdk_assistant_message_event(message)
            if _is_v1_rich_items_message(metadata):
                _scrub_v1_legacy_image_fields(message_meta)
            # A message carrying only an id has no side-channel metadata worth a
            # dedicated data part (the streamed text already conveyed the body).
            if message_meta and any(key != "id" for key in message_meta):
                yield _sse(
                    {
                        "type": "data-assistant-message",
                        "data": {"message": message_meta},
                    }
                )
