"""Canonical v3 stream → legacy public-event projection for the graph boundary.

``MultiAgentWorkflow.execute_request_stream``/``resume_with_decisions_stream``
consume canonical ``V3StreamEvent``s from ``iter_v3_events_from_graph`` (either
the experimental v3 protocol or the v1/v2 tuple fallback — see
``langchain_v3.py``) and must still emit the legacy public dicts
(``token``/``thinking``/``tool_start``/``tool_end``/``node_complete``/
``continuation_start``/``agent_selected``/``interrupt``/``complete``/
``error``) that ``AIService`` consumes today. This module holds that
projection so it is unit-testable independently of the graph. Delete once the
graph emits ``V3StreamEvent`` directly to the service layer.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from ...ai.utils import coerce_response_text, make_json_safe, normalize_tool_call

# Bound to MultiAgentWorkflow._tool_end_events_from_node_state — stays on the
# workflow because it is a tool-loop helper shared with non-streaming code.
ToolEndEventsFromNodeState = Callable[..., Iterator[dict[str, Any]]]


@dataclass
class StreamProjectionContext:
    """Mutable per-stream accumulator state shared across continuation rounds.

    Holds everything the legacy public-event mapping needs while consuming
    canonical ``V3StreamEvent``s from ``iter_v3_events_from_graph``. Read back
    after the stream loop for terminal-response recovery.
    """

    last_emitted_agent: str | None = None
    suppress_tokens: bool = False
    internal_content_only: bool = True
    accumulated_content: str = ""
    accumulated_thinking: str = ""
    last_state_values: dict[str, Any] | None = None
    current_tool_calls: dict[Any, dict[str, Any]] = field(default_factory=dict)
    emitted_tool_call_ids: set[str] = field(default_factory=set)
    emitted_tool_result_ids: set[str] = field(default_factory=set)


def _is_internal_stream_chunk(metadata: Any) -> bool:
    """Return True if this stream chunk originates from an internal (non-user-facing) LLM run.

    Checks, in priority order:
    1. The `tags` list for the 'internal' tag (set via RunnableConfig in generate_summary).
    2. The nested `metadata` dict's `internal` key (also set via RunnableConfig).
    """
    if not isinstance(metadata, dict):
        return False

    tags = metadata.get("tags") or []
    if "internal" in tags:
        return True

    nested_metadata = metadata.get("metadata") or {}
    return isinstance(nested_metadata, dict) and nested_metadata.get("internal") is True


def _consume_stream_text_chunk(
    accumulated_content: str, text_chunk: Any
) -> tuple[str, str | None]:
    """
    Return (new_accumulated_content, delta_to_emit) for a streaming text chunk.

    Handles both:
    - Cumulative chunks (Gemini): Each chunk contains all text accumulated so far
    - Incremental chunks (OpenAI): Each chunk contains only new text
    """
    if not text_chunk:
        return accumulated_content, None

    chunk_text = coerce_response_text(text_chunk)
    if not chunk_text:
        return accumulated_content, None

    if accumulated_content:
        # Exact repeat of what we've already accumulated - skip it
        if chunk_text == accumulated_content:
            return accumulated_content, None

        # Cumulative chunk: new chunk starts with what we already have (Gemini)
        if chunk_text.startswith(accumulated_content):
            delta = chunk_text[len(accumulated_content) :]
            return chunk_text, delta or None

        # Duplicate tail chunk - skip it
        if accumulated_content.endswith(chunk_text):
            return accumulated_content, None

    # Incremental chunk (OpenAI) or first chunk - append and emit
    return accumulated_content + chunk_text, chunk_text


class GraphPublicStreamProjector:
    """Projects canonical v3 stream events onto the legacy public dict vocabulary.

    ``tool_end_events_from_node_state`` stays injected rather than moved
    because it is a tool-loop helper shared with non-streaming workflow code
    (``MultiAgentWorkflow._tool_end_events_from_node_state``).
    """

    def __init__(
        self,
        *,
        tool_end_events_from_node_state: ToolEndEventsFromNodeState,
        suppress_internal_stream_chunks: bool,
    ) -> None:
        self._tool_end_events_from_node_state = tool_end_events_from_node_state
        self._suppress_internal_stream_chunks = suppress_internal_stream_chunks

    def map_event(self, event: Any, ctx: StreamProjectionContext):
        """Map one canonical ``V3StreamEvent`` to legacy public stream dicts.

        ``iter_v3_events_from_graph`` yields canonical events from either the
        experimental v3 protocol (clean ``message_delta`` / ``tool_call_available``
        / ``tool_execution_end`` / ``state_snapshot[kind=values]``) or the v1/v2
        tuple fallback (``state_snapshot[kind=messages_tuple|updates_tuple]``
        carrying raw chunks). Both converge here onto the legacy vocabulary that
        the service/adapter layers currently consume.
        """
        etype = event.type
        data = event.data or {}

        if etype == "message_delta":
            ctx.internal_content_only = False
            ctx.accumulated_content, delta = _consume_stream_text_chunk(
                ctx.accumulated_content, data.get("text", "")
            )
            if delta and not ctx.suppress_tokens:
                yield {"type": "token", "content": delta}
            return

        if etype == "reasoning_delta":
            ctx.internal_content_only = False
            text = data.get("text", "")
            if text:
                ctx.accumulated_thinking += text
                yield {"type": "thinking", "content": text}
            return

        if etype == "tool_call_available":
            yield from self._emit_tool_start_from_canonical(
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                args=data.get("args", {}),
                ctx=ctx,
            )
            return

        if etype == "tool_execution_end":
            tool_call_id = event.tool_call_id
            dedupe_key = str(tool_call_id) if tool_call_id else None
            if dedupe_key and dedupe_key in ctx.emitted_tool_result_ids:
                return
            if dedupe_key:
                ctx.emitted_tool_result_ids.add(dedupe_key)
            payload = {
                "type": "tool_end",
                "name": event.tool_name or "unknown",
                "tool_call_id": tool_call_id,
                "result": make_json_safe(data.get("output")),
            }
            if data.get("render"):
                payload["render"] = data["render"]
            yield payload
            return

        if etype == "state_snapshot":
            kind = data.get("kind")
            if kind == "messages_tuple":
                yield from self._map_legacy_message_chunk(
                    data.get("chunk"), data.get("metadata"), ctx
                )
            elif kind == "updates_tuple":
                yield from self._map_legacy_update_node(event.node, data.get("node_state"), ctx)
            elif kind == "values":
                yield from self._map_v3_values_snapshot(data, ctx)
            return

        if etype in (
            "subagent_start",
            "subagent_end",
            "subagent_message_delta",
            "subagent_tool_call_available",
            "subagent_tool_execution_start",
            "subagent_tool_execution_end",
        ):
            payload: dict[str, Any] = {"type": etype, "data": dict(data)}
            if event.subagent is not None:
                payload["subagent"] = event.subagent.model_dump(mode="json")
            if event.tool_call_id:
                payload["tool_call_id"] = event.tool_call_id
            if event.tool_name:
                payload["tool_name"] = event.tool_name
            yield payload
            return

        # message_start/end, reasoning_start/end, tool_call_delta carry no
        # legacy public projection in this bridge.

    def _emit_tool_start_from_canonical(
        self,
        *,
        tool_call_id: str | None,
        tool_name: str | None,
        args: Any,
        ctx: StreamProjectionContext,
    ):
        if tool_call_id and tool_call_id in ctx.emitted_tool_call_ids:
            return
        if tool_call_id:
            ctx.emitted_tool_call_ids.add(tool_call_id)
        yield {
            "type": "tool_start",
            "name": tool_name or "unknown",
            "tool_call_id": tool_call_id,
            "args": make_json_safe(args or {}),
        }

    def _map_legacy_message_chunk(
        self, message_chunk: Any, metadata: Any, ctx: StreamProjectionContext
    ):
        """Reproduce the legacy ``stream_mode="messages"`` chunk handling."""
        if isinstance(message_chunk, ToolMessage):
            return
        if self._suppress_internal_stream_chunks and _is_internal_stream_chunk(metadata):
            return

        ctx.internal_content_only = False

        if hasattr(message_chunk, "content_blocks") and message_chunk.content_blocks:
            for block in message_chunk.content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    ctx.accumulated_content, delta = _consume_stream_text_chunk(
                        ctx.accumulated_content, block.get("text", "")
                    )
                    if delta and not ctx.suppress_tokens:
                        yield {"type": "token", "content": delta}
                elif block_type == "thinking":
                    thinking_content = block.get("thinking", "") or block.get("text", "")
                    if thinking_content:
                        ctx.accumulated_thinking += thinking_content
                        yield {"type": "thinking", "content": thinking_content}
                elif block_type == "reasoning":
                    reasoning_content = block.get("reasoning", "") or block.get("text", "")
                    if reasoning_content:
                        ctx.accumulated_thinking += reasoning_content
                        yield {"type": "thinking", "content": reasoning_content}
                elif block_type == "tool_call_chunk":
                    tool_index = block.get("index", 0)
                    tool_id = block.get("id")
                    tool_name = block.get("name")
                    tool_args = block.get("args", "")
                    if tool_index not in ctx.current_tool_calls:
                        ctx.current_tool_calls[tool_index] = {
                            "id": tool_id,
                            "name": tool_name,
                            "args": "",
                        }
                    if tool_args:
                        ctx.current_tool_calls[tool_index]["args"] += tool_args
                    if tool_name and not ctx.current_tool_calls[tool_index]["name"]:
                        ctx.current_tool_calls[tool_index]["name"] = tool_name
                    if tool_id and not ctx.current_tool_calls[tool_index]["id"]:
                        ctx.current_tool_calls[tool_index]["id"] = tool_id

        elif hasattr(message_chunk, "content") and isinstance(message_chunk.content, list):
            for part in message_chunk.content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    if part_type == "thinking":
                        thinking_content = part.get("thinking", "") or part.get("text", "")
                        if thinking_content:
                            ctx.accumulated_thinking += thinking_content
                            yield {"type": "thinking", "content": thinking_content}
                    elif part_type == "reasoning":
                        reasoning_content = part.get("reasoning", "") or part.get("text", "")
                        if reasoning_content:
                            ctx.accumulated_thinking += reasoning_content
                            yield {"type": "thinking", "content": reasoning_content}
                    elif part_type == "text":
                        ctx.accumulated_content, delta = _consume_stream_text_chunk(
                            ctx.accumulated_content, part.get("text", "")
                        )
                        if delta and not ctx.suppress_tokens:
                            yield {"type": "token", "content": delta}
                elif isinstance(part, str) and part:
                    ctx.accumulated_content, delta = _consume_stream_text_chunk(
                        ctx.accumulated_content, part
                    )
                    if delta and not ctx.suppress_tokens:
                        yield {"type": "token", "content": delta}

        elif (
            hasattr(message_chunk, "content")
            and message_chunk.content
            and isinstance(message_chunk.content, str)
        ):
            content = coerce_response_text(message_chunk.content)
            ctx.accumulated_content, delta = _consume_stream_text_chunk(
                ctx.accumulated_content, content
            )
            if delta and not ctx.suppress_tokens:
                yield {"type": "token", "content": delta}

        if hasattr(message_chunk, "chunk_position") and message_chunk.chunk_position == "last":
            for tool_call in ctx.current_tool_calls.values():
                if tool_call["name"]:
                    tool_call_id = tool_call["id"]
                    if tool_call_id and tool_call_id in ctx.emitted_tool_call_ids:
                        continue
                    if tool_call_id:
                        ctx.emitted_tool_call_ids.add(tool_call_id)
                    try:
                        args = json.loads(tool_call["args"]) if tool_call["args"] else {}
                    except Exception:
                        args = tool_call["args"]
                    yield {
                        "type": "tool_start",
                        "name": tool_call["name"],
                        "tool_call_id": tool_call_id,
                        "args": make_json_safe(args),
                    }
            ctx.current_tool_calls = {}

    def _map_legacy_update_node(
        self, node_name: Any, node_state: Any, ctx: StreamProjectionContext
    ):
        """Reproduce the legacy ``stream_mode="updates"`` per-node handling."""
        if isinstance(node_state, dict):
            if ctx.last_state_values is None:
                ctx.last_state_values = {}
            ctx.last_state_values.update(node_state)

            new_agent = node_state.get("selected_agent")
            if isinstance(new_agent, str) and new_agent != ctx.last_emitted_agent:
                ctx.last_emitted_agent = new_agent
                yield {"type": "agent_selected", "agent": new_agent, "reason": "handoff"}

        if node_name in ("planning_agent", "planning_tools") and isinstance(node_state, dict):
            node_info: dict[str, Any] = {"node": node_name}
            if node_name == "planning_agent" and "messages" in node_state:
                messages = node_state.get("messages", [])
                if messages:
                    last_msg = messages[-1] if isinstance(messages, list) else messages
                    if (
                        isinstance(last_msg, AIMessage)
                        and hasattr(last_msg, "tool_calls")
                        and last_msg.tool_calls
                    ):
                        node_info["tool_calls"] = [
                            {
                                "name": normalized_tc.get("name"),
                                "id": normalized_tc.get("id"),
                                "args": make_json_safe(normalized_tc.get("args", {})),
                            }
                            for normalized_tc in (
                                normalize_tool_call(tc) for tc in last_msg.tool_calls
                            )
                        ]
            if node_name == "planning_tools":
                todos = node_state.get("todos", [])
                if todos:
                    node_info["todos_count"] = len(todos)
                    node_info["current_task_index"] = node_state.get("current_task_index")
            yield {"type": "node_complete", **node_info}

        if isinstance(node_state, dict) and "messages" in node_state:
            messages = node_state["messages"]
            if messages:
                last_msg = messages[-1] if isinstance(messages, list) else messages
                if isinstance(last_msg, AIMessage):
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            normalized_tool_call = normalize_tool_call(tool_call)
                            tool_call_id = normalized_tool_call.get("id")
                            if tool_call_id and tool_call_id not in ctx.emitted_tool_call_ids:
                                ctx.emitted_tool_call_ids.add(tool_call_id)
                                yield {
                                    "type": "tool_start",
                                    "name": normalized_tool_call.get("name", "unknown"),
                                    "tool_call_id": tool_call_id,
                                    "args": make_json_safe(normalized_tool_call.get("args", {})),
                                }
                elif isinstance(last_msg, ToolMessage):
                    yield from self._tool_end_events_from_node_state(
                        node_state=node_state,
                        last_state_values=ctx.last_state_values,
                        emitted_tool_result_ids=ctx.emitted_tool_result_ids,
                    )

    def _map_v3_values_snapshot(self, data: dict[str, Any], ctx: StreamProjectionContext):
        """Derive handoff / planning node_complete / tool_start from a v3 full-state
        ``values`` snapshot, since the v3 protocol has no per-node ``updates`` channel.

        Tool execution end and the per-token text are emitted from the dedicated
        canonical events (``tool_execution_end`` / ``message_delta``); only the
        state-derived signals are produced here.
        """
        values = data.get("values")
        new_messages = data.get("new_messages") or []
        selected_agent = data.get("selected_agent")

        if isinstance(values, dict):
            if ctx.last_state_values is None:
                ctx.last_state_values = {}
            ctx.last_state_values.update(values)

        if isinstance(selected_agent, str) and selected_agent != ctx.last_emitted_agent:
            ctx.last_emitted_agent = selected_agent
            yield {"type": "agent_selected", "agent": selected_agent, "reason": "handoff"}

        # Best-effort planning node_complete derived from the newly added messages.
        if selected_agent in ("planning_agent", "planning_tools"):
            planning_ai = next(
                (
                    msg
                    for msg in reversed(new_messages)
                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)
                ),
                None,
            )
            if planning_ai is not None:
                node_info: dict[str, Any] = {"node": "planning_agent"}
                node_info["tool_calls"] = [
                    {
                        "name": normalized_tc.get("name"),
                        "id": normalized_tc.get("id"),
                        "args": make_json_safe(normalized_tc.get("args", {})),
                    }
                    for normalized_tc in (normalize_tool_call(tc) for tc in planning_ai.tool_calls)
                ]
                yield {"type": "node_complete", **node_info}

        # Tool-call availability also surfaces via the messages channel; dedupe
        # guards against double emission when both paths see the same call.
        for msg in new_messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tool_call in msg.tool_calls:
                    normalized_tool_call = normalize_tool_call(tool_call)
                    tool_call_id = normalized_tool_call.get("id")
                    if tool_call_id and tool_call_id not in ctx.emitted_tool_call_ids:
                        ctx.emitted_tool_call_ids.add(tool_call_id)
                        yield {
                            "type": "tool_start",
                            "name": normalized_tool_call.get("name", "unknown"),
                            "tool_call_id": tool_call_id,
                            "args": make_json_safe(normalized_tool_call.get("args", {})),
                        }
