"""LangGraph stream → canonical v3 event normalization.

Two source paths feed the canonical :class:`V3StreamEvent` model:

1. **Experimental v3 protocol** (``await graph.astream_events(version="v3")``).
   This is the active production path for real compiled graphs. The protocol
   emits ``{type, method, params, seq}`` envelopes on the ``messages``,
   ``values`` and ``lifecycle`` channels (see the Implementation Log in
   ``event_streaming.md`` for the captured schema). :class:`V3ProtocolTranslator`
   converts those into canonical events.

2. **v1/v2 tuple fallback** (``astream(stream_mode=["messages","updates"])``).
   Used for runnables that do not implement the v3 protocol (e.g. test
   doubles). :class:`LangGraphV3Normalizer` and :func:`normalize_update_chunk`
   convert the tuples; :func:`iter_v3_events_from_graph` wraps the raw tuples in
   ``state_snapshot`` carrier events so the graph mapper can reproduce the exact
   legacy behavior.

:func:`iter_v3_events_from_graph` dispatches between the two and always yields
canonical :class:`V3StreamEvent` objects.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import AsyncGenerator, Iterable
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from .events import SubagentRef, V3StreamEvent, make_event

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _node_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("langgraph_node") or metadata.get("node")
    return str(value) if value else None


def _text_from_block(block: dict[str, Any]) -> str:
    for key in ("text", "content", "reasoning"):
        value = block.get(key)
        if isinstance(value, str):
            return value
    return ""


def _message_list_from_node_state(node_state: Any) -> list[Any]:
    if not isinstance(node_state, dict):
        return []
    messages = node_state.get("messages")
    return list(messages) if isinstance(messages, list) else []


# ---------------------------------------------------------------------------
# v1/v2 tuple fallback normalization
# ---------------------------------------------------------------------------


class LangGraphV3Normalizer:
    """Convert v1/v2 ``stream_mode="messages"`` chunks to canonical events.

    Used by the fallback path for runnables that don't implement the v3
    protocol. Kept deliberately small — the production graph runs the v3
    protocol path.
    """

    def __init__(self) -> None:
        self._sequence = 0

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def from_message_chunk(
        self,
        chunk: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> V3StreamEvent | None:
        node = _node_from_metadata(metadata)
        blocks = getattr(chunk, "content_blocks", None)
        if isinstance(blocks, list):
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type in {"thinking", "reasoning"}:
                    text = _text_from_block(block)
                    if text:
                        return make_event(
                            "reasoning_delta",
                            sequence=self.next_sequence(),
                            node=node,
                            agent=node,
                            data={"text": text},
                        )
                if block_type == "tool_call_chunk":
                    call_id = block.get("id") or block.get("tool_call_id")
                    return make_event(
                        "tool_call_delta",
                        sequence=self.next_sequence(),
                        node=node,
                        agent=node,
                        tool_call_id=str(call_id) if call_id else None,
                        tool_name=block.get("name"),
                        data={"args_delta": block.get("args") or ""},
                    )

        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content:
            return make_event(
                "message_delta",
                sequence=self.next_sequence(),
                node=node,
                agent=node,
                data={"text": content},
            )
        return None


def normalize_update_chunk(
    chunk: dict[str, Any],
    *,
    sequence_start: int,
) -> Iterable[V3StreamEvent]:
    sequence = sequence_start
    for node, node_state in chunk.items():
        for message in _message_list_from_node_state(node_state):
            if isinstance(message, ToolMessage):
                yield make_event(
                    "tool_execution_end",
                    sequence=sequence,
                    node=str(node),
                    agent=str(node),
                    tool_call_id=str(message.tool_call_id) if message.tool_call_id else None,
                    tool_name=getattr(message, "name", None) or "tool",
                    data={"output": message.content},
                )
                sequence += 1
            elif isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    yield make_event(
                        "tool_call_available",
                        sequence=sequence,
                        node=str(node),
                        agent=str(node),
                        tool_call_id=str(tool_call.get("id")) if tool_call.get("id") else None,
                        tool_name=tool_call.get("name"),
                        data={"args": tool_call.get("args") or {}},
                    )
                    sequence += 1


# ---------------------------------------------------------------------------
# Experimental v3 protocol translation
# ---------------------------------------------------------------------------


def _message_event_and_metadata(data: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """``messages`` channel data is ``(message_event, metadata)``.

    LangGraph emits this as a tuple at runtime (it serializes to a JSON array),
    so accept both tuples and lists. A bare mapping is also tolerated.
    """
    if isinstance(data, (list, tuple)) and data:
        from collections.abc import Mapping

        head = data[0] if isinstance(data[0], Mapping) else None
        meta = data[1] if len(data) > 1 and isinstance(data[1], Mapping) else {}
        return head, meta
    if isinstance(data, dict):
        return data, {}
    return None, {}


class V3ProtocolTranslator:
    """Translate experimental v3 protocol envelopes into canonical events.

    Stateful across a single run: tracks emitted sequence numbers, the
    ``ToolMessage`` ids already surfaced from ``values`` snapshots, and the
    last seen ``selected_agent`` so handoffs can be derived without an
    ``updates`` channel.
    """

    def __init__(self) -> None:
        self._sequence = 0
        self._seen_tool_message_ids: set[str] = set()
        self._seen_message_keys: set[int] = set()

    def _next(self) -> int:
        self._sequence += 1
        return self._sequence

    def translate(self, raw: dict[str, Any]) -> Iterable[V3StreamEvent]:
        if not isinstance(raw, dict) or raw.get("type") != "event":
            return
        method = raw.get("method")
        params = raw.get("params") or {}
        namespace = list(params.get("namespace") or [])
        data = params.get("data")

        if method == "messages":
            yield from self._translate_messages(data, namespace)
        elif method == "values":
            yield from self._translate_values(data, namespace, params.get("interrupts"))
        elif method == "lifecycle":
            yield from self._translate_lifecycle(data, namespace)
        elif method == "updates":
            yield from self._translate_updates(data, namespace)
        # tasks / checkpoints / custom channels are not consumed today.

    def _translate_messages(
        self, data: Any, namespace: list[str]
    ) -> Iterable[V3StreamEvent]:
        message_event, metadata = _message_event_and_metadata(data)
        if message_event is None:
            return
        node = _node_from_metadata(metadata)
        run_id = metadata.get("run_id") if isinstance(metadata, dict) else None
        event_name = message_event.get("event")

        if event_name == "content-block-delta":
            delta = message_event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text-delta":
                text = delta.get("text", "")
                if text:
                    yield make_event(
                        "message_delta",
                        sequence=self._next(),
                        node=node,
                        agent=node,
                        namespace=namespace,
                        run_id=run_id,
                        data={"text": text},
                    )
            elif dtype == "reasoning-delta":
                reasoning = delta.get("reasoning", "")
                if reasoning:
                    yield make_event(
                        "reasoning_delta",
                        sequence=self._next(),
                        node=node,
                        agent=node,
                        namespace=namespace,
                        run_id=run_id,
                        data={"text": reasoning},
                    )
            elif dtype == "block-delta":
                fields = delta.get("fields") or {}
                if fields.get("type") == "tool_call_chunk":
                    call_id = fields.get("id")
                    yield make_event(
                        "tool_call_delta",
                        sequence=self._next(),
                        node=node,
                        agent=node,
                        namespace=namespace,
                        run_id=run_id,
                        tool_call_id=str(call_id) if call_id else None,
                        tool_name=fields.get("name"),
                        data={"args_delta": fields.get("args") or ""},
                    )
        elif event_name == "content-block-finish":
            content = message_event.get("content") or {}
            ctype = content.get("type")
            if ctype == "tool_call":
                call_id = content.get("id")
                yield make_event(
                    "tool_call_available",
                    sequence=self._next(),
                    node=node,
                    agent=node,
                    namespace=namespace,
                    run_id=run_id,
                    tool_call_id=str(call_id) if call_id else None,
                    tool_name=content.get("name"),
                    data={"args": content.get("args") or {}},
                )
        elif event_name == "message-start":
            yield make_event(
                "message_start",
                sequence=self._next(),
                node=node,
                agent=node,
                namespace=namespace,
                run_id=run_id,
                message_id=message_event.get("id"),
                data={"role": message_event.get("role")},
            )
        elif event_name == "message-finish":
            yield make_event(
                "message_end",
                sequence=self._next(),
                node=node,
                agent=node,
                namespace=namespace,
                run_id=run_id,
                data={"usage": message_event.get("usage")},
            )

    def _translate_values(
        self, data: Any, namespace: list[str], interrupts: Any
    ) -> Iterable[V3StreamEvent]:
        if not isinstance(data, dict):
            return
        context = data.get("context") if isinstance(data.get("context"), dict) else {}
        render_results = context.get("tool_render_results") or {}
        messages = data.get("messages")
        new_messages: list[Any] = []
        if isinstance(messages, list):
            for message in messages:
                key = id(message)
                if key in self._seen_message_keys:
                    continue
                self._seen_message_keys.add(key)
                new_messages.append(message)
                if isinstance(message, ToolMessage):
                    tool_call_id = getattr(message, "tool_call_id", None)
                    if tool_call_id and tool_call_id not in self._seen_tool_message_ids:
                        self._seen_tool_message_ids.add(tool_call_id)
                        end_data: dict[str, Any] = {"output": message.content}
                        render = render_results.get(str(tool_call_id))
                        if render:
                            end_data["render"] = render
                        yield make_event(
                            "tool_execution_end",
                            sequence=self._next(),
                            namespace=namespace,
                            tool_call_id=str(tool_call_id),
                            tool_name=getattr(message, "name", None) or "tool",
                            data=end_data,
                        )

        # Carry the full state snapshot (+ this superstep's new messages) so the
        # graph mapper can derive handoffs, planning node_complete, and terminal
        # recovery without the missing `updates` channel.
        yield make_event(
            "state_snapshot",
            sequence=self._next(),
            namespace=namespace,
            node=namespace[-1] if namespace else None,
            data={
                "kind": "values",
                "values": data,
                "new_messages": new_messages,
                "selected_agent": data.get("selected_agent"),
                "interrupts": list(interrupts) if interrupts else [],
            },
        )

    def _translate_updates(
        self, data: Any, namespace: list[str]
    ) -> Iterable[V3StreamEvent]:
        if not isinstance(data, dict):
            return
        node = data.get("node")
        values = data.get("values")
        yield make_event(
            "state_snapshot",
            sequence=self._next(),
            namespace=namespace,
            node=str(node) if node else None,
            data={
                "kind": "updates_tuple",
                "node_state": values if isinstance(values, dict) else {},
            },
        )

    def _translate_lifecycle(
        self, data: Any, namespace: list[str]
    ) -> Iterable[V3StreamEvent]:
        if not isinstance(data, dict):
            return
        event_name = data.get("event")
        ns = list(data.get("namespace") or namespace)
        graph_name = data.get("graph_name") or (ns[-1] if ns else "subagent")
        task_id = ns[-1] if ns else graph_name
        if event_name in {"started", "running"}:
            yield make_event(
                "subagent_start",
                sequence=self._next(),
                namespace=ns,
                subagent=SubagentRef(
                    id=str(task_id),
                    name=str(graph_name),
                    path=list(ns),
                    status="running",
                ),
                data={"cause": data.get("cause")},
            )
        elif event_name in {"completed", "failed", "interrupted"}:
            status = {
                "completed": "completed",
                "failed": "failed",
                "interrupted": "requires_approval",
            }[event_name]
            yield make_event(
                "subagent_end",
                sequence=self._next(),
                namespace=ns,
                subagent=SubagentRef(
                    id=str(task_id),
                    name=str(graph_name),
                    path=list(ns),
                    status=status,
                ),
                data={"error": data.get("error")},
            )


# ---------------------------------------------------------------------------
# Stream entry point
# ---------------------------------------------------------------------------


async def _open_v3_stream(graph: Any, state: Any, *, config: dict[str, Any] | None) -> Any:
    """Open the experimental v3 stream, awaiting the v3 awaitable contract.

    Raises ``NotImplementedError`` (or ``AttributeError``/``TypeError``) when the
    graph does not implement the v3 protocol — the caller treats that as a
    signal to use the tuple fallback.
    """
    stream = graph.astream_events(state, config=config, version="v3")
    if inspect.isawaitable(stream):
        stream = await stream
    return stream


async def _aclose_quietly(stream: Any) -> None:
    """Finalize a partially-consumed async stream, ignoring shutdown errors."""
    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return
    # Best-effort cleanup of an already-dead stream; close failures are not actionable.
    with contextlib.suppress(Exception):
        await aclose()


async def _iter_tuple_fallback(
    graph: Any,
    state: Any,
    *,
    config: dict[str, Any] | None,
) -> AsyncGenerator[V3StreamEvent, None]:
    sequence = 0
    async for chunk in graph.astream(
        state,
        config=config,
        stream_mode=["messages", "updates"],
    ):
        if not isinstance(chunk, tuple) or len(chunk) != 2:
            continue
        mode, payload = chunk
        sequence += 1
        if mode == "messages":
            message_chunk, metadata = payload
            yield make_event(
                "state_snapshot",
                sequence=sequence,
                data={
                    "kind": "messages_tuple",
                    "chunk": message_chunk,
                    "metadata": metadata,
                },
            )
        elif mode == "updates" and isinstance(payload, dict):
            for node, node_state in payload.items():
                sequence += 1
                yield make_event(
                    "state_snapshot",
                    sequence=sequence,
                    node=str(node),
                    data={"kind": "updates_tuple", "node_state": node_state},
                )


async def iter_v3_events_from_graph(
    graph: Any,
    state: Any,
    *,
    config: dict[str, Any] | None = None,
) -> AsyncGenerator[V3StreamEvent, None]:
    """Yield canonical v3 events from a LangGraph graph.

    Prefers the experimental v3 protocol on graphs that implement it; falls
    back to the v1/v2 ``stream_mode=["messages","updates"]`` tuple path for
    runnables that don't (test doubles, older runnables). The fallback wraps
    raw tuples in ``state_snapshot`` carrier events so the graph mapper can
    reproduce the legacy public-event behavior exactly.
    """
    v3_stream = None
    if hasattr(graph, "astream_events"):
        try:
            v3_stream = await _open_v3_stream(graph, state, config=config)
        except (NotImplementedError, AttributeError, TypeError, ValueError):
            # Graph does not implement the v3 protocol — fall back to tuples.
            v3_stream = None

    if v3_stream is not None:
        translator = V3ProtocolTranslator()
        iterator = v3_stream.__aiter__()
        # langchain-core <1.4 / langgraph <1.2.4 return a *non-awaitable* async
        # generator whose v3 version check raises only once iteration starts, not
        # at open time above. Probe the first event so that case still falls back
        # to the v1/v2 tuple path instead of surfacing the version error.
        try:
            first = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except (NotImplementedError, AttributeError, TypeError, ValueError):
            await _aclose_quietly(v3_stream)
        else:
            for event in translator.translate(first):
                yield event
            async for raw in iterator:
                for event in translator.translate(raw):
                    yield event
            return

    async for event in _iter_tuple_fallback(graph, state, config=config):
        yield event
