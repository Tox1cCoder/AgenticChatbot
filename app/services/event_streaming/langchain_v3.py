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

from ...core.config import settings
from .events import SubagentRef, V3StreamEvent, make_event

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _node_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("langgraph_node") or metadata.get("node")
    return str(value) if value else None


def _subagent_ref_from_metadata(metadata: dict[str, Any]) -> SubagentRef | None:
    """Identify a planning-subagent worker model run from its merged metadata.

    The Planning worker runtime stamps every worker model call with
    ``purpose=planning_subagent`` + ``subagent_task_id``/``subagent_agent``,
    which LangGraph merges into the messages-channel metadata.
    """
    if metadata.get("purpose") != "planning_subagent":
        return None
    task_id = str(metadata.get("subagent_task_id") or "").strip()
    if not task_id:
        return None
    return SubagentRef(
        id=task_id,
        name=str(metadata.get("subagent_agent") or "unknown_agent"),
        path=["planning_agent", task_id],
        status="running",
    )


def _is_internal_run(metadata: dict[str, Any]) -> bool:
    if metadata.get("internal") is True:
        return True
    tags = metadata.get("tags")
    return isinstance(tags, (list, tuple)) and "internal" in tags


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
    last seen ``active_agent_id`` so transitions can be derived without an
    ``updates`` channel.
    """

    def __init__(self) -> None:
        self._sequence = 0
        self._seen_tool_message_ids: set[str] = set()
        self._seen_message_keys: set[int] = set()
        self._emitted_non_standard_reasoning: set[str] = set()

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
        elif method == "custom":
            yield from self._translate_custom(data, namespace)
        # tasks / checkpoints channels are not consumed today.

    def _translate_custom(self, data: Any, namespace: list[str]) -> Iterable[V3StreamEvent]:
        """Project a Planning node's typed event onto the public subagent stream.

        Fields are allowlisted rather than passed through. The writer is called
        from inside worker execution, where the objective, the worker's answer,
        an execution key and a provider receipt are all in scope -- publishing
        the payload wholesale would put private work in the user-visible trace
        the first time someone added a field.
        """
        if not isinstance(data, dict):
            return

        event_type = data.get("type")
        if event_type == "planning_worker":
            yield from self._translate_worker_event(data, namespace)
        elif event_type == "planning_worker_tool":
            yield from self._translate_worker_tool_event(data, namespace)
        elif event_type == "planning_dispatch":
            yield from self._translate_dispatch_event(data, namespace)
        elif event_type == "image_preview":
            yield from self._translate_image_preview(data, namespace)
        # Any other producer on this shared channel is not ours to publish.

    def _translate_image_preview(self, data: dict, namespace: list[str]) -> Iterable[V3StreamEvent]:
        """Republish an early image preview emitted from inside the graph.

        Previews used to travel on a side queue reached through a weak token in
        checkpoint state, because there was no in-graph channel for them. There
        is one now, and it survives a resume without the token being rebound.
        """
        payload = dict(data)
        payload.pop("type", None)
        if not payload:
            return
        yield make_event(
            "image_preview",
            sequence=self._next(),
            agent="image_generator_agent",
            node="image_generator_agent",
            namespace=namespace,
            data=payload,
        )

    def _worker_ref(self, data: dict, status: str) -> SubagentRef | None:
        dispatch_id = str(data.get("dispatch_id") or "")
        task_id = str(data.get("task_id") or "")
        if not dispatch_id or not task_id:
            return None
        return SubagentRef(
            # ``(dispatch_id, task_id)`` is worker identity everywhere else, so
            # the stream uses the same pair rather than inventing one.
            id=f"{dispatch_id}:{task_id}",
            # A custom agent's runtime id is ``custom_agent:<uuid>``; the
            # dispatch resolved its display name server-side.
            name=str(data.get("agent_name") or data.get("agent_id") or "worker"),
            path=[dispatch_id, task_id],
            status=status,
        )

    def _translate_worker_event(self, data: dict, namespace: list[str]) -> Iterable[V3StreamEvent]:
        phase = data.get("phase")
        if phase == "start":
            ref = self._worker_ref(data, "running")
            event_type = "subagent_start"
        elif phase == "interrupt":
            ref = self._worker_ref(data, "requires_approval")
            event_type = "subagent_end"
        elif phase == "end":
            status = str(data.get("status") or "completed")
            ref = self._worker_ref(data, "failed" if status == "failed" else "completed")
            event_type = "subagent_end"
        else:
            return

        if ref is None:
            return

        payload = {
            "dispatch_id": str(data.get("dispatch_id") or ""),
            "task_id": str(data.get("task_id") or ""),
        }
        error_code = data.get("error_code")
        if error_code:
            payload["error_code"] = str(error_code)

        yield make_event(
            event_type,
            sequence=self._next(),
            namespace=namespace,
            subagent=ref,
            data=payload,
        )

    def _translate_worker_tool_event(
        self, data: dict, namespace: list[str]
    ) -> Iterable[V3StreamEvent]:
        phase = data.get("phase")
        if phase not in ("start", "end"):
            return
        ref = self._worker_ref(data, "running")
        if ref is None:
            return

        payload = {
            "dispatch_id": str(data.get("dispatch_id") or ""),
            "task_id": str(data.get("task_id") or ""),
        }
        error_code = data.get("error_code")
        if error_code:
            payload["error_code"] = str(error_code)

        yield make_event(
            "subagent_tool_execution_start" if phase == "start" else "subagent_tool_execution_end",
            sequence=self._next(),
            namespace=namespace,
            subagent=ref,
            tool_call_id=str(data.get("tool_call_id") or "") or None,
            tool_name=str(data.get("tool_name") or "") or None,
            data=payload,
        )

    def _translate_dispatch_event(
        self, data: dict, namespace: list[str]
    ) -> Iterable[V3StreamEvent]:
        phase = data.get("phase")
        if phase not in ("validated", "collected"):
            return
        dispatch_id = str(data.get("dispatch_id") or "")
        if not dispatch_id:
            return

        ref = SubagentRef(
            id=dispatch_id,
            name="planning_dispatch",
            path=[dispatch_id],
            status="running" if phase == "validated" else "completed",
        )
        yield make_event(
            "subagent_start" if phase == "validated" else "subagent_end",
            sequence=self._next(),
            namespace=namespace,
            subagent=ref,
            data={
                "dispatch_id": dispatch_id,
                "wave": int(data.get("wave") or 0),
                "task_count": int(data.get("task_count") or 0),
            },
        )

    def _translate_messages(self, data: Any, namespace: list[str]) -> Iterable[V3StreamEvent]:
        message_event, metadata = _message_event_and_metadata(data)
        if message_event is None:
            return
        node = _node_from_metadata(metadata)
        run_id = metadata.get("run_id") if isinstance(metadata, dict) else None
        event_name = message_event.get("event")

        if isinstance(metadata, dict):
            # Planning-subagent worker model runs surface on this channel too
            # (nested calls inherit the graph's streaming callbacks). Re-route
            # their deltas to attributed subagent events instead of letting
            # them interleave into the main answer/thinking stream.
            subagent = _subagent_ref_from_metadata(metadata)
            if subagent is not None:
                yield from self._translate_subagent_delta(
                    message_event, subagent, namespace=namespace, run_id=run_id
                )
                return
            if settings.suppress_internal_stream_chunks and _is_internal_run(metadata):
                return

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
                else:
                    yield from self._reasoning_from_non_standard(
                        fields, node=node, namespace=namespace, run_id=run_id
                    )
        elif event_name == "content-block-finish":
            content = message_event.get("content") or {}
            ctype = content.get("type")
            if ctype == "non_standard":
                # A provider-specific thought block (e.g. Gemini ``thinking``)
                # that reached the wire unnormalized emits no delta at all, so
                # the terminal block is the only chance to surface it.
                yield from self._reasoning_from_non_standard(
                    content, node=node, namespace=namespace, run_id=run_id
                )
            elif ctype == "tool_call":
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

    def _reasoning_from_non_standard(
        self,
        block: Any,
        *,
        node: str | None,
        namespace: list[str],
        run_id: Any,
    ) -> Iterable[V3StreamEvent]:
        """Recover a thought summary from a ``non_standard`` content block.

        ``langchain-core`` wraps any provider-specific block it does not
        recognize — Gemini's ``{"type": "thinking"}`` among them — as
        ``{"type": "non_standard", "value": {...}}``. The model boundary
        normalizes these into standard ``reasoning`` blocks; this is the
        backstop for anything that slips through, so a summary degrades to
        one late delta rather than vanishing.

        Text already surfaced for the same value is not re-emitted, because the
        bridge can deliver the same accumulated block twice (delta + finish).
        """
        if not isinstance(block, dict) or block.get("type") != "non_standard":
            return
        value = block.get("value")
        if not isinstance(value, dict) or value.get("type") not in {"thinking", "reasoning"}:
            return

        text = ""
        for key in ("thinking", "reasoning", "summary", "text"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                text = candidate
                break
        if not text or text in self._emitted_non_standard_reasoning:
            return
        self._emitted_non_standard_reasoning.add(text)

        yield make_event(
            "reasoning_delta",
            sequence=self._next(),
            node=node,
            agent=node,
            namespace=namespace,
            run_id=run_id,
            data={"text": text},
        )

    def _translate_subagent_delta(
        self,
        message_event: dict[str, Any],
        subagent: SubagentRef,
        *,
        namespace: list[str],
        run_id: Any,
    ) -> Iterable[V3StreamEvent]:
        """Project a worker model-run envelope to ``subagent_message_delta``.

        Only text/reasoning deltas surface (``channel`` distinguishes them);
        worker message lifecycle and tool-call chunks stay private — worker
        tool activity is reported by the dispatcher's event sink.
        """
        if message_event.get("event") != "content-block-delta":
            return
        delta = message_event.get("delta") or {}
        dtype = delta.get("type")
        if dtype == "text-delta":
            text, channel = delta.get("text", ""), "text"
        elif dtype == "reasoning-delta":
            text, channel = delta.get("reasoning", ""), "reasoning"
        else:
            return
        if text:
            yield make_event(
                "subagent_message_delta",
                sequence=self._next(),
                namespace=namespace,
                run_id=run_id if isinstance(run_id, str) else None,
                subagent=subagent,
                data={"text": text, "channel": channel},
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
                "active_agent_id": data.get("active_agent_id"),
                "interrupts": list(interrupts) if interrupts else [],
            },
        )

    def _translate_updates(self, data: Any, namespace: list[str]) -> Iterable[V3StreamEvent]:
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

    def _translate_lifecycle(self, data: Any, namespace: list[str]) -> Iterable[V3StreamEvent]:
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


def _custom_channel_transformer() -> Any:
    """A v3 transformer whose only job is to subscribe to the custom channel.

    ``astream_events(version="v3")`` asks the graph for exactly the union of
    ``required_stream_modes`` across its transformers, and none of the four
    native ones wants ``custom``. Without this, every ``get_stream_writer()``
    call in the Planning nodes is discarded before it reaches the iterator --
    verified against LangGraph 1.2.9, where adding it changed the observed
    methods from ``{values}`` to ``{values, custom}``.

    It captures nothing itself; the events it enables are read off the main
    iterator by :meth:`V3ProtocolTranslator._translate_custom`.
    """
    from langgraph.pregel.main import StreamTransformer

    class CustomChannelTransformer(StreamTransformer):
        required_stream_modes = ("custom",)

        def init(self) -> dict[str, Any]:
            return {}

        def process(self, event: Any) -> bool:
            return True

    return CustomChannelTransformer


try:
    _CustomChannelTransformer: Any = _custom_channel_transformer()
except Exception:  # pragma: no cover - older langgraph without the mux
    _CustomChannelTransformer = None


async def _open_v3_stream(graph: Any, state: Any, *, config: dict[str, Any] | None) -> Any:
    """Open the experimental v3 stream, awaiting the v3 awaitable contract.

    Raises ``NotImplementedError`` (or ``AttributeError``/``TypeError``) when the
    graph does not implement the v3 protocol — the caller treats that as a
    signal to use the tuple fallback.
    """
    stream = graph.astream_events(
        state, config=config, version="v3", transformers=[_CustomChannelTransformer]
    )
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


#: Stateless apart from its sequence counter, which the fallback overrides.
_FALLBACK_TRANSLATOR = V3ProtocolTranslator()


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
        stream_mode=["messages", "updates", "custom"],
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
        elif mode == "custom":
            # Same projection the v3 path uses, so a runnable on the fallback
            # still surfaces worker lifecycle and image previews.
            for event in _FALLBACK_TRANSLATOR._translate_custom(payload, []):
                yield event


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
