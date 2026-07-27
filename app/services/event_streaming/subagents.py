"""Event sink for custom (non-subgraph) subagents.

The Planning Agent's ``dispatch_subagents`` tool runs workers through a custom
dispatcher rather than LangGraph subgraphs, so their activity does not surface
on the v3 ``lifecycle`` channel. This queue-backed sink lets the dispatcher emit
canonical subagent lifecycle events that the graph stream drains and forwards,
giving custom subagents the same event-level visibility as DeepAgents subgraph
subagents.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import weakref
from collections import deque
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from .events import IMAGE_PREVIEW_STATUS_PARTIAL, SubagentRef, V3StreamEvent, make_event

logger = logging.getLogger(__name__)

_SINK_CLOSED = object()

# Frames the client replaces in place (previews) or can lose without breaking
# the activity view (message deltas). These are dropped first under backpressure;
# lifecycle events (start / tool / end) are always kept.
_TRANSIENT_EVENT_TYPES = frozenset({"image_preview", "subagent_message_delta"})


def _event_item_id(event: V3StreamEvent) -> Any:
    return (event.data or {}).get("item_id")


def _is_droppable(event: V3StreamEvent) -> bool:
    """Whether an event may be discarded/coalesced under backpressure.

    Lifecycle events (start / tool / end) are never droppable. An
    ``image_preview`` is droppable ONLY while it is an in-progress ``partial``:
    a FINAL image delivered by protected reference is lossless (never dropped),
    and any other preview status (e.g. ``preview_skipped``) is kept because it
    carries no bulky base64 and is a structured observability signal.
    """
    if event.type not in _TRANSIENT_EVENT_TYPES:
        return False
    if event.type == "image_preview":
        return (event.data or {}).get("status") == IMAGE_PREVIEW_STATUS_PARTIAL
    return True


class SubagentEventSink:
    def __init__(self, maxsize: int = 0) -> None:
        # The queue stays unbounded; the soft cap is enforced in ``emit_event``
        # so lifecycle events are never rejected while transient frames are.
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._sequence = 0
        self._maxsize = max(0, int(maxsize))
        self.dropped_transient_count = 0

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    async def emit(
        self,
        event_type: str,
        *,
        task_id: str,
        agent_name: str,
        status: str,
        data: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> None:
        await self._queue.put(
            make_event(
                event_type,
                sequence=self._next_sequence(),
                subagent=SubagentRef(
                    id=task_id,
                    name=agent_name,
                    path=["planning_agent", task_id],
                    status=status,
                ),
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                data=data or {},
            )
        )

    def emit_event(self, event: V3StreamEvent) -> None:
        """Enqueue a prebuilt canonical event (e.g. ``image_preview``).

        Synchronous on purpose: producers inside graph nodes must not await the
        stream. Backpressure policy when the soft ``maxsize`` cap is reached:

        - Lossless events (lifecycle, FINAL image references) are ALWAYS
          enqueued; a stale transient is evicted first to keep memory bounded.
        - An in-progress ``partial`` prefers to REPLACE an already-queued stale
          partial for the SAME item (freshest preview wins) before any frame is
          discarded; only when no same-item frame exists is the incoming partial
          coalesced away. Unrelated events are never displaced by a partial.
        """
        if self._maxsize <= 0 or not _is_droppable(event):
            if self._maxsize > 0 and self._queue.qsize() >= self._maxsize:
                self._evict_stale_transient(prefer_item_id=_event_item_id(event))
            self._enqueue(event)
            return

        if self._queue.qsize() < self._maxsize:
            self._enqueue(event)
            return

        item_id = _event_item_id(event)
        if item_id is not None and self._evict_stale_partial(item_id):
            self._enqueue(event)
            return
        self._record_transient_drop()

    def _enqueue(self, event: V3StreamEvent) -> None:
        self._queue.put_nowait(event.model_copy(update={"sequence": self._next_sequence()}))

    def _pending(self) -> deque | None:
        """The queue's backing deque, or None if the runtime shape changed."""
        pending = getattr(self._queue, "_queue", None)
        return pending if isinstance(pending, deque) else None

    def _evict_stale_partial(self, item_id: Any) -> bool:
        """Discard the oldest queued in-progress partial for ``item_id``."""
        pending = self._pending()
        if pending is None:
            return False
        for index, queued in enumerate(pending):
            if (
                queued.type == "image_preview"
                and (queued.data or {}).get("status") == IMAGE_PREVIEW_STATUS_PARTIAL
                and (queued.data or {}).get("item_id") == item_id
            ):
                del pending[index]
                self._record_transient_drop()
                return True
        return False

    def _evict_stale_transient(self, *, prefer_item_id: Any) -> bool:
        """Make room for a lossless event by dropping one stale transient.

        Prefers a same-item partial (so an in-flight preview yields to its own
        final), otherwise the oldest droppable frame; lifecycle/final events are
        never candidates."""
        if prefer_item_id is not None and self._evict_stale_partial(prefer_item_id):
            return True
        pending = self._pending()
        if pending is None:
            return False
        for index, queued in enumerate(pending):
            if _is_droppable(queued):
                del pending[index]
                self._record_transient_drop()
                return True
        return False

    def _record_transient_drop(self) -> None:
        self.dropped_transient_count += 1
        if self.dropped_transient_count == 1:
            logger.info(
                "subagent event queue saturated (maxsize=%d); dropping/replacing "
                "transient frames code=subagent_event_queue_saturated",
                self._maxsize,
            )

    async def drain(self) -> list[V3StreamEvent]:
        events: list[V3StreamEvent] = []
        while not self._queue.empty():
            events.append(await self._queue.get())
        return events

    async def stream(self) -> AsyncGenerator[V3StreamEvent, None]:
        """Yield events as they are emitted, until :meth:`close` is called."""
        while True:
            event = await self._queue.get()
            if event is _SINK_CLOSED:
                return
            yield event

    def close(self) -> None:
        """Signal :meth:`stream` to stop after already-queued events drain."""
        self._queue.put_nowait(_SINK_CLOSED)


# Graph state must stay msgpack-serializable for LangGraph checkpointing (HITL
# interrupts persist the full state). The sink itself therefore never enters
# state — only an opaque string token does. The registry holds weak references:
# the streaming generator owns the only strong reference, so a sink (and its
# registry entry) dies with its stream and resumed runs simply resolve to None.
_SINK_REGISTRY: weakref.WeakValueDictionary[str, SubagentEventSink] = weakref.WeakValueDictionary()


def register_subagent_event_sink(sink: SubagentEventSink) -> str:
    """Register a sink and return the serializable token to store in state."""
    token = uuid4().hex
    _SINK_REGISTRY[token] = sink
    return token


def rebind_subagent_event_sink(token: str, sink: SubagentEventSink) -> None:
    """Bind a live sink under an EXISTING token (resume parity).

    The token persisted in a checkpoint outlives the sink it was created for:
    after the original stream ends its sink is garbage-collected and the token
    weakly resolves to None. On resume the same checkpointed graph state still
    carries that token, so rebinding this run's live sink under it lets the
    resumed image node resolve a live sink WITHOUT mutating the checkpoint. The
    caller must hold a strong reference for the stream's lifetime (the registry
    only holds a weak one)."""
    _SINK_REGISTRY[token] = sink


def resolve_subagent_event_sink(token: Any) -> SubagentEventSink | None:
    """Resolve a state-carried token back to its live sink, if any."""
    if not isinstance(token, str):
        return None
    return _SINK_REGISTRY.get(token)


async def stream_with_subagent_events(
    primary: AsyncGenerator[V3StreamEvent, None],
    sink: SubagentEventSink,
) -> AsyncGenerator[V3StreamEvent, None]:
    """Interleave ``primary`` graph events with ``sink`` subagent events live.

    Both sources feed one FIFO queue, so a subagent event surfaces the instant
    it is emitted instead of buffering until ``primary`` produces its next
    event. Ends when ``primary`` is exhausted; sink events already enqueued are
    flushed first. Feeder tasks are cancelled on early exit (client disconnect).
    """
    out: asyncio.Queue[Any] = asyncio.Queue()
    primary_done = object()

    async def _pump_primary() -> None:
        try:
            async for event in primary:
                await out.put(event)
        finally:
            await out.put(primary_done)

    async def _pump_sink() -> None:
        async for event in sink.stream():
            await out.put(event)

    primary_task = asyncio.ensure_future(_pump_primary())
    sink_task = asyncio.ensure_future(_pump_sink())
    closed = False
    try:
        while True:
            item = await out.get()
            if item is primary_done:
                break
            yield item
        sink.close()
        closed = True
        await sink_task
        while not out.empty():
            item = out.get_nowait()
            if item is not primary_done:
                yield item
        # Re-raise the primary's exception (GraphRecursionError drives
        # auto-continue; anything else must become a terminal error event).
        await primary_task
    finally:
        # Only close on early exit (disconnect/teardown); a second sentinel
        # after a clean run would kill the next auto-continue round's stream.
        if not closed:
            sink.close()
        for task in (primary_task, sink_task):
            if not task.done():
                task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(primary_task, sink_task, return_exceptions=True)
