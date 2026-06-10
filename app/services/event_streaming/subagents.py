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
import weakref
from typing import Any
from uuid import uuid4

from .events import SubagentRef, V3StreamEvent, make_event


class SubagentEventSink:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[V3StreamEvent] = asyncio.Queue()
        self._sequence = 0

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

    async def drain(self) -> list[V3StreamEvent]:
        events: list[V3StreamEvent] = []
        while not self._queue.empty():
            events.append(await self._queue.get())
        return events


# Graph state must stay msgpack-serializable for LangGraph checkpointing (HITL
# interrupts persist the full state). The sink itself therefore never enters
# state — only an opaque string token does. The registry holds weak references:
# the streaming generator owns the only strong reference, so a sink (and its
# registry entry) dies with its stream and resumed runs simply resolve to None.
_SINK_REGISTRY: weakref.WeakValueDictionary[str, SubagentEventSink] = (
    weakref.WeakValueDictionary()
)


def register_subagent_event_sink(sink: SubagentEventSink) -> str:
    """Register a sink and return the serializable token to store in state."""
    token = uuid4().hex
    _SINK_REGISTRY[token] = sink
    return token


def resolve_subagent_event_sink(token: Any) -> SubagentEventSink | None:
    """Resolve a state-carried token back to its live sink, if any."""
    if not isinstance(token, str):
        return None
    return _SINK_REGISTRY.get(token)
