"""Dependency probe: gate that the installed LangChain/LangGraph stack supports
the experimental v3 event-streaming protocol on compiled graphs.

The official `astream_events(version="v3")` contract in langchain-core 1.4.2 /
langgraph 1.2.4 differs from v1/v2:

- It is only implemented on ``BaseChatModel`` and ``CompiledGraph`` (a bare
  ``Runnable`` raises ``NotImplementedError`` via ``_astream_events_v3_unsupported``).
- The call returns an *awaitable* that must be ``await``ed before iterating
  (it yields an ``AsyncGraphRunStream``), not an async iterator directly.
- Emitted events use the JSON-RPC-like protocol shape
  ``{"type": "event", "method": <channel>, "params": {...}, "seq": <int>}`` —
  not the v1/v2 ``{"event": ..., "data": ...}`` shape.

These tests pin that contract so a dependency drift that breaks experimental v3
streaming fails loudly. See the Implementation Log in ``event_streaming.md``.
"""

from __future__ import annotations

import warnings
from typing import Annotated, TypedDict

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages


class _ProbeState(TypedDict):
    messages: Annotated[list, add_messages]


def _build_compiled_graph():
    def echo(state: _ProbeState) -> _ProbeState:
        return {"messages": []}

    graph = StateGraph(_ProbeState)
    graph.add_node("echo", echo)
    graph.add_edge(START, "echo")
    return graph.compile()


@pytest.mark.asyncio
async def test_compiled_graph_astream_events_emits_v3_protocol():
    compiled = _build_compiled_graph()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stream = await compiled.astream_events({"messages": []}, version="v3")
        events = [event async for event in stream]

    assert events
    assert all(isinstance(event, dict) for event in events)
    assert all(event.get("type") == "event" for event in events)
    assert all("method" in event for event in events)
    assert all("seq" in event for event in events)
    assert all("params" in event for event in events)
    # The default local mux emits the values channel at minimum.
    assert any(event.get("method") == "values" for event in events)


@pytest.mark.asyncio
async def test_plain_runnable_does_not_implement_v3():
    chain = RunnableLambda(lambda value: value)

    # On unsupported runnables the v3 entry point returns a coroutine whose
    # NotImplementedError surfaces on ``await`` (matching the v3 contract where
    # the awaitable resolves to the stream on subclasses that implement it).
    with pytest.raises(NotImplementedError):
        await chain.astream_events("hello", version="v3")
