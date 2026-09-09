from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.services.event_streaming.langchain_v3 import (
    LangGraphV3Normalizer,
    V3ProtocolTranslator,
    iter_v3_events_from_graph,
    normalize_update_chunk,
)

# ---------------------------------------------------------------------------
# Fallback (v1/v2 tuple) helpers — these back the non-graph / FakeGraph path.
# ---------------------------------------------------------------------------


def test_message_chunk_text_becomes_message_delta():
    normalizer = LangGraphV3Normalizer()
    chunk = AIMessageChunk(content="hello")
    event = normalizer.from_message_chunk(chunk, metadata={"langgraph_node": "chat_agent"})

    assert event is not None
    assert event.type == "message_delta"
    assert event.node == "chat_agent"
    assert event.data["text"] == "hello"


def test_reasoning_content_block_becomes_reasoning_delta():
    normalizer = LangGraphV3Normalizer()
    chunk = SimpleNamespace(
        content="",
        content_blocks=[{"type": "reasoning", "reasoning": "thinking"}],
    )

    event = normalizer.from_message_chunk(chunk, metadata={"langgraph_node": "chat_agent"})

    assert event is not None
    assert event.type == "reasoning_delta"
    assert event.data["text"] == "thinking"


def test_tool_call_chunk_is_not_tool_execution_start():
    normalizer = LangGraphV3Normalizer()
    chunk = SimpleNamespace(
        content="",
        content_blocks=[
            {
                "type": "tool_call_chunk",
                "id": "call-1",
                "name": "search_documents",
                "args": '{"query":"x"}',
            }
        ],
    )

    event = normalizer.from_message_chunk(chunk, metadata={"langgraph_node": "rag_agent"})

    assert event is not None
    assert event.type == "tool_call_delta"
    assert event.tool_call_id == "call-1"
    assert event.tool_name == "search_documents"
    assert event.data["args_delta"] == '{"query":"x"}'


def test_tool_message_becomes_tool_execution_end():
    events = list(
        normalize_update_chunk(
            {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content="result",
                            tool_call_id="call-1",
                            name="search_documents",
                        )
                    ]
                }
            },
            sequence_start=10,
        )
    )

    assert len(events) == 1
    assert events[0].type == "tool_execution_end"
    assert events[0].sequence == 10
    assert events[0].tool_call_id == "call-1"
    assert events[0].tool_name == "search_documents"
    assert events[0].data["output"] == "result"


def test_final_ai_message_tool_calls_become_tool_call_available():
    events = list(
        normalize_update_chunk(
            {
                "chat_agent": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {"id": "call-1", "name": "search_documents", "args": {"query": "x"}}
                            ],
                        )
                    ]
                }
            },
            sequence_start=1,
        )
    )

    assert events[0].type == "tool_call_available"
    assert events[0].data["args"] == {"query": "x"}


# ---------------------------------------------------------------------------
# Experimental v3 protocol translator — the active production path.
# ---------------------------------------------------------------------------


def _v3(method, data, *, seq, namespace=None, interrupts=None):
    params = {"namespace": namespace or [], "timestamp": 0, "data": data}
    if interrupts is not None:
        params["interrupts"] = interrupts
    return {"type": "event", "method": method, "params": params, "seq": seq}


def _messages_event(message_event, *, node="chat_agent", seq=1, namespace=None):
    # LangGraph emits messages-channel data as a tuple at runtime, not a list.
    return _v3(
        "messages",
        (message_event, {"langgraph_node": node, "run_id": "r1"}),
        seq=seq,
        namespace=namespace,
    )


def test_translator_text_delta_becomes_message_delta():
    t = V3ProtocolTranslator()
    events = list(
        t.translate(
            _messages_event(
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "delta": {"type": "text-delta", "text": "hi"},
                }
            )
        )
    )
    assert [e.type for e in events] == ["message_delta"]
    assert events[0].data["text"] == "hi"
    assert events[0].node == "chat_agent"


def test_translator_reasoning_delta_becomes_reasoning_delta():
    t = V3ProtocolTranslator()
    events = list(
        t.translate(
            _messages_event(
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "delta": {"type": "reasoning-delta", "reasoning": "plan"},
                }
            )
        )
    )
    assert [e.type for e in events] == ["reasoning_delta"]
    assert events[0].data["text"] == "plan"


def test_translator_block_delta_tool_call_chunk_becomes_tool_call_delta():
    t = V3ProtocolTranslator()
    events = list(
        t.translate(
            _messages_event(
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "delta": {
                        "type": "block-delta",
                        "fields": {
                            "type": "tool_call_chunk",
                            "id": "call-1",
                            "name": "search_documents",
                            "args": '{"query"',
                        },
                    },
                }
            )
        )
    )
    assert [e.type for e in events] == ["tool_call_delta"]
    assert events[0].tool_call_id == "call-1"
    assert events[0].tool_name == "search_documents"
    assert events[0].data["args_delta"] == '{"query"'


def _worker_messages_event(
    message_event,
    *,
    seq=1,
    dispatch_id="d1",
    task_id="w1",
    agent="search_agent",
):
    return _v3(
        "messages",
        (
            message_event,
            {
                "langgraph_node": "planning_agent",
                "run_id": "r1",
                "internal": True,
                "purpose": "planning_subagent",
                "subagent": True,
                "subagent_agent": agent,
                "subagent_dispatch_id": dispatch_id,
                "subagent_task_id": task_id,
                "tags": ["internal", "planning_subagent"],
            },
        ),
        seq=seq,
    )


def test_translator_routes_worker_reasoning_delta_to_subagent_message_delta():
    t = V3ProtocolTranslator()
    events = list(
        t.translate(
            _worker_messages_event(
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "delta": {"type": "reasoning-delta", "reasoning": "worker plan"},
                }
            )
        )
    )
    assert [e.type for e in events] == ["subagent_message_delta"]
    assert events[0].data == {"text": "worker plan", "channel": "reasoning"}
    assert events[0].subagent is not None
    assert events[0].subagent.id == "d1:w1"
    assert events[0].subagent.name == "search_agent"
    assert events[0].subagent.path == ["d1", "w1"]
    assert events[0].subagent.status == "running"


def test_translator_routes_worker_text_delta_to_subagent_message_delta():
    t = V3ProtocolTranslator()
    events = list(
        t.translate(
            _worker_messages_event(
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "delta": {"type": "text-delta", "text": "worker answer"},
                }
            )
        )
    )
    assert [e.type for e in events] == ["subagent_message_delta"]
    assert events[0].data == {"text": "worker answer", "channel": "text"}


def test_translator_distinguishes_same_task_id_across_dispatches():
    message_event = {
        "event": "content-block-delta",
        "index": 0,
        "delta": {"type": "text-delta", "text": "worker answer"},
    }
    translator = V3ProtocolTranslator()

    refs = [
        list(
            translator.translate(
                _worker_messages_event(message_event, dispatch_id=dispatch_id)
            )
        )[0].subagent
        for dispatch_id in ("d1", "d2")
    ]

    assert [ref.id for ref in refs] == ["d1:w1", "d2:w1"]
    assert [ref.path for ref in refs] == [["d1", "w1"], ["d2", "w1"]]


def test_translator_suppresses_worker_message_lifecycle_and_tool_chunks():
    t = V3ProtocolTranslator()
    for message_event in (
        {"event": "message-start", "id": "m1", "role": "assistant"},
        {"event": "message-finish", "usage": {}},
        {
            "event": "content-block-delta",
            "index": 0,
            "delta": {
                "type": "block-delta",
                "fields": {"type": "tool_call_chunk", "id": "c1", "name": "t", "args": "{"},
            },
        },
    ):
        assert list(t.translate(_worker_messages_event(message_event))) == []


def test_translator_drops_internal_non_subagent_run_deltas():
    t = V3ProtocolTranslator()
    event = _v3(
        "messages",
        (
            {
                "event": "content-block-delta",
                "index": 0,
                "delta": {"type": "text-delta", "text": "summary text"},
            },
            {"langgraph_node": "chat_agent", "internal": True, "tags": ["internal"]},
        ),
        seq=1,
    )
    assert list(t.translate(event)) == []


def test_translator_content_block_finish_tool_call_becomes_tool_call_available():
    t = V3ProtocolTranslator()
    events = list(
        t.translate(
            _messages_event(
                {
                    "event": "content-block-finish",
                    "index": 0,
                    "content": {
                        "type": "tool_call",
                        "id": "call-1",
                        "name": "search_documents",
                        "args": {"query": "x"},
                    },
                },
                node="rag_agent",
            )
        )
    )
    assert [e.type for e in events] == ["tool_call_available"]
    assert events[0].tool_call_id == "call-1"
    assert events[0].tool_name == "search_documents"
    assert events[0].data["args"] == {"query": "x"}


def test_translator_new_tool_message_in_values_becomes_tool_execution_end():
    t = V3ProtocolTranslator()
    ai = AIMessage(
        content="", tool_calls=[{"id": "call-1", "name": "search_documents", "args": {}}]
    )
    tool_msg = ToolMessage(content="results", tool_call_id="call-1", name="search_documents")
    # First snapshot: just the AI message (tool call available).
    list(t.translate(_v3("values", {"messages": [ai]}, seq=1)))
    # Second snapshot: the tool result arrives.
    events = list(t.translate(_v3("values", {"messages": [ai, tool_msg]}, seq=2)))
    end = [e for e in events if e.type == "tool_execution_end"]
    assert len(end) == 1
    assert end[0].tool_call_id == "call-1"
    assert end[0].tool_name == "search_documents"
    assert end[0].data["output"] == "results"


def test_translator_lifecycle_started_and_completed_become_subagent_events():
    t = V3ProtocolTranslator()
    ns = ["subflow:abc"]
    started = list(
        t.translate(
            _v3(
                "lifecycle",
                {"event": "started", "namespace": ns, "graph_name": "subflow"},
                seq=1,
                namespace=[],
            )
        )
    )
    completed = list(
        t.translate(_v3("lifecycle", {"event": "completed", "namespace": ns}, seq=2, namespace=[]))
    )
    assert [e.type for e in started] == ["subagent_start"]
    assert started[0].subagent is not None
    assert started[0].subagent.status == "running"
    assert [e.type for e in completed] == ["subagent_end"]
    assert completed[0].subagent.status == "completed"


# ---------------------------------------------------------------------------
# iter_v3_events_from_graph dispatch: v3 graph vs tuple fallback.
# ---------------------------------------------------------------------------


class _FakeV3Graph:
    """A graph that implements the experimental v3 contract (await then iterate)."""

    def __init__(self, raw_events):
        self._raw_events = raw_events

    async def astream_events(self, _input, *, config=None, version="v2", **kwargs):
        events = self._raw_events

        async def _gen():
            for event in events:
                yield event

        return _gen()


class _FakeTupleGraph:
    """A graph that only implements astream() (no v3) — exercises the fallback."""

    def __init__(self, chunks):
        self._chunks = chunks

    async def astream(self, _input, *, config=None, stream_mode=None, **kwargs):
        for chunk in self._chunks:
            yield chunk


class _FakeLazyVersionErrorGraph:
    """Mirrors langchain-core <1.4 / langgraph <1.2.4.

    ``astream_events(version="v3")`` returns a *non-awaitable* async generator
    whose version check raises ``NotImplementedError`` only on the first
    iteration (not at call time), exactly like the installed base ``Runnable``.
    It also implements ``astream()`` so the tuple fallback can run.
    """

    def __init__(self, chunks):
        self._chunks = chunks

    def astream_events(self, _input, *, config=None, version="v2", **kwargs):
        async def _gen():
            raise NotImplementedError(
                'Only versions "v1" and "v2" of the schema is currently supported.'
            )
            yield  # pragma: no cover - marks _gen as an async generator

        return _gen()

    async def astream(self, _input, *, config=None, stream_mode=None, **kwargs):
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_iter_v3_uses_protocol_path_for_v3_graph():
    graph = _FakeV3Graph(
        [
            _messages_event(
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "delta": {"type": "text-delta", "text": "hi"},
                }
            ),
        ]
    )
    events = [e async for e in iter_v3_events_from_graph(graph, {"messages": []}, config={})]
    assert any(e.type == "message_delta" and e.data["text"] == "hi" for e in events)


@pytest.mark.asyncio
async def test_iter_v3_falls_back_to_tuple_path_when_no_astream_events():
    graph = _FakeTupleGraph(
        [
            ("messages", (AIMessageChunk(content="hello"), {"langgraph_node": "chat_agent"})),
            ("updates", {"chat_agent": {"messages": [AIMessage(content="done")]}}),
        ]
    )
    events = [e async for e in iter_v3_events_from_graph(graph, {"messages": []}, config={})]
    # Fallback wraps both message chunks and node updates as state_snapshot carriers.
    assert events
    assert all(e.type == "state_snapshot" for e in events)
    assert events[0].data["kind"] == "messages_tuple"
    assert events[1].data["kind"] == "updates_tuple"


@pytest.mark.asyncio
async def test_iter_v3_falls_back_when_v3_version_check_raises_on_iteration():
    """Old langchain raises the v3 version error lazily on first iteration, not
    at open time. The dispatcher must still fall back to the tuple path instead
    of letting that NotImplementedError surface to the stream."""
    graph = _FakeLazyVersionErrorGraph(
        [
            ("messages", (AIMessageChunk(content="hello"), {"langgraph_node": "chat_agent"})),
            ("updates", {"chat_agent": {"messages": [AIMessage(content="done")]}}),
        ]
    )
    events = [e async for e in iter_v3_events_from_graph(graph, {"messages": []}, config={})]
    assert events
    assert all(e.type == "state_snapshot" for e in events)
    assert events[0].data["kind"] == "messages_tuple"
    assert events[1].data["kind"] == "updates_tuple"
