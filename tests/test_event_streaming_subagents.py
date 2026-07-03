from __future__ import annotations

import asyncio

import pytest

from app.ai.planning_subagents import (
    DispatchSubagentsInput,
    PlanningSubagentDispatcher,
    PlanningSubagentTask,
)
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.services.event_streaming.events import make_event
from app.services.event_streaming.subagents import (
    SubagentEventSink,
    register_subagent_event_sink,
    resolve_subagent_event_sink,
    stream_with_subagent_events,
)


class FakeWorkflow:
    async def _run_agent_in_isolated_context(self, **kwargs):
        return AgentResponse(
            agent_type=AgentType.SEARCH,
            agent_id=kwargs["agent_name"],
            message=AgentMessage(role=MessageRole.ASSISTANT, content="worker answer"),
            metadata={"provider": "openai", "model": "gpt-5"},
            tool_artifacts=[
                {
                    "tool_call_id": "sub-call-1",
                    "tool": "search_documents",
                    "output": "source text",
                    "status": "success",
                }
            ],
        )


@pytest.mark.asyncio
async def test_dispatcher_emits_subagent_start_and_end_events():
    sink = SubagentEventSink()
    dispatcher = PlanningSubagentDispatcher(workflow=FakeWorkflow(), event_sink=sink)
    request = DispatchSubagentsInput(
        tasks=[
            PlanningSubagentTask(
                id="worker-a",
                agent="search_agent",
                task="Find source material.",
            )
        ]
    )

    result = await dispatcher.dispatch(request, parent_state={"context": {}})
    events = await sink.drain()

    assert result.status == "completed"
    assert [event.type for event in events] == [
        "subagent_start",
        "subagent_tool_execution_end",
        "subagent_end",
    ]
    assert events[0].subagent.id == "worker-a"
    assert events[0].subagent.name == "search_agent"
    assert events[0].subagent.path == ["planning_agent", "worker-a"]
    assert events[-1].subagent.status == "completed"
    # The end event carries the full answer as `summary`; `output` is
    # reserved for worker tool events.
    assert events[-1].data["summary"] == "worker answer"
    assert "output" not in events[-1].data


def test_sink_token_keeps_graph_state_checkpoint_serializable():
    """HITL interrupts persist graph state via msgpack — the sink itself must
    never enter state, only the registry token (regression: 'Type is not
    msgpack serializable: SubagentEventSink' broke every interrupt)."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    sink = SubagentEventSink()
    token = register_subagent_event_sink(sink)

    state_fragment = {"context": {"subagent_event_sink_token": token}}
    JsonPlusSerializer().dumps_typed(("state", state_fragment))

    assert resolve_subagent_event_sink(token) is sink


def test_resolve_sink_returns_none_for_dead_or_unknown_tokens():
    assert resolve_subagent_event_sink(None) is None
    assert resolve_subagent_event_sink("unknown") is None

    token = register_subagent_event_sink(SubagentEventSink())
    # The only strong reference was the local above — entry dies with it.
    assert resolve_subagent_event_sink(token) is None


@pytest.mark.asyncio
async def test_stream_with_subagent_events_yields_sink_event_while_primary_blocked():
    sink = SubagentEventSink()
    gate = asyncio.Event()

    async def primary():
        yield make_event("message_delta", sequence=1, data={"text": "a"})
        await gate.wait()  # primary is blocked here while the subagent runs
        yield make_event("message_delta", sequence=2, data={"text": "b"})

    merged = stream_with_subagent_events(primary(), sink)
    first = await merged.__anext__()
    await sink.emit("subagent_start", task_id="w1", agent_name="search_agent", status="running")
    second = await merged.__anext__()  # must be the subagent event, not "b"
    gate.set()
    rest = [event async for event in merged]

    assert first.type == "message_delta"
    assert second.type == "subagent_start"
    assert [event.type for event in rest] == ["message_delta"]


@pytest.mark.asyncio
async def test_stream_with_subagent_events_propagates_primary_exception():
    """Graph exceptions (GraphRecursionError → auto-continue, anything else →
    terminal ``error`` event) must escape the merge, not die inside the pump."""
    sink = SubagentEventSink()

    class PrimaryBoom(RuntimeError):
        pass

    async def primary():
        yield make_event("message_delta", sequence=1, data={"text": "a"})
        raise PrimaryBoom("graph blew up")

    received = []
    with pytest.raises(PrimaryBoom):
        async for event in stream_with_subagent_events(primary(), sink):
            received.append(event.type)

    assert received == ["message_delta"]


@pytest.mark.asyncio
async def test_sink_remains_usable_for_next_auto_continue_round():
    """execute_request_stream reuses one sink across auto-continue rounds; a
    completed merge must not leave a stray close-sentinel behind that would
    kill the next round's live stream."""
    sink = SubagentEventSink()

    async def round_one():
        yield make_event("message_delta", sequence=1, data={"text": "round-1"})

    async def round_two():
        await sink.emit("subagent_start", task_id="w1", agent_name="search_agent", status="running")
        yield make_event("message_delta", sequence=2, data={"text": "round-2"})

    first = [e.type async for e in stream_with_subagent_events(round_one(), sink)]
    second = [e.type async for e in stream_with_subagent_events(round_two(), sink)]

    assert first == ["message_delta"]
    assert "subagent_start" in second
    assert "message_delta" in second
