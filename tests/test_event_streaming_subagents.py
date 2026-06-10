from __future__ import annotations

import pytest

from app.ai.planning_subagents import (
    DispatchSubagentsInput,
    PlanningSubagentDispatcher,
    PlanningSubagentTask,
)
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.services.event_streaming.subagents import (
    SubagentEventSink,
    register_subagent_event_sink,
    resolve_subagent_event_sink,
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
    assert events[-1].data["output"] == "worker answer"


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
