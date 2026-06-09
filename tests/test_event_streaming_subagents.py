from __future__ import annotations

import pytest

from app.ai.planning_subagents import (
    DispatchSubagentsInput,
    PlanningSubagentDispatcher,
    PlanningSubagentTask,
)
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.services.event_streaming.subagents import SubagentEventSink


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
