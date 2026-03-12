"""Regression tests for graph terminal response recovery."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.ai.graph import MultiAgentWorkflow


def _snapshot(*, values=None, next_nodes=None):
    return SimpleNamespace(values=values or {}, next=next_nodes or [])


class _FakeGraph:
    def __init__(self, *, ainvoke_result=None, snapshots=None):
        self._ainvoke_result = ainvoke_result
        self._snapshots = list(snapshots or [])

    async def ainvoke(self, *_args, **_kwargs):
        return self._ainvoke_result

    async def aget_state(self, *_args, **_kwargs):
        if not self._snapshots:
            return _snapshot()
        if len(self._snapshots) == 1:
            return self._snapshots[0]
        return self._snapshots.pop(0)

    async def astream(self, *_args, **_kwargs):
        if False:
            yield None


class TestTerminalResponseRecovery:
    @pytest.mark.asyncio
    async def test_execute_recovers_response_from_final_ai_message(self):
        workflow = object.__new__(MultiAgentWorkflow)
        workflow.checkpointer = None
        workflow.graph = _FakeGraph(
            ainvoke_result={
                "selected_agent": "chat_agent",
                "messages": [
                    HumanMessage(content="Hi"),
                    AIMessage(content="Recovered final answer"),
                ],
                "context": {},
            }
        )

        response = await workflow.execute(message="Hi")

        assert response is not None
        assert response.agent_id == "chat_agent"
        assert response.message.content == "Recovered final answer"

    @pytest.mark.asyncio
    async def test_execute_stream_recovers_response_from_checkpoint_messages(self):
        workflow = object.__new__(MultiAgentWorkflow)
        workflow.checkpointer = object()
        workflow.graph = _FakeGraph(
            snapshots=[
                _snapshot(
                    values={
                        "selected_agent": "search_agent",
                        "messages": [
                            HumanMessage(content="Hi"),
                            AIMessage(content="Recovered streamed answer"),
                        ],
                        "context": {},
                    }
                )
            ]
        )

        async def _route_node(state):
            state["selected_agent"] = "chat_agent"
            return state

        workflow._route_node = _route_node

        events = [
            event async for event in workflow.execute_stream(message="Hi", thread_id="thread-1")
        ]

        assert [event["type"] for event in events] == ["agent_selected", "complete"]
        assert events[0]["agent"] == "chat_agent"
        assert events[1]["response"].agent_id == "search_agent"
        assert events[1]["response"].message.content == "Recovered streamed answer"

    @pytest.mark.asyncio
    async def test_resume_recovers_response_from_final_ai_message(self):
        workflow = object.__new__(MultiAgentWorkflow)
        workflow.checkpointer = object()
        final_state = {
            "selected_agent": "planning_agent",
            "messages": [
                HumanMessage(content="Continue"),
                AIMessage(content="Recovered resumed answer"),
            ],
            "context": {},
        }
        workflow.graph = _FakeGraph(
            ainvoke_result=final_state,
            snapshots=[
                _snapshot(values={"messages": []}),
                _snapshot(values=final_state),
            ],
        )

        response = await workflow.resume(thread_id="thread-1")

        assert response is not None
        assert response.agent_id == "planning_agent"
        assert response.message.content == "Recovered resumed answer"
