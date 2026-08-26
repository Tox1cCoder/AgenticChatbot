"""Streaming-level regression tests for inter-agent hand_off.

Covers:
- When a Planning Agent ``hand_off`` re-routes the turn to the Search Agent,
  the stream emits a second ``agent_selected`` event with ``reason="handoff"``
  and the ``complete`` event carries the delegated agent's final answer
  (not the source agent's transfer narration).
- ``_recover_terminal_response`` never promotes the handoff narration
  ``AIMessage`` (which has a ``hand_off`` tool call) into the final
  assistant message — that prose is control-plane chatter, not a reply.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import (
    AgentMessage,
    AgentResponse,
    AgentType,
    MessageRole,
    WorkflowExecutionRequest,
)


def _make_workflow() -> MultiAgentWorkflow:
    return MultiAgentWorkflow.__new__(MultiAgentWorkflow)


@pytest.mark.asyncio
async def test_planning_handoff_stream_returns_delegated_agent_answer():
    """Planning Agent → Search Agent hand_off should produce the Search Agent's
    final answer in the same turn, plus a second ``agent_selected`` event when
    the active agent changes mid-stream."""
    workflow = _make_workflow()
    workflow.checkpointer = None
    workflow.agents = {"planning_agent": object(), "search_agent": object()}
    workflow._build_graph_config = lambda thread_id=None: None
    workflow._resolve_thread_id = (
        lambda thread_id, conversation_id, turn_id=None: thread_id or conversation_id
    )
    workflow._build_initial_state_from_request = lambda request: {
        "messages": [HumanMessage(content=request.message)],
        "conversation_id": request.conversation_id,
        "active_agent_id": None,
        "context": {},
    }
    workflow._get_conversation_history = AsyncMock(return_value=[])
    # Routing runs inside the graph; the adapter only supplies runtime context.
    workflow.routing_service = object()
    workflow.routing_context_builder = object()
    workflow.history_provider = None
    workflow.document_repository = None
    workflow._attach_context_outputs = lambda state, response: response
    workflow._get_agent_type = lambda name: AgentType.SEARCH
    workflow._attach_planning_state_metadata = lambda response, state: response

    final_response = AgentResponse(
        agent_type=AgentType.SEARCH,
        agent_id="search_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content="Search Agent final answer."),
        metadata={},
    )
    handoff_context = {
        "handoff": {
            "active": True,
            "source_agent": "planning_agent",
            "target_agent": "search_agent",
            "tool_call_id": "h1",
        }
    }
    final_state = {
        "active_agent_id": "search_agent",
        "messages": [
            HumanMessage(content="latest info please"),
            AIMessage(
                content="",
                tool_calls=[{"id": "h1", "name": "hand_off", "args": {}}],
            ),
            ToolMessage(
                content='{"hand_off":"search_agent"}',
                tool_call_id="h1",
                name="hand_off",
            ),
            AIMessage(content="Search Agent final answer."),
        ],
        "response": final_response,
        "context": handoff_context,
    }

    class FakeGraph:
        async def astream(self, *_args, **_kwargs):
            # The route node's state update is what the stream adapter derives
            # the turn's initial selection from — it never pre-runs routing.
            yield ("updates", {"route": {"active_agent_id": "planning_agent"}})
            yield (
                "updates",
                {
                    "planning_tools": {
                        "active_agent_id": "search_agent",
                        "context": handoff_context,
                    }
                },
            )
            yield ("updates", {"search_agent": final_state})

    workflow.graph = FakeGraph()

    events: list[Any] = []
    async for event in workflow.execute_request_stream(
        WorkflowExecutionRequest(message="latest info please", conversation_id="conv-1")
    ):
        events.append(event)
        if event.type == "complete":
            break

    agent_selected_events = [event for event in events if event.type == "agent_selected"]
    # The turn's first selection is the routing decision; every later one is an
    # accepted transition. Both are emitted, and they are never conflated.
    assert [(event.agent, event.data["cause"]) for event in agent_selected_events] == [
        ("planning_agent", "route"),
        ("search_agent", "handoff"),
    ]

    complete = next(event for event in events if event.type == "complete")
    assert complete.data["response"].message.content == "Search Agent final answer."


def test_recover_terminal_response_ignores_handoff_narration():
    """When the only AI content in state is a hand_off narration AIMessage
    (it carries a ``hand_off`` tool call), it must NOT become the recovered
    final assistant response — there is no real reply to deliver yet."""
    workflow = _make_workflow()
    state = {
        "active_agent_id": "search_agent",
        "messages": [
            HumanMessage(content="latest info please"),
            AIMessage(
                content="Transfering your request to Search Agent...",
                tool_calls=[{"id": "h1", "name": "hand_off", "args": {}}],
            ),
            ToolMessage(
                content='{"hand_off":"search_agent"}',
                tool_call_id="h1",
                name="hand_off",
            ),
        ],
        "context": {
            "handoff": {
                "active": True,
                "target_agent": "search_agent",
                "tool_call_id": "h1",
            }
        },
    }

    assert workflow._recover_terminal_response(state) is None
