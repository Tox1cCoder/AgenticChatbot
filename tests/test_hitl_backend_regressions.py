from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.ai import graph as graph_module
from app.ai.schemas import InterruptDecision
from app.services.message_service import MessageService


def _paused_snapshot(pending_node: str, *, tool_call_id: str = "call-1"):
    """A checkpoint waiting on one human decision, whatever node asked."""
    item = SimpleNamespace(
        id="int-1",
        value={"action_requests": [{"name": "write", "tool_call_id": tool_call_id}]},
    )
    task = SimpleNamespace(name=pending_node, id="task-1", interrupts=(item,), result=None)
    return SimpleNamespace(
        next=(pending_node,),
        tasks=(task,),
        interrupts=(item,),
        values={"active_agent_id": "chat_agent", "conversation_id": "conv-1"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pending_node",
    ["chat_agent", "custom_agent", "planning_worker", "rag_agent", "a_node_added_tomorrow"],
)
async def test_decision_stream_accepts_a_pause_from_any_node(pending_node):
    """Resume is gated on pending interrupts, not on a node-name allowlist.

    The allowlist this replaced never listed ``planning_worker``, so once
    Planning fan-out became parent topology a paused worker could not be
    resumed at all.
    """
    workflow = graph_module.MultiAgentWorkflow.__new__(graph_module.MultiAgentWorkflow)
    workflow.checkpointer = object()
    workflow.graph = SimpleNamespace(
        aget_state=lambda _config: _async_value(_paused_snapshot(pending_node))
    )
    workflow._build_graph_config = lambda _thread_id: {}

    stream = workflow.resume_with_decisions_stream(
        "thread-1",
        [InterruptDecision(type="approve", tool_call_id="call-1")],
    )

    first = await anext(stream)
    assert first.type == "agent_selected"
    assert first.agent == "chat_agent"
    assert first.data == {"agent": "chat_agent"}
    await stream.aclose()


@pytest.mark.asyncio
async def test_decision_stream_refuses_a_turn_that_is_not_waiting():
    """A turn merely mid-execution has a next node but nothing to approve."""
    workflow = graph_module.MultiAgentWorkflow.__new__(graph_module.MultiAgentWorkflow)
    workflow.checkpointer = object()
    workflow.graph = SimpleNamespace(
        aget_state=lambda _config: _async_value(
            SimpleNamespace(next=("chat_agent",), tasks=(), interrupts=(), values={})
        )
    )
    workflow._build_graph_config = lambda _thread_id: {}

    stream = workflow.resume_with_decisions_stream(
        "thread-1", [InterruptDecision(type="approve", tool_call_id="call-1")]
    )
    with pytest.raises(ValueError, match="not waiting"):
        await anext(stream)


async def _async_value(value):
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_node", ["planning_worker", "rag_agent"])
async def test_auto_resume_returns_followup_planning_or_rag_interrupt(pending_node):
    workflow = graph_module.MultiAgentWorkflow.__new__(graph_module.MultiAgentWorkflow)
    workflow.checkpointer = object()
    initial_snapshot = SimpleNamespace(
        next=(pending_node,), tasks=(), interrupts=(), values={"messages": []}
    )
    followup_snapshot = _paused_snapshot(pending_node)
    workflow.graph = SimpleNamespace(
        aget_state=AsyncMock(side_effect=[initial_snapshot, followup_snapshot]),
        ainvoke=AsyncMock(return_value={}),
    )
    workflow._build_graph_config = lambda _thread_id: {}
    # A resume now hands the graph a runtime context built from the
    # checkpointed values, because the transition resolver reads the *live*
    # inventory from it. This stub is a bare `__new__` instance, so give it the
    # collaborators that build requires.
    workflow.agents = {"chat_agent": object(), pending_node: object()}
    workflow.routing_service = object()
    workflow.routing_context_builder = object()
    expected = object()
    workflow._build_interrupt_agent_response = lambda _snapshot, _thread_id, _conversation_id: (
        expected
    )

    assert await workflow.resume("thread-1", user_input="continue") is expected


def test_server_mcp_qualified_id_is_not_client_runtime_provenance():
    provenance = {
        "tool_origin": "server_mcp",
        "server_name": "calculator",
        "qualified_tool_id": "calculator::calculate",
    }

    assert MessageService._is_client_runtime_provenance_entry(provenance) is False
    assert MessageService._derive_interrupt_execution_scope(
        {"tool_provenance": {"call-1": provenance}}
    ) == {
        "session_id": None,
        "catalog_version": None,
        "tool_instance_id": None,
    }
    assert (
        MessageService._is_client_runtime_provenance_entry(
            {"qualified_tool_id": "calculator::calculate"}
        )
        is False
    )
