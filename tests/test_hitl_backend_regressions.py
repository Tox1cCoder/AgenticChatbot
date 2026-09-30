from types import SimpleNamespace

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


def test_the_non_streaming_resume_chain_stays_deleted():
    """No route called it, and it could not have worked.

    It resumed ``thread_id=str(conversation_id)``, but checkpoint threads are
    per turn (``routing-v2:{conversation}:{turn}``), so it addressed a thread
    that never exists. Resuming goes through the streaming path only.
    """
    from app.interfaces.message_service_interface import IMessageService
    from app.interfaces.workflow_runtime_interface import IWorkflowRuntime
    from app.services.ai_service import AIService

    assert not hasattr(AIService, "resume_workflow")
    assert not hasattr(MessageService, "resume_workflow")
    assert not hasattr(IMessageService, "resume_workflow")
    assert not hasattr(graph_module.MultiAgentWorkflow, "resume")
    assert not hasattr(IWorkflowRuntime, "resume")


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
