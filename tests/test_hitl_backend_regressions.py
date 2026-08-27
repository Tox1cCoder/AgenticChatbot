from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.ai import graph as graph_module
from app.ai.schemas import InterruptDecision
from app.services.message_service import MessageService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pending_node", ["chat_agent", "custom_agent", "planning_tools", "rag_tools"]
)
async def test_decision_stream_accepts_all_approval_interrupt_nodes(pending_node):
    workflow = graph_module.MultiAgentWorkflow.__new__(graph_module.MultiAgentWorkflow)
    workflow.checkpointer = object()
    workflow.graph = SimpleNamespace(
        aget_state=lambda _config: _async_value(
            SimpleNamespace(
                next=(pending_node,),
                values={"active_agent_id": "chat_agent", "conversation_id": "conv-1"},
            )
        )
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


async def _async_value(value):
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_node", ["planning_tools", "rag_tools"])
async def test_auto_resume_returns_followup_planning_or_rag_interrupt(pending_node):
    workflow = graph_module.MultiAgentWorkflow.__new__(graph_module.MultiAgentWorkflow)
    workflow.checkpointer = object()
    initial_snapshot = SimpleNamespace(next=(pending_node,), values={"messages": []})
    followup_snapshot = SimpleNamespace(
        next=(pending_node,),
        values={"conversation_id": "conv-1"},
    )
    workflow.graph = SimpleNamespace(
        aget_state=AsyncMock(side_effect=[initial_snapshot, followup_snapshot]),
        ainvoke=AsyncMock(return_value={}),
    )
    workflow._build_graph_config = lambda _thread_id: {}
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
