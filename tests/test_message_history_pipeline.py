"""Memory refactor Tasks 4–8 guards: end-to-end ID plumbing and compaction.

These tests do not require a live database. They use the workflow boundary
schemas directly to assert that stable DB ids flow into the graph state and
that the final ``AIMessage`` carries the reserved assistant id.
"""

from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import (
    AgentMessage,
    AgentResponse,
    AgentType,
    MessageRole,
    WorkflowExecutionRequest,
)


def _make_workflow() -> MultiAgentWorkflow:
    """Construct a workflow shell without invoking heavy ``__init__`` deps."""
    return MultiAgentWorkflow.__new__(MultiAgentWorkflow)


def test_initial_state_carries_user_and_assistant_ids():
    """Workflow request fields must propagate into both ``GraphState`` and the
    initial ``HumanMessage`` so subsequent turns can identify the current
    prompt deterministically."""
    workflow = _make_workflow()
    user_id = str(uuid4())
    assistant_id = str(uuid4())

    request = WorkflowExecutionRequest(
        message="hello",
        conversation_id=str(uuid4()),
        user_id=str(uuid4()),
        user_message_id=user_id,
        assistant_message_id=assistant_id,
    )

    state = workflow._build_initial_state_from_request(request)

    assert state["user_message_id"] == user_id
    assert state["assistant_message_id"] == assistant_id
    [human] = state["messages"]
    assert isinstance(human, HumanMessage)
    assert human.id == user_id


def test_finalize_response_stamps_only_final_assistant_message():
    """Tool-calling AI messages must NOT carry the reserved assistant id —
    only the final user-visible reply gets it."""
    workflow = _make_workflow()
    assistant_id = str(uuid4())

    final_response = AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content="all done"),
    )
    state_with_final: dict = {
        "messages": [],
        "assistant_message_id": assistant_id,
    }
    workflow._finalize_agent_response(state_with_final, final_response)
    [final] = state_with_final["messages"]
    assert isinstance(final, AIMessage)
    assert final.id == assistant_id

    tool_response = AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[{"id": "abc", "name": "search", "args": {}}],
        ),
    )
    state_with_tool: dict = {
        "messages": [],
        "assistant_message_id": assistant_id,
    }
    workflow._finalize_agent_response(state_with_tool, tool_response)
    [intermediate] = state_with_tool["messages"]
    assert isinstance(intermediate, AIMessage)
    assert intermediate.id != assistant_id


async def _run_compaction(workflow, *, snapshot, config, thread_id):
    workflow.checkpointer = object()
    captured = {}

    async def fake_get_state(_config):
        return snapshot

    async def fake_update_state(_config, payload):
        captured["payload"] = payload

    workflow.graph = type(
        "FakeGraph",
        (),
        {
            "aget_state": staticmethod(fake_get_state),
            "aupdate_state": staticmethod(fake_update_state),
        },
    )()
    await workflow._compact_checkpoint_after_terminal_response(config=config, thread_id=thread_id)
    return captured


def test_checkpoint_compaction_runs_after_complete_not_after_interrupt(monkeypatch):
    """Compaction issues a RemoveMessage for every checkpoint message id when
    the snapshot has no pending ``next`` node. When the graph is paused mid-turn
    (next non-empty), compaction must be a no-op."""
    import asyncio
    from types import SimpleNamespace

    from langgraph.graph.message import RemoveMessage

    workflow = _make_workflow()

    msg_a = AIMessage(content="hello", id="m-a")
    msg_b = AIMessage(content="world", id="m-b")
    snapshot_done = SimpleNamespace(
        next=[],
        values={"messages": [msg_a, msg_b]},
    )
    captured = asyncio.run(
        _run_compaction(
            workflow,
            snapshot=snapshot_done,
            config={"configurable": {"thread_id": "conv-1"}},
            thread_id="conv-1",
        )
    )
    removals = captured["payload"]["messages"]
    assert all(isinstance(r, RemoveMessage) for r in removals)
    assert {r.id for r in removals} == {"m-a", "m-b"}

    snapshot_paused = SimpleNamespace(
        next=["approval"],
        values={"messages": [msg_a]},
    )
    captured_paused = asyncio.run(
        _run_compaction(
            workflow,
            snapshot=snapshot_paused,
            config={"configurable": {"thread_id": "conv-1"}},
            thread_id="conv-1",
        )
    )
    assert "payload" not in captured_paused, "Compaction must skip on interrupt"


def test_workflow_request_round_trip_through_ai_schema():
    """Ensure the AI-layer ``WorkflowExecutionRequest`` mirrors the new
    ID fields on the service-layer schema (round-trip via ``model_dump``)."""
    from app.ai.schemas import WorkflowExecutionRequest as AIWorkflowExecutionRequest
    from app.services.ai_service import AIService

    request = WorkflowExecutionRequest(
        message="hi",
        conversation_id=str(uuid4()),
        user_id=str(uuid4()),
        user_message_id=str(uuid4()),
        assistant_message_id=str(uuid4()),
    )

    ai_request = AIService._to_ai_request(request)
    assert isinstance(ai_request, AIWorkflowExecutionRequest)
    assert ai_request.user_message_id == request.user_message_id
    assert ai_request.assistant_message_id == request.assistant_message_id
