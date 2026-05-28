"""Memory refactor Tasks 4–8 guards: end-to-end ID plumbing and compaction.

These tests do not require a live database. They use the workflow boundary
schemas directly to assert that stable DB ids flow into the graph state and
that the final ``AIMessage`` carries the reserved assistant id.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
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


@pytest.mark.asyncio
async def test_streaming_workflow_does_not_compact_before_complete_event(monkeypatch):
    """Checkpoint compaction must not happen inside the workflow before the
    service has durably persisted the terminal assistant message."""

    from types import SimpleNamespace

    workflow = _make_workflow()
    workflow.checkpointer = object()

    response = AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content="done"),
        metadata={},
    )
    final_state = {
        "messages": [AIMessage(content="done")],
        "selected_agent": "chat_agent",
        "response": response,
        "context": {},
    }

    class FakeGraph:
        async def astream(self, *_args, **_kwargs):
            yield ("updates", {"chat_agent": final_state})

        async def aget_state(self, _config):
            return SimpleNamespace(next=[], values=final_state)

    workflow.graph = FakeGraph()
    workflow._route_node = AsyncMock(return_value={"selected_agent": "chat_agent"})
    workflow._get_conversation_history = AsyncMock(return_value=[])

    events: list[str] = []

    async def fake_compact(*, config, thread_id):
        events.append("compact")

    monkeypatch.setattr(workflow, "_compact_checkpoint_after_terminal_response", fake_compact)

    request = WorkflowExecutionRequest(
        message="hello",
        conversation_id="conv-1",
    )

    async for event in workflow.execute_request_stream(request):
        events.append(event["type"])
        if event["type"] == "complete":
            break

    assert events == ["agent_selected", "complete"]


@pytest.mark.asyncio
async def test_message_service_compacts_checkpoint_after_persist(monkeypatch):
    """Once the service has persisted the assistant row, it can safely clear
    LangGraph's transient checkpoint transcript."""

    from types import SimpleNamespace

    from app.schemas.workflow import (
        WorkflowExecutionRequest as ServiceWorkflowExecutionRequest,
    )
    from app.schemas.workflow import (
        WorkflowPlanningContext,
        WorkflowResponse,
        WorkflowResponseMessage,
    )
    from app.services.message_service import MessageService

    service = MessageService.__new__(MessageService)
    events: list[tuple[str, str | None]] = []
    conversation_id = uuid4()
    assistant_message_id = uuid4()

    class FakeAIService:
        async def compact_checkpoint_after_terminal_response(self, thread_id):
            events.append(("compact", str(thread_id) if thread_id is not None else None))

    service.ai_service = FakeAIService()
    service.task_plan_service = None
    service._sync_response_plan_state = lambda **_kwargs: False
    service._generate_and_add_suggestions = AsyncMock()
    service._schedule_summary_refresh = lambda **kwargs: events.append(
        ("summary", str(kwargs["through_message_id"]))
    )

    def fake_create_bot_response_message(**kwargs):
        events.append(("persist", str(kwargs.get("message_id"))))
        return SimpleNamespace(id=kwargs.get("message_id"))

    service._create_bot_response_message = fake_create_bot_response_message

    workflow_request = ServiceWorkflowExecutionRequest(
        message="hello",
        conversation_id=str(conversation_id),
        user_id=str(uuid4()),
        thread_id=None,
        planning=WorkflowPlanningContext(),
    )
    bot_response = WorkflowResponse(
        message=WorkflowResponseMessage(content="done"),
        metadata={},
    )

    await service._persist_completed_workflow_response(
        conversation_id=conversation_id,
        user_id=uuid4(),
        bot_response=bot_response,
        sanitized_persona=None,
        workflow_request=workflow_request,
        message_id=assistant_message_id,
    )
    await service._compact_checkpoint_after_persist(
        workflow_request=workflow_request,
        conversation_id=conversation_id,
    )

    assert events == [
        ("persist", str(assistant_message_id)),
        ("summary", str(assistant_message_id)),
        ("compact", str(conversation_id)),
    ]


@pytest.mark.asyncio
async def test_capable_widget_response_does_not_invent_missing_marker(monkeypatch):
    from types import SimpleNamespace

    from app.core.config import settings
    from app.schemas.workflow import (
        WorkflowExecutionRequest as ServiceWorkflowExecutionRequest,
    )
    from app.schemas.workflow import (
        WorkflowPlanningContext,
        WorkflowResponse,
        WorkflowResponseMessage,
    )
    from app.services.message_service import MessageService

    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    service = MessageService.__new__(MessageService)
    captured: dict = {}
    service.task_plan_service = None
    service._sync_response_plan_state = lambda **_kwargs: False
    service._generate_and_add_suggestions = AsyncMock()
    service._schedule_summary_refresh = lambda **_kwargs: None

    def fake_create_bot_response_message(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    service._create_bot_response_message = fake_create_bot_response_message
    workflow_request = ServiceWorkflowExecutionRequest(
        message="Explain compound interest with a chart.",
        conversation_id=str(uuid4()),
        user_id=str(uuid4()),
        planning=WorkflowPlanningContext(),
        inline_rich_response_v1=True,
    )
    bot_response = WorkflowResponse(
        message=WorkflowResponseMessage(
            content="Compound interest builds over time.\n\nThe later years diverge sharply."
        ),
        metadata={"_inline_rich_response_v1": True},
        tool_artifacts=[
            {
                "tool_call_id": "widget-call",
                "tool": "widget_create",
                "output": (
                    '{"widget_id":"w-inline","session_id":"conv-1","widget_type":"chart",'
                    '"title":"Growth comparison","status":"active","version":1}'
                ),
                "status": "success",
            }
        ],
    )

    await service._persist_completed_workflow_response(
        conversation_id=uuid4(),
        user_id=uuid4(),
        bot_response=bot_response,
        sanitized_persona=None,
        workflow_request=workflow_request,
    )

    assert captured["content"] == (
        "Compound interest builds over time.\n\nThe later years diverge sharply."
    )
    assert "<!--rich:" not in captured["content"]
    assert captured["metadata"]["rich_items_version"] == 1
    assert captured["metadata"]["rich_items"][0]["id"] == "widget:w-inline"


@pytest.mark.asyncio
async def test_checkpoint_compaction_disables_langsmith_tracing(monkeypatch):
    """Checkpoint compaction must wrap ``aupdate_state`` in
    ``tracing_context(enabled=False)`` so the cleanup does not surface as a
    noisy ``LangGraphUpdateState`` entry with ``remove - No data`` rows in
    LangSmith. The cleanup *state mutation* still has to run — only its
    LangSmith trace visibility is suppressed."""
    from types import SimpleNamespace

    workflow = _make_workflow()
    workflow.checkpointer = object()
    calls: list[tuple[str, object]] = []

    class FakeTracingContext:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def __enter__(self):
            calls.append(("enter", None))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", None))
            return False

    async def fake_get_state(_config):
        return SimpleNamespace(
            next=[],
            values={"messages": [AIMessage(content="x", id="m1")]},
        )

    async def fake_update_state(_config, _payload):
        calls.append(("update", None))

    workflow.graph = SimpleNamespace(
        aget_state=fake_get_state,
        aupdate_state=fake_update_state,
    )
    monkeypatch.setattr("app.ai.graph.tracing_context", FakeTracingContext)

    await workflow._compact_checkpoint_after_terminal_response(
        config={"configurable": {"thread_id": "conv-1"}},
        thread_id="conv-1",
    )

    assert ("init", {"enabled": False}) in calls
    assert (
        calls.index(("enter", None)) < calls.index(("update", None)) < calls.index(("exit", None))
    )


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
