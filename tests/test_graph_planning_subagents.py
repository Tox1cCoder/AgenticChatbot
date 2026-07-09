"""Graph-level tests for Planning-mode subagent integration.

Covers:
- The Planning Agent only receives ``dispatch_subagents`` when Planning
  mode is active and the feature flag is enabled.
- ``MultiAgentWorkflow._run_agent_in_isolated_context`` builds a child
  state that does NOT inherit the parent's chat ``messages`` and that
  carries scoped identifiers (``conversation_id``, ``user_id``,
  ``device_id``).
- The dispatcher rejects ``planning_agent`` as a worker target.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import (
    AgentMessage,
    AgentResponse,
    AgentType,
    MessageRole,
)
from app.core.config import settings


def _ok(content: str = "ok") -> AgentResponse:
    return AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content=content),
        metadata={},
    )


# ---------------------------------------------------------------------------
# Planning node binding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planning_node_binds_dispatch_in_planning_and_executing_when_plan_exists(
    monkeypatch,
):
    monkeypatch.setattr(settings, "planning_subagents_enabled", True)

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow._get_conversation_history = AsyncMock(return_value=[])
    workflow.planning_agent = SimpleNamespace()
    workflow.planning_agent.invoke_model_with_history = AsyncMock(
        return_value=AgentResponse(
            agent_type=AgentType.PLANNING,
            agent_id="planning_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={},
        )
    )

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(
        workflow, "_finalize_forced_final_response", lambda state, response: response
    )
    monkeypatch.setattr(workflow, "_merge_tool_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(workflow, "_finalize_agent_response", lambda state, response: state)
    monkeypatch.setattr(workflow, "_final_response_kwargs", lambda state: {})

    from langchain_core.messages import HumanMessage

    base_state: dict[str, Any] = {
        "messages": [HumanMessage(content="execute the plan")],
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "todos": [
            {
                "id": "todo-1",
                "description": "Research the migration constraints",
                "status": "pending",
                "order": 0,
            }
        ],
        "current_task_index": 0,
        "planning_call_count": 0,
        "planning_mode_enabled": True,
        "has_existing_plan": True,
        "context": {},
    }

    # Planning phase still binds the dispatcher so the model can decide whether
    # the current turn is execution/delegation work instead of being forced into
    # a no-subagents answer by a hard graph gate.
    state_planning = dict(base_state)
    state_planning["planning_phase"] = "planning"
    await workflow._planning_node(state_planning)
    call_kwargs = workflow.planning_agent.invoke_model_with_history.call_args.kwargs
    internal_tools = call_kwargs.get("internal_tools") or []
    tool_names = [t.name for t in internal_tools]
    assert "dispatch_subagents" in tool_names

    # Executing phase: should bind the dispatcher.
    state_exec = dict(base_state)
    state_exec["planning_phase"] = "executing"
    workflow.planning_agent.invoke_model_with_history.reset_mock()
    await workflow._planning_node(state_exec)
    call_kwargs = workflow.planning_agent.invoke_model_with_history.call_args.kwargs
    internal_tools = call_kwargs.get("internal_tools") or []
    tool_names = [t.name for t in internal_tools]
    assert "dispatch_subagents" in tool_names


@pytest.mark.asyncio
async def test_planning_node_binds_dispatch_when_planning_mode_disabled(
    monkeypatch,
):
    """planning_subagents_enabled is the only binding gate.

    An explicit "use subagents" request routes to planning_agent without
    Planning mode being pre-enabled, so the dispatch tool must still bind.
    """
    monkeypatch.setattr(settings, "planning_subagents_enabled", True)

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow._get_conversation_history = AsyncMock(return_value=[])
    workflow.planning_agent = SimpleNamespace()
    workflow.planning_agent.invoke_model_with_history = AsyncMock(
        return_value=AgentResponse(
            agent_type=AgentType.PLANNING,
            agent_id="planning_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={},
        )
    )
    monkeypatch.setattr(
        workflow, "_finalize_forced_final_response", lambda state, response: response
    )
    monkeypatch.setattr(workflow, "_merge_tool_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(workflow, "_finalize_agent_response", lambda state, response: state)
    monkeypatch.setattr(workflow, "_final_response_kwargs", lambda state: {})

    await workflow._planning_node(
        {
            "messages": [HumanMessage(content="create a plan")],
            "conversation_id": "conv-1",
            "user_id": "user-1",
            "device_id": "device-1",
            "todos": [],
            "current_task_index": None,
            "planning_call_count": 0,
            "planning_mode_enabled": False,
            "has_existing_plan": False,
            "planning_phase": "planning",
            "context": {},
        }
    )

    call_kwargs = workflow.planning_agent.invoke_model_with_history.call_args.kwargs
    tool_names = [tool.name for tool in call_kwargs.get("internal_tools") or []]
    assert "dispatch_subagents" in tool_names


@pytest.mark.asyncio
async def test_planning_node_binds_dispatch_without_existing_plan(monkeypatch):
    """Plan presence is NO LONGER a gate — users can test subagents on fresh plans."""
    monkeypatch.setattr(settings, "planning_subagents_enabled", True)

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow._get_conversation_history = AsyncMock(return_value=[])
    workflow.planning_agent = SimpleNamespace()
    workflow.planning_agent.invoke_model_with_history = AsyncMock(
        return_value=AgentResponse(
            agent_type=AgentType.PLANNING,
            agent_id="planning_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={},
        )
    )
    monkeypatch.setattr(
        workflow, "_finalize_forced_final_response", lambda state, response: response
    )
    monkeypatch.setattr(workflow, "_merge_tool_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(workflow, "_finalize_agent_response", lambda state, response: state)
    monkeypatch.setattr(workflow, "_final_response_kwargs", lambda state: {})

    await workflow._planning_node(
        {
            "messages": [HumanMessage(content="use a subagent to search")],
            "conversation_id": "conv-1",
            "user_id": "user-1",
            "device_id": "device-1",
            "todos": [],
            "current_task_index": None,
            "planning_call_count": 0,
            "planning_mode_enabled": True,
            "has_existing_plan": False,
            "planning_phase": "planning",
            "context": {},
        }
    )

    call_kwargs = workflow.planning_agent.invoke_model_with_history.call_args.kwargs
    tool_names = [tool.name for tool in call_kwargs.get("internal_tools") or []]
    assert "dispatch_subagents" in tool_names


@pytest.mark.asyncio
async def test_planning_node_binds_dispatch_schema_without_constructing_dispatcher(monkeypatch):
    monkeypatch.setattr(settings, "planning_subagents_enabled", True)

    class _UnexpectedDispatcher:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dispatcher should not be constructed for schema binding")

    monkeypatch.setattr(
        "app.ai.planning_subagents.PlanningSubagentDispatcher",
        _UnexpectedDispatcher,
    )

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow._get_conversation_history = AsyncMock(return_value=[])
    workflow.planning_agent = SimpleNamespace()
    workflow.planning_agent.invoke_model_with_history = AsyncMock(
        return_value=AgentResponse(
            agent_type=AgentType.PLANNING,
            agent_id="planning_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={},
        )
    )
    monkeypatch.setattr(
        workflow, "_finalize_forced_final_response", lambda state, response: response
    )
    monkeypatch.setattr(workflow, "_merge_tool_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(workflow, "_finalize_agent_response", lambda state, response: state)
    monkeypatch.setattr(workflow, "_final_response_kwargs", lambda state: {})

    await workflow._planning_node(
        {
            "messages": [HumanMessage(content="execute")],
            "conversation_id": "conv-1",
            "user_id": "user-1",
            "device_id": "device-1",
            "todos": [],
            "current_task_index": 0,
            "planning_call_count": 0,
            "planning_mode_enabled": True,
            "planning_phase": "executing",
            "context": {},
        }
    )

    call_kwargs = workflow.planning_agent.invoke_model_with_history.call_args.kwargs
    tool_names = [tool.name for tool in call_kwargs.get("internal_tools") or []]
    assert "dispatch_subagents" in tool_names


@pytest.mark.asyncio
async def test_planning_node_binds_dispatch_when_planning_mode_disabled_in_executing_phase(
    monkeypatch,
):
    monkeypatch.setattr(settings, "planning_subagents_enabled", True)

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow._get_conversation_history = AsyncMock(return_value=[])
    workflow.planning_agent = SimpleNamespace()
    workflow.planning_agent.invoke_model_with_history = AsyncMock(
        return_value=AgentResponse(
            agent_type=AgentType.PLANNING,
            agent_id="planning_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={},
        )
    )
    monkeypatch.setattr(
        workflow, "_finalize_forced_final_response", lambda state, response: response
    )
    monkeypatch.setattr(workflow, "_merge_tool_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(workflow, "_finalize_agent_response", lambda state, response: state)
    monkeypatch.setattr(workflow, "_final_response_kwargs", lambda state: {})

    from langchain_core.messages import HumanMessage

    state: dict[str, Any] = {
        "messages": [HumanMessage(content="execute")],
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "todos": [],
        "current_task_index": 0,
        "planning_call_count": 0,
        "planning_mode_enabled": False,
        "planning_phase": "executing",
        "context": {},
    }

    await workflow._planning_node(state)
    call_kwargs = workflow.planning_agent.invoke_model_with_history.call_args.kwargs
    internal_tools = call_kwargs.get("internal_tools") or []
    tool_names = [t.name for t in internal_tools]
    assert "dispatch_subagents" in tool_names


@pytest.mark.asyncio
async def test_planning_node_does_not_bind_dispatch_when_feature_disabled(monkeypatch):
    monkeypatch.setattr(settings, "planning_subagents_enabled", False)

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow._get_conversation_history = AsyncMock(return_value=[])
    workflow.planning_agent = SimpleNamespace()
    workflow.planning_agent.invoke_model_with_history = AsyncMock(
        return_value=AgentResponse(
            agent_type=AgentType.PLANNING,
            agent_id="planning_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
            metadata={},
        )
    )
    monkeypatch.setattr(
        workflow, "_finalize_forced_final_response", lambda state, response: response
    )
    monkeypatch.setattr(workflow, "_merge_tool_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(workflow, "_finalize_agent_response", lambda state, response: state)
    monkeypatch.setattr(workflow, "_final_response_kwargs", lambda state: {})

    from langchain_core.messages import HumanMessage

    state: dict[str, Any] = {
        "messages": [HumanMessage(content="execute")],
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "todos": [],
        "current_task_index": 0,
        "planning_call_count": 0,
        "planning_mode_enabled": True,
        "planning_phase": "executing",
        "context": {},
    }

    await workflow._planning_node(state)
    call_kwargs = workflow.planning_agent.invoke_model_with_history.call_args.kwargs
    internal_tools = call_kwargs.get("internal_tools") or []
    tool_names = [t.name for t in internal_tools]
    assert "dispatch_subagents" not in tool_names


def _planning_tool_state(
    *,
    planning_call_count: int,
    tool_name: str = "write_todos",
    tool_content: str = "done",
) -> dict[str, Any]:
    return {
        "selected_agent": "planning_agent",
        "planning_call_count": planning_call_count,
        "planning_mode_enabled": True,
        "planning_phase": "executing",
        "todos": [
            {"id": "t1", "description": "done task", "status": "completed", "order": 0},
            {"id": "t2", "description": "remaining task", "status": "pending", "order": 1},
        ],
        "current_task_index": 1,
        "messages": [
            HumanMessage(content="execute the plan"),
            AIMessage(
                content="",
                tool_calls=[{"id": "tc-1", "name": tool_name, "args": {}}],
            ),
            ToolMessage(content=tool_content, tool_call_id="tc-1", name=tool_name),
        ],
        "context": {},
    }


def test_should_continue_planning_does_not_end_on_soft_budget_with_tool_result(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    monkeypatch.setattr(settings, "auto_continue_enabled", True)
    monkeypatch.setattr(settings, "auto_continue_soft_limit_ratio", 0.5)
    monkeypatch.setattr(settings, "planning_max_iterations", 10)

    state = _planning_tool_state(planning_call_count=5)

    route = workflow._should_continue_planning(state)

    assert route == "planning_agent"
    context = state.get("context") or {}
    assert "continuation_signal" not in context
    assert "force_final_response" not in context


def test_should_continue_planning_hard_budget_forces_final_synthesis_after_tool_result(
    monkeypatch,
):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    monkeypatch.setattr(settings, "auto_continue_enabled", True)
    monkeypatch.setattr(settings, "planning_max_iterations", 10)

    state = _planning_tool_state(planning_call_count=10)

    route = workflow._should_continue_planning(state)

    assert route == "planning_agent"
    context = state.get("context") or {}
    assert context["force_final_response"] is True
    assert context["tool_budget"]["reason"] == "max_iterations_reached"
    assert context["tool_budget"]["scope"] == "planning"


def test_should_continue_planning_ignores_budget_when_disabled_after_tool_result(
    monkeypatch,
):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    monkeypatch.setattr(settings, "auto_continue_enabled", True)
    monkeypatch.setattr(settings, "auto_continue_soft_limit_ratio", 0.5)
    monkeypatch.setattr(settings, "planning_max_iterations", 0)

    state = _planning_tool_state(planning_call_count=999)

    route = workflow._should_continue_planning(state)

    assert route == "planning_agent"
    context = state.get("context") or {}
    assert "force_final_response" not in context
    assert "tool_budget" not in context
    assert "pause_reason" not in context
    assert "continuation_signal" not in context


def test_should_continue_planning_plan_mutation_ignores_budget_when_disabled(
    monkeypatch,
):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    monkeypatch.setattr(settings, "planning_max_iterations", 0)

    state = _planning_tool_state(planning_call_count=999)
    state["context"]["plan_just_modified"] = True

    route = workflow._should_continue_planning(state)

    assert route == "planning_agent"
    context = state.get("context") or {}
    assert context["generate_plan_response"] is True
    assert "force_final_response" not in context
    assert "tool_budget" not in context
    assert "pause_reason" not in context


def test_recover_terminal_response_ignores_empty_tool_call_response():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state = _planning_tool_state(planning_call_count=10)
    state["response"] = AgentResponse(
        agent_type=AgentType.PLANNING,
        agent_id="planning_agent",
        message=AgentMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[{"id": "tc-1", "name": "write_todos", "args": {}}],
        ),
        metadata={},
    )

    response = workflow._recover_terminal_response(state)

    assert response is None


def test_attach_planning_state_metadata_includes_subagent_activity():
    response = AgentResponse(
        agent_type=AgentType.PLANNING,
        agent_id="planning_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content="summary"),
        metadata={},
    )
    state: dict[str, Any] = {
        "todos": [],
        "planning_call_count": 3,
        "context": {
            "subagent_dispatches": [
                {
                    "rationale": "parallel checks",
                    "task_ids": ["w1", "w2"],
                    "agents": ["search_agent", "chat_agent"],
                    "status": "partial",
                }
            ],
            "subagent_results": [
                {
                    "id": "w1",
                    "agent": "search_agent",
                    "status": "completed",
                    "elapsed_ms": 120,
                    "summary": "done",
                },
                {
                    "id": "w2",
                    "agent": "chat_agent",
                    "status": "timeout",
                    "elapsed_ms": 1000,
                    "summary": "timed out",
                    "error": "timeout",
                },
            ],
        },
    }

    enriched = MultiAgentWorkflow._attach_planning_state_metadata(response, state)

    assert enriched.metadata["subagent_dispatches"][0]["status"] == "partial"
    assert [item["id"] for item in enriched.metadata["subagent_results"]] == ["w1", "w2"]


# ---------------------------------------------------------------------------
# Isolated worker runner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_agent_in_isolated_context_does_not_pollute_parent_messages(monkeypatch):
    """Worker intermediate messages must NOT be appended to the parent state."""
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow._get_conversation_history = AsyncMock(return_value=[])

    async def fake_invoke(messages, conversation_history, persona, **kwargs):
        # The worker ought to have received its own isolated state with
        # a single HumanMessage carrying the task prompt.
        assert len(messages) == 1
        return _ok("worker done")

    chat_agent = SimpleNamespace(
        invoke_model_with_history=fake_invoke,
        agent_config_key="chat",
        agent_id="chat_agent",
    )
    workflow.chat_agent = chat_agent
    workflow.agents = {"chat_agent": chat_agent}

    from langchain_core.messages import AIMessage, HumanMessage

    parent_messages = [
        HumanMessage(content="parent prompt"),
        AIMessage(content="parent reply"),
    ]
    parent_state: dict[str, Any] = {
        "messages": list(parent_messages),
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "context": {},
    }

    response = await workflow._run_agent_in_isolated_context(
        agent_name="chat_agent",
        task_prompt="this is the worker prompt — find files",
        parent_state=parent_state,
    )

    assert response.message.content == "worker done"
    # Parent state messages must not have been mutated by the worker.
    assert parent_state["messages"] == parent_messages


@pytest.mark.asyncio
async def test_run_agent_in_isolated_context_stamps_subagent_task_id_in_run_config():
    """Worker model runs must carry subagent_task_id so the v3 stream
    translator can attribute their deltas to the right worker row."""
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow._get_conversation_history = AsyncMock(return_value=[])

    captured_kwargs: dict[str, Any] = {}

    async def fake_invoke(messages, conversation_history, persona, **kwargs):
        captured_kwargs.update(kwargs)
        return _ok("done")

    agent = SimpleNamespace(
        invoke_model_with_history=fake_invoke,
        agent_config_key="chat",
        agent_id="chat_agent",
    )
    workflow.chat_agent = agent
    workflow.agents = {"chat_agent": agent}

    await workflow._run_agent_in_isolated_context(
        agent_name="chat_agent",
        task_prompt="worker task",
        parent_state={"messages": [], "context": {}},
        task_id="w7",
    )

    run_config = captured_kwargs["run_config"]
    assert run_config["metadata"]["purpose"] == "planning_subagent"
    assert run_config["metadata"]["subagent_task_id"] == "w7"
    assert run_config["metadata"]["subagent_agent"] == "chat_agent"


@pytest.mark.asyncio
async def test_run_agent_in_isolated_context_inherits_scoped_identifiers_without_history_summary(
    monkeypatch,
):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow._get_conversation_history = AsyncMock(return_value=[])

    captured_kwargs: dict[str, Any] = {}

    async def fake_invoke(messages, conversation_history, persona, **kwargs):
        captured_kwargs.update(kwargs)
        return _ok("done")

    agent = SimpleNamespace(
        invoke_model_with_history=fake_invoke,
        agent_config_key="search",
        agent_id="search_agent",
    )
    workflow.search_agent = agent
    workflow.agents = {"search_agent": agent}

    parent_state: dict[str, Any] = {
        "conversation_id": "conv-42",
        "user_id": "user-99",
        "device_id": "device-7",
        "persona": "Alice",
        "model_request": {"chat": {"model": "x"}},
        "history_summary": "older chat summary",
        "context": {},
        "messages": [],
    }

    await workflow._run_agent_in_isolated_context(
        agent_name="search_agent",
        task_prompt="find docs about API migration constraints",
        parent_state=parent_state,
    )

    assert captured_kwargs["conversation_id"] == "conv-42"
    assert captured_kwargs["user_id"] == "user-99"
    assert captured_kwargs["device_id"] == "device-7"
    assert captured_kwargs["model_request"] == {"chat": {"model": "x"}}
    assert captured_kwargs["history_summary"] is None


@pytest.mark.asyncio
async def test_run_agent_in_isolated_context_tags_worker_invocations_internal(monkeypatch):
    """Subagent worker model runs must be tagged internal so graph streaming
    suppresses nested worker chunks and only the supervisor output is user-facing.
    """
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow._get_conversation_history = AsyncMock(return_value=[])

    captured_kwargs: dict[str, Any] = {}

    async def fake_invoke(messages, conversation_history, persona, **kwargs):
        captured_kwargs.update(kwargs)
        return _ok("done")

    agent = SimpleNamespace(
        invoke_model_with_history=fake_invoke,
        agent_config_key="search",
        agent_id="search_agent",
    )
    workflow.search_agent = agent
    workflow.agents = {"search_agent": agent}

    parent_state: dict[str, Any] = {
        "conversation_id": "conv-42",
        "user_id": "user-99",
        "device_id": "device-7",
        "context": {},
        "messages": [],
    }

    await workflow._run_agent_in_isolated_context(
        agent_name="search_agent",
        task_prompt="find docs about API migration constraints",
        parent_state=parent_state,
    )

    run_config = captured_kwargs.get("run_config")
    assert isinstance(run_config, dict)
    assert "internal" in (run_config.get("tags") or [])
    assert "planning_subagent" in (run_config.get("tags") or [])
    assert (run_config.get("metadata") or {}).get("internal") is True
    assert (run_config.get("metadata") or {}).get("purpose") == "planning_subagent"


@pytest.mark.asyncio
async def test_run_agent_in_isolated_context_tags_rag_worker_internal(monkeypatch):
    """RAG subagents use process_message, so their internal stream tag must
    travel through the AgentMessage metadata.
    """
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    captured: dict[str, Any] = {}

    async def fake_process_message(message, conversation_id):
        captured["metadata"] = dict(message.metadata)
        captured["conversation_id"] = conversation_id
        return _ok("done")

    rag_agent = SimpleNamespace(
        process_message=fake_process_message,
        agent_config_key="rag",
        agent_id="rag_agent",
    )
    workflow.rag_agent = rag_agent
    workflow.agents = {"rag_agent": rag_agent}

    await workflow._run_agent_in_isolated_context(
        agent_name="rag_agent",
        task_prompt="search documents for relevant facts",
        parent_state={
            "conversation_id": "conv-rag",
            "user_id": "user-1",
            "device_id": "device-1",
            "context": {},
            "messages": [],
        },
    )

    run_config = captured["metadata"].get("run_config")
    assert isinstance(run_config, dict)
    assert "internal" in (run_config.get("tags") or [])
    assert "planning_subagent" in (run_config.get("tags") or [])
    assert (run_config.get("metadata") or {}).get("internal") is True
    assert captured["metadata"].get("history_summary") is None


@pytest.mark.asyncio
async def test_run_agent_in_isolated_context_drives_rag_search_loop(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    captured_metadata: list[dict[str, Any]] = []

    async def fake_process_message(message, conversation_id):
        captured_metadata.append(dict(message.metadata))
        if len(captured_metadata) == 1:
            return AgentResponse(
                agent_type=AgentType.RAG,
                agent_id="rag_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[
                        {
                            "id": "rag-tool-1",
                            "name": "search_documents",
                            "args": {"action": "scan_all"},
                        }
                    ],
                ),
                metadata={},
            )
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="final answer"),
            metadata={},
        )

    rag_agent = SimpleNamespace(
        process_message=fake_process_message,
        agent_config_key="rag",
        agent_id="rag_agent",
    )
    workflow.rag_agent = rag_agent
    workflow.agents = {"rag_agent": rag_agent}

    async def fake_execute_search_documents_action(**kwargs):
        assert kwargs["conversation_id"] == "conv-rag"
        assert kwargs["user_id"] == "user-1"
        return "SEARCH RESULT", "scan_all", {"documents": [{"document_id": "doc-1"}]}

    monkeypatch.setattr(
        "app.ai.graph.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    monkeypatch.setattr(
        "app.ai.graph.apply_tool_output_offload",
        lambda **kwargs: (kwargs["output_text"], None),
    )

    response = await workflow._run_agent_in_isolated_context(
        agent_name="rag_agent",
        task_prompt="search documents for relevant facts",
        parent_state={
            "conversation_id": "conv-rag",
            "user_id": "user-1",
            "device_id": "device-1",
            "history_summary": "parent memory should not be injected",
            "context": {},
            "messages": [],
        },
    )

    assert response.message.content == "final answer"
    assert len(captured_metadata) == 2
    assert captured_metadata[0]["tool_context"] == []
    assert captured_metadata[1]["tool_context"] == ["SEARCH RESULT"]
    assert captured_metadata[1]["history_summary"] is None
    assert response.tool_artifacts is not None
    assert response.tool_artifacts[0]["tool"] == "search_documents"


@pytest.mark.asyncio
async def test_run_agent_in_isolated_context_allows_generic_worker_under_iteration_cap(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    call_count = 0

    async def fake_invoke(messages, conversation_history, persona, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return AgentResponse(
                agent_type=AgentType.SEARCH,
                agent_id="search_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[
                        {
                            "id": f"tool-{call_count}",
                            "name": "lookup",
                            "args": {"step": call_count},
                        }
                    ],
                ),
                metadata={},
            )
        return AgentResponse(
            agent_type=AgentType.SEARCH,
            agent_id="search_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="final answer"),
            metadata={},
        )

    agent = SimpleNamespace(
        invoke_model_with_history=fake_invoke,
        agent_config_key="search",
        agent_id="search_agent",
    )
    workflow.search_agent = agent
    workflow.agents = {"search_agent": agent}

    async def fake_ensure_map(*args, **kwargs):
        async def lookup(args):
            return f"step {args['step']} complete"

        return {"lookup": SimpleNamespace(name="lookup", ainvoke=lookup)}

    monkeypatch.setattr("app.ai.graph.ensure_agent_tool_map", fake_ensure_map)

    response = await workflow._run_agent_in_isolated_context(
        agent_name="search_agent",
        task_prompt="perform a multi-step lookup",
        parent_state={
            "conversation_id": "conv-1",
            "user_id": "user-1",
            "device_id": "device-1",
            "context": {},
            "messages": [],
        },
    )

    assert response.error is None
    assert response.message.content == "final answer"
    assert call_count == 3


@pytest.mark.asyncio
async def test_run_agent_in_isolated_context_allows_rag_worker_under_iteration_cap(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    call_count = 0

    async def fake_process_message(message, conversation_id):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return AgentResponse(
                agent_type=AgentType.RAG,
                agent_id="rag_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[
                        {
                            "id": f"rag-tool-{call_count}",
                            "name": "search_documents",
                            "args": {"action": "scan_all"},
                        }
                    ],
                ),
                metadata={},
            )
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="final answer"),
            metadata={},
        )

    rag_agent = SimpleNamespace(
        process_message=fake_process_message,
        agent_config_key="rag",
        agent_id="rag_agent",
    )
    workflow.rag_agent = rag_agent
    workflow.agents = {"rag_agent": rag_agent}

    async def fake_execute_search_documents_action(**kwargs):
        return "SEARCH RESULT", "scan_all", {"documents": [{"document_id": "doc-1"}]}

    monkeypatch.setattr(
        "app.ai.graph.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    monkeypatch.setattr(
        "app.ai.graph.apply_tool_output_offload",
        lambda **kwargs: (kwargs["output_text"], None),
    )

    response = await workflow._run_agent_in_isolated_context(
        agent_name="rag_agent",
        task_prompt="perform a multi-step document search",
        parent_state={
            "conversation_id": "conv-rag",
            "user_id": "user-1",
            "device_id": "device-1",
            "context": {},
            "messages": [],
        },
    )

    assert response.error is None
    assert response.message.content == "final answer"
    assert call_count == 3


@pytest.mark.asyncio
async def test_run_agent_in_isolated_context_rejects_planning_agent(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.agents = {"planning_agent": object(), "chat_agent": object()}

    with pytest.raises(ValueError):
        await workflow._run_agent_in_isolated_context(
            agent_name="planning_agent",
            task_prompt="recursive planning is forbidden",
            parent_state={},
        )


# ---------------------------------------------------------------------------
# Sequential tool execution regression (deferred tool search)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planning_tools_node_executes_dispatch_subagents(monkeypatch):
    """The dispatch tool must be reachable through the existing planning_tools
    node so the Planning Agent can call it like any internal tool.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    monkeypatch.setattr(settings, "planning_subagents_enabled", True)

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.planning_agent = SimpleNamespace(
        tools=[],
        agent_config_key="planning",
        agent_id="planning_agent",
    )

    async def fake_ensure_map(*args, **kwargs):
        return {}

    monkeypatch.setattr("app.ai.workflow.planning_loop.ensure_agent_tool_map", fake_ensure_map)

    captured: dict[str, Any] = {}

    async def fake_runner(**kwargs):
        captured.setdefault("calls", []).append(kwargs)
        return _ok(f"done: {kwargs['task_prompt'][:30]}")

    workflow._run_agent_in_isolated_context = fake_runner  # type: ignore[assignment]

    state: dict[str, Any] = {
        "messages": [
            HumanMessage(content="Run independent research"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-1",
                        "name": "dispatch_subagents",
                        "args": {
                            "tasks": [
                                {
                                    "id": "w1",
                                    "agent": "search_agent",
                                    "task": "search for migration constraints",
                                },
                                {
                                    "id": "w2",
                                    "agent": "chat_agent",
                                    "task": "outline likely files to change",
                                },
                            ],
                            "rationale": "two independent threads",
                        },
                    }
                ],
            ),
        ],
        "todos": [{"id": "t1", "description": "do work", "status": "pending", "order": 0}],
        "current_task_index": 0,
        "planning_call_count": 0,
        "planning_mode_enabled": True,
        "planning_phase": "executing",
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "context": {},
    }

    result_state = await workflow._planning_tools_node(state)

    # The dispatch tool must have produced a ToolMessage in state.
    from langchain_core.messages import ToolMessage as _ToolMessage

    tool_messages = [m for m in result_state["messages"] if isinstance(m, _ToolMessage)]
    assert len(tool_messages) == 1
    payload = tool_messages[0].content
    assert "results" in payload
    assert "w1" in payload and "w2" in payload

    # Both workers were dispatched.
    assert len(captured["calls"]) == 2
    agent_names = {call["agent_name"] for call in captured["calls"]}
    assert agent_names == {"search_agent", "chat_agent"}

    # Subagent dispatch summary attached to context for debug visibility.
    ctx = result_state["context"]
    assert ctx.get("subagent_dispatches"), "expected subagent_dispatches summary"
    assert ctx.get("subagent_results"), "expected subagent_results log"


@pytest.mark.asyncio
async def test_planning_tools_node_skips_dispatch_tool_build_when_no_dispatch_call(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.planning_agent = SimpleNamespace(
        tools=[],
        agent_config_key="planning",
        agent_id="planning_agent",
    )

    async def fake_ensure_map(*args, **kwargs):
        return {}

    monkeypatch.setattr("app.ai.workflow.planning_loop.ensure_agent_tool_map", fake_ensure_map)

    def unexpected_build(*args, **kwargs):
        raise AssertionError("dispatch tool should not be built without dispatch_subagents call")

    monkeypatch.setattr(workflow, "_build_planning_internal_tools", unexpected_build)

    state: dict[str, Any] = {
        "messages": [
            HumanMessage(content="mark task done"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "todo-1",
                        "name": "write_todos",
                        "args": {
                            "action": "complete_todo",
                            "todo_id": "t1",
                        },
                    }
                ],
            ),
        ],
        "todos": [{"id": "t1", "description": "do work", "status": "in_progress", "order": 0}],
        "current_task_index": 0,
        "planning_call_count": 0,
        "planning_mode_enabled": True,
        "planning_phase": "executing",
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "context": {},
    }

    result_state = await workflow._planning_tools_node(state)

    tool_messages = [m for m in result_state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "write_todos"


@pytest.mark.asyncio
async def test_run_agent_in_isolated_context_reuses_worker_tool_map(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    responses = [
        AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id="chat_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[{"id": "search-1", "name": "tool_search", "args": {"query": "x"}}],
            ),
            metadata={},
        ),
        AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id="chat_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[{"id": "loaded-1", "name": "loaded_tool", "args": {}}],
            ),
            metadata={},
        ),
        _ok("done"),
    ]
    invoke_count = 0

    async def fake_invoke(messages, conversation_history, persona, **kwargs):
        nonlocal invoke_count
        response = responses[invoke_count]
        invoke_count += 1
        return response

    agent = SimpleNamespace(
        invoke_model_with_history=fake_invoke,
        agent_config_key="chat",
        agent_id="chat_agent",
    )
    workflow.agents = {"chat_agent": agent}

    ensure_calls = 0
    shared_tool_map: dict[str, Any] = {"tool_search": object()}

    async def fake_ensure_map(*args, **kwargs):
        nonlocal ensure_calls
        ensure_calls += 1
        return shared_tool_map

    execute_seen_loaded = False

    async def fake_execute_tool_calls(*, tool_calls, tool_map, **kwargs):
        nonlocal execute_seen_loaded
        tool_name = tool_calls[0]["name"]
        if tool_name == "tool_search":
            tool_map["loaded_tool"] = object()
            return (
                [{"tool_call_id": "search-1", "name": "tool_search", "content": "loaded"}],
                [],
                [],
            )
        if tool_name == "loaded_tool":
            execute_seen_loaded = "loaded_tool" in tool_map
            return [{"tool_call_id": "loaded-1", "name": "loaded_tool", "content": "ok"}], [], []
        raise AssertionError(f"unexpected tool call: {tool_name}")

    monkeypatch.setattr("app.ai.graph.ensure_agent_tool_map", fake_ensure_map)
    monkeypatch.setattr("app.ai.graph.execute_tool_calls", fake_execute_tool_calls)

    response = await workflow._run_agent_in_isolated_context(
        agent_name="chat_agent",
        task_prompt="use tool search and then the loaded tool",
        parent_state={
            "conversation_id": "conv-1",
            "user_id": "user-1",
            "device_id": "device-1",
            "context": {},
            "messages": [],
        },
    )

    assert response.message.content == "done"
    assert ensure_calls == 1
    assert execute_seen_loaded is True


@pytest.mark.asyncio
async def test_execute_tool_calls_remains_sequential_for_dependent_tools(monkeypatch):
    """Two tools where the second relies on a mutation done by the first must
    still run sequentially under ``execute_tool_calls`` — the dispatcher's
    parallelism must NOT have leaked into generic tool execution.
    """
    from app.ai.tool_execution import execute_tool_calls

    state: dict[str, Any] = {"calls": []}

    class _MutatingTool:
        name = "first_tool"

        async def ainvoke(self, args):
            state["calls"].append("first_tool")
            state["dependent_loaded"] = True
            return "first_done"

    class _DependentTool:
        name = "dependent_tool"

        async def ainvoke(self, args):
            state["calls"].append("dependent_tool")
            assert state.get("dependent_loaded") is True
            return "dependent_done"

    tool_map = {"first_tool": _MutatingTool(), "dependent_tool": _DependentTool()}

    monkeypatch.setattr(
        "app.ai.tool_execution._mark_tool_used_if_deferred",
        lambda *a, **k: None,
    )

    outputs, _, _ = await execute_tool_calls(
        tool_calls=[
            {"id": "t1", "name": "first_tool", "args": {}},
            {"id": "t2", "name": "dependent_tool", "args": {}},
        ],
        tool_map=tool_map,
    )

    assert state["calls"] == ["first_tool", "dependent_tool"]
    assert [o["tool_call_id"] for o in outputs] == ["t1", "t2"]


# ---------------------------------------------------------------------------
# Planning hand_off integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planning_tools_node_applies_hand_off_to_target_agent(monkeypatch):
    """When the Planning Agent calls ``hand_off``, planning_tools must swap
    ``selected_agent`` so the conditional edge routes to the target.
    """
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.planning_agent = SimpleNamespace(
        tools=[],
        agent_config_key="planning",
        agent_id="planning_agent",
    )

    async def fake_ensure_map(*args, **kwargs):
        return {
            "hand_off": SimpleNamespace(
                name="hand_off",
                ainvoke=AsyncMock(return_value='{"hand_off": "chat_agent", "reason": "off-plan"}'),
            )
        }

    monkeypatch.setattr("app.ai.workflow.planning_loop.ensure_agent_tool_map", fake_ensure_map)

    state: dict[str, Any] = {
        "selected_agent": "planning_agent",
        "messages": [
            HumanMessage(content="actually, just answer this normally"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-1",
                        "name": "hand_off",
                        "args": {"target_agent": "chat_agent", "reason": "off-plan"},
                    }
                ],
            ),
        ],
        "todos": [],
        "current_task_index": 0,
        "planning_call_count": 0,
        "planning_mode_enabled": True,
        "planning_phase": "executing",
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "context": {},
    }

    workflow.agents = {"planning_agent": object(), "chat_agent": object()}

    result_state = await workflow._planning_tools_node(state)

    assert result_state["selected_agent"] == "chat_agent"


def test_should_continue_planning_routes_to_delegated_agent():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.agents = {"planning_agent": object(), "chat_agent": object()}

    state: dict[str, Any] = {
        "selected_agent": "chat_agent",
        "planning_call_count": 1,
        "planning_mode_enabled": True,
        "planning_phase": "executing",
        "todos": [],
        "messages": [
            HumanMessage(content="off-plan question"),
            AIMessage(
                content="",
                tool_calls=[{"id": "tc-1", "name": "hand_off", "args": {}}],
            ),
            ToolMessage(
                content='{"hand_off": "chat_agent"}',
                tool_call_id="tc-1",
                name="hand_off",
            ),
        ],
        "context": {},
    }

    assert workflow._should_continue_planning(state) == "chat_agent"


@pytest.mark.asyncio
async def test_planning_tools_node_handles_hand_off_alongside_dispatch(monkeypatch):
    """When the Planning Agent emits BOTH ``dispatch_subagents`` and
    ``hand_off`` in the same response, the dispatch must still run and produce
    a ToolMessage, AND the hand_off must reroute ``selected_agent`` so the
    conditional edge transfers control out of the planning loop.
    """
    monkeypatch.setattr(settings, "planning_subagents_enabled", True)

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.planning_agent = SimpleNamespace(
        tools=[], agent_config_key="planning", agent_id="planning_agent"
    )
    workflow.agents = {
        "planning_agent": object(),
        "chat_agent": object(),
        "search_agent": object(),
    }

    async def fake_ensure_map(*args, **kwargs):
        return {
            "hand_off": SimpleNamespace(
                name="hand_off",
                ainvoke=AsyncMock(return_value='{"hand_off": "chat_agent", "reason": "off-plan"}'),
            )
        }

    monkeypatch.setattr("app.ai.workflow.planning_loop.ensure_agent_tool_map", fake_ensure_map)

    dispatched: list[str] = []

    async def fake_runner(**kwargs):
        dispatched.append(kwargs["agent_name"])
        return _ok("worker done")

    workflow._run_agent_in_isolated_context = fake_runner  # type: ignore[assignment]

    state: dict[str, Any] = {
        "selected_agent": "planning_agent",
        "messages": [
            HumanMessage(content="research X and then hand me back to chat"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc-disp",
                        "name": "dispatch_subagents",
                        "args": {
                            "tasks": [
                                {
                                    "id": "w1",
                                    "agent": "search_agent",
                                    "task": "look up X",
                                },
                            ],
                            "rationale": "single-shot research before handoff",
                        },
                    },
                    {
                        "id": "tc-handoff",
                        "name": "hand_off",
                        "args": {
                            "target_agent": "chat_agent",
                            "reason": "off-plan",
                        },
                    },
                ],
            ),
        ],
        "todos": [],
        "current_task_index": 0,
        "planning_call_count": 0,
        "planning_mode_enabled": True,
        "planning_phase": "executing",
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "context": {},
    }

    result_state = await workflow._planning_tools_node(state)

    # Dispatch must have executed.
    assert dispatched == ["search_agent"]

    # Both tool calls produced ToolMessages so the model history stays valid.
    tool_messages = [m for m in result_state["messages"] if isinstance(m, ToolMessage)]
    tool_names = sorted(tm.name or "" for tm in tool_messages)
    assert tool_names == ["dispatch_subagents", "hand_off"]

    # hand_off rerouted selected_agent away from planning so the conditional
    # edge transfers control to chat_agent.
    assert result_state["selected_agent"] == "chat_agent"

    # _should_continue_planning honors the new selected_agent.
    assert workflow._should_continue_planning(result_state) == "chat_agent"

    # Dispatch summary metadata is preserved on context even when followed by
    # hand_off — the UI activity panel needs it to render the worker results.
    ctx = result_state["context"]
    assert ctx.get("subagent_dispatches"), "expected subagent_dispatches summary"
    assert ctx.get("subagent_results"), "expected subagent_results log"


@pytest.mark.asyncio
async def test_planning_consecutive_errors_only_warns_near_threshold(caplog, monkeypatch):
    """First error increment is INFO; only the final step that arms the circuit
    breaker should escalate to WARNING.
    """
    import logging

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    async def fake_ensure_map(*args, **kwargs):
        async def _raise(*_args, **_kwargs):
            raise RuntimeError("boom")

        return {"flaky_tool": SimpleNamespace(name="flaky_tool", ainvoke=_raise)}

    monkeypatch.setattr("app.ai.workflow.planning_loop.ensure_agent_tool_map", fake_ensure_map)
    monkeypatch.setattr(settings, "planning_consecutive_errors_limit", 3)

    workflow.planning_agent = SimpleNamespace(
        tools=[], agent_config_key="planning", agent_id="planning_agent"
    )
    workflow.agents = {"planning_agent": object()}

    def _make_state() -> dict[str, Any]:
        return {
            "selected_agent": "planning_agent",
            "messages": [
                HumanMessage(content="try again"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tc-1", "name": "flaky_tool", "args": {}}],
                ),
            ],
            "todos": [],
            "current_task_index": 0,
            "planning_call_count": 0,
            "planning_mode_enabled": True,
            "planning_phase": "executing",
            "context": {},
        }

    # First error: under threshold-1 → INFO.
    with caplog.at_level(logging.INFO, logger="app.ai.graph"):
        await workflow._planning_tools_node(_make_state())
    first_msgs = [r for r in caplog.records if "consecutive errors" in r.getMessage()]
    assert first_msgs and first_msgs[-1].levelno == logging.INFO
    caplog.clear()

    # Second error (now 1 step before limit=3): should escalate to WARNING.
    state = _make_state()
    state["context"]["consecutive_errors"] = 1
    with caplog.at_level(logging.INFO, logger="app.ai.graph"):
        await workflow._planning_tools_node(state)
    near_limit_msgs = [r for r in caplog.records if "consecutive errors" in r.getMessage()]
    assert near_limit_msgs and near_limit_msgs[-1].levelno == logging.WARNING

    # Third error reaches the breaker; the increment itself should not warn
    # because ``_should_continue_planning`` owns the "breaker fired" WARNING.
    caplog.clear()
    state = _make_state()
    state["context"]["consecutive_errors"] = 2
    with caplog.at_level(logging.INFO, logger="app.ai.graph"):
        await workflow._planning_tools_node(state)
    breaker_msgs = [r for r in caplog.records if "consecutive errors" in r.getMessage()]
    assert breaker_msgs and breaker_msgs[-1].levelno == logging.INFO


@pytest.mark.asyncio
async def test_planning_consecutive_errors_small_limit_does_not_warn_on_first(caplog, monkeypatch):
    """A small ``planning_consecutive_errors_limit`` must still log the first
    error as INFO. Earlier logic flagged the first error as WARNING when the
    limit was 2, which was noise — the circuit breaker hasn't tripped yet.
    """
    import logging

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    async def fake_ensure_map(*args, **kwargs):
        async def _raise(*_args, **_kwargs):
            raise RuntimeError("boom")

        return {"flaky_tool": SimpleNamespace(name="flaky_tool", ainvoke=_raise)}

    monkeypatch.setattr("app.ai.workflow.planning_loop.ensure_agent_tool_map", fake_ensure_map)
    monkeypatch.setattr(settings, "planning_consecutive_errors_limit", 2)

    workflow.planning_agent = SimpleNamespace(
        tools=[], agent_config_key="planning", agent_id="planning_agent"
    )
    workflow.agents = {"planning_agent": object()}

    state: dict[str, Any] = {
        "selected_agent": "planning_agent",
        "messages": [
            HumanMessage(content="try again"),
            AIMessage(
                content="",
                tool_calls=[{"id": "tc-1", "name": "flaky_tool", "args": {}}],
            ),
        ],
        "todos": [],
        "current_task_index": 0,
        "planning_call_count": 0,
        "planning_mode_enabled": True,
        "planning_phase": "executing",
        "context": {},
    }

    with caplog.at_level(logging.INFO, logger="app.ai.graph"):
        await workflow._planning_tools_node(state)
    msgs = [r for r in caplog.records if "consecutive errors" in r.getMessage()]
    assert msgs, "expected a consecutive-errors log line"
    assert msgs[-1].levelno == logging.INFO, "first error must not warn even on small limits"


# ---------------------------------------------------------------------------
# Phase 10 — Per-subagent model assignment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_agent_in_isolated_context_applies_generic_worker_model_override(
    monkeypatch,
):
    """A search_agent worker with an OpenAI override must receive a worker-local
    model_request override on the agent's invoke_model_with_history call.
    """
    from app.ai.planning_subagents import SubagentModelOverride

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    captured: dict[str, Any] = {}

    async def fake_invoke(messages, conversation_history, persona, **kwargs):
        captured.update(kwargs)
        return _ok("worker done")

    agent = SimpleNamespace(
        invoke_model_with_history=fake_invoke,
        agent_config_key="search",
        agent_id="search_agent",
    )
    workflow.search_agent = agent
    workflow.agents = {"search_agent": agent}

    override = SubagentModelOverride.model_validate(
        {"provider": "openai", "model": "gpt-5.4", "reasoning_effort": "high"}
    )

    parent_state: dict[str, Any] = {
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "model_request": {"chat": {"provider": "gemini", "model": "x"}},
        "context": {},
        "messages": [],
    }

    response = await workflow._run_agent_in_isolated_context(
        agent_name="search_agent",
        task_prompt="research API migration constraints",
        parent_state=parent_state,
        model_override=override,
    )

    worker_request = captured["model_request"]
    assert worker_request is not None
    # Parent siblings preserved
    assert worker_request["chat"] == {"provider": "gemini", "model": "x"}
    # Search key overlaid by the override
    assert worker_request["search"]["provider"] == "openai"
    assert worker_request["search"]["model"] == "gpt-5.4"
    assert worker_request["search"]["reasoning_effort"] == "high"
    assert worker_request["search"]["allow_custom_model"] is True
    # Parent state model_request itself must NOT be mutated.
    assert "search" not in parent_state["model_request"]
    assert response.message.content == "worker done"


@pytest.mark.asyncio
async def test_run_agent_in_isolated_context_applies_rag_worker_model_override(
    monkeypatch,
):
    """RAG subagents go through process_message; the worker model_request must
    reach AgentMessage.metadata["model_request"] for the RAG path too.
    """
    from app.ai.planning_subagents import SubagentModelOverride

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    captured: list[dict[str, Any]] = []

    async def fake_process_message(message, conversation_id):
        captured.append(dict(message.metadata))
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="final"),
            metadata={},
        )

    rag_agent = SimpleNamespace(
        process_message=fake_process_message,
        agent_config_key="rag",
        agent_id="rag_agent",
    )
    workflow.rag_agent = rag_agent
    workflow.agents = {"rag_agent": rag_agent}

    override = SubagentModelOverride.model_validate(
        {"provider": "gemini", "model": "gemini-3.1-pro-preview", "reasoning_effort": "high"}
    )

    parent_state: dict[str, Any] = {
        "conversation_id": "conv-rag",
        "user_id": "user-1",
        "device_id": "device-1",
        "model_request": None,
        "context": {},
        "messages": [],
    }

    await workflow._run_agent_in_isolated_context(
        agent_name="rag_agent",
        task_prompt="search documents for facts",
        parent_state=parent_state,
        model_override=override,
    )

    assert captured, "expected at least one process_message invocation"
    request = captured[0].get("model_request")
    assert isinstance(request, dict)
    assert request["rag"]["provider"] == "gemini"
    assert request["rag"]["model"] == "gemini-3.1-pro-preview"
    assert request["rag"]["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_dispatcher_isolates_sibling_worker_overrides(monkeypatch):
    """Two sibling workers with different overrides must each receive their
    own worker model_request without leaking into one another or the parent.
    """
    from app.ai.planning_subagents import (
        DispatchSubagentsInput,
        PlanningSubagentName,
        PlanningSubagentTask,
        SubagentModelOverride,
    )

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    seen_per_agent: dict[str, list[dict[str, Any]]] = {}

    async def fake_runner(
        *,
        agent_name,
        task_prompt,
        parent_state,
        related_todo_ids=None,
        model_override=None,
        task_id=None,
    ):
        from app.ai.planning_subagents import build_worker_model_request

        worker_request = build_worker_model_request(
            parent_model_request=parent_state.get("model_request"),
            agent_key=agent_name.replace("_agent", ""),
            override=model_override,
        )
        seen_per_agent.setdefault(agent_name, []).append(worker_request or {})
        return _ok(f"done {agent_name}")

    workflow._run_agent_in_isolated_context = fake_runner  # type: ignore[assignment]

    dispatcher = SimpleNamespace()
    dispatcher = __import__(
        "app.ai.planning_subagents", fromlist=["PlanningSubagentDispatcher"]
    ).PlanningSubagentDispatcher(workflow=workflow, settings=settings)

    parent_state: dict[str, Any] = {
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "model_request": {"chat": {"provider": "gemini", "model": "parent-x"}},
        "context": {},
    }

    request = DispatchSubagentsInput(
        tasks=[
            PlanningSubagentTask(
                id="w1",
                agent=PlanningSubagentName.SEARCH_AGENT,
                task="search the docs in depth",
                model_override=SubagentModelOverride.model_validate(
                    {"provider": "openai", "model": "gpt-5.4"}
                ),
            ),
            PlanningSubagentTask(
                id="w2",
                agent=PlanningSubagentName.CHAT_AGENT,
                task="outline the file map in depth",
                model_override=SubagentModelOverride.model_validate(
                    {"provider": "gemini", "model": "gemini-3-flash-preview"}
                ),
            ),
        ]
    )

    await dispatcher.dispatch(request, parent_state=parent_state)

    # Parent state untouched
    assert parent_state["model_request"] == {"chat": {"provider": "gemini", "model": "parent-x"}}

    search_request = seen_per_agent["search_agent"][0]
    chat_request = seen_per_agent["chat_agent"][0]

    # Each worker request carries its own override on the right key.
    assert search_request["search"]["model"] == "gpt-5.4"
    assert "search" not in chat_request
    assert chat_request["chat"]["model"] == "gemini-3-flash-preview"
    # And neither worker observed the other worker's override.
    assert (
        "chat" not in search_request or search_request["chat"]["model"] != "gemini-3-flash-preview"
    )


@pytest.mark.asyncio
async def test_dispatch_result_includes_requested_and_resolved_model(monkeypatch):
    from app.ai.planning_subagents import (
        DispatchSubagentsInput,
        PlanningSubagentDispatcher,
        PlanningSubagentName,
        PlanningSubagentTask,
        SubagentModelOverride,
    )

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    async def fake_runner(
        *,
        agent_name,
        task_prompt,
        parent_state,
        related_todo_ids=None,
        model_override=None,
        task_id=None,
    ):
        return AgentResponse(
            agent_type=AgentType.SEARCH,
            agent_id="search_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="ok"),
            metadata={
                "provider": "openai",
                "model": "gpt-5.4",
                "config_source": "request",
                "reasoning_effort": "high",
                "context_window": {
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "context_window_tokens": 400000,
                    "max_input_tokens": 400000,
                    "max_output_tokens": 128000,
                    "source": "registry",
                    "known": True,
                    "used_tokens": 4096,
                    "used_token_source": "actual_total",
                    "usage_ratio": 4096 / 400000,
                    "display_state": "ok",
                },
            },
        )

    workflow._run_agent_in_isolated_context = fake_runner  # type: ignore[assignment]

    dispatcher = PlanningSubagentDispatcher(workflow=workflow, settings=settings)
    request = DispatchSubagentsInput(
        tasks=[
            PlanningSubagentTask(
                id="w1",
                agent=PlanningSubagentName.SEARCH_AGENT,
                task="research migration risk in depth",
                model_override=SubagentModelOverride.model_validate(
                    {"provider": "openai", "model": "gpt-5.4", "reasoning_effort": "high"}
                ),
            )
        ]
    )

    result = await dispatcher.dispatch(request, parent_state={})
    entry = result.results[0]
    assert entry.requested_model is not None
    assert entry.requested_model["model"] == "gpt-5.4"
    assert entry.requested_model["reasoning_effort"] == "high"
    assert entry.resolved_model is not None
    assert entry.resolved_model["model"] == "gpt-5.4"
    assert entry.resolved_model["reasoning_effort"] == "high"
    assert entry.resolved_model["context_window"]["used_tokens"] == 4096
    assert entry.resolved_model["context_window"]["used_token_source"] == "actual_total"
    # Sanitized: no API keys ever surfaced.
    assert "api_key" not in entry.resolved_model


# ---------------------------------------------------------------------------
# Handoff control-plane scoping
# ---------------------------------------------------------------------------


def test_delegated_agent_messages_strip_handoff_control_messages():
    """When a hand_off is the routing reason for this turn, the delegated
    agent must receive only the user's request — not the source agent's
    transfer narration AIMessage or the matching hand_off ToolMessage.
    """
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    messages = [
        HumanMessage(content="what changed in the latest release?"),
        AIMessage(
            content="Transfering your request to Search Agent...",
            tool_calls=[
                {"id": "handoff-1", "name": "hand_off", "args": {"target_agent": "search_agent"}}
            ],
        ),
        ToolMessage(
            content='{"hand_off": "search_agent", "reason": "needs current info"}',
            tool_call_id="handoff-1",
            name="hand_off",
        ),
    ]
    state: dict[str, Any] = {
        "selected_agent": "search_agent",
        "messages": messages,
        "context": {
            "handoff": {
                "active": True,
                "source_agent": "planning_agent",
                "target_agent": "search_agent",
                "tool_call_id": "handoff-1",
            }
        },
    }

    delegated = workflow._messages_for_selected_agent(state, "search_agent", messages)

    assert delegated == [messages[0]]


def test_apply_hand_off_records_control_metadata():
    """Successful hand_off must stamp ``state['context']['handoff']`` so the
    streamer and delegated-agent scoping can react to it deterministically.
    """
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.agents = {"search_agent": object(), "planning_agent": object()}

    state: dict[str, Any] = {
        "selected_agent": "planning_agent",
        "messages": [],
        "context": {},
    }
    tool_outputs = [
        {
            "tool_call_id": "handoff-9",
            "name": "hand_off",
            "content": '{"hand_off": "search_agent", "reason": "needs current info"}',
        }
    ]

    new_state = workflow._apply_hand_off_if_present(state, tool_outputs)

    assert new_state["selected_agent"] == "search_agent"
    handoff = new_state["context"]["handoff"]
    assert handoff == {
        "active": True,
        "source_agent": "planning_agent",
        "target_agent": "search_agent",
        "reason": "needs current info",
        "tool_call_id": "handoff-9",
    }


def test_delegated_agent_messages_passthrough_when_no_active_handoff():
    """Without an active handoff entry, scoping must behave like
    ``_get_current_turn_messages`` — return the slice from the last
    ``HumanMessage`` onward, untouched."""
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    messages = [
        HumanMessage(content="old"),
        AIMessage(content="prior answer"),
        HumanMessage(content="new question"),
        ToolMessage(content="some result", tool_call_id="t1", name="search_documents"),
    ]
    state: dict[str, Any] = {
        "selected_agent": "search_agent",
        "messages": messages,
        "context": {},
    }

    delegated = workflow._messages_for_selected_agent(state, "search_agent", messages)

    assert delegated == messages[2:]


def test_should_continue_planning_routes_handoff_to_custom_agent():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    rid = f"custom_agent:{uuid4()}"
    workflow.agents = {
        "planning_agent": object(),
        "chat_agent": object(),
        "search_agent": object(),
    }
    state = {
        "selected_agent": rid,
        "custom_agents": {
            rid: {
                "id": rid.split(":", 1)[1],
                "runtime_agent_id": rid,
                "name": "Data Analyst",
            }
        },
        "planning_call_count": 1,
        "planning_phase": "executing",
        "context": {},
        "messages": [],
    }

    assert workflow._should_continue_planning(state) == "custom_agent"


def test_worker_loop_uses_tool_error_limit_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 4, raising=False)
    assert settings.tool_execution_consecutive_errors_limit == 4


@pytest.mark.asyncio
async def test_generic_worker_stops_after_repeated_tool_errors(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 2, raising=False)

    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)

    class _LoopingAgent:
        agent_config_key = "chat"
        tool_state_key = "chat"
        agent_type = AgentType.CHAT
        agent_id = "chat_agent"

        async def invoke_model_with_history(self, **kwargs):
            return AgentResponse(
                agent_type=AgentType.CHAT,
                agent_id="chat_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[{"id": "call-1", "name": "missing_tool", "args": {}}],
                ),
                metadata={},
            )

    graph.agents = {"chat_agent": _LoopingAgent()}

    async def _empty_map(*args, **kwargs):
        return {}

    monkeypatch.setattr("app.ai.graph.ensure_agent_tool_map", _empty_map)

    response = await graph._run_agent_in_isolated_context(
        parent_state={"conversation_id": "conv-1", "user_id": "user-1"},
        agent_name="chat_agent",
        task_prompt="try missing tool",
    )

    assert response.metadata["pause_reason"] == "consecutive_tool_errors"


@pytest.mark.asyncio
async def test_generic_worker_stops_after_iteration_limit(monkeypatch):
    monkeypatch.setattr(settings, "react_agent_max_iterations", 2)

    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    call_count = 0

    class _LoopingAgent:
        agent_config_key = "chat"
        tool_state_key = "chat"
        agent_type = AgentType.CHAT
        agent_id = "chat_agent"

        async def invoke_model_with_history(self, **kwargs):
            nonlocal call_count
            call_count += 1
            return AgentResponse(
                agent_type=AgentType.CHAT,
                agent_id="chat_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[{"id": f"call-{call_count}", "name": "lookup", "args": {}}],
                ),
                metadata={},
            )

    graph.agents = {"chat_agent": _LoopingAgent()}

    class _Lookup:
        name = "lookup"

        async def ainvoke(self, args):
            return "ok"

    async def _tool_map(*args, **kwargs):
        return {"lookup": _Lookup()}

    monkeypatch.setattr("app.ai.graph.ensure_agent_tool_map", _tool_map)

    response = await graph._run_agent_in_isolated_context(
        parent_state={"conversation_id": "conv-1", "user_id": "user-1"},
        agent_name="chat_agent",
        task_prompt="loop until capped",
    )

    assert response.metadata["pause_reason"] == "worker_max_iterations"
    assert call_count == 2


@pytest.mark.asyncio
async def test_rag_worker_stops_after_iteration_limit(monkeypatch):
    monkeypatch.setattr(settings, "agentic_max_iterations", 2)

    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    call_count = 0

    async def fake_process_message(message, conversation_id):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            return AgentResponse(
                agent_type=AgentType.RAG,
                agent_id="rag_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[
                        {
                            "id": f"rag-call-{call_count}",
                            "name": "search_documents",
                            "args": {"action": "scan_all"},
                        }
                    ],
                ),
                metadata={},
            )
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="final answer"),
            metadata={},
        )

    rag_agent = type(
        "RagWorker",
        (),
        {
            "agent_config_key": "rag",
            "tool_state_key": "rag",
            "agent_type": AgentType.RAG,
            "agent_id": "rag_agent",
            "process_message": staticmethod(fake_process_message),
        },
    )()
    graph.rag_agent = rag_agent
    graph.agents = {"rag_agent": rag_agent}

    async def fake_execute_search_documents_action(**kwargs):
        return "SEARCH RESULT", "scan_all", {}

    monkeypatch.setattr(
        "app.ai.graph.execute_search_documents_action",
        fake_execute_search_documents_action,
    )
    monkeypatch.setattr(
        "app.ai.graph.apply_tool_output_offload",
        lambda **kwargs: (kwargs["output_text"], None),
    )

    response = await graph._run_agent_in_isolated_context(
        parent_state={"conversation_id": "conv-1", "user_id": "user-1", "context": {}},
        agent_name="rag_agent",
        task_prompt="loop document search until capped",
    )

    assert response.metadata["pause_reason"] == "worker_max_iterations"
    assert call_count == 2
