from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.ai.agents.base_agent import BaseAgent
from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.core.config import settings
from app.core.runtime_modeling import ResolvedRuntimeModelConfig


def _tool_call() -> dict:
    return {"id": "tool-call-1", "name": "lookup", "args": {"query": "status"}}


def test_graph_config_clamps_recursion_limit_for_final_synthesis(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.checkpointer = None
    monkeypatch.setattr(settings, "react_agent_max_iterations", 10)
    monkeypatch.setattr(settings, "planning_max_iterations", 10)
    monkeypatch.setattr(settings, "react_agent_recursion_limit", 15)

    config = workflow._build_graph_config()

    assert config["recursion_limit"] == 25


def test_graph_config_accounts_for_planning_iteration_budget(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.checkpointer = None
    monkeypatch.setattr(settings, "react_agent_max_iterations", 5)
    monkeypatch.setattr(settings, "planning_max_iterations", 20)
    monkeypatch.setattr(settings, "react_agent_recursion_limit", 0)

    config = workflow._build_graph_config()

    assert config["recursion_limit"] == 45


def _forced_final_state() -> dict:
    return {
        "active_agent_id": "chat_agent",
        "messages": [
            HumanMessage(content="What happened?"),
            AIMessage(content="", tool_calls=[_tool_call()]),
            ToolMessage(content="The lookup result", tool_call_id="tool-call-1", name="lookup"),
        ],
        "context": {
            "force_final_response": True,
            "tool_budget": {
                "reason": "soft_budget",
                "scope": "runtime",
                "count": 5,
                "limit": 5,
            },
        },
    }


def _budget_workflow() -> MultiAgentWorkflow:
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow._get_conversation_history = AsyncMock(return_value=[])
    workflow.agents = {"chat_agent": object()}
    workflow.chat_agent = SimpleNamespace(_convert_history_to_langchain_messages=lambda history: [])
    return workflow


@pytest.mark.asyncio
async def test_forced_final_response_unbinds_every_tool():
    """A budget-exhausted turn must be unable to call another tool.

    The guard lives in the tool factory, so it holds no matter which model the
    framework loop ends up calling.
    """
    workflow = _budget_workflow()
    request = await workflow._specialist_request_for("chat_agent", _forced_final_state())

    assert request.extras["disable_tools"] is True
    assert "tool-use budget" in request.extras["system_prompt_kwargs"]["tool_budget_notice"].lower()

    from app.ai.agents.chat_agent import build_chat_specialist_definition

    definition = build_chat_specialist_definition(
        SimpleNamespace(_init_tools=AsyncMock(), _get_tools_for_binding=lambda **kw: ["a_tool"])
    )
    assert await definition.tool_factory(request) == []


@pytest.mark.asyncio
async def test_forced_final_response_records_the_exhausted_budget():
    workflow = _budget_workflow()
    state = _forced_final_state()
    response = AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content="Final answer"),
        metadata={},
    )

    finalized = workflow._finalize_forced_final_response(state, response)

    assert finalized.metadata["tool_budget_exhausted"]["limit"] == 5


class _DummyAgent(BaseAgent):
    def _init_gemini(self) -> None:
        self.gemini_client = None
        self.langchain_model = None

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CHAT

    @property
    def agent_id(self) -> str:
        return "dummy_agent"

    def _get_base_system_prompt(self) -> str:
        return "Dummy system prompt."


class _FakeLLM:
    def __init__(self) -> None:
        self.messages = None
        self.config = None

    async def ainvoke(self, messages, config=None):
        self.messages = messages
        self.config = config
        return AIMessage(content="Final answer from plain model")


@pytest.mark.asyncio
async def test_base_agent_disable_tools_uses_plain_model_and_budget_notice(monkeypatch):
    agent = _DummyAgent(agent_config_key="chat")
    fake_llm = _FakeLLM()

    monkeypatch.setattr(agent, "_init_tools", AsyncMock())
    monkeypatch.setattr(
        agent,
        "_resolve_runtime_model_config",
        lambda _user_id, _model_request=None: ResolvedRuntimeModelConfig(
            agent_key="chat",
            provider="gemini",
            model="dummy-model",
            temperature=0.0,
            api_key=None,
            key_source="none",
            source="test",
            capabilities={"supports_tool_calling": True},
        ),
    )
    monkeypatch.setattr(
        agent,
        "_create_langchain_model_from_runtime",
        lambda _runtime_config, **_kwargs: (fake_llm, False),
    )
    monkeypatch.setattr(
        agent,
        "_get_llm_with_tools",
        Mock(side_effect=AssertionError("tools must not be bound for final synthesis")),
    )
    monkeypatch.setattr(
        agent,
        "_get_tools_for_binding",
        Mock(side_effect=AssertionError("tool schemas must not be counted when disabled")),
    )

    response = await agent.invoke_model_with_history(
        messages=[HumanMessage(content="Use the result")],
        conversation_history=[],
        persona=None,
        disable_tools=True,
        tool_budget_notice="Tool-use budget reached. Produce the final answer now.",
    )

    assert response.message.content == "Final answer from plain model"
    assert isinstance(fake_llm.messages[0], SystemMessage)
    assert "Tool-use budget reached" in fake_llm.messages[0].content


@pytest.mark.asyncio
async def test_base_agent_ainvoke_with_retries_forwards_run_config():
    agent = _DummyAgent(agent_config_key="chat")
    fake_llm = _FakeLLM()
    run_config = {
        "tags": ["internal", "planning_subagent"],
        "metadata": {"internal": True, "purpose": "planning_subagent"},
    }

    await agent._ainvoke_with_retries(
        fake_llm,
        [HumanMessage(content="worker prompt")],
        run_config=run_config,
    )

    assert fake_llm.config is run_config


@pytest.mark.asyncio
async def test_an_unresolvable_tool_answers_the_model_instead_of_raising(monkeypatch):
    """A call the execution map cannot resolve is model-visible feedback.

    The model can only recover from a bad call if it gets told; raising would
    end the turn on an exception the model never sees.
    """
    from app.ai.workflow.middleware import SpecialistToolScope, ToolExecutionMiddleware

    scope = SpecialistToolScope(
        agent=None,
        agent_key="chat",
        conversation_id="conversation-1",
        user_id="user-1",
        device_id=None,
    )

    async def _no_tools():
        return []

    middleware = ToolExecutionMiddleware(scope=scope, tool_factory=_no_tools)
    request = SimpleNamespace(
        tool_call={"id": "call-1", "name": "missing_tool", "args": {}},
        state={},
        runtime=SimpleNamespace(context=None),
    )

    async def _must_not_execute(_request):
        raise AssertionError("the framework must not execute an unresolved tool")

    message = await middleware.awrap_tool_call(request, _must_not_execute)

    assert message.tool_call_id == "call-1"
    assert message.status == "error"
    assert "missing_tool" in message.content


@pytest.mark.asyncio
async def test_canvas_edit_never_binds_the_widget_mutations(monkeypatch):
    """Editing a canvas returns the whole artifact, so a widget mutation in
    the same turn would be overwritten by the rewrite. The tools are withheld
    rather than refused after the model has already spent a call on one."""
    from app.ai.canvas_state import CANVAS_EDIT_DENIED_TOOL_NAMES

    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.agents = {"canvas_agent": object()}
    workflow.chat_agent = SimpleNamespace(_convert_history_to_langchain_messages=lambda h: [])

    async def _history(*_args, **_kwargs):
        return []

    async def _snapshot(*_args, **_kwargs):
        return SimpleNamespace(content="<html>previous</html>")

    workflow._get_conversation_history = _history
    workflow._get_active_canvas_snapshot = _snapshot

    state = {
        "active_agent_id": "canvas_agent",
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "messages": [HumanMessage(content="tweak the heading")],
        "context": {},
    }

    request = await workflow._specialist_request_for("canvas_agent", state)

    assert request.extras["excluded_tool_names"] == CANVAS_EDIT_DENIED_TOOL_NAMES


@pytest.mark.asyncio
async def test_a_fresh_canvas_still_binds_the_widget_tools():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.agents = {"canvas_agent": object()}
    workflow.chat_agent = SimpleNamespace(_convert_history_to_langchain_messages=lambda h: [])

    async def _history(*_args, **_kwargs):
        return []

    async def _no_snapshot(*_args, **_kwargs):
        return None

    workflow._get_conversation_history = _history
    workflow._get_active_canvas_snapshot = _no_snapshot

    state = {
        "active_agent_id": "canvas_agent",
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "messages": [HumanMessage(content="make me a dashboard")],
        "context": {},
    }

    request = await workflow._specialist_request_for("canvas_agent", state)

    assert request.extras["excluded_tool_names"] is None


def test_apply_tool_outputs_tracks_same_error_streak(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 2, raising=False)
    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state = {"messages": [], "context": {}}

    artifact = {
        "tool_call_id": "call-1",
        "tool": "read_file",
        "args": {"path": "missing.txt"},
        "status": "error",
        "error_type": "not_found",
        "output": "missing",
    }

    graph._apply_tool_outputs_to_state(
        state,
        tool_outputs=[{"tool_call_id": "call-1", "name": "read_file", "content": "missing"}],
        tool_artifacts=[artifact],
    )
    graph._apply_tool_outputs_to_state(
        state,
        tool_outputs=[{"tool_call_id": "call-2", "name": "read_file", "content": "missing"}],
        tool_artifacts=[{**artifact, "tool_call_id": "call-2"}],
    )

    streak = state["context"]["tool_error_streak"]
    assert streak["count"] == 2
    assert streak["limit"] == 2
    assert streak["signature"]["tool"] == "read_file"
    assert streak["signature"]["args"] == '{"path":"missing.txt"}'


def test_apply_tool_outputs_preserves_flagged_skill_terminal_errors(monkeypatch):
    monkeypatch.setattr(settings, "tool_result_max_chars", 120, raising=False)
    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state = {"messages": [], "context": {}}
    terminal_content = '{"status":"error","untrusted_terminal_output":"' + "x" * 4000 + '"}'

    graph._apply_tool_outputs_to_state(
        state,
        tool_outputs=[
            {
                "tool_call_id": "call-skill",
                "name": "client__demo__run_skill_command",
                "content": terminal_content,
                "preserve_full_content": True,
            },
            {
                "tool_call_id": "call-2",
                "name": "big_tool",
                "content": "y" * 4000,
            },
        ],
        truncate_outputs=True,
    )

    skill_message, other_message = state["messages"]
    assert skill_message.content == terminal_content
    assert len(other_message.content) <= 120


def test_apply_tool_outputs_lifts_but_does_not_persist_internal_rich_candidates():
    graph = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state = {"messages": [], "context": {}}
    candidate = {
        "id": "image:tool-content:call-1:0",
        "type": "image",
        "display_policy": "inline_only",
        "payload": {"data": "YWJj", "mime_type": "image/png"},
    }
    artifact = {
        "tool_call_id": "call-1",
        "tool": "image_tool",
        "args": {},
        "status": "success",
        "output": "[image/png image]",
        "_rich_item_candidates": [candidate],
    }

    graph._apply_tool_outputs_to_state(
        state,
        tool_outputs=[
            {
                "tool_call_id": "call-1",
                "name": "image_tool",
                "content": "[image/png image]",
            }
        ],
        tool_artifacts=[artifact],
    )

    assert state["context"]["rich_item_candidates"] == [candidate]
    assert "_rich_item_candidates" not in artifact
    assert "_rich_item_candidates" not in state["context"]["tool_artifacts"][0]
