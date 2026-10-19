"""Graph-level helpers that outlived the Planning subagent dispatcher.

Most of this file tested ``_run_agent_in_isolated_context`` and the
``PlanningSubagentDispatcher``; both were deleted in the cutover, and what they
guaranteed is now asserted where the behaviour lives:

* worker isolation and scoped identifiers -> ``build_worker_request`` in
  ``test_planning_execution_graph.py``;
* ``planning_agent`` refused as a worker target -> ``validate_dispatch_call``
  in ``test_planning_execution_graph.py`` and ``test_custom_agents_planning.py``.

What remains here is the handoff/tool plumbing that is still live: control
message stripping, sequential tool execution, planning
state metadata, and the finalized-response accessor.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import (
    AgentMessage,
    AgentResponse,
    AgentType,
    MessageRole,
)
from app.ai.workflow.contracts import AgentTransition


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


def _planning_tool_state(
    *,
    planning_call_count: int,
    tool_name: str = "write_todos",
    tool_content: str = "done",
) -> dict[str, Any]:
    return {
        "active_agent_id": "planning_agent",
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


def test_finalized_response_ignores_empty_tool_call_response():
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

    response = workflow._finalized_response(state)

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


# ---------------------------------------------------------------------------
# Sequential tool execution regression (deferred tool search)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 10 — Per-subagent model assignment
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Handoff control-plane scoping
# ---------------------------------------------------------------------------


def _accepted_handoff(source: str, target: str, tool_call_id: str) -> AgentTransition:
    return AgentTransition(
        from_agent_id=source, to_agent_id=target, source="handoff", tool_call_id=tool_call_id
    )


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
            content='{"hand_off": "search_agent"}',
            tool_call_id="handoff-1",
            name="hand_off",
        ),
    ]
    state: dict[str, Any] = {
        "active_agent_id": "search_agent",
        "messages": messages,
        "agent_history": [_accepted_handoff("planning_agent", "search_agent", "handoff-1")],
    }

    delegated = workflow._messages_for_active_agent(state, "search_agent", messages)

    assert delegated == [messages[0]]


def test_a_handoff_tool_result_never_reaches_the_agent_without_its_tool_call():
    """``hand_off`` returns a parent command, so the call that produced the
    result lives in the source agent's private messages and never reaches the
    parent. The delegated agent would otherwise be sent a tool result with no
    call, which OpenAI rejects: "No tool call found for function call output".
    """
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    messages = [
        HumanMessage(content="open the site and summarise it"),
        ToolMessage(
            id="handoff:call_4NHaeEgazno6MBxoJJsXA8Lb",
            content="handoff_requested",
            tool_call_id="call_4NHaeEgazno6MBxoJJsXA8Lb",
            name="hand_off",
        ),
    ]
    state: dict[str, Any] = {
        "active_agent_id": "chat_agent",
        "messages": messages,
        "context": {},
        "agent_history": [
            _accepted_handoff("search_agent", "chat_agent", "call_4NHaeEgazno6MBxoJJsXA8Lb")
        ],
    }

    delegated = workflow._messages_for_active_agent(state, "chat_agent", messages)

    assert delegated == [messages[0]]


def test_a_handoff_to_another_agent_leaves_this_agents_messages_alone():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    messages = [
        HumanMessage(content="question"),
        ToolMessage(content="refused", tool_call_id="c1", name="hand_off", status="error"),
    ]
    state: dict[str, Any] = {
        "active_agent_id": "search_agent",
        "messages": messages,
        "context": {},
        "agent_history": [_accepted_handoff("search_agent", "chat_agent", "c0")],
    }

    assert workflow._messages_for_active_agent(state, "search_agent", messages) == messages


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
        "active_agent_id": "search_agent",
        "messages": messages,
        "context": {},
    }

    delegated = workflow._messages_for_active_agent(state, "search_agent", messages)

    assert delegated == messages[2:]
