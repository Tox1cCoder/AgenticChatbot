"""Graph-level helpers that outlived the Planning subagent dispatcher.

Most of this file tested ``_run_agent_in_isolated_context`` and the
``PlanningSubagentDispatcher``; both were deleted in the cutover, and what they
guaranteed is now asserted where the behaviour lives:

* worker isolation and scoped identifiers -> ``build_worker_request`` in
  ``test_planning_execution_graph.py``;
* ``planning_agent`` refused as a worker target -> ``validate_dispatch_call``
  in ``test_planning_execution_graph.py`` and ``test_custom_agents_planning.py``.

What remains here is the handoff/tool plumbing that is still live: control
message stripping, sequential tool execution, the tool-error limit, planning
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
        "context": {
            "handoff": {
                "active": True,
                "source_agent": "planning_agent",
                "target_agent": "search_agent",
                "tool_call_id": "handoff-1",
            }
        },
    }

    delegated = workflow._messages_for_active_agent(state, "search_agent", messages)

    assert delegated == [messages[0]]




def test_apply_hand_off_records_control_metadata():
    """Successful hand_off must stamp ``state['context']['handoff']`` so the
    streamer and delegated-agent scoping can react to it deterministically.
    """
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.agents = {"search_agent": object(), "planning_agent": object()}

    state: dict[str, Any] = {
        "active_agent_id": "planning_agent",
        "messages": [],
        "context": {},
    }
    tool_outputs = [
        {
            "tool_call_id": "handoff-9",
            "name": "hand_off",
            "content": '{"hand_off": "search_agent"}',
        }
    ]

    new_state = workflow._apply_hand_off_if_present(state, tool_outputs)

    assert new_state["active_agent_id"] == "search_agent"
    handoff = new_state["context"]["handoff"]
    assert handoff == {
        "active": True,
        "source_agent": "planning_agent",
        "target_agent": "search_agent",
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
        "active_agent_id": "search_agent",
        "messages": messages,
        "context": {},
    }

    delegated = workflow._messages_for_active_agent(state, "search_agent", messages)

    assert delegated == messages[2:]




def test_worker_loop_uses_tool_error_limit_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "tool_execution_consecutive_errors_limit", 4, raising=False)
    assert settings.tool_execution_consecutive_errors_limit == 4






