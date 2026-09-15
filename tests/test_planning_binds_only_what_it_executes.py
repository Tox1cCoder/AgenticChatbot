"""Planning may only be offered tools its own node loop can execute.

`72664fa3` (2026-09-03) moved Planning fan-out into parent topology and, in its
own words, "planning_tools is gone from the parent graph" -- the node that
executed Planning's ordinary tool calls. What replaced it routes exactly three
tool names and answers everything else with a synthetic
``unsupported_planning_tool`` control message.

The *binding* was never narrowed to match. ``PlanningAgent`` kept delegating to
``BaseAgent``, so the model was still offered ``tool_search``, MCP tools, web
tools, client tools and skills -- and the shared tool-discovery guidance still
instructed it to call ``tool_search``. It did, got a synthetic refusal, looped,
and the turn ended with no answer at all: ``empty_public_content`` with
``agent_id: planning_agent``.

Planning plans and delegates. Real tool work reaches a full pipeline through
``planning_dispatch`` -> ``planning_worker`` -> a specialist.
"""

from __future__ import annotations

import ast
import pathlib

PLANNING_EXECUTION = pathlib.Path("app/ai/workflow/planning_execution.py")
PLANNING_AGENT = pathlib.Path("app/ai/agents/planning_agent.py")


def _routed_tool_names() -> set[str]:
    """The tool names ``planning_model`` actually has a branch for."""
    from app.ai.workflow.planning_execution import (
        DISPATCH_CONTROL_TOOL_NAME,
        HANDOFF_TOOL_NAME,
        WRITE_TODOS_TOOL_NAME,
    )

    return {DISPATCH_CONTROL_TOOL_NAME, HANDOFF_TOOL_NAME, WRITE_TODOS_TOOL_NAME}


def test_the_agents_executable_set_matches_the_nodes_routed_set():
    """Two files, one fact. The agent names them to avoid a workflow import."""
    from app.ai.agents.planning_agent import EXECUTABLE_PLANNING_TOOLS

    assert set(EXECUTABLE_PLANNING_TOOLS) == _routed_tool_names()


def test_planning_overrides_tool_binding_instead_of_inheriting_everything():
    """The regression in one assertion: it used to call straight through."""
    source = PLANNING_AGENT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_get_tools_for_binding"
    )
    body = ast.get_source_segment(source, method) or ""

    assert "super()._get_tools_for_binding" not in body, (
        "Planning cannot execute the tools BaseAgent binds; inheriting the full "
        "set is what offered it tool_search it could never run"
    )


def test_planning_binds_only_its_control_tools():
    from langchain_core.tools import tool

    from app.ai.agents.planning_agent import PlanningAgent

    @tool
    def hand_off(target: str) -> str:
        """Transfer the conversation."""
        return target

    @tool
    def dispatch_subagents(tasks: str) -> str:
        """Propose worker tasks."""
        return tasks

    @tool
    def tool_search(query: str) -> str:
        """Discover tools -- Planning cannot run this."""
        return query

    bound = PlanningAgent._get_tools_for_binding(
        object.__new__(PlanningAgent),
        internal_tools=[hand_off, dispatch_subagents, tool_search],
    )
    names = {getattr(item, "name", None) for item in bound}

    assert "write_todos" in names, "write_todos is mandatory for Planning"
    unroutable = names - _routed_tool_names()
    assert not unroutable, f"bound tools Planning cannot execute: {unroutable}"
    assert "tool_search" not in names


def test_every_bound_tool_has_a_branch_in_the_model_node():
    """The invariant that broke: offered set must be a subset of routed set."""
    from langchain_core.tools import tool

    from app.ai.agents.planning_agent import PlanningAgent

    @tool
    def hand_off(target: str) -> str:
        """Transfer the conversation."""
        return target

    bound = PlanningAgent._get_tools_for_binding(
        object.__new__(PlanningAgent), internal_tools=[hand_off]
    )
    unroutable = {getattr(item, "name", None) for item in bound} - _routed_tool_names()

    assert not unroutable, f"these would hit unsupported_planning_tool: {unroutable}"


def test_planning_is_not_told_to_discover_tools_it_cannot_call():
    from app.ai.prompts import TOOL_EXPLORATION_SUFFIX

    source = PLANNING_AGENT.read_text(encoding="utf-8")

    assert "_strip_tool_exploration" in source or "TOOL_EXPLORATION_SUFFIX" in source, (
        "Planning must actively remove the shared tool-discovery guidance"
    )
    assert "tool_search" in TOOL_EXPLORATION_SUFFIX, "guarding the right block"


def test_the_unsupported_tool_backstop_survives():
    """A model that calls something anyway must still get a paired result."""
    assert "unsupported_planning_tool" in PLANNING_EXECUTION.read_text(encoding="utf-8")
