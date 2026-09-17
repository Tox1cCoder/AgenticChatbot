"""Every Planning tool call must come back with exactly one result.

Reported as a turn that spawned subagents, ran for two and a half minutes, then
answered "Error: No response generated". The checkpoint for that turn holds the
whole story in three messages:

    HumanMessage  "ok"
    AIMessage     tool_calls=[write_todos (call_50046),
                              dispatch_subagents (call_50061)]
    ToolMessage   tool_call_id=call_50061

``call_50046`` was never answered. Gemini is handed a function call with no
function response and returns a candidate with no parts at all -- the usage row
for that call reads ``output_tokens=0``, ``finish_reason=STOP``, and the empty
content reaches ``planning_model`` as the no-text-no-tool-calls branch, two
nodes before ``empty_public_content`` names the symptom.

``_route_dispatch`` appends the model's message with *all* of its tool calls but
pairs only the dispatch one. Its own rejection path, twenty lines above, already
pairs every call; only the success path forgot. ``planning_actions`` has the
same hole from the other side -- it pairs only the calls named ``write_todos``.

This worked before ``72664fa3`` moved fan-out into parent topology: the
``planning_tools`` node it removed executed every tool call the model made, so
each one got a result whatever combination the model chose.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.ai.workflow.inventory import AgentDescriptor, RoutingInventory
from app.ai.workflow.planning_execution import (
    NOT_RUN_THIS_TURN,
    PlanningLimits,
    PlanningNodeFactory,
    TodoActionOutcome,
)

DISPATCH = "dispatch_subagents"
WRITE_TODOS = "write_todos"


def _limits() -> PlanningLimits:
    return PlanningLimits(
        max_tasks=8,
        max_concurrency=4,
        max_dispatch_waves=2,
        objective_max_chars=4000,
        parent_context_max_chars=12000,
    )


def _inventory() -> RoutingInventory:
    return RoutingInventory.from_descriptors(
        AgentDescriptor(
            agent_id=agent_id,
            display_name=agent_id,
            capability_description="",
            enabled=True,
            attached=True,
            kind="base",
        )
        for agent_id in ("chat_agent", "search_agent", "planning_agent")
    )


async def _apply_todo_actions(state, calls) -> TodoActionOutcome:
    """Stands in for the real applier: reports what it actually did."""
    return TodoActionOutcome(
        todos=[{"id": "t1", "status": "in_progress"}],
        current_task_index=0,
        tool_messages=tuple(
            ToolMessage(
                content=f"Applied {call['args']['action']}",
                tool_call_id=str(call.get("id") or ""),
                name=WRITE_TODOS,
            )
            for call in calls
        ),
        actions=tuple(call["args"]["action"] for call in calls),
    )


def _factory(response: Any) -> PlanningNodeFactory:
    async def call_model(state):
        return response

    return PlanningNodeFactory(
        call_model=call_model,
        worker_runtime=SimpleNamespace(),
        limits=_limits(),
        inventory_for=lambda state: _inventory(),
        resolve_allowed_tools=lambda agent_id, state: ("search::web",),
        apply_todo_actions=_apply_todo_actions,
        review_rubric=None,
    )


def _response(tool_calls) -> Any:
    return SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=[dict(c) for c in tool_calls]),
        metadata={},
    )


def _dispatch_call(call_id: str = "call_50061") -> dict[str, Any]:
    return {
        "name": DISPATCH,
        "id": call_id,
        "type": "tool_call",
        "args": {
            "tasks": [
                {"task_id": "t1", "objective": "read the file", "agent_id": "chat_agent"},
            ]
        },
    }


def _write_todos_call(call_id: str = "call_50046", action: str = "set_todos") -> dict[str, Any]:
    return {
        "name": WRITE_TODOS,
        "id": call_id,
        "type": "tool_call",
        "args": {"action": action, "todos": [{"content": "read the file"}]},
    }


def _results(messages) -> dict[str, str]:
    return {
        m.tool_call_id: str(m.content) for m in messages if isinstance(m, ToolMessage)
    }


def _state() -> dict[str, Any]:
    """The parent state a Planning turn runs against, after one user message."""
    return {
        "messages": [HumanMessage(content="ok")],
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "planning_dispatch_waves": 0,
        "planning_dispatched_task_count": 0,
        "todos": [],
    }


def _unanswered(messages) -> set[str]:
    """Tool-call ids in the appended AI messages with no paired result."""
    called: set[str] = set()
    answered: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            called.update(str(call.get("id") or "") for call in (message.tool_calls or []))
        elif isinstance(message, ToolMessage):
            answered.add(str(message.tool_call_id or ""))
    return called - answered


async def _through_collect(factory, state) -> list:
    """Every message the model sees on its next turn.

    The dispatch call is answered by ``planning_collect``, one node later, so a
    check that stopped at ``planning_model`` would report the dispatch itself as
    stranded and miss the call that actually was.
    """
    command = await factory.planning_model(state)
    messages = list(command.update["messages"])
    if command.goto != "planning_dispatch":
        return messages

    collected = factory.planning_collect(
        {**state, "messages": messages, "planning_dispatch": command.update["planning_dispatch"]}
    )
    return messages + list(collected.update["messages"])


@pytest.mark.asyncio
async def test_dispatch_alongside_write_todos_answers_both():
    """The reported failure, in the order the model actually emitted them."""
    factory = _factory(_response([_write_todos_call(), _dispatch_call()]))

    messages = await _through_collect(factory, _state())

    assert not _unanswered(messages), (
        "a tool call with no result strands the next provider request"
    )


@pytest.mark.asyncio
async def test_dispatch_first_also_answers_the_sibling():
    """Order must not decide whether a call is answered."""
    factory = _factory(_response([_dispatch_call(), _write_todos_call()]))

    messages = await _through_collect(factory, _state())

    assert not _unanswered(messages)


@pytest.mark.asyncio
async def test_no_call_is_answered_twice():
    """One result per call: a second would be as malformed as none."""
    factory = _factory(_response([_write_todos_call(), _dispatch_call()]))

    messages = await _through_collect(factory, _state())
    answered = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]

    assert sorted(answered) == ["call_50046", "call_50061"]


@pytest.mark.asyncio
async def test_write_todos_alongside_an_unroutable_call_answers_both():
    """The same hole from the ``planning_actions`` side."""
    stray = {"name": "hand_off", "id": "call_9", "type": "tool_call", "args": {}}
    factory = _factory(_response([_write_todos_call(), stray]))
    state = _state()

    command = await factory.planning_model(state)
    messages = list(command.update["messages"])
    if command.goto == "planning_actions":
        follow_up = await factory.planning_actions({**state, "messages": messages})
        messages.extend(follow_up.update["messages"])

    assert not _unanswered(messages)


@pytest.mark.asyncio
async def test_a_lone_dispatch_still_gets_exactly_one_result():
    """The fix must not start emitting spare results for the dispatch call."""
    factory = _factory(_response([_dispatch_call()]))
    state = _state()

    command = await factory.planning_model(state)
    results = [m for m in command.update["messages"] if isinstance(m, ToolMessage)]

    assert results == []  # answered later, by planning_collect
    assert command.goto == "planning_dispatch"


@pytest.mark.asyncio
async def test_todo_bookkeeping_runs_even_when_dispatch_takes_the_route():
    """Marking a todo done and fanning out work is one ordinary model turn.

    Refusing the bookkeeping because the dispatch won the route makes the user
    watch their own tool calls come back rejected, and costs a model turn to
    re-issue them. Reported from a real run: three ``start_todo`` /
    ``complete_todo`` calls answered ``not_run_this_turn``, two more answered
    ``dispatch_rejected``, and only the third turn actually applied any of them.
    """
    factory = _factory(
        _response([
            _write_todos_call("call_a", "complete_todo"),
            _write_todos_call("call_b", "start_todo"),
            _dispatch_call(),
        ])
    )

    command = await factory.planning_model(_state())
    results = _results(command.update["messages"])

    assert results["call_a"] == "Applied complete_todo"
    assert results["call_b"] == "Applied start_todo"
    assert command.update["todos"] == [{"id": "t1", "status": "in_progress"}]
    assert command.goto == "planning_dispatch"


@pytest.mark.asyncio
async def test_todo_bookkeeping_runs_even_when_the_dispatch_is_rejected():
    """A bad dispatch proposal says nothing about the todo calls beside it.

    They were answered ``dispatch_rejected`` -- a code about someone else's
    failure, on a call that would have succeeded.
    """
    bad_dispatch = {"name": DISPATCH, "id": "call_bad", "type": "tool_call", "args": {}}
    factory = _factory(_response([_write_todos_call("call_a", "complete_todo"), bad_dispatch]))

    command = await factory.planning_model(_state())
    results = _results(command.update["messages"])

    assert results["call_a"] == "Applied complete_todo"
    assert results["call_bad"] != "dispatch_rejected"  # the real validation code
    assert command.goto == "planning_model"


@pytest.mark.asyncio
async def test_no_route_reports_another_calls_failure():
    """``dispatch_rejected`` on a call that is not the dispatch is misleading."""
    stray = {"name": "hand_off", "id": "call_9", "type": "tool_call", "args": {}}
    bad_dispatch = {"name": DISPATCH, "id": "call_bad", "type": "tool_call", "args": {}}
    factory = _factory(_response([stray, bad_dispatch]))

    command = await factory.planning_model(_state())

    assert _results(command.update["messages"])["call_9"] == NOT_RUN_THIS_TURN
