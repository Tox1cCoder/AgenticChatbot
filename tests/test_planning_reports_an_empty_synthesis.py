"""Planning must say so when its own model turn produces nothing.

Reported as a turn that planned, dispatched workers, collected good results,
and then stopped with "Error: No response generated" -- and a console that was
otherwise clean.

``planning_model`` took the no-tool-call branch, built ``AIMessage(content="")``
and went to ``planning_package`` without a word. The failure surfaced two nodes
later as ``empty_public_content``, which names the symptom and nothing about
what the model actually returned. The specialist path already logs this; this
is the same line for Planning.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage


class _Response:
    def __init__(self, content, tool_calls=None):
        self.message = type("M", (), {"content": content, "tool_calls": tool_calls or []})()


def _factory(response):
    from app.ai.workflow.planning_execution import PlanningNodeFactory

    factory = object.__new__(PlanningNodeFactory)

    async def call_model(_state):
        return response

    factory._call_model = call_model
    return factory


@pytest.mark.asyncio
async def test_an_empty_synthesis_is_logged_with_what_came_back(caplog):
    factory = _factory(_Response(""))
    state = {"messages": [HumanMessage(content="plan something"), AIMessage(content="ok")]}

    with caplog.at_level(logging.WARNING, logger="app.ai.workflow.planning_execution"):
        command = await factory.planning_model(state)

    assert command.goto == "planning_package"
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "neither tool calls nor text" in logged
    assert "empty_public_content" in logged, "name the failure it is about to cause"


@pytest.mark.asyncio
async def test_a_real_answer_is_not_logged(caplog):
    factory = _factory(_Response("Here is the finished plan."))
    state = {"messages": [HumanMessage(content="plan something")]}

    with caplog.at_level(logging.WARNING, logger="app.ai.workflow.planning_execution"):
        command = await factory.planning_model(state)

    assert command.goto == "planning_package"
    assert not [r for r in caplog.records if "neither tool calls nor text" in r.getMessage()]


@pytest.mark.asyncio
async def test_whitespace_only_counts_as_empty(caplog):
    factory = _factory(_Response("   \n  "))
    state = {"messages": []}

    with caplog.at_level(logging.WARNING, logger="app.ai.workflow.planning_execution"):
        await factory.planning_model(state)

    assert "neither tool calls nor text" in " ".join(r.getMessage() for r in caplog.records)
