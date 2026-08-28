"""Control flow is not a tool failure.

``execute_tool_calls`` deliberately catches ``BaseException`` so a broken tool
becomes model-visible feedback instead of ending the turn. LangGraph carries an
interrupt and a parent command as exceptions too, and turning one of those into
a tool error loses it: a turn waiting for a human reports that the tool failed,
and the agent carries on without the human.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphBubbleUp, GraphInterrupt

from app.ai.tool_execution import execute_tool_calls


def _raising_tool(name: str, error: BaseException) -> StructuredTool:
    async def _run() -> str:
        raise error

    return StructuredTool.from_function(
        coroutine=_run, name=name, description="Raises for the test."
    )


def _call(name: str) -> dict:
    return {"id": "call-1", "name": name, "args": {}}


@pytest.mark.parametrize(
    "error",
    [GraphBubbleUp("pausing"), GraphInterrupt(())],
    ids=["bubble_up", "interrupt"],
)
async def test_control_flow_exceptions_are_not_turned_into_tool_errors(error):
    tool = _raising_tool("pause_me", error)

    with pytest.raises(type(error)):
        await execute_tool_calls(tool_calls=[_call("pause_me")], tool_map={"pause_me": tool})


async def test_an_ordinary_tool_failure_is_still_model_visible_feedback():
    """The catch-all stays: a broken tool must not end the turn."""
    tool = _raising_tool("explode", RuntimeError("kaboom"))

    outputs, artifacts, _images = await execute_tool_calls(
        tool_calls=[_call("explode")], tool_map={"explode": tool}
    )

    assert artifacts[0]["status"] == "error"
    assert outputs[0]["tool_call_id"] == "call-1"
