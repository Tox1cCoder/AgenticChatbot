"""Regression tests for two graph-layer contracts.

- A streamed turn that raises must not publish the exception text: the message
  service persists the event's ``error`` as the assistant message, and provider
  exceptions carry URLs, request ids and credential fragments.
- Each ``write_todos`` ToolMessage reports its own call's outcome, not whether
  any earlier call in the same turn failed.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage

import app.ai.todo_actions as todo_actions
from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import WorkflowExecutionRequest
from app.ai.workflow.contracts import WorkflowRoutingException
from app.ai.workflow.errors import workflow_error

_SECRET_TEXT = "401 from https://api.example.test/v1?key=sk-live-SECRET123"


def _streaming_workflow(graph: Any) -> MultiAgentWorkflow:
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.checkpointer = None
    workflow.agents = {"chat_agent": object()}
    workflow.history_provider = None
    workflow.document_repository = None
    workflow.routing_service = object()
    workflow.routing_context_builder = object()
    workflow.chat_image_service = None
    workflow._build_graph_config = lambda thread_id=None: None
    workflow._build_initial_state_from_request = lambda request: {
        "messages": [HumanMessage(content=request.message)],
        "context": {},
    }
    workflow.graph = graph
    return workflow


async def _collect(workflow: MultiAgentWorkflow) -> list[Any]:
    request = WorkflowExecutionRequest(message="hello", conversation_id=None)
    return [event async for event in workflow.execute_request_stream(request)]


class _RaisingGraph:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def astream(self, *_args: Any, **_kwargs: Any):
        raise self._exc
        yield  # pragma: no cover - makes this an async generator


@pytest.mark.asyncio
async def test_execution_failure_does_not_publish_exception_text():
    workflow = _streaming_workflow(_RaisingGraph(RuntimeError(_SECRET_TEXT)))

    events = await _collect(workflow)

    assert [event.type for event in events] == ["error"]
    published = events[0].data["error"]
    assert "SECRET123" not in published
    assert "api.example.test" not in published
    assert "RuntimeError" in published


@pytest.mark.asyncio
async def test_preparation_failure_does_not_publish_exception_text():
    workflow = _streaming_workflow(_RaisingGraph(AssertionError("unreachable")))

    async def _fail(_state: Any) -> Any:
        raise ConnectionError(_SECRET_TEXT)

    workflow._prepare_turn_runtime_context = _fail

    events = await _collect(workflow)

    assert [event.type for event in events] == ["error"]
    assert "SECRET123" not in events[0].data["error"]


@pytest.mark.asyncio
async def test_typed_workflow_error_keeps_its_code():
    error = workflow_error("routing_timeout", request_id="request-1")
    workflow = _streaming_workflow(_RaisingGraph(WorkflowRoutingException(error)))

    events = await _collect(workflow)

    assert events[0].data["error"] == "workflow_error:routing_timeout"


@pytest.mark.asyncio
async def test_a_failed_write_todos_call_does_not_mark_later_calls_failed(monkeypatch):
    real_apply = todo_actions.apply_write_todos_action
    calls_seen: list[str] = []

    def _apply(**kwargs: Any):
        calls_seen.append(kwargs["tool_args"].get("action"))
        if len(calls_seen) == 1:
            raise ValueError("malformed todo")
        return real_apply(**kwargs)

    monkeypatch.setattr(todo_actions, "apply_write_todos_action", _apply)
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    calls = [
        {"id": "call-1", "name": "write_todos", "args": {"action": "add_todo"}},
        {
            "id": "call-2",
            "name": "write_todos",
            "args": {
                "action": "set_todos",
                "todos": [{"id": "t1", "description": "Do it", "order": 0}],
            },
        },
    ]

    outcome = await workflow._planning_apply_todo_actions({"messages": []}, calls)

    statuses = [(message.tool_call_id, message.status) for message in outcome.tool_messages]
    assert statuses == [("call-1", "error"), ("call-2", "success")]
    assert outcome.had_error is True
