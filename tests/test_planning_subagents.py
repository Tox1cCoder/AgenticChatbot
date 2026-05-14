"""Tests for the Planning-mode subagent dispatcher and schemas.

These tests cover:
- Pydantic schema validation (allowed/disallowed agents, blank ids/tasks,
  duplicate ids, max-tasks bound).
- Dispatcher concurrency (workers actually overlap).
- Result ordering (results match input order even when timing varies).
- Failure / timeout / requires_approval propagation.
- Tool contract: ``dispatch_subagents`` returns a JSON string with the
  expected aggregate ``status`` and an ordered ``results`` list.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from pydantic import ValidationError

from app.ai.planning_subagents import (
    DispatchSubagentsInput,
    DispatchSubagentsResult,
    PlanningSubagentDispatcher,
    PlanningSubagentName,
    PlanningSubagentResult,
    PlanningSubagentTask,
    create_dispatch_subagents_tool,
)
from app.ai.schemas import (
    AgentMessage,
    AgentResponse,
    AgentType,
    MessageRole,
)
from app.core.config import settings

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _valid_task(task_id: str = "w1", agent: str = "search_agent") -> dict[str, Any]:
    return {
        "id": task_id,
        "agent": agent,
        "task": "Investigate the API migration constraints in detail",
    }


def test_planning_subagent_name_excludes_planning_agent():
    assert "planning_agent" not in [member.value for member in PlanningSubagentName]
    assert {member.value for member in PlanningSubagentName} == {
        "chat_agent",
        "rag_agent",
        "search_agent",
        "image_generator_agent",
        "canvas_agent",
    }


def test_dispatch_input_accepts_known_worker_agents():
    payload = {
        "tasks": [
            _valid_task("worker-1", "chat_agent"),
            _valid_task("worker-2", "rag_agent"),
            _valid_task("worker-3", "search_agent"),
            _valid_task("worker-4", "image_generator_agent"),
            _valid_task("worker-5", "canvas_agent"),
        ],
    }
    parsed = DispatchSubagentsInput.model_validate(payload)
    assert len(parsed.tasks) == 5


def test_dispatch_input_rejects_planning_agent_target():
    payload = {"tasks": [_valid_task("worker-1", "planning_agent")]}
    with pytest.raises(ValidationError):
        DispatchSubagentsInput.model_validate(payload)


def test_dispatch_input_rejects_empty_tasks_list():
    with pytest.raises(ValidationError):
        DispatchSubagentsInput.model_validate({"tasks": []})


def test_dispatch_input_rejects_too_many_tasks(monkeypatch):
    monkeypatch.setattr(settings, "planning_subagents_max_tasks", 2)
    payload = {
        "tasks": [
            _valid_task("worker-1"),
            _valid_task("worker-2"),
            _valid_task("worker-3"),
        ]
    }
    with pytest.raises(ValidationError):
        DispatchSubagentsInput.model_validate(payload)


def test_dispatch_input_rejects_duplicate_task_ids():
    payload = {
        "tasks": [
            _valid_task("dup"),
            _valid_task("dup"),
        ]
    }
    with pytest.raises(ValidationError):
        DispatchSubagentsInput.model_validate(payload)


def test_dispatch_input_rejects_blank_task_description():
    payload = {
        "tasks": [
            {"id": "w1", "agent": "chat_agent", "task": "   "},
        ]
    }
    with pytest.raises(ValidationError):
        DispatchSubagentsInput.model_validate(payload)


def test_dispatch_input_rejects_blank_task_id():
    payload = {
        "tasks": [
            {"id": "  ", "agent": "chat_agent", "task": "Investigate something concrete"},
        ]
    }
    with pytest.raises(ValidationError):
        DispatchSubagentsInput.model_validate(payload)


def test_planning_subagent_task_context_must_be_serializable():
    # set object containing set is not JSON-serializable
    with pytest.raises(ValidationError):
        PlanningSubagentTask.model_validate(
            {
                "id": "w1",
                "agent": "chat_agent",
                "task": "do something concrete enough",
                "context": {"weird": {1, 2, 3}},
            }
        )


# ---------------------------------------------------------------------------
# Aggregate status logic
# ---------------------------------------------------------------------------


def test_dispatch_result_status_completed_when_all_workers_completed():
    results = [
        PlanningSubagentResult(
            id="w1",
            agent=PlanningSubagentName.CHAT_AGENT,
            status="completed",
            elapsed_ms=10,
            summary="ok",
        ),
        PlanningSubagentResult(
            id="w2",
            agent=PlanningSubagentName.SEARCH_AGENT,
            status="completed",
            elapsed_ms=12,
            summary="ok",
        ),
    ]
    aggregated = DispatchSubagentsResult.from_results(results)
    assert aggregated.status == "completed"


def test_dispatch_result_status_partial_with_mixed_outcomes():
    results = [
        PlanningSubagentResult(
            id="w1",
            agent=PlanningSubagentName.CHAT_AGENT,
            status="completed",
            elapsed_ms=10,
            summary="ok",
        ),
        PlanningSubagentResult(
            id="w2",
            agent=PlanningSubagentName.SEARCH_AGENT,
            status="failed",
            elapsed_ms=12,
            summary="boom",
            error="boom",
        ),
    ]
    aggregated = DispatchSubagentsResult.from_results(results)
    assert aggregated.status == "partial"


def test_dispatch_result_status_failed_when_no_worker_completed():
    results = [
        PlanningSubagentResult(
            id="w1",
            agent=PlanningSubagentName.CHAT_AGENT,
            status="failed",
            elapsed_ms=10,
            summary="boom",
            error="boom",
        ),
        PlanningSubagentResult(
            id="w2",
            agent=PlanningSubagentName.SEARCH_AGENT,
            status="timeout",
            elapsed_ms=12,
            summary="timed out",
        ),
    ]
    aggregated = DispatchSubagentsResult.from_results(results)
    assert aggregated.status == "failed"


# ---------------------------------------------------------------------------
# Dispatcher concurrency / runtime semantics
# ---------------------------------------------------------------------------


class _StubWorkflow:
    """Minimal MultiAgentWorkflow surface used by the dispatcher."""

    def __init__(self, runner):
        self._runner = runner

    async def _run_agent_in_isolated_context(
        self,
        *,
        agent_name: str,
        task_prompt: str,
        parent_state: dict[str, Any],
        related_todo_ids: list[str] | None = None,
    ) -> AgentResponse:
        return await self._runner(
            agent_name=agent_name,
            task_prompt=task_prompt,
            parent_state=parent_state,
            related_todo_ids=related_todo_ids,
        )


def _ok_response(content: str) -> AgentResponse:
    return AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content=content),
        metadata={},
    )


@pytest.mark.asyncio
async def test_dispatcher_runs_workers_concurrently(monkeypatch):
    monkeypatch.setattr(settings, "planning_subagents_max_parallel", 5)
    monkeypatch.setattr(settings, "planning_subagents_worker_timeout_seconds", 5)
    monkeypatch.setattr(settings, "planning_subagents_max_iterations", 5)
    monkeypatch.setattr(settings, "planning_subagents_result_max_chars", 6000)

    sleep_seconds = 0.3

    async def runner(**kwargs):
        await asyncio.sleep(sleep_seconds)
        return _ok_response(f"done {kwargs['agent_name']}")

    workflow = _StubWorkflow(runner)
    dispatcher = PlanningSubagentDispatcher(workflow=workflow, settings=settings)

    request = DispatchSubagentsInput(
        tasks=[
            PlanningSubagentTask(
                id="w1", agent=PlanningSubagentName.CHAT_AGENT, task="task one is long enough"
            ),
            PlanningSubagentTask(
                id="w2", agent=PlanningSubagentName.SEARCH_AGENT, task="task two is long enough"
            ),
            PlanningSubagentTask(
                id="w3", agent=PlanningSubagentName.RAG_AGENT, task="task three is long enough"
            ),
        ]
    )

    start = time.perf_counter()
    result = await dispatcher.dispatch(request, parent_state={})
    elapsed = time.perf_counter() - start

    # Three workers, each sleeping 0.3s, should overlap. Wall clock must be
    # well under the sequential 0.9s. Allow generous slack for CI noise.
    assert elapsed < 0.7, f"workers did not run concurrently: {elapsed:.2f}s"
    assert result.status == "completed"
    assert [r.id for r in result.results] == ["w1", "w2", "w3"]


@pytest.mark.asyncio
async def test_dispatcher_preserves_input_order_despite_finish_order(monkeypatch):
    monkeypatch.setattr(settings, "planning_subagents_max_parallel", 5)
    monkeypatch.setattr(settings, "planning_subagents_worker_timeout_seconds", 5)
    monkeypatch.setattr(settings, "planning_subagents_result_max_chars", 6000)

    delays = {"w1": 0.3, "w2": 0.05, "w3": 0.15}

    async def runner(**kwargs):
        # Invert finish order vs. input order.
        await asyncio.sleep(delays.get(_strip_id(kwargs["task_prompt"]), 0.05))
        return _ok_response(f"done {kwargs['agent_name']}")

    workflow = _StubWorkflow(runner)
    dispatcher = PlanningSubagentDispatcher(workflow=workflow, settings=settings)

    request = DispatchSubagentsInput(
        tasks=[
            PlanningSubagentTask(
                id="w1",
                agent=PlanningSubagentName.CHAT_AGENT,
                task="[w1] fairly long task description",
            ),
            PlanningSubagentTask(
                id="w2",
                agent=PlanningSubagentName.SEARCH_AGENT,
                task="[w2] fairly long task description",
            ),
            PlanningSubagentTask(
                id="w3",
                agent=PlanningSubagentName.RAG_AGENT,
                task="[w3] fairly long task description",
            ),
        ]
    )

    result = await dispatcher.dispatch(request, parent_state={})
    assert [r.id for r in result.results] == ["w1", "w2", "w3"]


def _strip_id(prompt: str) -> str:
    # Helper for test runner: extracts "w<n>" id wrapped in [] if present.
    if "[" in prompt and "]" in prompt:
        return prompt.split("[", 1)[1].split("]", 1)[0]
    return ""


@pytest.mark.asyncio
async def test_dispatcher_failure_does_not_cancel_siblings(monkeypatch):
    monkeypatch.setattr(settings, "planning_subagents_max_parallel", 5)
    monkeypatch.setattr(settings, "planning_subagents_worker_timeout_seconds", 5)
    monkeypatch.setattr(settings, "planning_subagents_result_max_chars", 6000)

    async def runner(**kwargs):
        if kwargs["agent_name"] == "search_agent":
            raise RuntimeError("boom")
        return _ok_response("ok")

    workflow = _StubWorkflow(runner)
    dispatcher = PlanningSubagentDispatcher(workflow=workflow, settings=settings)

    request = DispatchSubagentsInput(
        tasks=[
            PlanningSubagentTask(
                id="w1",
                agent=PlanningSubagentName.CHAT_AGENT,
                task="this is a long task description",
            ),
            PlanningSubagentTask(
                id="w2",
                agent=PlanningSubagentName.SEARCH_AGENT,
                task="this is a long task description",
            ),
            PlanningSubagentTask(
                id="w3",
                agent=PlanningSubagentName.RAG_AGENT,
                task="this is a long task description",
            ),
        ]
    )

    result = await dispatcher.dispatch(request, parent_state={})

    statuses = {r.id: r.status for r in result.results}
    assert statuses == {"w1": "completed", "w2": "failed", "w3": "completed"}
    assert result.status == "partial"
    failed = next(r for r in result.results if r.id == "w2")
    assert failed.error and "boom" in failed.error


@pytest.mark.asyncio
async def test_dispatcher_timeout_reported_as_structured_result(monkeypatch):
    monkeypatch.setattr(settings, "planning_subagents_max_parallel", 5)
    monkeypatch.setattr(settings, "planning_subagents_worker_timeout_seconds", 1)
    monkeypatch.setattr(settings, "planning_subagents_result_max_chars", 6000)

    async def runner(**kwargs):
        if kwargs["agent_name"] == "rag_agent":
            await asyncio.sleep(5)
        return _ok_response("ok")

    workflow = _StubWorkflow(runner)
    dispatcher = PlanningSubagentDispatcher(
        workflow=workflow, settings=settings, default_timeout_seconds=0.2
    )

    request = DispatchSubagentsInput(
        tasks=[
            PlanningSubagentTask(
                id="w-fast",
                agent=PlanningSubagentName.CHAT_AGENT,
                task="this is a long task description",
            ),
            PlanningSubagentTask(
                id="w-slow",
                agent=PlanningSubagentName.RAG_AGENT,
                task="this is a long task description",
            ),
        ]
    )

    result = await dispatcher.dispatch(request, parent_state={})
    statuses = {r.id: r.status for r in result.results}
    assert statuses["w-fast"] == "completed"
    assert statuses["w-slow"] == "timeout"


@pytest.mark.asyncio
async def test_dispatcher_truncates_summary_to_configured_size(monkeypatch):
    monkeypatch.setattr(settings, "planning_subagents_max_parallel", 5)
    monkeypatch.setattr(settings, "planning_subagents_worker_timeout_seconds", 5)
    monkeypatch.setattr(settings, "planning_subagents_result_max_chars", 100)

    async def runner(**kwargs):
        return _ok_response("x" * 10_000)

    workflow = _StubWorkflow(runner)
    dispatcher = PlanningSubagentDispatcher(workflow=workflow, settings=settings)

    request = DispatchSubagentsInput(
        tasks=[
            PlanningSubagentTask(
                id="w1",
                agent=PlanningSubagentName.CHAT_AGENT,
                task="this is a long enough task description",
            )
        ]
    )

    result = await dispatcher.dispatch(request, parent_state={})
    assert len(result.results[0].summary) <= 100


@pytest.mark.asyncio
async def test_dispatcher_does_not_embed_worker_artifacts_in_results(monkeypatch):
    monkeypatch.setattr(settings, "planning_subagents_max_parallel", 5)
    monkeypatch.setattr(settings, "planning_subagents_worker_timeout_seconds", 5)
    monkeypatch.setattr(settings, "planning_subagents_result_max_chars", 6000)

    huge_artifact = {
        "tool_call_id": "worker-tool-1",
        "tool": "search_documents",
        "output": "x" * 1000,
        "render": {
            "type": "json",
            "structured_content": {"chunks": ["y" * 20_000]},
        },
    }

    async def runner(**kwargs):
        response = _ok_response("worker summary")
        response.tool_artifacts = [huge_artifact]
        return response

    workflow = _StubWorkflow(runner)
    dispatcher = PlanningSubagentDispatcher(workflow=workflow, settings=settings)
    request = DispatchSubagentsInput(
        tasks=[
            PlanningSubagentTask(
                id="w1",
                agent=PlanningSubagentName.CHAT_AGENT,
                task="this is a long enough task description",
            )
        ]
    )

    result = await dispatcher.dispatch(request, parent_state={})

    assert result.results[0].summary == "worker summary"
    assert result.results[0].artifacts == []
    assert result.results[0].images == []


# ---------------------------------------------------------------------------
# Tool contract: dispatch_subagents returns valid JSON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_subagents_tool_returns_valid_json_string(monkeypatch):
    monkeypatch.setattr(settings, "planning_subagents_max_parallel", 5)
    monkeypatch.setattr(settings, "planning_subagents_worker_timeout_seconds", 5)
    monkeypatch.setattr(settings, "planning_subagents_result_max_chars", 6000)

    async def runner(**kwargs):
        return _ok_response("done")

    workflow = _StubWorkflow(runner)
    dispatcher = PlanningSubagentDispatcher(workflow=workflow, settings=settings)
    parent_state: dict[str, Any] = {}

    tool = create_dispatch_subagents_tool(
        dispatcher=dispatcher,
        parent_state_provider=lambda: parent_state,
    )

    payload = {
        "tasks": [
            _valid_task("w1", "chat_agent"),
            _valid_task("w2", "search_agent"),
        ],
        "rationale": "independent research tasks",
    }

    result_json = await tool.ainvoke(payload)
    parsed = json.loads(result_json)
    assert parsed["status"] in ("completed", "partial", "failed")
    assert isinstance(parsed["results"], list)
    assert [r["id"] for r in parsed["results"]] == ["w1", "w2"]
    for entry in parsed["results"]:
        assert entry["status"] in ("completed", "failed", "timeout", "requires_approval")
        assert "elapsed_ms" in entry
        assert "summary" in entry
        assert "agent" in entry


@pytest.mark.asyncio
async def test_dispatch_subagents_tool_stashes_compact_results_without_artifacts(monkeypatch):
    monkeypatch.setattr(settings, "planning_subagents_max_parallel", 5)
    monkeypatch.setattr(settings, "planning_subagents_worker_timeout_seconds", 5)
    monkeypatch.setattr(settings, "planning_subagents_result_max_chars", 6000)

    async def runner(**kwargs):
        response = _ok_response("done")
        response.tool_artifacts = [
            {
                "tool_call_id": "worker-tool-1",
                "tool": "rag_search",
                "output": "x" * 1000,
                "render": {
                    "type": "json",
                    "structured_content": {"chunks": ["y" * 20_000]},
                },
            }
        ]
        return response

    workflow = _StubWorkflow(runner)
    dispatcher = PlanningSubagentDispatcher(workflow=workflow, settings=settings)
    parent_state: dict[str, Any] = {}
    tool = create_dispatch_subagents_tool(
        dispatcher=dispatcher,
        parent_state_provider=lambda: parent_state,
    )

    result_json = await tool.ainvoke(
        {
            "tasks": [_valid_task("w1", "chat_agent")],
            "rationale": "independent research task",
        }
    )

    parsed = json.loads(result_json)
    assert parsed["results"][0]["summary"] == "done"
    assert "artifacts" not in parsed["results"][0]
    assert "images" not in parsed["results"][0]

    context_results = parent_state["context"]["subagent_results"]
    assert context_results[0]["summary"] == "done"
    assert "artifacts" not in context_results[0]
    assert "images" not in context_results[0]


# ---------------------------------------------------------------------------
# PlanningAgent prompt invariants
# ---------------------------------------------------------------------------


def _make_planning_agent_for_prompt():
    from app.ai.agents.planning_agent import PlanningAgent

    agent = PlanningAgent.__new__(PlanningAgent)
    agent.agent_config_key = "planning"
    agent.model_name = "test-model"
    agent.runtime_model_resolver = None
    agent.gemini_client = None
    agent.langchain_model = None
    agent.mcp_manager = None
    agent.tools = []
    agent._tools_generation_seen = 0
    return agent


def test_planning_prompt_executing_phase_mentions_dispatch_subagents():
    agent = _make_planning_agent_for_prompt()
    prompt = agent._build_system_prompt(
        persona=None,
        has_tool_context=False,
        todos=[],
        current_task_index=0,
        planning_phase="executing",
    )
    assert "dispatch_subagents" in prompt
    assert "INDEPENDENT" in prompt or "independent" in prompt
    assert "write_todos" in prompt


def test_planning_prompt_does_not_advertise_hand_off():
    agent = _make_planning_agent_for_prompt()
    prompt = agent._build_system_prompt(
        persona=None,
        has_tool_context=False,
        todos=[],
        current_task_index=0,
        planning_phase="executing",
    )
    assert "hand_off" not in prompt
    assert "INTER-AGENT DELEGATION" not in prompt


def test_planning_agent_binding_excludes_hand_off(monkeypatch):
    from app.ai.agents.planning_agent import PlanningAgent

    monkeypatch.setattr(
        "app.ai.agents.base_agent.should_use_deferred_loading",
        lambda _agent_key: True,
    )
    monkeypatch.setattr(
        "app.ai.agents.base_agent.get_available_skill_summaries",
        lambda **kwargs: [],
    )

    agent = PlanningAgent.__new__(PlanningAgent)
    agent.agent_config_key = "planning"
    agent.mcp_manager = None
    agent.tools = []

    tools = agent._get_tools_for_binding(conversation_id="conversation-1")
    tool_names = [tool.name for tool in tools]

    assert "write_todos" in tool_names
    assert "hand_off" not in tool_names


def test_planning_prompt_planning_phase_exposes_dispatch_with_explicit_trigger():
    agent = _make_planning_agent_for_prompt()
    prompt = agent._build_system_prompt(
        persona=None,
        has_tool_context=False,
        todos=[],
        current_task_index=0,
        planning_phase="planning",
    )
    prompt_lower = prompt.lower()
    assert "dispatch_subagents" in prompt
    # Planning-phase prompt must instruct the model to actually CALL the tool
    # when the user explicitly asks (no more "do not use it" framing that
    # made the model narrate instead of dispatching).
    assert "call `dispatch_subagents`" in prompt_lower or "call dispatch_subagents" in prompt_lower
    assert "delegate" in prompt_lower or "subagent" in prompt_lower


def test_planning_prompt_states_workers_cannot_mutate_todos():
    agent = _make_planning_agent_for_prompt()
    prompt = agent._build_system_prompt(
        persona=None,
        has_tool_context=False,
        todos=[],
        current_task_index=0,
        planning_phase="executing",
    )
    lowered = prompt.lower()
    assert "cannot mutate todos" in lowered or "cannot update todos" in lowered


def test_planning_prompt_says_supervisor_must_reconcile_with_write_todos():
    agent = _make_planning_agent_for_prompt()
    prompt = agent._build_system_prompt(
        persona=None,
        has_tool_context=False,
        todos=[],
        current_task_index=0,
        planning_phase="executing",
    )
    # Either "supervisor" framing OR the equivalent "you are the only actor
    # allowed to call write_todos" reconciliation guidance.
    lowered = prompt.lower()
    assert "supervisor" in lowered or "only actor allowed to call" in lowered
    assert "write_todos" in prompt
