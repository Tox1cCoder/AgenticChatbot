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
    SubagentModelOverride,
    build_worker_model_request,
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


def test_dispatch_input_accepts_more_than_legacy_task_limit():
    payload = {
        "tasks": [
            _valid_task("worker-1"),
            _valid_task("worker-2"),
            _valid_task("worker-3"),
        ]
    }
    parsed = DispatchSubagentsInput.model_validate(payload)

    assert [task.id for task in parsed.tasks] == ["worker-1", "worker-2", "worker-3"]


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


def test_planning_subagent_task_coerces_string_context_to_dict():
    """The Planning Agent sometimes passes a raw text blob (crawled page,
    document excerpt) as ``context``. Accept it by wrapping in
    ``{"text": value}`` instead of failing the whole dispatch call.
    """
    task = PlanningSubagentTask.model_validate(
        {
            "id": "w1",
            "agent": "chat_agent",
            "task": "Summarize the article below",
            "context": "Here is the crawled data ...\n\nLine 2\nLine 3",
        }
    )
    assert isinstance(task.context, dict)
    assert task.context["text"] == "Here is the crawled data ...\n\nLine 2\nLine 3"


def test_planning_subagent_task_coerces_list_context_to_dict():
    """List context (e.g. a JSON array of references) should be wrapped under
    ``items`` so the worker still receives the structured data.
    """
    task = PlanningSubagentTask.model_validate(
        {
            "id": "w1",
            "agent": "search_agent",
            "task": "Cross-check the citations",
            "context": [{"url": "https://example.com"}, {"url": "https://example.org"}],
        }
    )
    assert isinstance(task.context, dict)
    assert task.context["items"] == [
        {"url": "https://example.com"},
        {"url": "https://example.org"},
    ]


def test_planning_subagent_task_treats_none_context_as_empty_dict():
    task = PlanningSubagentTask.model_validate(
        {
            "id": "w1",
            "agent": "chat_agent",
            "task": "Run a self-contained task with no extra context",
            "context": None,
        }
    )
    assert task.context == {}


def test_planning_subagent_task_rejects_obviously_bad_context_type():
    """Coercion is permissive but not infinite — a primitive that isn't text,
    list, or dict (e.g. a raw int) must still surface as a validation error so
    callers don't silently lose data.
    """
    with pytest.raises(ValidationError):
        PlanningSubagentTask.model_validate(
            {
                "id": "w1",
                "agent": "chat_agent",
                "task": "do something concrete enough",
                "context": 42,
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
        model_override: Any = None,
    ) -> AgentResponse:
        return await self._runner(
            agent_name=agent_name,
            task_prompt=task_prompt,
            parent_state=parent_state,
            related_todo_ids=related_todo_ids,
            model_override=model_override,
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
    monkeypatch.setattr(settings, "tool_result_max_chars", 6000)

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
    monkeypatch.setattr(settings, "tool_result_max_chars", 6000)

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
    monkeypatch.setattr(settings, "tool_result_max_chars", 6000)

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
async def test_dispatcher_does_not_apply_subagent_specific_timeout(monkeypatch):
    monkeypatch.setattr(settings, "tool_result_max_chars", 6000)

    async def runner(**kwargs):
        if kwargs["agent_name"] == "rag_agent":
            await asyncio.sleep(0.05)
        return _ok_response("ok")

    workflow = _StubWorkflow(runner)
    dispatcher = PlanningSubagentDispatcher(workflow=workflow, settings=settings)

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
    assert statuses["w-slow"] == "completed"


@pytest.mark.asyncio
async def test_dispatcher_truncates_summary_to_configured_size(monkeypatch):
    monkeypatch.setattr(settings, "tool_result_max_chars", 100)

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
async def test_dispatcher_keeps_worker_artifacts_for_ui_but_not_in_model_json(monkeypatch):
    """Worker artifacts must reach the UI without bloating the model context.

    The dispatcher captures every worker's tool_artifacts on the
    ``PlanningSubagentResult`` so the Streamlit UI can expand them per worker.
    The model-facing JSON (and the ``subagent_results`` context entries built
    from ``_compact_result_payload``) must still omit those nested artifacts to
    keep supervisor prompts compact.
    """
    monkeypatch.setattr(settings, "tool_result_max_chars", 6000)

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
    parent_state: dict[str, Any] = {}
    tool = create_dispatch_subagents_tool(
        dispatcher=dispatcher,
        parent_state_provider=lambda: parent_state,
    )

    payload = {
        "tasks": [
            _valid_task("w1", "chat_agent"),
        ]
    }

    model_json = await tool.ainvoke(payload)
    parsed = json.loads(model_json)

    # Model context: compact, no artifacts/images leaked.
    assert parsed["results"][0]["summary"] == "worker summary"
    assert "artifacts" not in parsed["results"][0]
    assert "images" not in parsed["results"][0]

    # subagent_results in context is built from the same compact payload.
    assert parent_state["context"]["subagent_results"][0].get("artifacts") in (None, [])

    # subagent_worker_artifacts in context carries the full artifact list so
    # the UI activity panel can render them under each worker.
    worker_artifacts = parent_state["context"]["subagent_worker_artifacts"]
    assert "w1" in worker_artifacts
    assert worker_artifacts["w1"][0]["tool_call_id"] == "worker-tool-1"


# ---------------------------------------------------------------------------
# Tool contract: dispatch_subagents returns valid JSON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_subagents_tool_returns_valid_json_string(monkeypatch):
    monkeypatch.setattr(settings, "tool_result_max_chars", 6000)

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
    monkeypatch.setattr(settings, "tool_result_max_chars", 6000)

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


def test_planning_prompt_advertises_hand_off_with_clear_disambiguation():
    """Planning agent now owns BOTH dispatch_subagents and hand_off.

    The executing-phase prompt must mention both and clearly distinguish
    parallel fan-out (``dispatch_subagents``) from full-conversation
    delegation (``hand_off``), so the supervisor picks the right primitive.
    """
    agent = _make_planning_agent_for_prompt()
    prompt = agent._build_system_prompt(
        persona=None,
        has_tool_context=False,
        todos=[],
        current_task_index=0,
        planning_phase="executing",
    )
    assert "dispatch_subagents" in prompt
    assert "hand_off" in prompt


def test_planning_agent_binding_includes_hand_off(monkeypatch):
    """The Planning Agent is now wired into graph-level delegation, so the
    ``hand_off`` tool must appear in its bound toolset alongside the always-on
    ``write_todos`` internal tool.
    """
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
    assert "hand_off" in tool_names


def test_planning_agent_binding_can_opt_out_of_hand_off(monkeypatch):
    """Subagent workers pass ``include_hand_off=False`` so graph-level
    delegation does not leak into isolated worker tool maps.
    """
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

    tools = agent._get_tools_for_binding(
        conversation_id="conversation-1",
        include_hand_off=False,
    )
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


def test_planning_prompt_clarifies_context_shape():
    """Supervisor was dumping raw text into ``context`` and tripping a Pydantic
    dict-type error. The prompt must call out the expected JSON-object shape.
    """
    agent = _make_planning_agent_for_prompt()
    prompt = agent._build_system_prompt(
        persona=None,
        has_tool_context=False,
        todos=[],
        current_task_index=0,
        planning_phase="executing",
    )
    lowered = prompt.lower()
    assert "context" in lowered
    # Either explicit instruction to use a JSON object or naming the wrapper
    # keys the validator coerces into.
    assert "json object" in lowered or '{"text"' in prompt or '{"items"' in prompt


def test_planning_prompt_lists_official_subagent_model_ids():
    """The supervisor was hallucinating ``gpt-5.5-medium`` (model+effort smush)
    and ``gemini-3-flash`` (missing ``-preview``). The executing-phase prompt
    must list the exact official model ids and the separate reasoning_effort
    values so the model picks valid combinations.
    """
    agent = _make_planning_agent_for_prompt()
    prompt = agent._build_system_prompt(
        persona=None,
        has_tool_context=False,
        todos=[],
        current_task_index=0,
        planning_phase="executing",
    )

    # OpenAI catalog
    assert "gpt-5.5" in prompt
    assert "gpt-5.4" in prompt
    assert "gpt-5.4-mini" in prompt

    # Gemini catalog — full ``-preview`` suffix is required.
    assert "gemini-3.1-pro-preview" in prompt
    assert "gemini-3-flash-preview" in prompt

    # reasoning_effort values must be enumerated separately so the model
    # doesn't paste them onto the model id.
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh"):
        assert effort in prompt

    # The exact failure modes must be explicitly called out (as "wrong"
    # examples). Showing both the right and wrong shapes is what stops the
    # supervisor from re-hallucinating them.
    lowered = prompt.lower()
    assert "wrong" in lowered or "invalid" in lowered or "never" in lowered
    # Concrete guard text: reasoning_effort is its own field, NOT a model suffix.
    assert "reasoning_effort" in prompt
    assert "separate field" in lowered or "never append" in lowered or "do not append" in lowered


# ---------------------------------------------------------------------------
# Phase 10 — Per-subagent model assignment
# ---------------------------------------------------------------------------


def test_subagent_model_override_accepts_openai_with_reasoning_effort():
    override = SubagentModelOverride.model_validate(
        {"provider": "openai", "model": "gpt-5.4", "reasoning_effort": "high"}
    )
    assert override.provider == "openai"
    assert override.model == "gpt-5.4"
    assert override.reasoning_effort == "high"
    assert override.allow_custom_model is True


def test_subagent_model_override_accepts_gemini_models():
    override = SubagentModelOverride.model_validate(
        {"provider": "gemini", "model": "gemini-3.1-pro-preview", "reasoning_effort": "low"}
    )
    assert override.provider == "gemini"
    assert override.model == "gemini-3.1-pro-preview"
    assert override.reasoning_effort == "low"


def test_subagent_model_override_accepts_all_reasoning_effort_levels():
    for level in ("none", "minimal", "low", "medium", "high", "xhigh"):
        override = SubagentModelOverride.model_validate(
            {"model": "gpt-5.4", "reasoning_effort": level}
        )
        assert override.reasoning_effort == level


def test_subagent_model_override_rejects_blank_model():
    with pytest.raises(ValidationError):
        SubagentModelOverride.model_validate({"model": "  "})


def test_subagent_model_override_rejects_unknown_provider():
    with pytest.raises(ValidationError):
        SubagentModelOverride.model_validate({"provider": "anthropic", "model": "claude-3"})


def test_subagent_model_override_rejects_invalid_temperature():
    with pytest.raises(ValidationError):
        SubagentModelOverride.model_validate({"model": "gpt-5.4", "temperature": "hot"})


def test_subagent_model_override_rejects_invalid_reasoning_effort():
    with pytest.raises(ValidationError):
        SubagentModelOverride.model_validate({"model": "gpt-5.4", "reasoning_effort": "extreme"})


def test_subagent_task_accepts_model_override():
    task = PlanningSubagentTask.model_validate(
        {
            "id": "w1",
            "agent": "search_agent",
            "task": "Research migration risk in depth",
            "model_override": {
                "provider": "openai",
                "model": "gpt-5.4",
                "reasoning_effort": "high",
            },
        }
    )
    assert task.model_override is not None
    assert task.model_override.model == "gpt-5.4"
    assert task.model_override.reasoning_effort == "high"


def test_subagent_task_without_model_override_still_validates():
    task = PlanningSubagentTask.model_validate(
        {
            "id": "w1",
            "agent": "search_agent",
            "task": "Research migration risk in depth",
        }
    )
    assert task.model_override is None


def test_dispatch_input_round_trips_model_override():
    payload = {
        "tasks": [
            {
                "id": "w1",
                "agent": "search_agent",
                "task": "Research migration risk in depth",
                "model_override": {
                    "provider": "gemini",
                    "model": "gemini-3-flash-preview",
                    "reasoning_effort": "minimal",
                    "temperature": 0.5,
                },
            }
        ]
    }
    parsed = DispatchSubagentsInput.model_validate(payload)
    assert parsed.tasks[0].model_override is not None
    assert parsed.tasks[0].model_override.provider == "gemini"
    assert parsed.tasks[0].model_override.temperature == 0.5


# ---------------------------------------------------------------------------
# Phase 10 — build_worker_model_request
# ---------------------------------------------------------------------------


def test_build_worker_model_request_returns_none_when_no_override_and_no_parent():
    assert build_worker_model_request(
        parent_model_request=None,
        agent_key="search",
        override=None,
    ) is None


def test_build_worker_model_request_returns_copy_of_parent_when_no_override():
    parent = {"chat": {"model": "x"}, "all": {"temperature": 0.5}}
    result = build_worker_model_request(
        parent_model_request=parent,
        agent_key="search",
        override=None,
    )
    assert result == parent
    assert result is not parent
    # Mutating result must not mutate parent.
    result["chat"]["model"] = "MUTATED"
    assert parent["chat"]["model"] == "x"


def test_build_worker_model_request_overlays_only_target_agent_key():
    parent = {
        "chat": {"model": "gemini-3-flash-preview"},
        "all": {"temperature": 0.7},
    }
    override = SubagentModelOverride.model_validate(
        {"provider": "openai", "model": "gpt-5.4", "reasoning_effort": "high"}
    )
    result = build_worker_model_request(
        parent_model_request=parent,
        agent_key="search",
        override=override,
    )
    assert result is not None
    # parent siblings preserved
    assert result["chat"] == {"model": "gemini-3-flash-preview"}
    assert result["all"] == {"temperature": 0.7}
    # target overlaid
    assert result["search"]["provider"] == "openai"
    assert result["search"]["model"] == "gpt-5.4"
    assert result["search"]["reasoning_effort"] == "high"
    # parent itself untouched
    assert "search" not in parent


def test_build_worker_model_request_does_not_mutate_parent_input():
    parent = {"chat": {"model": "x"}}
    override = SubagentModelOverride.model_validate(
        {"provider": "openai", "model": "gpt-5.4"}
    )
    build_worker_model_request(
        parent_model_request=parent,
        agent_key="search",
        override=override,
    )
    assert parent == {"chat": {"model": "x"}}


def test_build_worker_model_request_infers_provider_for_openai_gpt_models():
    override = SubagentModelOverride.model_validate({"model": "gpt-5.4"})
    result = build_worker_model_request(
        parent_model_request=None,
        agent_key="search",
        override=override,
    )
    assert result is not None
    assert result["search"]["provider"] == "openai"


def test_build_worker_model_request_infers_provider_for_openai_o_models():
    override = SubagentModelOverride.model_validate({"model": "o3-mini"})
    result = build_worker_model_request(
        parent_model_request=None,
        agent_key="search",
        override=override,
    )
    assert result is not None
    assert result["search"]["provider"] == "openai"


def test_build_worker_model_request_infers_provider_for_gemini_models():
    override = SubagentModelOverride.model_validate({"model": "gemini-3.1-pro-preview"})
    result = build_worker_model_request(
        parent_model_request=None,
        agent_key="search",
        override=override,
    )
    assert result is not None
    assert result["search"]["provider"] == "gemini"


def test_build_worker_model_request_infers_provider_for_gemini_flash_preview():
    override = SubagentModelOverride.model_validate({"model": "gemini-3-flash-preview"})
    result = build_worker_model_request(
        parent_model_request=None,
        agent_key="search",
        override=override,
    )
    assert result is not None
    assert result["search"]["provider"] == "gemini"


def test_build_worker_model_request_preserves_explicit_provider():
    """If the override carries an explicit provider, do not re-infer."""
    override = SubagentModelOverride.model_validate(
        {"provider": "openai", "model": "custom-private-name"}
    )
    result = build_worker_model_request(
        parent_model_request=None,
        agent_key="search",
        override=override,
    )
    assert result is not None
    assert result["search"]["provider"] == "openai"


@pytest.mark.asyncio
async def test_dispatch_tool_stashes_per_task_model_summaries(monkeypatch):
    """The compact dispatch summary on parent context should include per-task
    model information so the response/UI can show which model each worker used.
    """
    monkeypatch.setattr(settings, "tool_result_max_chars", 6000)

    async def runner(**kwargs):
        response = _ok_response(f"done {kwargs['agent_name']}")
        response.metadata = {
            "provider": "openai",
            "model": "gpt-5.4",
            "config_source": "request",
            "reasoning_effort": "high",
        }
        return response

    workflow = _StubWorkflow(runner)
    dispatcher = PlanningSubagentDispatcher(workflow=workflow, settings=settings)
    parent_state: dict[str, Any] = {}

    tool = create_dispatch_subagents_tool(
        dispatcher=dispatcher,
        parent_state_provider=lambda: parent_state,
    )

    payload = {
        "tasks": [
            {
                "id": "w1",
                "agent": "search_agent",
                "task": "research migration risk in depth",
                "model_override": {
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                },
            }
        ],
        "rationale": "explicit assignment",
    }

    result_json = await tool.ainvoke(payload)
    parsed = json.loads(result_json)

    # Model-facing JSON for the supervisor must include requested/resolved.
    entry = parsed["results"][0]
    assert entry["requested_model"]["model"] == "gpt-5.4"
    assert entry["resolved_model"]["model"] == "gpt-5.4"

    # The compact dispatch-level summary stashed on the parent context for UI
    # rendering must include per-task requested model info too.
    dispatches = parent_state["context"]["subagent_dispatches"]
    assert len(dispatches) == 1
    assert "models" in dispatches[0]
    # Per-task model summary keyed by task id.
    assert dispatches[0]["models"]["w1"]["requested"]["model"] == "gpt-5.4"
    assert dispatches[0]["models"]["w1"]["resolved"]["model"] == "gpt-5.4"


@pytest.mark.asyncio
async def test_dispatch_tool_dispatches_summary_omits_models_when_no_override(monkeypatch):
    """When no override is set and the worker has no resolved metadata, the
    dispatch summary should not include a stray empty ``models`` block.
    """
    monkeypatch.setattr(settings, "tool_result_max_chars", 6000)

    async def runner(**kwargs):
        # No model metadata; default-path worker.
        return _ok_response("done")

    workflow = _StubWorkflow(runner)
    dispatcher = PlanningSubagentDispatcher(workflow=workflow, settings=settings)
    parent_state: dict[str, Any] = {}

    tool = create_dispatch_subagents_tool(
        dispatcher=dispatcher,
        parent_state_provider=lambda: parent_state,
    )

    await tool.ainvoke(
        {
            "tasks": [{"id": "w1", "agent": "chat_agent", "task": "outline files to touch"}],
            "rationale": "default routing",
        }
    )

    dispatches = parent_state["context"]["subagent_dispatches"]
    assert "models" not in dispatches[0]
