# Planning Rubric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native Planning Rubric runtime that mirrors LangChain DeepAgents rubric behavior for Planning mode without migrating this repository to `deepagents`.

**Architecture:** Keep the existing LangGraph workflow, `PlanningAgent`, `write_todos` tool, `TaskPlanService`, and Planning subagent dispatcher. Add a focused Planning rubric runtime that first builds a context-specific rubric from the user request, current plan snapshot, lifecycle, and prior feedback, then grades candidate todo plans, injects actionable feedback into the existing planning loop, retries until the rubric is satisfied or capped, and exposes rubric status/evaluations through response metadata. Deterministic code is limited to structural safety and schema validation; plan-quality judgment stays LLM-driven.

**Tech Stack:** Python 3.10+, LangChain chat models, LangGraph, Pydantic v2, FastAPI service layer, SQLAlchemy-backed task plans, pytest, pytest-asyncio.

---

## Spec-Kit Feature Summary

### User Story

As a user of Planning mode, I want generated or modified task plans to be checked against clear planning-quality criteria before they are persisted or presented as ready, so that vague, incomplete, over-broad, or incorrectly mutated plans are revised automatically.

### Acceptance Scenarios

1. **New plan satisfies rubric on first pass**
   - Given Planning mode is enabled and no plan exists
   - When the planning runtime generates a candidate todo list
   - Then the native rubric grader evaluates the candidate
   - And the final candidate plan is persisted with its terminal rubric status
   - And assistant metadata includes `planning_rubric.status == "satisfied"` when all criteria pass.

2. **New plan needs revision**
   - Given the first candidate todo list contains vague tasks
   - When the grader returns `needs_revision`
   - Then grader feedback is injected into a forced `write_todos` revision pass
   - And the revised todo list is graded again
   - And the loop stops on `satisfied` or `max_iterations_reached`.

3. **Modified plan preserves existing task identity**
   - Given an existing plan has stable task IDs and completed tasks
   - When the user asks for a plan modification
   - Then the rubric checks that unchanged tasks preserve IDs and statuses
   - And failed preservation produces `needs_revision` feedback before persistence.

4. **In-chat Planning tool mutation is reviewed**
   - Given the Planning Agent calls `write_todos` inside the graph
   - When the action creates, adds, updates, or removes plan tasks
   - Then the graph stores rubric feedback in `GraphState.context`
   - And routes back to `planning_agent` for revision when the verdict is `needs_revision`.

5. **Rubric infrastructure failure is visible and non-destructive**
   - Given the grader provider raises an exception
   - When rubric evaluation fails
   - Then the terminal status is `grader_error`
   - And metadata includes the explanation
   - And structural safeguards such as valid todo shape and allowed statuses still apply.

6. **Rubric criteria are contextual**
   - Given the user asks for a small, narrow plan
   - When the runtime resolves the planning rubric
   - Then the criteria are tailored to that request instead of applying a fixed universal checklist
   - And metadata includes whether the rubric source was `caller`, `generated`, or `fallback`.

### Non-Goals

- Do not add the `deepagents` dependency.
- Do not replace `MultiAgentWorkflow` with `create_deep_agent`.
- Do not grade every non-planning chat answer.
- Do not let the rubric grader mutate persisted `TaskPlan` rows directly.
- Do not add database tables for rubric evaluations in the first implementation.
- Do not add grader tools for running external commands in the first implementation.
- Do not hard-code one static planning-quality checklist as the only rubric. The runtime may keep small invariant guardrails, but the actual criteria used for each attempt must be generated or overridden from context.

---

## References Checked

- LangChain DeepAgents rubric docs: `https://docs.langchain.com/oss/python/deepagents/rubric`
- Existing implementation plan style: `plans/plan_2.md`, `plans/subagents.md`, `plans/tool_search_v2.md`
- Current Planning mode docs: `README.md` Planning Mode & Task Plans section

DeepAgents rubric behavior to mirror:

- A caller supplies a rubric string.
- A dedicated grader model returns verdicts.
- `needs_revision` feedback is injected and the worker runs again.
- Terminal statuses include `satisfied`, `max_iterations_reached`, `failed`, and `grader_error`.
- Per-pass evaluations include iteration, explanation, and per-criterion results.

---

## Current Code Findings

- `app/ai/agents/planning_agent.py` owns plan generation/modification through `generate_plan()`, `modify_plan()`, and `_generate_or_modify_plan()`.
- `PlanningAgent._generate_or_modify_plan()` already forces a `write_todos` tool call and extracts candidate todos from tool calls.
- `PlanningAgent._validate_task_descriptions()` is a deterministic quality gate for under-specified tasks, but it is rejection-only and currently fires before any model-driven revision loop can repair vague tasks.
- `app/ai/planning_runtime_adapter.py` adapts service calls into `PlanningAgent.generate_plan()` / `modify_plan()`, but currently returns only `todos`.
- `app/services/task_plan_service.py` persists generated/modified tasks after receiving `PlanningRuntimeResult.todos`.
- `app/ai/graph.py::_planning_tools_node()` applies in-chat `write_todos` calls and then routes back through `_should_continue_planning()`.
- `app/ai/graph.py::_attach_planning_state_metadata()` is the right final response hook for graph-level rubric metadata.
- `app/schemas/workflow.py::WorkflowPlanningContext` is the service boundary used by `MessageService`; `app/ai/schemas.py::WorkflowPlanningContext` is the AI boundary used after `AIService._to_ai_request()`. Rubric metadata must be declared in both schemas.
- `app/ai/schemas.py::GraphContext` already carries Planning subagent metadata and can be extended with turn-scoped rubric metadata.
- `app/core/response_constants.py::build_bot_metadata()` already preserves unknown response metadata keys. Message persistence only needs a regression test unless later code starts filtering `planning_rubric`.
- `deepagents` is not in `pyproject.toml` and is not installed in the current environment.

---

## Constitution Check

No `.specify/memory/constitution.md` exists in this repository. Apply the repository's current engineering constraints.

Simplicity Gate:

- Pass: add one focused runtime module for rubric schemas, prompt construction, parsing, and loop results.
- Pass: use existing `PlanningAgent` model creation and retry helpers instead of a second model stack.
- Pass: no database migration in the first implementation.

Integration-First Gate:

- Pass: service-owned plan generation is reviewed before persistence.
- Pass: graph-owned plan mutation is reviewed before final assistant metadata is attached.
- Pass: rubric metadata follows existing `AgentResponse.metadata` and `GraphContext` patterns.

Safety Gate:

- Pass: only `PlanningAgent` can call `write_todos`; the grader never mutates todos.
- Pass: rubric loops have an explicit iteration cap independent from `planning_max_iterations`.
- Pass: grader errors become metadata and do not silently discard deterministic validation failures.

---

## Research Decisions

### Decision 1: Implement a Native Rubric Runtime

Create `app/ai/planning_rubric.py` instead of adding `deepagents`.

Rationale:

- Current planning is not a DeepAgents app.
- The repo already has LangGraph state, persistence, streaming, HITL, and custom subagents.
- A native module can copy the useful behavior while preserving existing boundaries.

### Decision 2: Use Planning Agent Runtime Model Infrastructure

Use `PlanningAgent._create_langchain_model_from_runtime()` and `_ainvoke_with_retries()` for grader calls.

Rationale:

- Keeps API-key/provider fallback behavior aligned with the rest of the app.
- Avoids a new model resolver surface.
- Supports tests with fake model calls.

### Decision 3: Represent Rubric Results as Response Metadata First

Expose rubric output in metadata:

```json
{
  "planning_rubric": {
    "status": "satisfied",
    "grading_run_id": "uuid",
    "iterations": 1,
    "source": "generate_plan",
    "evaluations": [
      {
        "iteration": 0,
        "result": "satisfied",
        "explanation": "All criteria passed.",
        "criteria": [
          {"name": "request_fit", "passed": true}
        ]
      }
    ]
  }
}
```

Rationale:

- No migration required.
- Existing message metadata paths preserve unknown fields.
- UI/API clients can add rendering later without changing the runtime contract.

### Decision 4: Review Both Service-Level Plans and Graph-Level `write_todos` Mutations

Service-level review covers automatic plan creation/modification through `TaskPlanService`. Graph-level review covers chat turns where the Planning Agent uses `write_todos` inside `MultiAgentWorkflow`.

Rationale:

- Covering only `generate_plan()` would miss in-chat plan edits.
- Covering only graph mutations would miss pre-workflow auto-created plans.

### Decision 5: Generate Rubrics Dynamically Per Attempt

Do not make the grader depend on one hard-coded checklist. The runtime builds a rubric contract for each attempt from:

- the user's latest request,
- the existing persisted task snapshot when one exists,
- the candidate todo list,
- the current planning lifecycle/phase,
- optional caller-supplied `planning_rubric` metadata, and
- prior rubric feedback when a revision is being graded.

Rationale:

- Planning quality is contextual. A plan for a small refactor, a research task, and a multi-agent execution run should not be judged by identical task granularity rules.
- The LLM can decide what “complete”, “specific”, and “preserved” mean for the actual request while code enforces only structural invariants such as valid JSON, allowed statuses, and max task count.
- This mirrors the DeepAgents rubric concept more closely: a caller supplies a rubric, but in this app the caller can be explicit user metadata or a contextual rubric-authoring prompt.

---

## Data Model

Add these Pydantic models to `app/ai/planning_rubric.py`.

```python
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


PlanningRubricVerdict = Literal[
    "satisfied",
    "needs_revision",
    "failed",
    "grader_error",
]

PlanningRubricTerminalStatus = Literal[
    "satisfied",
    "needs_revision",
    "max_iterations_reached",
    "failed",
    "grader_error",
    "disabled",
]


class PlanningRubricCriterionResult(BaseModel):
    name: str = Field(..., min_length=1)
    passed: bool
    gap: str | None = None


class PlanningRubricEvaluation(BaseModel):
    iteration: int = Field(..., ge=0)
    result: PlanningRubricVerdict
    explanation: str = ""
    criteria: list[PlanningRubricCriterionResult] = Field(default_factory=list)


class PlanningRubricContract(BaseModel):
    rubric: str = Field(..., min_length=1)
    source: Literal["caller", "generated", "fallback"]
    rationale: str = ""


class PlanningRubricAttempt(BaseModel):
    status: PlanningRubricTerminalStatus
    grading_run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    iterations: int = Field(..., ge=0)
    source: Literal["generate_plan", "modify_plan", "planning_tools"]
    rubric: str
    rubric_source: Literal["caller", "generated", "fallback"] = "generated"
    rubric_rationale: str = ""
    evaluations: list[PlanningRubricEvaluation] = Field(default_factory=list)
    feedback: str | None = None
    error: str | None = None

    def metadata(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)
```

Extend existing schemas:

- `app/schemas/task_plan.py::PlanningRuntimeResult`
  - Add `metadata: dict[str, Any] = Field(default_factory=dict)`.
- `app/schemas/workflow.py::WorkflowPlanningContext`
  - Add `rubric_metadata: dict[str, Any] | None = None`.
- `app/ai/schemas.py::WorkflowPlanningContext`
  - Add `rubric_metadata: dict[str, Any] | None = None`.
- `app/ai/schemas.py::GraphContext`
  - Add `planning_rubric: dict[str, Any]`.
  - Add `planning_rubric_feedback: str`.

---

## Rubric Contract

Rubric text is resolved per attempt, not hard-coded as a single checklist.

Resolution order:

1. If `message.metadata["planning_rubric"]` is a non-empty string, use it as the caller-supplied rubric.
2. Otherwise call a lightweight rubric-author prompt using the same runtime model infrastructure. It receives the user request, current plan lifecycle, existing todos, candidate todos, and any prior feedback, then returns a concise checklist tailored to this attempt.
3. If rubric authoring fails, use a small fallback rubric that contains only invariant planning safety requirements. This fallback is a resilience path, not the normal quality model.

Fallback rubric text:

```python
FALLBACK_PLANNING_RUBRIC = """\
- The candidate todo list is valid for the current planning phase.
- The candidate tasks are understandable in the context of the user's latest request.
- Existing completed, skipped, or in-progress work is not silently destroyed.
"""
```

The rubric-author prompt must require JSON only:

```text
Return JSON matching this shape:
{
  "rubric": "- criterion_name: context-specific criterion\n- another_criterion: ...",
  "rationale": "short explanation of why these criteria fit this request"
}
```

The generated rubric should avoid fixed universal thresholds such as minimum task counts or exact description lengths. It may mention identity/status preservation only when an existing plan is being modified, and it may mention granularity only relative to the user's actual goal.

The grader prompt must require JSON only:

```text
Return JSON matching this shape:
{
  "result": "satisfied" | "needs_revision" | "failed",
  "explanation": "short summary",
  "criteria": [
    {"name": "criterion_name", "passed": true},
    {"name": "criterion_name", "passed": false, "gap": "actionable fix"}
  ]
}
```

`grader_error` is produced by code when the grader call or parsing fails.

---

## Files To Add

- `app/ai/planning_rubric.py`
- `tests/test_planning_rubric.py`
- `tests/test_planning_agent_rubric.py`
- `tests/test_graph_planning_rubric.py`
- `tests/test_message_service_planning_rubric.py`

## Files To Modify

- `app/core/config.py`
- `app/schemas/workflow.py`
- `app/ai/schemas.py`
- `app/schemas/task_plan.py`
- `app/ai/planning_runtime_adapter.py`
- `app/ai/agents/planning_agent.py`
- `app/ai/graph.py`
- `app/core/response_constants.py`
- `app/services/message_service.py`
- `app/services/task_plan_service.py`
- `README.md`

---

## Implementation Tasks

### Task 1: Add Rubric Schemas And Pure Helpers

**Files:**

- Create: `app/ai/planning_rubric.py`
- Test: `tests/test_planning_rubric.py`

> **STATUS: COMPLETE** (commit `5d90111`). All 6 unit tests pass. Verification env: system Python 3.14.4 via `python -m pytest` (the project `.venv` at 3.13.7 has no pytest installed).

- [x] **Step 1: Write schema and helper tests**

Create `tests/test_planning_rubric.py`:

```python
import json

from app.ai.planning_rubric import (
    FALLBACK_PLANNING_RUBRIC,
    PlanningRubricAttempt,
    PlanningRubricContract,
    PlanningRubricEvaluation,
    build_planning_rubric_feedback,
    parse_planning_rubric_evaluation,
)


def test_fallback_rubric_is_minimal_and_invariant_only():
    assert "valid for the current planning phase" in FALLBACK_PLANNING_RUBRIC
    assert "not silently destroyed" in FALLBACK_PLANNING_RUBRIC
    assert "20 chars" not in FALLBACK_PLANNING_RUBRIC


def test_planning_rubric_contract_accepts_generated_source():
    contract = PlanningRubricContract(
        rubric="- request_fit: Tasks fit this user's actual request.",
        source="generated",
        rationale="The user's request is small and does not need a large checklist.",
    )

    assert contract.source == "generated"
    assert "request_fit" in contract.rubric


def test_parse_planning_rubric_evaluation_from_json_text():
    payload = {
        "result": "needs_revision",
        "explanation": "Two tasks are vague.",
        "criteria": [
            {
                "name": "concrete_backend_scope",
                "passed": False,
                "gap": "Replace 'fix backend' with concrete files and behavior.",
            },
            {"name": "preserve_existing_behavior", "passed": True},
        ],
    }

    parsed = parse_planning_rubric_evaluation(json.dumps(payload), iteration=2)

    assert parsed.iteration == 2
    assert parsed.result == "needs_revision"
    assert parsed.criteria[0].name == "concrete_backend_scope"
    assert parsed.criteria[0].gap == "Replace 'fix backend' with concrete files and behavior."


def test_parse_planning_rubric_evaluation_strips_markdown_fence():
    raw = (
        "```json\n"
        '{"result":"satisfied","explanation":"ok","criteria":[{"name":"request_fit","passed":true}]}'
        "\n```"
    )

    parsed = parse_planning_rubric_evaluation(raw, iteration=0)

    assert parsed.result == "satisfied"
    assert parsed.criteria[0].passed is True


def test_build_planning_rubric_feedback_includes_failed_criteria_only():
    evaluation = PlanningRubricEvaluation(
        iteration=0,
        result="needs_revision",
        explanation="Needs work.",
        criteria=[
            {"name": "concrete_backend_scope", "passed": False, "gap": "Make task 1 concrete."},
            {"name": "preserve_existing_behavior", "passed": True},
        ],
    )

    feedback = build_planning_rubric_feedback(evaluation)

    assert "Needs work." in feedback
    assert "concrete_backend_scope" in feedback
    assert "Make task 1 concrete." in feedback
    assert "preserve_existing_behavior" not in feedback


def test_attempt_metadata_excludes_none_fields():
    attempt = PlanningRubricAttempt(
        status="satisfied",
        grading_run_id="run-1",
        iterations=1,
        source="generate_plan",
        rubric=FALLBACK_PLANNING_RUBRIC,
        evaluations=[
            PlanningRubricEvaluation(
                iteration=0,
                result="satisfied",
                explanation="All criteria passed.",
                criteria=[{"name": "request_fit", "passed": True}],
            )
        ],
    )

    metadata = attempt.metadata()

    assert metadata["status"] == "satisfied"
    assert metadata["evaluations"][0]["criteria"][0]["name"] == "request_fit"
    assert "error" not in metadata
```

- [x] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_planning_rubric.py -q
```

Expected: fail because `app.ai.planning_rubric` does not exist.

- [x] **Step 3: Implement schemas and pure helpers**

Create `app/ai/planning_rubric.py` with:

```python
from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


PlanningRubricVerdict = Literal["satisfied", "needs_revision", "failed", "grader_error"]
PlanningRubricTerminalStatus = Literal[
    "satisfied",
    "needs_revision",
    "max_iterations_reached",
    "failed",
    "grader_error",
    "disabled",
]


FALLBACK_PLANNING_RUBRIC = """\
- The candidate todo list is valid for the current planning phase.
- The candidate tasks are understandable in the context of the user's latest request.
- Existing completed, skipped, or in-progress work is not silently destroyed.
"""


class PlanningRubricCriterionResult(BaseModel):
    name: str = Field(..., min_length=1)
    passed: bool
    gap: str | None = None


class PlanningRubricEvaluation(BaseModel):
    iteration: int = Field(..., ge=0)
    result: PlanningRubricVerdict
    explanation: str = ""
    criteria: list[PlanningRubricCriterionResult] = Field(default_factory=list)


class PlanningRubricContract(BaseModel):
    rubric: str = Field(..., min_length=1)
    source: Literal["caller", "generated", "fallback"]
    rationale: str = ""


class PlanningRubricAttempt(BaseModel):
    status: PlanningRubricTerminalStatus
    grading_run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    iterations: int = Field(..., ge=0)
    source: Literal["generate_plan", "modify_plan", "planning_tools"]
    rubric: str
    rubric_source: Literal["caller", "generated", "fallback"] = "generated"
    rubric_rationale: str = ""
    evaluations: list[PlanningRubricEvaluation] = Field(default_factory=list)
    feedback: str | None = None
    error: str | None = None

    def metadata(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def _strip_json_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_planning_rubric_evaluation(raw_text: str, *, iteration: int) -> PlanningRubricEvaluation:
    cleaned = _strip_json_fence(raw_text)
    try:
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("grader JSON root must be an object")
        payload["iteration"] = iteration
        return PlanningRubricEvaluation.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        return PlanningRubricEvaluation(
            iteration=iteration,
            result="grader_error",
            explanation=f"Failed to parse planning rubric grader output: {exc}",
            criteria=[],
        )


def build_planning_rubric_feedback(evaluation: PlanningRubricEvaluation) -> str:
    lines = ["Planning rubric review requires a revision."]
    if evaluation.explanation.strip():
        lines.append(f"Summary: {evaluation.explanation.strip()}")
    for criterion in evaluation.criteria:
        if criterion.passed:
            continue
        gap = criterion.gap or "Criterion failed without a specific gap."
        lines.append(f"- {criterion.name}: {gap}")
    return "\n".join(lines)
```

- [x] **Step 4: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_planning_rubric.py -q
```

Expected: `6 passed`.

- [x] **Step 5: Commit**

Run:

```powershell
git add app/ai/planning_rubric.py tests/test_planning_rubric.py
git commit -m "feat: add planning rubric schemas"
```

---

### Task 2: Add Configuration And Metadata Schema Plumbing

**Files:**

- Modify: `app/core/config.py`
- Modify: `app/schemas/workflow.py`
- Modify: `app/ai/schemas.py`
- Modify: `app/schemas/task_plan.py`
- Modify: `app/ai/planning_runtime_adapter.py`
- Test: `tests/test_config_redis.py`
- Test: `tests/test_message_service_planning_rubric.py`

> **STATUS: COMPLETE** (commit `fe9b066`). `test_config_redis.py` + `test_message_service_planning_rubric.py` = 9 passed.
> Decisions: (1) `planning_rubric_max_iterations` was added to the existing `_positive_int` `@field_validator` list, which enforces a minimum of 1 (rejects 0). (2) `GraphContext` gained `planning_rubric: dict[str, Any]` and `planning_rubric_feedback: str`; both stay optional because `GraphContext` is a `TypedDict(total=False)`.

- [x] **Step 1: Write config tests**

Append to `tests/test_config_redis.py`:

```python
def test_planning_rubric_defaults_enabled():
    from app.core.config import Settings

    settings = Settings(secret_key="test-secret", environment="development")

    assert settings.planning_rubric_enabled is True
    assert settings.planning_rubric_max_iterations == 3


def test_planning_rubric_max_iterations_accepts_one_as_minimum_cap():
    from app.core.config import Settings

    settings = Settings(
        secret_key="test-secret",
        environment="development",
        planning_rubric_max_iterations=1,
    )

    assert settings.planning_rubric_max_iterations == 1
```

- [x] **Step 2: Write metadata propagation test for runtime adapter**

Create `tests/test_message_service_planning_rubric.py`:

```python
import pytest

from app.ai.planning_runtime_adapter import PlanningRuntimeAdapter
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


@pytest.mark.asyncio
async def test_planning_runtime_adapter_preserves_planning_rubric_metadata():
    class _PlanningAgent:
        async def generate_plan(self, message, conversation_id=None):
            assert message.role == MessageRole.USER
            return AgentResponse(
                agent_type=AgentType.PLANNING,
                agent_id="planning_agent",
                message=AgentMessage(role=MessageRole.ASSISTANT, content="plan"),
                metadata={
                    "todos": [
                        {
                            "id": "t1",
                            "description": "Implement the native rubric evaluator",
                            "status": "pending",
                            "order": 0,
                        }
                    ],
                    "planning_rubric": {"status": "satisfied", "iterations": 1},
                },
            )

    from app.schemas.task_plan import PlanningRuntimeRequest

    result = await PlanningRuntimeAdapter(_PlanningAgent()).generate_plan(
        PlanningRuntimeRequest(
            user_message="Add rubric grading",
            conversation_id="conv-1",
            user_id="user-1",
        )
    )

    assert result.todos[0]["id"] == "t1"
    assert result.metadata["planning_rubric"]["status"] == "satisfied"
```

- [x] **Step 3: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_config_redis.py::test_planning_rubric_defaults_enabled tests/test_config_redis.py::test_planning_rubric_max_iterations_accepts_one_as_minimum_cap tests/test_message_service_planning_rubric.py -q
```

Expected: fail because config fields and `PlanningRuntimeResult.metadata` do not exist.

- [x] **Step 4: Add config fields**

In `app/core/config.py`, add near existing Planning Agent settings:

```python
    planning_rubric_enabled: bool = Field(
        default=True,
        description=(
            "Enable native Planning rubric grading. When enabled, generated or "
            "modified todo plans are evaluated against a planning-quality rubric "
            "and revised before persistence or final response when possible."
        ),
    )
    planning_rubric_max_iterations: int = Field(
        default=3,
        description=(
            "Maximum Planning rubric grading passes per attempt. Minimum 1. "
            "Set planning_rubric_enabled=False to disable grading."
        ),
    )
```

Also add `"planning_rubric_max_iterations"` to the positive integer field validation list that already contains `"react_agent_max_iterations"`.

- [x] **Step 5: Extend schema fields**

In `app/schemas/task_plan.py`, change `PlanningRuntimeResult` to:

```python
class PlanningRuntimeResult(BaseModel):
    """Service-owned result for planning runtime operations."""

    todos: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Canonical todo payload returned by the planning runtime",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Planning runtime metadata such as rubric evaluation status",
    )
```

In `app/schemas/workflow.py`, extend the service-facing `WorkflowPlanningContext`:

```python
    rubric_metadata: dict[str, Any] | None = None
```

In `app/ai/schemas.py`, extend the AI-facing `WorkflowPlanningContext`:

```python
    rubric_metadata: dict[str, Any] | None = None
```

In `app/ai/schemas.py`, extend `GraphContext`:

```python
    planning_rubric: dict[str, Any]
    planning_rubric_feedback: str
```

- [x] **Step 6: Preserve metadata in runtime adapter**

In `app/ai/planning_runtime_adapter.py`, change both result builders:

```python
        return PlanningRuntimeResult(
            todos=self._extract_todos(response),
            metadata=dict(getattr(response, "metadata", None) or {}),
        )
```

- [x] **Step 7: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_config_redis.py tests/test_message_service_planning_rubric.py -q
```

Expected: config and metadata tests pass.

- [x] **Step 8: Commit**

Run:

```powershell
git add app/core/config.py app/schemas/workflow.py app/ai/schemas.py app/schemas/task_plan.py app/ai/planning_runtime_adapter.py tests/test_config_redis.py tests/test_message_service_planning_rubric.py
git commit -m "feat: add planning rubric metadata plumbing"
```

---

### Task 3: Add Rubric Grader Call And Revision Loop For Service-Level Plans

**Files:**

- Modify: `app/ai/planning_rubric.py`
- Modify: `app/ai/agents/planning_agent.py`
- Test: `tests/test_planning_agent_rubric.py`

> **STATUS: COMPLETE** (commit `446bafd`). `test_planning_agent_rubric.py` + `test_planning_rubric.py` = 10 passed; existing `test_planning_subagents.py` + `test_custom_agents_planning.py` = 65 passed (no regression).
> Decisions: (1) The inline rubric loop from Step 5 was extracted into a dedicated `PlanningAgent._run_planning_rubric_loop()` helper instead of inlined into `_generate_or_modify_plan()`. Rationale: the inline block is ~80 lines and would push `_generate_or_modify_plan()` over the 100-line function limit; the extracted method keeps both functions readable and the logic identical. (2) The dead `_validate_task_descriptions()` method, its `_MIN_DESC_LEN` constant, and the now-unused `tokenize_text` import were deleted (no remaining references in app/ or tests/) to keep linters clean — the plan's directive was to replace the hard rejection with the rubric loop.

- [x] **Step 1: Write tests for satisfied, revision, max, and grader-error paths**

Create `tests/test_planning_agent_rubric.py`:

```python
import pytest
from langchain_core.messages import AIMessage

from app.ai.agents.planning_agent import PlanningAgent
from app.ai.schemas import AgentMessage, MessageRole, TodoStatus


def _agent():
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


class _FakeToolModel:
    def bind_tools(self, *_args, **_kwargs):
        return self


def _todo_call(description, todo_id="t1"):
    return {
        "id": "call-1",
        "name": "write_todos",
        "args": {
            "action": "set_todos",
            "todos": [
                {
                    "id": todo_id,
                    "description": description,
                    "status": TodoStatus.PENDING.value,
                    "order": 0,
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_generate_plan_attaches_satisfied_rubric_metadata(monkeypatch):
    agent = _agent()

    class _Runtime:
        provider = "gemini"
        model = "test-model"
        temperature = 0
        api_key = None
        key_source = "none"
        source = "default"
        warnings = []
        is_custom_model = False
        provider_fallback = None
        context_window = None
        fallback_config = None

    monkeypatch.setattr(agent, "_resolve_runtime_model_config", lambda *_args, **_kwargs: _Runtime())
    monkeypatch.setattr(agent, "_create_langchain_model_from_runtime", lambda *_args, **_kwargs: (_FakeToolModel(), False))

    responses = [
        AIMessage(content="", tool_calls=[_todo_call("Implement native planning rubric grading")]),
        AIMessage(
            content='{"rubric":"- request_fit: The tasks fit the user request.","rationale":"Small feature plan."}'
        ),
        AIMessage(
            content='{"result":"satisfied","explanation":"ok","criteria":[{"name":"request_fit","passed":true}]}'
        ),
    ]

    async def fake_invoke(_model, _messages, run_config=None):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_ainvoke_with_retries", fake_invoke)

    response = await agent.generate_plan(
        AgentMessage(role=MessageRole.USER, content="Add rubric grading"),
        conversation_id="conv-1",
    )

    assert response.error is None
    assert response.metadata["todos"][0]["description"] == "Implement native planning rubric grading"
    assert response.metadata["planning_rubric"]["status"] == "satisfied"
    assert response.metadata["planning_rubric"]["iterations"] == 1


@pytest.mark.asyncio
async def test_generate_plan_revises_after_needs_revision(monkeypatch):
    agent = _agent()

    class _Runtime:
        provider = "gemini"
        model = "test-model"
        temperature = 0
        api_key = None
        key_source = "none"
        source = "default"
        warnings = []
        is_custom_model = False
        provider_fallback = None
        context_window = None
        fallback_config = None

    monkeypatch.setattr(agent, "_resolve_runtime_model_config", lambda *_args, **_kwargs: _Runtime())
    monkeypatch.setattr(agent, "_create_langchain_model_from_runtime", lambda *_args, **_kwargs: (_FakeToolModel(), False))

    responses = [
        AIMessage(content="", tool_calls=[_todo_call("Fix backend")]),
        AIMessage(
            content='{"rubric":"- concrete_backend_scope: Backend tasks name concrete behavior.","rationale":"The candidate is vague."}'
        ),
        AIMessage(
            content='{"result":"needs_revision","explanation":"vague","criteria":[{"name":"concrete_backend_scope","passed":false,"gap":"Name the concrete backend behavior."}]}'
        ),
        AIMessage(
            content="",
            tool_calls=[_todo_call("Add native Planning rubric evaluator and metadata contract")],
        ),
        AIMessage(
            content='{"result":"satisfied","explanation":"ok","criteria":[{"name":"concrete_backend_scope","passed":true}]}'
        ),
    ]

    async def fake_invoke(_model, messages, run_config=None):
        if len(responses) == 2:
            assert "Name the concrete backend behavior" in str(messages[-1].content)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_ainvoke_with_retries", fake_invoke)

    response = await agent.generate_plan(
        AgentMessage(role=MessageRole.USER, content="Add rubric grading"),
        conversation_id="conv-1",
    )

    assert response.error is None
    assert response.metadata["todos"][0]["description"] == (
        "Add native Planning rubric evaluator and metadata contract"
    )
    assert response.metadata["planning_rubric"]["status"] == "satisfied"
    assert response.metadata["planning_rubric"]["iterations"] == 2


@pytest.mark.asyncio
async def test_generate_plan_marks_max_iterations_reached(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "planning_rubric_max_iterations", 1)
    agent = _agent()

    class _Runtime:
        provider = "gemini"
        model = "test-model"
        temperature = 0
        api_key = None
        key_source = "none"
        source = "default"
        warnings = []
        is_custom_model = False
        provider_fallback = None
        context_window = None
        fallback_config = None

    monkeypatch.setattr(agent, "_resolve_runtime_model_config", lambda *_args, **_kwargs: _Runtime())
    monkeypatch.setattr(agent, "_create_langchain_model_from_runtime", lambda *_args, **_kwargs: (_FakeToolModel(), False))

    responses = [
        AIMessage(content="", tool_calls=[_todo_call("Fix backend")]),
        AIMessage(
            content='{"rubric":"- concrete_scope: Tasks should name the concrete behavior for this request.","rationale":"The candidate is vague."}'
        ),
        AIMessage(
            content='{"result":"needs_revision","explanation":"vague","criteria":[{"name":"concrete_scope","passed":false,"gap":"Be concrete."}]}'
        ),
    ]

    async def fake_invoke(_model, _messages, run_config=None):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_ainvoke_with_retries", fake_invoke)

    response = await agent.generate_plan(
        AgentMessage(role=MessageRole.USER, content="Add rubric grading"),
        conversation_id="conv-1",
    )

    assert response.metadata["planning_rubric"]["status"] == "max_iterations_reached"
    assert response.metadata["planning_rubric"]["feedback"]


@pytest.mark.asyncio
async def test_generate_plan_marks_grader_error(monkeypatch):
    agent = _agent()

    class _Runtime:
        provider = "gemini"
        model = "test-model"
        temperature = 0
        api_key = None
        key_source = "none"
        source = "default"
        warnings = []
        is_custom_model = False
        provider_fallback = None
        context_window = None
        fallback_config = None

    monkeypatch.setattr(agent, "_resolve_runtime_model_config", lambda *_args, **_kwargs: _Runtime())
    monkeypatch.setattr(agent, "_create_langchain_model_from_runtime", lambda *_args, **_kwargs: (_FakeToolModel(), False))

    responses = [
        AIMessage(content="", tool_calls=[_todo_call("Implement native planning rubric grading")]),
        AIMessage(
            content='{"rubric":"- request_fit: Tasks fit this request.","rationale":"Generated from context."}'
        ),
        RuntimeError("grader unavailable"),
    ]

    async def fake_invoke(_model, _messages, run_config=None):
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(agent, "_ainvoke_with_retries", fake_invoke)

    response = await agent.generate_plan(
        AgentMessage(role=MessageRole.USER, content="Add rubric grading"),
        conversation_id="conv-1",
    )

    assert response.error is None
    assert response.metadata["planning_rubric"]["status"] == "grader_error"
    assert "grader unavailable" in response.metadata["planning_rubric"]["error"]
```

- [x] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_planning_agent_rubric.py -q
```

Expected: fail because the rubric loop is not implemented.

- [x] **Step 3: Add grader prompt helpers**

In `app/ai/planning_rubric.py`, add:

```python
def build_planning_rubric_author_prompt(
    *,
    user_message: str,
    candidate_todos: list[dict[str, Any]],
    existing_todos: list[dict[str, Any]] | None,
    plan_modified: bool,
    lifecycle: str | None = None,
    prior_feedback: str | None = None,
) -> str:
    payload = {
        "user_message": user_message,
        "plan_modified": plan_modified,
        "lifecycle": lifecycle,
        "existing_todos": existing_todos or [],
        "candidate_todos": candidate_todos,
        "prior_feedback": prior_feedback,
    }
    return (
        "You are writing a concise, context-specific grading rubric for a "
        "Planning-mode todo plan. Do not use fixed global thresholds such as "
        "minimum character counts or required task counts. Judge what matters "
        "for this user's request and the current plan state. Return JSON only "
        "with keys rubric and rationale.\n\n"
        f"{json.dumps(payload, indent=2, default=str)}"
    )


def parse_planning_rubric_contract(raw_text: str) -> PlanningRubricContract:
    cleaned = _strip_json_fence(raw_text)
    try:
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("rubric JSON root must be an object")
        payload.setdefault("source", "generated")
        return PlanningRubricContract.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError):
        return PlanningRubricContract(
            rubric=FALLBACK_PLANNING_RUBRIC,
            source="fallback",
            rationale="Rubric authoring failed; using minimal invariant fallback.",
        )


def format_todos_for_rubric(todos: list[dict[str, Any]]) -> str:
    return json.dumps(todos, indent=2, default=str)


def build_planning_rubric_grader_prompt(
    *,
    rubric: str,
    user_message: str,
    candidate_todos: list[dict[str, Any]],
    existing_todos: list[dict[str, Any]] | None,
    plan_modified: bool,
) -> str:
    payload = {
        "user_message": user_message,
        "plan_modified": plan_modified,
        "existing_todos": existing_todos or [],
        "candidate_todos": candidate_todos,
        "rubric": rubric,
    }
    return (
        "You are a Planning-mode rubric grader. Grade the candidate todo plan "
        "against the rubric. Return JSON only with keys result, explanation, "
        "and criteria. result must be one of satisfied, needs_revision, failed.\n\n"
        f"{json.dumps(payload, indent=2, default=str)}"
    )


def build_planning_rubric_revision_prompt(
    *,
    feedback: str,
    candidate_todos: list[dict[str, Any]],
) -> str:
    return (
        "Revise the task plan by calling write_todos. Keep valid existing task "
        "ids and statuses. Apply this rubric feedback exactly:\n\n"
        f"{feedback}\n\n"
        "Current candidate todos:\n"
        f"{format_todos_for_rubric(candidate_todos)}"
    )
```

- [x] **Step 4: Add PlanningAgent rubric methods**

In `app/ai/agents/planning_agent.py`, import helpers:

```python
from ..planning_rubric import (
    FALLBACK_PLANNING_RUBRIC,
    PlanningRubricAttempt,
    PlanningRubricContract,
    PlanningRubricEvaluation,
    build_planning_rubric_author_prompt,
    build_planning_rubric_feedback,
    build_planning_rubric_grader_prompt,
    build_planning_rubric_revision_prompt,
    parse_planning_rubric_contract,
    parse_planning_rubric_evaluation,
)
```

Add methods to `PlanningAgent`:

```python
    async def _resolve_planning_rubric_contract(
        self,
        *,
        llm: Any,
        caller_rubric: str | None,
        user_message: str,
        candidate_todos: list[dict[str, Any]],
        existing_todos: list[dict[str, Any]] | None,
        plan_modified: bool,
        lifecycle: str | None = None,
        prior_feedback: str | None = None,
    ) -> PlanningRubricContract:
        if isinstance(caller_rubric, str) and caller_rubric.strip():
            return PlanningRubricContract(
                rubric=caller_rubric.strip(),
                source="caller",
                rationale="Caller supplied the planning rubric.",
            )

        prompt = build_planning_rubric_author_prompt(
            user_message=user_message,
            candidate_todos=candidate_todos,
            existing_todos=existing_todos,
            plan_modified=plan_modified,
            lifecycle=lifecycle,
            prior_feedback=prior_feedback,
        )
        try:
            raw_response = await self._ainvoke_with_retries(
                llm,
                [SystemMessage(content=prompt)],
            )
            raw_text = coerce_response_text(getattr(raw_response, "content", ""))
            return parse_planning_rubric_contract(raw_text)
        except Exception:
            return PlanningRubricContract(
                rubric=FALLBACK_PLANNING_RUBRIC,
                source="fallback",
                rationale="Rubric authoring failed; using minimal invariant fallback.",
            )


    async def _grade_planning_todos(
        self,
        *,
        llm: Any,
        rubric: str,
        user_message: str,
        candidate_todos: list[dict[str, Any]],
        existing_todos: list[dict[str, Any]] | None,
        plan_modified: bool,
        iteration: int,
    ) -> PlanningRubricEvaluation:
        prompt = build_planning_rubric_grader_prompt(
            rubric=rubric,
            user_message=user_message,
            candidate_todos=candidate_todos,
            existing_todos=existing_todos,
            plan_modified=plan_modified,
        )
        try:
            raw_response = await self._ainvoke_with_retries(
                llm,
                [SystemMessage(content=prompt)],
            )
            raw_text = coerce_response_text(getattr(raw_response, "content", ""))
            return parse_planning_rubric_evaluation(raw_text, iteration=iteration)
        except Exception as exc:
            return PlanningRubricEvaluation(
                iteration=iteration,
                result="grader_error",
                explanation=f"Planning rubric grader failed: {exc}",
                criteria=[],
            )

    async def _revise_todos_from_rubric_feedback(
        self,
        *,
        llm: Any,
        write_todos_tool: BaseTool,
        base_system_prompt: str,
        user_message: str,
        base_todos: list[dict[str, Any]],
        candidate_todos: list[dict[str, Any]],
        feedback: str,
    ) -> list[dict[str, Any]]:
        revision_prompt = build_planning_rubric_revision_prompt(
            feedback=feedback,
            candidate_todos=candidate_todos,
        )
        bound_llm = ModelFactory.bind_tools_to_model(
            llm,
            [write_todos_tool],
            tool_choice="write_todos",
        )
        raw_response = await self._ainvoke_with_retries(
            bound_llm,
            [
                SystemMessage(content=base_system_prompt),
                HumanMessage(content=user_message),
                HumanMessage(content=revision_prompt),
            ],
        )
        return self._canonicalize_todos(
            self._apply_write_todos_calls(
                base_todos=base_todos,
                tool_calls=getattr(raw_response, "tool_calls", None) or [],
            )
        )
```

- [x] **Step 5: Thread rubric loop into `_generate_or_modify_plan()`**

Replace the current pre-response hard rejection around `_validate_task_descriptions()` with a rubric-aware revision loop. Keep only structural validation in code; do not reject based on fixed description length or fixed token counts before the LLM has a chance to revise.

Add a helper near the existing quality-gate methods:

```python
    def _build_planning_structural_feedback(
        self,
        todos: list[dict[str, Any]],
    ) -> str | None:
        if not todos:
            return "The plan must contain at least one todo."
        seen_ids: set[str] = set()
        for index, todo in enumerate(todos, start=1):
            todo_id = str(todo.get("id") or "").strip()
            if not todo_id:
                return f"Task {index} is missing a stable id."
            if todo_id in seen_ids:
                return f"Task {index} reuses duplicate id {todo_id}."
            seen_ids.add(todo_id)
            status = str(todo.get("status") or "").strip().lower()
            if status not in {
                TodoStatus.PENDING.value,
                TodoStatus.IN_PROGRESS.value,
                TodoStatus.COMPLETED.value,
                TodoStatus.SKIPPED.value,
            }:
                return f"Task {index} has unsupported status {status!r}."
        return None
```

Before formatting the response text, add:

```python
        rubric_attempt: PlanningRubricAttempt | None = None
        if getattr(settings, "planning_rubric_enabled", True):
            caller_rubric = message.metadata.get("planning_rubric")
            caller_rubric = caller_rubric if isinstance(caller_rubric, str) else None
            max_iterations = max(1, int(getattr(settings, "planning_rubric_max_iterations", 3)))
            evaluations: list[PlanningRubricEvaluation] = []
            feedback: str | None = None
            status = "satisfied"

            contract = await self._resolve_planning_rubric_contract(
                llm=llm,
                caller_rubric=caller_rubric,
                user_message=message_content,
                candidate_todos=canonical_todos,
                existing_todos=todos,
                plan_modified=plan_modified,
                lifecycle="planning",
            )

            for iteration in range(max_iterations):
                structural_feedback = self._build_planning_structural_feedback(canonical_todos)
                if structural_feedback:
                    evaluation = PlanningRubricEvaluation(
                        iteration=iteration,
                        result="needs_revision",
                        explanation="Structural planning guardrail requires revision.",
                        criteria=[
                            {
                                "name": "structural_validity",
                                "passed": False,
                                "gap": structural_feedback,
                            }
                        ],
                    )
                else:
                    evaluation = await self._grade_planning_todos(
                        llm=llm,
                        rubric=contract.rubric,
                        user_message=message_content,
                        candidate_todos=canonical_todos,
                        existing_todos=todos,
                        plan_modified=plan_modified,
                        iteration=iteration,
                    )
                evaluations.append(evaluation)

                if evaluation.result == "satisfied":
                    status = "satisfied"
                    break
                if evaluation.result in {"failed", "grader_error"}:
                    status = evaluation.result
                    feedback = evaluation.explanation
                    break

                feedback = build_planning_rubric_feedback(evaluation)
                if iteration >= max_iterations - 1:
                    status = "max_iterations_reached"
                    break

                canonical_todos = await self._revise_todos_from_rubric_feedback(
                    llm=llm,
                    write_todos_tool=write_todos_tool,
                    base_system_prompt=system_prompt,
                    user_message=message_content,
                    base_todos=canonical_todos,
                    candidate_todos=canonical_todos,
                    feedback=feedback,
                )

            rubric_attempt = PlanningRubricAttempt(
                status=status,
                iterations=len(evaluations),
                source="modify_plan" if plan_modified else "generate_plan",
                rubric=contract.rubric,
                rubric_source=contract.source,
                rubric_rationale=contract.rationale,
                evaluations=evaluations,
                feedback=feedback,
                error=feedback if status == "grader_error" else None,
            )
```

Then attach metadata:

```python
        if rubric_attempt is not None:
            metadata["planning_rubric"] = rubric_attempt.metadata()
```

- [x] **Step 6: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_planning_agent_rubric.py tests/test_planning_rubric.py -q
```

Expected: rubric unit and PlanningAgent tests pass.

- [x] **Step 7: Commit**

Run:

```powershell
git add app/ai/planning_rubric.py app/ai/agents/planning_agent.py tests/test_planning_agent_rubric.py tests/test_planning_rubric.py
git commit -m "feat: grade generated planning todos"
```

---

### Task 4: Carry Service-Level Rubric Metadata Into Workflow Responses

**Files:**

- Modify: `app/services/message_service.py`
- Modify: `app/schemas/workflow.py`
- Modify: `app/ai/schemas.py`
- Modify: `app/ai/graph.py`
- Test: `tests/test_message_service_planning_rubric.py`
- Test: `tests/test_graph_planning_rubric.py`

> **STATUS: COMPLETE** (commit `b16af47`). Task 4 tests = 3 passed; `test_graph_planning_subagents.py` = 37 passed (no regression).
> Decisions: (1) No manual field mapping was needed in `AIService._to_ai_request()` — it already does a full `model_dump(mode="python")` -> `model_validate`, so the new `rubric_metadata` field on both `WorkflowPlanningContext` schemas propagates automatically. (2) `_attach_planning_state_metadata` is a classmethod, so the plan's test call `MultiAgentWorkflow._attach_planning_state_metadata(response, state)` binds `cls` implicitly and works unchanged.

- [x] **Step 1: Add tests for service metadata carrying**

Append to `tests/test_message_service_planning_rubric.py`:

```python
@pytest.mark.asyncio
async def test_prepare_planning_context_carries_created_plan_rubric_metadata():
    from types import SimpleNamespace
    from uuid import uuid4

    from app.services.message_service import MessageService

    conversation_id = uuid4()
    user_id = uuid4()

    class _TaskPlanService:
        def __init__(self):
            self.created = False

        def get_conversation_tasks(self, *_args, **_kwargs):
            if self.created:
                return [
                    SimpleNamespace(
                        id=uuid4(),
                        description="Implement native planning rubric evaluator",
                        status=SimpleNamespace(value="pending"),
                        task_order=0,
                    )
                ]
            return []

        async def create_task_plan(self, *_args, **_kwargs):
            self.created = True
            return [
                SimpleNamespace(
                    id=uuid4(),
                    description="Implement native planning rubric evaluator",
                    status=SimpleNamespace(value="pending"),
                    task_order=0,
                )
            ]

        def get_active_or_next_task(self, *_args, **_kwargs):
            return None

        def consume_last_planning_runtime_metadata(self):
            return {"planning_rubric": {"status": "satisfied", "iterations": 1}}

    service = MessageService.__new__(MessageService)
    service.task_plan_service = _TaskPlanService()

    result = await MessageService._prepare_planning_context(
        service,
        conversation_id=conversation_id,
        user_id=user_id,
        message_content="Plan the rubric work",
        planning_mode_enabled=True,
        plan_lifecycle=None,
    )

    assert result.rubric_metadata == {"status": "satisfied", "iterations": 1}
```

Create `tests/test_graph_planning_rubric.py`:

```python
from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


def test_attach_planning_state_metadata_includes_planning_rubric():
    response = AgentResponse(
        agent_type=AgentType.PLANNING,
        agent_id="planning_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content="Plan ready"),
        metadata={},
    )
    state = {
        "planning_call_count": 1,
        "context": {
            "planning_rubric": {
                "status": "satisfied",
                "iterations": 1,
                "evaluations": [],
            }
        },
    }

    enriched = MultiAgentWorkflow._attach_planning_state_metadata(response, state)

    assert enriched.metadata["planning_rubric"]["status"] == "satisfied"
```

- [x] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_message_service_planning_rubric.py tests/test_graph_planning_rubric.py -q
```

Expected: fail because planning rubric metadata is not carried through.

- [x] **Step 3: Store last runtime metadata in TaskPlanService**

In `app/services/task_plan_service.py`, initialize an instance field in `__init__`:

```python
        self._last_planning_runtime_metadata: dict[str, Any] = {}
```

After each `planning_result = await self.planning_runtime.generate_plan(...)` and `modify_plan(...)`, add:

```python
        self._last_planning_runtime_metadata = dict(planning_result.metadata or {})
```

Add method:

```python
    def consume_last_planning_runtime_metadata(self) -> dict[str, Any]:
        metadata = dict(self._last_planning_runtime_metadata or {})
        self._last_planning_runtime_metadata = {}
        return metadata
```

- [x] **Step 4: Carry metadata from MessageService into WorkflowPlanningContext**

In `app/services/message_service.py::_prepare_planning_context()`, after `create_task_plan()` or `modify_task_plan()` calls, read the metadata if available:

```python
                consume_metadata = getattr(
                    self.task_plan_service,
                    "consume_last_planning_runtime_metadata",
                    None,
                )
                if callable(consume_metadata):
                    runtime_metadata = consume_metadata()
                    rubric_metadata = runtime_metadata.get("planning_rubric")
                    if isinstance(rubric_metadata, dict):
                        result.rubric_metadata = rubric_metadata
```

- [x] **Step 5: Seed graph context with planning rubric metadata**

In `app/ai/graph.py::_create_initial_state()` or the method that builds `initial_state`, after `initial_state["plan_lifecycle"]` is set, add:

```python
        if request.planning.rubric_metadata:
            context = dict(initial_state.get("context") or {})
            context["planning_rubric"] = request.planning.rubric_metadata
            initial_state["context"] = context
```

- [x] **Step 6: Attach final metadata**

In `app/ai/graph.py::_attach_planning_state_metadata()`, copy the context field:

```python
        planning_rubric = context.get("planning_rubric")
        if isinstance(planning_rubric, dict) and planning_rubric:
            response.metadata["planning_rubric"] = make_json_safe(planning_rubric)
```

- [x] **Step 7: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_message_service_planning_rubric.py tests/test_graph_planning_rubric.py -q
```

Expected: tests pass.

- [x] **Step 8: Commit**

Run:

```powershell
git add app/services/task_plan_service.py app/services/message_service.py app/schemas/workflow.py app/ai/schemas.py app/ai/graph.py tests/test_message_service_planning_rubric.py tests/test_graph_planning_rubric.py
git commit -m "feat: surface planning rubric metadata"
```

---

### Task 5: Add Graph-Level Rubric Review For `write_todos` Mutations

**Files:**

- Modify: `app/ai/graph.py`
- Modify: `app/ai/agents/planning_agent.py`
- Test: `tests/test_graph_planning_rubric.py`

> **STATUS: COMPLETE** (commit `232b208`). Task 5 graph tests + `test_graph_planning_subagents.py` = 40 passed; `test_planning_agent_rubric.py` + `test_planning_rubric.py` + `test_planning_subagents.py` + `test_custom_agents_planning.py` = 75 passed (no regression).
> Decisions: (1) The `source` parameter on `_review_planning_todos_with_rubric` (graph) and `review_todos_with_planning_rubric` (agent) is now actually threaded through to the returned `PlanningRubricAttempt` instead of hardcoding the `"planning_tools"` literal — the plan's draft left `source` unused (a static-analysis hint); behavior is identical because the only caller passes `"planning_tools"`. (2) Steps 5 and 6 (set feedback on needs_revision / clear on satisfied) were merged into one review block in `_planning_tools_node`, with `max_iterations_reached` also clearing feedback so the graph loop terminates at the cap (matches the loop-prevention risk mitigation). (3) `planning_rubric_feedback` reaches `_build_system_prompt` via the existing `invoke_model_with_history(**system_prompt_kwargs)` forwarding — no signature change needed.

- [x] **Step 1: Add graph tests for revision feedback and satisfied pass**

Append to `tests/test_graph_planning_rubric.py`:

```python
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.ai.graph import MultiAgentWorkflow
from app.ai.planning_rubric import PlanningRubricAttempt, PlanningRubricEvaluation


@pytest.mark.asyncio
async def test_planning_tools_node_stores_rubric_feedback_for_plan_mutation(monkeypatch):
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.planning_agent = object()

    async def fake_ensure_map(*_args, **_kwargs):
        return {}

    monkeypatch.setattr("app.ai.graph.ensure_agent_tool_map", fake_ensure_map)

    async def fake_review(**_kwargs):
        return PlanningRubricAttempt(
            status="needs_revision",
            grading_run_id="run-1",
            iterations=1,
            source="planning_tools",
            rubric="- concrete_plan_scope",
            evaluations=[
                PlanningRubricEvaluation(
                    iteration=0,
                    result="needs_revision",
                    explanation="Task is vague.",
                    criteria=[
                        {
                            "name": "concrete_plan_scope",
                            "passed": False,
                            "gap": "Replace vague task with concrete behavior.",
                        }
                    ],
                )
            ],
            feedback="Replace vague task with concrete behavior.",
        )

    workflow._review_planning_todos_with_rubric = fake_review  # type: ignore[assignment]

    state = {
        "messages": [
            HumanMessage(content="make a plan"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "todo-1",
                        "name": "write_todos",
                        "args": {
                            "action": "set_todos",
                            "todos": [
                                {
                                    "id": "t1",
                                    "description": "Fix backend",
                                    "status": "pending",
                                    "order": 0,
                                }
                            ],
                        },
                    }
                ],
            ),
        ],
        "todos": [],
        "current_task_index": None,
        "planning_call_count": 0,
        "planning_mode_enabled": True,
        "planning_phase": "planning",
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "context": {},
    }

    result = await workflow._planning_tools_node(state)

    assert result["context"]["planning_rubric"]["status"] == "needs_revision"
    assert "Replace vague task" in result["context"]["planning_rubric_feedback"]
    assert result["context"]["plan_just_modified"] is False


def test_should_continue_planning_routes_back_for_rubric_feedback():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow.agents = {"planning_agent": object()}
    state = {
        "planning_call_count": 1,
        "planning_mode_enabled": True,
        "planning_phase": "planning",
        "todos": [{"id": "t1", "description": "Fix backend", "status": "pending", "order": 0}],
        "messages": [HumanMessage(content="plan"), AIMessage(content="Plan updated")],
        "context": {"planning_rubric_feedback": "Make task concrete."},
    }

    assert workflow._should_continue_planning(state) == "planning_agent"
```

- [x] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_graph_planning_rubric.py -q
```

Expected: fail because graph review is not implemented.

- [x] **Step 3: Add graph review helper**

In `app/ai/graph.py`, add:

```python
    async def _review_planning_todos_with_rubric(
        self,
        *,
        state: GraphState,
        todos: list[dict[str, Any]],
        source: str = "planning_tools",
    ) -> Any:
        if not getattr(settings, "planning_rubric_enabled", True):
            from .planning_rubric import FALLBACK_PLANNING_RUBRIC, PlanningRubricAttempt

            return PlanningRubricAttempt(
                status="disabled",
                iterations=0,
                source="planning_tools",
                rubric=FALLBACK_PLANNING_RUBRIC,
                rubric_source="fallback",
                evaluations=[],
            )

        # Reuse PlanningAgent's grader method so provider/model behavior stays centralized.
        context = GraphStateView(state).context_copy()
        previous = context.get("planning_rubric")
        previous_iterations = 0
        if (
            isinstance(previous, dict)
            and previous.get("source") == "planning_tools"
            and previous.get("status") == "needs_revision"
        ):
            try:
                previous_iterations = int(previous.get("iterations") or 0)
            except (TypeError, ValueError):
                previous_iterations = 0
        try:
            return await self.planning_agent.review_todos_with_planning_rubric(
                user_message=self._latest_user_text(state),
                candidate_todos=todos,
                existing_todos=state.get("all_tasks") or [],
                plan_modified=bool(state.get("has_existing_plan")),
                source="planning_tools",
                start_iteration=previous_iterations,
                user_id=state.get("user_id"),
                model_request=state.get("model_request"),
            )
        except Exception as exc:
            from .planning_rubric import FALLBACK_PLANNING_RUBRIC, PlanningRubricAttempt

            return PlanningRubricAttempt(
                status="grader_error",
                iterations=previous_iterations,
                source="planning_tools",
                rubric=FALLBACK_PLANNING_RUBRIC,
                rubric_source="fallback",
                evaluations=[],
                feedback=f"Planning rubric review failed: {exc}",
                error=f"Planning rubric review failed: {exc}",
            )
```

Add helper if one does not already exist:

```python
    @staticmethod
    def _latest_user_text(state: GraphState) -> str:
        for message in reversed(state.get("messages", []) or []):
            if isinstance(message, HumanMessage):
                return str(message.content or "")
        return ""
```

- [x] **Step 4: Add public PlanningAgent review method**

In `app/ai/agents/planning_agent.py`, add:

```python
    async def review_todos_with_planning_rubric(
        self,
        *,
        user_message: str,
        candidate_todos: list[dict[str, Any]],
        existing_todos: list[dict[str, Any]] | None,
        plan_modified: bool,
        source: str = "planning_tools",
        start_iteration: int = 0,
        user_id: str | None = None,
        model_request: dict[str, Any] | None = None,
    ) -> PlanningRubricAttempt:
        runtime_config = self._resolve_runtime_model_config(user_id, model_request)
        llm, _ = self._create_langchain_model_from_runtime(
            runtime_config,
            user_id=user_id,
            enable_reasoning_summary=False,
        )
        max_iterations = max(1, int(getattr(settings, "planning_rubric_max_iterations", 3)))
        evaluations: list[PlanningRubricEvaluation] = []
        feedback: str | None = None
        status = "satisfied"

        iteration = max(0, int(start_iteration or 0))
        if iteration >= max_iterations:
            return PlanningRubricAttempt(
                status="max_iterations_reached",
                iterations=iteration,
                source="planning_tools",
                rubric=FALLBACK_PLANNING_RUBRIC,
                rubric_source="fallback",
                rubric_rationale="Graph-level rubric pass cap was already reached.",
                evaluations=[],
                feedback=None,
            )

        contract = await self._resolve_planning_rubric_contract(
            llm=llm,
            caller_rubric=None,
            user_message=user_message,
            candidate_todos=candidate_todos,
            existing_todos=existing_todos,
            plan_modified=plan_modified,
            lifecycle="planning_tools",
        )
        evaluation = await self._grade_planning_todos(
            llm=llm,
            rubric=contract.rubric,
            user_message=user_message,
            candidate_todos=candidate_todos,
            existing_todos=existing_todos,
            plan_modified=plan_modified,
            iteration=iteration,
        )
        evaluations.append(evaluation)
        if evaluation.result == "satisfied":
            status = "satisfied"
        elif evaluation.result in {"failed", "grader_error"}:
            status = evaluation.result
            feedback = evaluation.explanation
        else:
            feedback = build_planning_rubric_feedback(evaluation)
            if iteration >= max_iterations - 1:
                status = "max_iterations_reached"
            else:
                status = "needs_revision"

        return PlanningRubricAttempt(
            status=status,
            iterations=iteration + 1,
            source="planning_tools",
            rubric=contract.rubric,
            rubric_source=contract.source,
            rubric_rationale=contract.rationale,
            evaluations=evaluations,
            feedback=feedback,
            error=feedback if status == "grader_error" else None,
        )
```

- [x] **Step 5: Invoke review after plan-mutating `write_todos` actions**

In `app/ai/graph.py::_planning_tools_node()`, after write actions update `state["todos"]`, add:

```python
        if any(action in plan_modifying_actions for action in write_todos_actions):
            rubric_attempt = await self._review_planning_todos_with_rubric(
                state=state,
                todos=todos,
                source="planning_tools",
            )
            if getattr(rubric_attempt, "status", None) != "disabled":
                context["planning_rubric"] = rubric_attempt.metadata()
            if getattr(rubric_attempt, "status", None) in {
                "needs_revision",
            } and getattr(rubric_attempt, "feedback", None):
                context["planning_rubric_feedback"] = rubric_attempt.feedback
                context["plan_just_modified"] = False
                context.pop("generate_plan_response", None)
            if getattr(rubric_attempt, "status", None) == "max_iterations_reached":
                context.pop("planning_rubric_feedback", None)
```

In `_should_continue_planning()`, add this check before the existing `plan_just_modified` branch so rubric feedback wins over the normal prose summary pass:

```python
        if context.get("planning_rubric_feedback"):
            logger.debug("[Should Continue Planning] Decision: planning_agent (rubric feedback)")
            return "planning_agent"
```

In `_planning_node()`, pass the feedback to the prompt:

```python
            planning_rubric_feedback=context.get("planning_rubric_feedback"),
```

In `PlanningAgent._build_system_prompt()`, append:

```python
        planning_rubric_feedback = kwargs.get("planning_rubric_feedback")
        if isinstance(planning_rubric_feedback, str) and planning_rubric_feedback.strip():
            prompt += (
                "\n\n# PLANNING RUBRIC FEEDBACK\n"
                "Revise the plan by calling write_todos. Do not answer in prose until "
                "the rubric feedback is resolved.\n"
                + planning_rubric_feedback.strip()
            )
```

- [x] **Step 6: Clear feedback after a new mutation**

In `_planning_tools_node()`, when a subsequent review returns `satisfied`, remove stale feedback:

```python
            if getattr(rubric_attempt, "status", None) == "satisfied":
                context.pop("planning_rubric_feedback", None)
```

- [x] **Step 7: Run graph tests**

Run:

```powershell
python -m pytest tests/test_graph_planning_rubric.py tests/test_graph_planning_subagents.py -q
```

Expected: rubric tests pass and existing Planning subagent tests remain green.

- [x] **Step 8: Commit**

Run:

```powershell
git add app/ai/graph.py app/ai/agents/planning_agent.py tests/test_graph_planning_rubric.py
git commit -m "feat: review planning tool mutations with rubric"
```

---

### Task 6: Preserve Rubric Metadata In Message Persistence

**Files:**

- Modify: `app/core/response_constants.py`
- Test: `tests/test_message_service_planning_rubric.py`

> **STATUS: COMPLETE** (commit `e7d03d0`). `test_message_service_planning_rubric.py` = 3 passed.
> Decision: No code change to `response_constants.py` was required (Step 3's conditional). `build_bot_metadata()` begins with `metadata = dict(response.metadata)` (response_constants.py:344), which already copies the `planning_rubric` key into persisted message metadata. Only the regression-guard test was added/committed.

- [x] **Step 1: Add message metadata test**

Append to `tests/test_message_service_planning_rubric.py`:

```python
def test_bot_metadata_preserves_planning_rubric():
    from app.core.response_constants import build_bot_metadata

    response = type(
        "Response",
        (),
        {
            "metadata": {
                "planning_rubric": {
                    "status": "satisfied",
                    "iterations": 1,
                    "evaluations": [],
                }
            },
            "tool_artifacts": None,
            "suggested_questions": None,
            "agent_id": "planning_agent",
        },
    )()

    metadata = build_bot_metadata(response, persona=None)

    assert metadata["planning_rubric"]["status"] == "satisfied"
```

- [x] **Step 2: Run test**

Run:

```powershell
python -m pytest tests/test_message_service_planning_rubric.py::test_bot_metadata_preserves_planning_rubric -q
```

Expected: pass because `build_bot_metadata()` currently copies unknown response metadata keys before rich-response finalization.

- [x] **Step 3: Implement preservation only if needed**

If the test fails, update `build_bot_metadata()` in `app/core/response_constants.py` so `planning_rubric` is copied from `bot_response.metadata` into persisted message metadata:

```python
    planning_rubric = response_metadata.get("planning_rubric")
    if isinstance(planning_rubric, dict) and planning_rubric:
        metadata["planning_rubric"] = planning_rubric
```

- [x] **Step 4: Run message-service rubric tests**

Run:

```powershell
python -m pytest tests/test_message_service_planning_rubric.py -q
```

Expected: tests pass.

- [x] **Step 5: Commit**

Run:

```powershell
git add app/core/response_constants.py tests/test_message_service_planning_rubric.py
git commit -m "test: preserve planning rubric metadata"
```

---

### Task 7: Documentation And Manual Validation

**Files:**

- Modify: `README.md`
- Modify: `planning_rubric.md`

> **STATUS: COMPLETE** (docs). README Planning section gained a "Planning Rubric Grading" subsection; the Manual Validation Checklist was appended to this plan (below). `test_demo_plan_widget.py` + `test_take100_api.py` = 18 passed.

- [x] **Step 1: Update README Planning section**

Add to the Planning Mode section:

```markdown
### Planning Rubric Grading

Planning mode includes a native rubric grader for generated and modified task plans.
For each planning attempt, the runtime resolves a context-specific rubric from the
user request, existing plan state, candidate todos, and optional caller-supplied
rubric metadata. The grader evaluates candidate `write_todos` output against that
rubric and returns actionable feedback. If the grader returns `needs_revision`,
the Planning Agent receives the feedback and revises the plan until it is satisfied
or `planning_rubric_max_iterations` is reached.

Rubric results are exposed on assistant message metadata under `planning_rubric`:

    {
      "status": "satisfied",
      "iterations": 1,
      "evaluations": [
        {
          "iteration": 0,
          "result": "satisfied",
          "explanation": "All criteria passed.",
          "criteria": [{"name": "request_fit", "passed": true}]
        }
      ]
    }
```

- [x] **Step 2: Add manual validation checklist**

Append this checklist to this plan after implementation:

```markdown
## Manual Validation Checklist

- Enable Planning mode on a new conversation.
- Send: "Create a plan to add native rubric grading to planning mode."
- Confirm the assistant response metadata contains `planning_rubric.status`.
- Send a plan-edit request that could produce a vague task, such as "make it shorter".
- Confirm vague tasks are revised or the metadata reports `max_iterations_reached`.
- Confirm existing completed tasks keep their IDs and completed status after a plan modification.
```

- [x] **Step 3: Run documentation-adjacent tests**

Run:

```powershell
python -m pytest tests/test_demo_plan_widget.py tests/test_take100_api.py -q
```

Expected: tests pass.

- [x] **Step 4: Commit**

Run:

```powershell
git add README.md planning_rubric.md
git commit -m "docs: document planning rubric grading"
```

---

### Task 8: Final Verification Matrix

**Files:**

- No code changes unless failures identify a regression.

> **STATUS: COMPLETE**. Verification matrix results:
> - Step 1 focused rubric suites (`test_planning_rubric` + `test_planning_agent_rubric` + `test_graph_planning_rubric` + `test_message_service_planning_rubric`): **16 passed**.
> - Step 2 planning regression (fallback set: `test_planning_subagents` + `test_graph_planning_subagents` + `test_custom_agents_planning` + `test_take100_api`; `test_task_plan_service.py` does not exist): **106 passed**.
> - Step 3 message-service + API regression (`test_message_service_subagent_streaming` + `test_custom_agents_message_service` + `test_take100_api`): **14 passed**.
> - Step 4 full suite (`python -m pytest -q`): **1010 passed, 1 failed**.
>
> **Documented pre-existing failure (unrelated):** `tests/client_backend/test_live_server_integration.py::test_live_document_upload_list_get_task_and_delete_flow` fails with `KeyError: 'document'` at line 472 — the live `/documents/upload` response payload's `data` dict lacks a `document` key. This is a live-server document-upload integration test in the documents domain. `git diff --name-only 9453eeb..HEAD` confirms none of the 7 rubric commits touched any documents/upload/rag file (only planning-domain modules, tests, and docs), so this failure is not a regression from this work. The other 6 tests in that live-integration file pass; the failure is environment/live-server dependent.

- [x] **Step 1: Run focused rubric suites**

Run:

```powershell
python -m pytest tests/test_planning_rubric.py tests/test_planning_agent_rubric.py tests/test_graph_planning_rubric.py tests/test_message_service_planning_rubric.py -q
```

Expected: all focused rubric tests pass.

- [x] **Step 2: Run existing Planning regression suites**

Run:

```powershell
python -m pytest tests/test_planning_subagents.py tests/test_graph_planning_subagents.py tests/test_custom_agents_planning.py tests/test_task_plan_service.py -q
```

Expected: all tests pass. If `tests/test_task_plan_service.py` does not exist, run:

```powershell
python -m pytest tests/test_take100_api.py tests/test_custom_agents_planning.py -q
```

- [x] **Step 3: Run message-service and API regression suites**

Run:

```powershell
python -m pytest tests/test_message_service_subagent_streaming.py tests/test_custom_agents_message_service.py tests/test_take100_api.py -q
```

Expected: all tests pass.

- [x] **Step 4: Run full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass, or any pre-existing unrelated failure is documented with the failing test name and reason.

- [x] **Step 5: Final commit**

Run:

```powershell
git status --short
git add app tests README.md planning_rubric.md
git commit -m "feat: add native planning rubric grading"
```

---

## Manual Validation Checklist

- Enable Planning mode on a new conversation.
- Send: "Create a plan to add native rubric grading to planning mode."
- Confirm the assistant response metadata contains `planning_rubric.status`.
- Send a plan-edit request that could produce a vague task, such as "make it shorter".
- Confirm vague tasks are revised or the metadata reports `max_iterations_reached`.
- Confirm existing completed tasks keep their IDs and completed status after a plan modification.

> Note: this checklist requires a live model provider (API key) and is not exercised by the
> automated suite, which stubs grader calls. Run it manually before shipping to production.

## Rollout Notes

- Default `planning_rubric_enabled=True` gives immediate quality coverage.
- Setting `planning_rubric_enabled=False` restores previous behavior.
- `planning_rubric_max_iterations` is the maximum number of grader passes per attempt; minimum value is 1.
- `planning_max_iterations` still controls Planning Agent graph loop budget; rubric iteration count is separate and stored under `planning_rubric.iterations`.
- If grader calls are too expensive, switch the default to `planning_rubric_enabled=False` and enable per environment.

## Risks And Mitigations

- **Risk:** Rubric grading increases latency during plan creation.
  - **Mitigation:** cap iterations at 3, allow disabling, and reuse existing provider retry settings.

- **Risk:** Grader feedback causes repeated plan rewrites.
  - **Mitigation:** separate rubric cap from Planning graph budget and stop with `max_iterations_reached`.

- **Risk:** Graph-level rubric feedback creates a loop with normal plan response generation.
  - **Mitigation:** store feedback in `GraphContext`, clear it on `satisfied`, and test `_should_continue_planning()`.

- **Risk:** Grader provider failure blocks plan creation.
  - **Mitigation:** represent provider failures as `grader_error` metadata and keep structural validation active.

## Success Metrics

- Focused rubric suites pass.
- Existing Planning subagent suite remains green.
- Assistant metadata includes `planning_rubric` for generated, modified, and in-chat mutated plans.
- Vague candidate plans receive at least one rubric feedback pass before finalization.
