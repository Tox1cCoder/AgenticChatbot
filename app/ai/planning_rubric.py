"""Native Planning-mode rubric runtime.

Mirrors the LangChain DeepAgents rubric behavior for Planning mode without
adding the ``deepagents`` dependency: a context-specific rubric is resolved per
attempt, candidate todo plans are graded against it, and ``needs_revision``
feedback is injected back into the planning loop until the rubric is satisfied
or an iteration cap is reached.

This module holds the Pydantic schemas, prompt builders, and pure parsing
helpers. The LLM grader calls and the revision loop live on ``PlanningAgent``.
"""

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
