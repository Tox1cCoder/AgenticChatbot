"""Planning-mode subagent dispatcher.

This module implements the supervisor-side primitive that lets the Planning
Agent fan out *independent* worker tasks to existing graph agents inside
the same chat turn. Workers run concurrently, each in an isolated execution
context, and only structured summaries are returned to the Planning Agent
through the ``dispatch_subagents`` tool.

The dispatcher is **NOT** a generic parallel tool runner: it is a
purpose-built supervisor that enforces ordered output and keeps generic
``execute_tool_calls()`` behavior sequential.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal, Protocol

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator, model_validator

from ..core.config import settings as global_settings
from .schemas import AgentResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class PlanningSubagentName(str, Enum):
    """Worker agents the Planning supervisor is allowed to dispatch to.

    ``planning_agent`` is intentionally excluded to prevent recursive
    planning supervisors.
    """

    CHAT_AGENT = "chat_agent"
    RAG_AGENT = "rag_agent"
    SEARCH_AGENT = "search_agent"
    IMAGE_GENERATOR_AGENT = "image_generator_agent"
    CANVAS_AGENT = "canvas_agent"


class PlanningSubagentTask(BaseModel):
    """One independent worker task the Planning Agent wants to dispatch."""

    id: str = Field(..., description="Stable identifier for correlating results back to a todo.")
    agent: PlanningSubagentName = Field(..., description="Worker agent to execute this task.")
    task: str = Field(..., description="Concrete instructions for the worker.")
    related_todo_ids: list[str] = Field(
        default_factory=list,
        description="Optional advisory list of todo ids this worker is helping complete.",
    )
    expected_output: str | None = Field(
        default=None,
        description="Optional hint about the shape/format the supervisor wants back.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional JSON-serializable context the supervisor wants the worker to see.",
    )

    @field_validator("id")
    @classmethod
    def _id_must_be_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("PlanningSubagentTask.id must be non-empty.")
        return value.strip()

    @field_validator("task")
    @classmethod
    def _task_must_be_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("PlanningSubagentTask.task must be a non-empty instruction.")
        return value.strip()

    @field_validator("context")
    @classmethod
    def _context_must_be_json_serializable(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PlanningSubagentTask.context must be JSON-serializable: {exc}")
        return value


class DispatchSubagentsInput(BaseModel):
    """Input schema for the ``dispatch_subagents`` Planning tool."""

    tasks: list[PlanningSubagentTask] = Field(
        ...,
        description="Independent worker tasks to dispatch in parallel.",
    )
    rationale: str | None = Field(
        default=None,
        description="Optional reason explaining why these tasks are independent.",
    )

    @model_validator(mode="after")
    def _validate_task_count_and_uniqueness(self) -> DispatchSubagentsInput:
        if not self.tasks:
            raise ValueError("dispatch_subagents requires at least one task.")
        seen: set[str] = set()
        for task in self.tasks:
            if task.id in seen:
                raise ValueError(f"Duplicate task id in dispatch_subagents call: {task.id!r}.")
            seen.add(task.id)
        return self


class PlanningSubagentResult(BaseModel):
    """Structured outcome of a single worker execution."""

    id: str
    agent: PlanningSubagentName
    status: Literal["completed", "failed", "timeout", "requires_approval"]
    elapsed_ms: int
    summary: str
    related_todo_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    images: list[dict[str, Any]] = Field(default_factory=list)


class DispatchSubagentsResult(BaseModel):
    """Aggregate dispatch result returned to the Planning Agent."""

    status: Literal["completed", "partial", "failed"]
    results: list[PlanningSubagentResult]
    rationale: str | None = None

    @classmethod
    def from_results(
        cls,
        results: list[PlanningSubagentResult],
        *,
        rationale: str | None = None,
    ) -> DispatchSubagentsResult:
        if not results:
            return cls(status="failed", results=[], rationale=rationale)
        completed = sum(1 for r in results if r.status == "completed")
        if completed == len(results):
            status: Literal["completed", "partial", "failed"] = "completed"
        elif completed == 0:
            status = "failed"
        else:
            status = "partial"
        return cls(status=status, results=results, rationale=rationale)


# ---------------------------------------------------------------------------
# Workflow protocol — narrows the dispatcher's coupling to MultiAgentWorkflow.
# ---------------------------------------------------------------------------


class _IsolatedAgentRunner(Protocol):
    async def _run_agent_in_isolated_context(
        self,
        *,
        agent_name: str,
        task_prompt: str,
        parent_state: dict[str, Any],
        related_todo_ids: list[str] | None = None,
    ) -> AgentResponse: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TRUNCATION_NOTICE = "\n…[truncated]"


def _truncate_summary(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(_TRUNCATION_NOTICE))
    return text[:keep] + _TRUNCATION_NOTICE


def _compact_result_payload(result: PlanningSubagentResult) -> dict[str, Any]:
    """Return the model/UI activity payload without nested worker artifacts."""

    return result.model_dump(
        mode="json",
        exclude={"artifacts", "images"},
        exclude_none=True,
    )


def _compact_dispatch_payload(result: DispatchSubagentsResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": result.status,
        "results": [_compact_result_payload(entry) for entry in result.results],
    }
    if result.rationale is not None:
        payload["rationale"] = result.rationale
    return payload


def _build_task_prompt(task: PlanningSubagentTask) -> str:
    parts: list[str] = [task.task.strip()]
    if task.expected_output:
        parts.append("\nExpected output:")
        parts.append(task.expected_output.strip())
    if task.context:
        try:
            ctx_json = json.dumps(task.context, indent=2, default=str)
        except (TypeError, ValueError):
            ctx_json = str(task.context)
        parts.append("\nContext (JSON):")
        parts.append(ctx_json)
    return "\n".join(parts).strip()


def _is_requires_approval_response(response: AgentResponse) -> bool:
    metadata = response.metadata or {}
    if metadata.get("requires_approval") is True:
        return True
    return metadata.get("pause_reason") == "awaiting_approval"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class PlanningSubagentDispatcher:
    """Run Planning-supervised worker tasks concurrently."""

    def __init__(
        self,
        *,
        workflow: _IsolatedAgentRunner,
        settings: Any | None = None,
    ) -> None:
        self._workflow = workflow
        self._settings = settings or global_settings

    def _result_max_chars(self) -> int:
        return int(getattr(self._settings, "tool_result_max_chars", 0) or 0)

    async def dispatch(
        self,
        request: DispatchSubagentsInput,
        *,
        parent_state: dict[str, Any],
    ) -> DispatchSubagentsResult:
        """Run all worker tasks concurrently and return ordered results."""
        coros = [self.run_one(task, parent_state=parent_state) for task in request.tasks]
        results = await asyncio.gather(*coros, return_exceptions=False)
        return DispatchSubagentsResult.from_results(list(results), rationale=request.rationale)

    async def run_one(
        self,
        task: PlanningSubagentTask,
        *,
        parent_state: dict[str, Any],
    ) -> PlanningSubagentResult:
        """Run a single worker, converting exceptions/timeouts into structured results."""
        wall_start = time.perf_counter()
        max_chars = self._result_max_chars()

        def _result(
            status: Literal["completed", "failed", "timeout", "requires_approval"],
            summary: str,
            error: str | None = None,
            artifacts: list[dict[str, Any]] | None = None,
        ) -> PlanningSubagentResult:
            return PlanningSubagentResult(
                id=task.id,
                agent=task.agent,
                status=status,
                elapsed_ms=int((time.perf_counter() - wall_start) * 1000),
                summary=_truncate_summary(summary, max_chars),
                related_todo_ids=list(task.related_todo_ids),
                error=error,
                artifacts=list(artifacts or []),
            )

        prompt = _build_task_prompt(task)
        try:
            response = await self._workflow._run_agent_in_isolated_context(
                agent_name=task.agent.value,
                task_prompt=prompt,
                parent_state=parent_state,
                related_todo_ids=list(task.related_todo_ids),
            )
        except asyncio.TimeoutError:
            return _result(
                "timeout",
                f"Worker {task.id} ({task.agent.value}) timed out in an underlying operation.",
                error="timeout",
            )
        except Exception as exc:  # pragma: no cover - sanity net
            logger.warning("Subagent worker %s raised: %s", task.id, exc)
            return _result(
                "failed",
                f"Worker {task.id} ({task.agent.value}) failed: {exc}",
                error=str(exc),
            )

        worker_artifacts = list(response.tool_artifacts or [])

        if response.error:
            return _result(
                "failed",
                response.message.content or response.error or "(no output)",
                error=response.error,
                artifacts=worker_artifacts,
            )

        if _is_requires_approval_response(response):
            return _result(
                "requires_approval",
                response.message.content
                or "Worker stopped awaiting human approval; supervisor must handle directly.",
                error="requires_approval",
                artifacts=worker_artifacts,
            )

        return _result("completed", response.message.content or "", artifacts=worker_artifacts)


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


_DISPATCH_TOOL_DESCRIPTION = (
    "Dispatch independent worker tasks to other graph agents during PLANNING "
    "execution. Workers run in parallel and the call blocks until every "
    "worker completes, fails, times out, or signals it needs human approval. "
    "Workers receive their own isolated context — they do NOT see the "
    "Planning Agent's chat history and they CANNOT update todos directly. "
    "After this tool returns, the Planning Agent must read each result and "
    "call `write_todos` to mark the related todos completed (or leave them "
    "pending with a clear blocker if the worker failed). Use only for "
    "INDEPENDENT tasks — dependent tasks must be executed sequentially."
)


def create_dispatch_subagents_tool(
    *,
    dispatcher: PlanningSubagentDispatcher | None = None,
    parent_state_provider: Callable[[], dict[str, Any]] | None = None,
) -> StructuredTool:
    """Build the LangChain ``dispatch_subagents`` tool.

    The closure captures ``dispatcher`` and a ``parent_state_provider`` so
    the tool can fetch the current Planning state at call time without
    circular references between the workflow and the tool.
    """

    async def _dispatch(
        tasks: list[PlanningSubagentTask],
        rationale: str | None = None,
    ) -> str:
        if dispatcher is None or parent_state_provider is None:
            raise RuntimeError("dispatch_subagents is not executable in this context.")

        request = DispatchSubagentsInput(tasks=tasks, rationale=rationale)
        parent_state = parent_state_provider()
        if parent_state is None:
            parent_state = {}
        result = await dispatcher.dispatch(request, parent_state=parent_state)
        compact_payload = _compact_dispatch_payload(result)

        # Stash a compact summary on parent state so the graph integration
        # can attach `subagent_dispatches`/`subagent_results` to GraphContext
        # and the final AgentResponse metadata. This is best-effort — the
        # canonical model-facing payload is the JSON returned below.
        try:
            context = parent_state.get("context")
            if not isinstance(context, dict):
                context = {}
                parent_state["context"] = context
            dispatches = list(context.get("subagent_dispatches") or [])
            dispatches.append(
                {
                    "rationale": rationale,
                    "task_ids": [task.id for task in request.tasks],
                    "agents": [task.agent.value for task in request.tasks],
                    "status": result.status,
                }
            )
            context["subagent_dispatches"] = dispatches

            results_log = list(context.get("subagent_results") or [])
            for entry in result.results:
                results_log.append(_compact_result_payload(entry))
            context["subagent_results"] = results_log

            # Surface worker tool artifacts in the UI activity panel without
            # bloating the model context: the model JSON above stays compact,
            # but a dedicated metadata bucket carries the full per-worker
            # artifact lists so the Streamlit renderer can expand them.
            worker_artifacts_log = dict(context.get("subagent_worker_artifacts") or {})
            for entry in result.results:
                if not entry.artifacts:
                    continue
                existing = list(worker_artifacts_log.get(entry.id) or [])
                seen_ids = {
                    art.get("tool_call_id")
                    for art in existing
                    if isinstance(art, dict) and art.get("tool_call_id")
                }
                for artifact in entry.artifacts:
                    if not isinstance(artifact, dict):
                        continue
                    tc_id = artifact.get("tool_call_id")
                    if tc_id and tc_id in seen_ids:
                        continue
                    existing.append(artifact)
                    if tc_id:
                        seen_ids.add(tc_id)
                worker_artifacts_log[entry.id] = existing
            if worker_artifacts_log:
                context["subagent_worker_artifacts"] = worker_artifacts_log
        except Exception:  # pragma: no cover - defensive logging path
            logger.debug("Failed to stash subagent dispatch summary on parent state.")

        return json.dumps(compact_payload)

    return StructuredTool.from_function(
        coroutine=_dispatch,
        name="dispatch_subagents",
        description=_DISPATCH_TOOL_DESCRIPTION,
        args_schema=DispatchSubagentsInput,
    )
