"""Planning-mode subagent dispatcher.

This module implements the supervisor-side primitive that lets the Planning
Agent fan out *independent* worker tasks to existing graph agents inside
the same chat turn. Workers run concurrently with a bounded semaphore,
each in an isolated execution context, and only structured summaries are
returned to the Planning Agent through the ``dispatch_subagents`` tool.

The dispatcher is **NOT** a generic parallel tool runner: it is a
purpose-built supervisor that enforces ordered output, per-worker
timeout, and result-size truncation. Generic ``execute_tool_calls()``
behavior must remain sequential.
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
        max_tasks = int(getattr(global_settings, "planning_subagents_max_tasks", 5))
        if not self.tasks:
            raise ValueError("dispatch_subagents requires at least one task.")
        if len(self.tasks) > max_tasks:
            raise ValueError(
                f"dispatch_subagents accepts at most {max_tasks} tasks per call; "
                f"received {len(self.tasks)}."
            )
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
    """Run Planning-supervised worker tasks concurrently with bounded fan-out."""

    def __init__(
        self,
        *,
        workflow: _IsolatedAgentRunner,
        settings: Any | None = None,
        default_timeout_seconds: float | None = None,
    ) -> None:
        self._workflow = workflow
        self._settings = settings or global_settings
        self._default_timeout = default_timeout_seconds

    def _max_parallel(self) -> int:
        return max(1, int(getattr(self._settings, "planning_subagents_max_parallel", 3)))

    def _result_max_chars(self) -> int:
        return int(getattr(self._settings, "planning_subagents_result_max_chars", 6000))

    def _timeout_seconds(self) -> float:
        if self._default_timeout is not None:
            return float(self._default_timeout)
        return float(getattr(self._settings, "planning_subagents_worker_timeout_seconds", 120))

    async def dispatch(
        self,
        request: DispatchSubagentsInput,
        *,
        parent_state: dict[str, Any],
    ) -> DispatchSubagentsResult:
        """Run all worker tasks concurrently and return ordered results."""
        semaphore = asyncio.Semaphore(self._max_parallel())
        timeout = self._timeout_seconds()

        async def _bounded(task: PlanningSubagentTask) -> PlanningSubagentResult:
            async with semaphore:
                return await self.run_one(task, parent_state=parent_state, timeout=timeout)

        coros = [_bounded(task) for task in request.tasks]
        results = await asyncio.gather(*coros, return_exceptions=False)
        return DispatchSubagentsResult.from_results(list(results), rationale=request.rationale)

    async def run_one(
        self,
        task: PlanningSubagentTask,
        *,
        parent_state: dict[str, Any],
        timeout: float | None = None,
    ) -> PlanningSubagentResult:
        """Run a single worker, converting exceptions/timeouts into structured results."""
        wall_start = time.perf_counter()
        timeout = timeout if timeout is not None else self._timeout_seconds()
        max_chars = self._result_max_chars()

        def _result(
            status: Literal["completed", "failed", "timeout", "requires_approval"],
            summary: str,
            error: str | None = None,
        ) -> PlanningSubagentResult:
            return PlanningSubagentResult(
                id=task.id,
                agent=task.agent,
                status=status,
                elapsed_ms=int((time.perf_counter() - wall_start) * 1000),
                summary=_truncate_summary(summary, max_chars),
                related_todo_ids=list(task.related_todo_ids),
                error=error,
            )

        prompt = _build_task_prompt(task)
        try:
            response = await asyncio.wait_for(
                self._workflow._run_agent_in_isolated_context(
                    agent_name=task.agent.value,
                    task_prompt=prompt,
                    parent_state=parent_state,
                    related_todo_ids=list(task.related_todo_ids),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return _result(
                "timeout",
                f"Worker {task.id} ({task.agent.value}) exceeded {timeout:.1f}s timeout.",
                error="timeout",
            )
        except Exception as exc:  # pragma: no cover - sanity net
            logger.warning("Subagent worker %s raised: %s", task.id, exc)
            return _result(
                "failed",
                f"Worker {task.id} ({task.agent.value}) failed: {exc}",
                error=str(exc),
            )

        if response.error:
            return _result(
                "failed",
                response.message.content or response.error or "(no output)",
                error=response.error,
            )

        if _is_requires_approval_response(response):
            return _result(
                "requires_approval",
                response.message.content
                or "Worker stopped awaiting human approval; supervisor must handle directly.",
                error="requires_approval",
            )

        return _result("completed", response.message.content or "")


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
        except Exception:  # pragma: no cover - defensive logging path
            logger.debug("Failed to stash subagent dispatch summary on parent state.")

        return json.dumps(compact_payload)

    return StructuredTool.from_function(
        coroutine=_dispatch,
        name="dispatch_subagents",
        description=_DISPATCH_TOOL_DESCRIPTION,
        args_schema=DispatchSubagentsInput,
    )
