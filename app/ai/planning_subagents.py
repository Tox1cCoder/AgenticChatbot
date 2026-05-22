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
import copy
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


class SubagentModelOverride(BaseModel):
    """Per-subagent-task model override.

    The Planning Agent uses this to attach a concrete model assignment to a
    single worker task without mutating parent or sibling routing. Values are
    request-scoped and are NEVER persisted to ``agent_model_configs``.
    """

    provider: Literal["gemini", "openai"] | None = Field(
        default=None,
        description="Provider to use; inferred from the model id when omitted.",
    )
    model: str = Field(..., description="Concrete provider model id (e.g. 'gpt-5.4').")
    temperature: float | None = Field(
        default=None,
        description="Optional override for sampling temperature (0.0 - 2.0).",
    )
    allow_custom_model: bool = Field(
        default=True,
        description="Allow models outside the synced catalog (subagent overrides are explicit).",
    )
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = Field(
        default=None,
        description=(
            "Provider-agnostic reasoning intensity. Normalized to OpenAI "
            "reasoning.effort or Gemini thinking_level per provider rules."
        ),
    )

    @field_validator("model")
    @classmethod
    def _model_must_be_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("SubagentModelOverride.model must be non-empty.")
        return value.strip()

    @field_validator("temperature")
    @classmethod
    def _temperature_within_supported_range(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not 0.0 <= float(value) <= 2.0:
            raise ValueError("SubagentModelOverride.temperature must be between 0.0 and 2.0.")
        return float(value)


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
    model_override: SubagentModelOverride | None = Field(
        default=None,
        description=(
            "Optional task-local model override. Affects only this worker; does not"
            " mutate parent state or sibling workers."
        ),
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

    @field_validator("context", mode="before")
    @classmethod
    def _coerce_context_to_dict(cls, value: Any) -> Any:
        """Permissive shape for ``context``.

        The Planning Agent occasionally passes raw scraped/crawled text or a
        bare list of references instead of a JSON object. Rather than failing
        the whole dispatch call, wrap recognized primitives into a dict so the
        worker still receives the structured data:

        - ``None`` / missing -> ``{}``
        - already a dict -> returned as-is
        - ``str`` -> ``{"text": value}``
        - ``list`` / ``tuple`` -> ``{"items": list(value)}``

        Other primitive types (int, float, bool) still fall through to the
        normal pydantic dict validator and surface a clear ValidationError.
        """
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return {"text": value}
        if isinstance(value, (list, tuple)):
            return {"items": list(value)}
        return value

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
    requested_model: dict[str, Any] | None = Field(
        default=None,
        description="Compact view of the requested task-local model override, when set.",
    )
    resolved_model: dict[str, Any] | None = Field(
        default=None,
        description="Compact view of the model that actually answered the worker call.",
    )


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
        model_override: SubagentModelOverride | None = None,
    ) -> AgentResponse: ...


# ---------------------------------------------------------------------------
# Model-request override helpers
# ---------------------------------------------------------------------------


_AGENT_NAME_TO_KEY = {
    "chat_agent": "chat",
    "rag_agent": "rag",
    "search_agent": "search",
    "image_generator_agent": "image_generator",
    "canvas_agent": "canvas",
}


def _infer_provider_from_model_id(model_id: str) -> str | None:
    """Best-effort provider inference for obvious model id prefixes."""
    if not model_id:
        return None
    lowered = model_id.strip().lower()
    if lowered.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if lowered.startswith("gemini"):
        return "gemini"
    return None


def build_worker_model_request(
    *,
    parent_model_request: dict[str, Any] | None,
    agent_key: str,
    override: SubagentModelOverride | None,
) -> dict[str, Any] | None:
    """Compose a worker-local ``model_request`` from parent + task override.

    - Returns a deep copy of ``parent_model_request`` (or ``None`` if no parent
      and no override).
    - When ``override`` is provided, overlays its sanitized payload on the
      target ``agent_key`` only; sibling agent entries and ``all`` are
      preserved untouched.
    - Never mutates the input ``parent_model_request``.
    """
    request: dict[str, Any] = copy.deepcopy(parent_model_request) if parent_model_request else {}

    if override is None:
        return request or None

    payload = override.model_dump(exclude_none=True)
    if not payload.get("provider"):
        inferred = _infer_provider_from_model_id(override.model)
        if inferred:
            payload["provider"] = inferred

    request[agent_key] = payload
    return request


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
    """Return the model/UI activity payload without nested worker artifacts.

    ``requested_model``/``resolved_model`` are kept so the supervisor (and the
    UI) can see exactly which model answered each worker, but the underlying
    runtime API key never makes it into ``PlanningSubagentResult`` and so it
    cannot leak here.
    """

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


def _summarize_requested_model(
    override: SubagentModelOverride | None,
) -> dict[str, Any] | None:
    if override is None:
        return None
    return override.model_dump(exclude_none=True)


# Worker response metadata keys carried back as ``resolved_model`` for the
# Planning Agent / UI. API keys, raw runtime config objects, and warnings are
# intentionally excluded — supervisors only need provider/model/effort.
_RESOLVED_MODEL_FIELDS = (
    "provider",
    "model",
    "config_source",
    "reasoning_effort",
    "context_window",
)


def _summarize_resolved_model(response: AgentResponse) -> dict[str, Any] | None:
    metadata = response.metadata or {}
    snapshot: dict[str, Any] = {}
    for key in _RESOLVED_MODEL_FIELDS:
        value = metadata.get(key)
        if value not in (None, ""):
            snapshot[key] = value
    return snapshot or None


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

        requested_model = _summarize_requested_model(task.model_override)

        def _result(
            status: Literal["completed", "failed", "timeout", "requires_approval"],
            summary: str,
            error: str | None = None,
            artifacts: list[dict[str, Any]] | None = None,
            resolved_model: dict[str, Any] | None = None,
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
                requested_model=requested_model,
                resolved_model=resolved_model,
            )

        prompt = _build_task_prompt(task)
        try:
            response = await self._workflow._run_agent_in_isolated_context(
                agent_name=task.agent.value,
                task_prompt=prompt,
                parent_state=parent_state,
                related_todo_ids=list(task.related_todo_ids),
                model_override=task.model_override,
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
        resolved_model = _summarize_resolved_model(response)

        if response.error:
            return _result(
                "failed",
                response.message.content or response.error or "(no output)",
                error=response.error,
                artifacts=worker_artifacts,
                resolved_model=resolved_model,
            )

        if _is_requires_approval_response(response):
            return _result(
                "requires_approval",
                response.message.content
                or "Worker stopped awaiting human approval; supervisor must handle directly.",
                error="requires_approval",
                artifacts=worker_artifacts,
                resolved_model=resolved_model,
            )

        return _result(
            "completed",
            response.message.content or "",
            artifacts=worker_artifacts,
            resolved_model=resolved_model,
        )


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
            dispatch_entry: dict[str, Any] = {
                "rationale": rationale,
                "task_ids": [task.id for task in request.tasks],
                "agents": [task.agent.value for task in request.tasks],
                "status": result.status,
            }

            # Per-task model summary so the response/UI can show which model
            # answered each worker. Keyed by task id; entries only contain
            # ``requested``/``resolved`` when at least one is set (no API keys,
            # no warnings, no runtime fallback config).
            models_by_task: dict[str, dict[str, Any]] = {}
            for entry in result.results:
                model_entry: dict[str, Any] = {}
                if entry.requested_model:
                    model_entry["requested"] = entry.requested_model
                if entry.resolved_model:
                    model_entry["resolved"] = entry.resolved_model
                if model_entry:
                    models_by_task[entry.id] = model_entry
            if models_by_task:
                dispatch_entry["models"] = models_by_task

            dispatches.append(dispatch_entry)
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
