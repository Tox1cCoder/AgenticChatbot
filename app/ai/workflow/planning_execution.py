"""Planning as an orchestrator over isolated workers.

Planning owns delegation and synthesis. Independent tasks fan out through
LangGraph ``Send``; each worker runs in its own per-invocation subgraph and
returns a typed private result.

Three things a worker deliberately cannot do: publish a public assistant
message, perform a parent-level handoff, or recurse into Planning. All three
would take a decision that belongs to the orchestrator and hide it inside a
branch.

Worker objectives and results are delimited as untrusted data. A worker result
is content the orchestrator reads, never instructions it follows.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from langgraph.types import Send
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, TypedDict

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.contracts import OutcomeProvenance, ResponseOutcome, WorkerResult

logger = logging.getLogger(__name__)

__all__ = [
    "PLANNING_AGENT_ID",
    "PlanningLimits",
    "PlanningOrchestrator",
    "PlanningState",
    "WorkerTask",
    "collect_worker_results",
    "dispatch_workers",
]

PLANNING_AGENT_ID = "planning_agent"
RAG_AGENT_ID = "rag_agent"

_UNTRUSTED_OPEN = "BEGIN UNTRUSTED WORKER RESULT"
_UNTRUSTED_CLOSE = "END UNTRUSTED WORKER RESULT"


class PlanningLimits(BaseModel):
    """Bounds on one Planning turn. Every value is validated positive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tasks: int = Field(gt=0, le=64)
    max_concurrency: int = Field(gt=0, le=32)
    objective_max_chars: int = Field(gt=0, le=64_000)
    parent_context_max_chars: int = Field(gt=0, le=200_000)

    @classmethod
    def from_settings(cls, settings: Any) -> PlanningLimits:
        return cls(
            max_tasks=int(getattr(settings, "planning_worker_max_tasks", 8)),
            max_concurrency=int(getattr(settings, "planning_worker_max_concurrency", 4)),
            objective_max_chars=int(
                getattr(settings, "planning_worker_objective_max_chars", 4000)
            ),
            parent_context_max_chars=int(
                getattr(settings, "planning_parent_context_max_chars", 12000)
            ),
        )


class WorkerTask(BaseModel):
    """One independent unit of delegated work."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1, max_length=160)
    objective: str
    agent_id: str = Field(min_length=1, max_length=160)
    allowed_tool_ids: tuple[str, ...] = ()
    model_request: dict[str, Any] | None = None
    related_todo_ids: tuple[str, ...] = ()


class PlanningState(TypedDict):
    """Planning-graph state. Worker messages never appear here."""

    worker_tasks: list[WorkerTask]
    limits: PlanningLimits
    runtime_request: dict[str, Any]
    worker_results: NotRequired[list[WorkerResult]]
    planning_result: NotRequired[Any]


def dispatch_workers(state: PlanningState) -> list[Send]:
    """Fan independent tasks out to per-invocation worker subgraphs.

    Bounding happens here rather than inside the worker so an over-long plan is
    truncated once, visibly, instead of each worker discovering its own limit.
    """
    limits = state["limits"]
    tasks = list(state["worker_tasks"])

    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            raise ValueError(f"duplicate worker task id: {task.task_id!r}")
        seen.add(task.task_id)

    bounded = tasks[: limits.max_tasks]
    if len(tasks) > len(bounded):
        logger.info(
            "Planning dispatch bounded to %d of %d tasks", len(bounded), len(tasks)
        )

    return [
        Send(
            "worker",
            {
                "task": task.model_copy(
                    update={"objective": task.objective[: limits.objective_max_chars]}
                ),
                "runtime_request": state["runtime_request"],
            },
        )
        for task in bounded
    ]


def collect_worker_results(
    tasks: Sequence[WorkerTask], results: Sequence[WorkerResult]
) -> list[WorkerResult]:
    """Order results by original task position, not completion order.

    Synthesis reads the plan in the order it was written; completion order is
    an accident of latency and would make the same plan synthesize differently
    on different runs.
    """
    by_task_id = {result.task_id: result for result in results}
    return [by_task_id[task.task_id] for task in tasks if task.task_id in by_task_id]


class PlanningOrchestrator:
    """Runs one Planning turn's workers and synthesizes their results."""

    def __init__(
        self,
        *,
        specialist_factory: Any,
        rag_execution_factory: Any,
        limits: PlanningLimits,
    ) -> None:
        self._specialist_factory = specialist_factory
        self._rag_execution_factory = rag_execution_factory
        self._limits = limits

    async def run_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute one dispatched task and return only its typed result."""
        task: WorkerTask = payload["task"]
        runtime_request: dict[str, Any] = payload.get("runtime_request") or {}

        if task.agent_id == PLANNING_AGENT_ID:
            return {"worker_results": [self._failed(task, "recursive_planning")]}

        try:
            if task.agent_id == RAG_AGENT_ID:
                result = await self._run_rag_worker(task, runtime_request)
            else:
                result = await self._run_specialist_worker(task, runtime_request)
        except TimeoutError:
            return {"worker_results": [self._failed(task, "worker_timeout")]}
        except Exception as exc:  # noqa: BLE001 - normalized into a typed result
            logger.warning("Planning worker %s failed: %s", task.task_id, exc)
            return {"worker_results": [self._failed(task, "tool_execution_failed")]}

        return {"worker_results": [result]}

    async def _run_specialist_worker(
        self, task: WorkerTask, runtime_request: dict[str, Any]
    ) -> WorkerResult:
        from app.ai.workflow.specialists import SpecialistRequest

        request = SpecialistRequest(
            agent_id=task.agent_id,
            conversation_id=runtime_request.get("conversation_id"),
            user_id=runtime_request.get("user_id"),
            device_id=runtime_request.get("device_id"),
            persona=runtime_request.get("persona"),
            model_request=task.model_request or runtime_request.get("model_request"),
            messages=[],
            history=[],
            state={},
            extras={"objective": task.objective, "allowed_tool_ids": task.allowed_tool_ids},
        )
        return await self._specialist_factory.invoke_worker(request, task_id=task.task_id)

    async def _run_rag_worker(
        self, task: WorkerTask, runtime_request: dict[str, Any]
    ) -> WorkerResult:
        """RAG workers use the same graph and grounding policy as top-level RAG."""
        from app.ai.workflow.rag_execution import RagExecutionRequest

        run = self._rag_execution_factory.build()
        result = await run.ainvoke(
            RagExecutionRequest(
                objective=task.objective,
                conversation_id=runtime_request.get("conversation_id"),
                user_id=runtime_request.get("user_id"),
                device_id=runtime_request.get("device_id"),
                model_request=task.model_request or runtime_request.get("model_request"),
                allowed_tool_ids=task.allowed_tool_ids,
                mode="worker",
                task_id=task.task_id,
            )
        )
        return WorkerResult(
            task_id=task.task_id,
            agent_id=task.agent_id,
            status="completed",
            content=str(getattr(result, "content", "")),
            evidence=tuple(getattr(result, "evidence", ()) or ()),
            artifacts=tuple(getattr(result, "artifacts", ()) or ()),
        )

    @staticmethod
    def _failed(task: WorkerTask, error_code: str) -> WorkerResult:
        return WorkerResult(
            task_id=task.task_id,
            agent_id=task.agent_id,
            status="failed",
            content="",
            error_code=error_code,
        )

    # -- synthesis -------------------------------------------------------

    async def synthesize(
        self,
        *,
        objective: str,
        results: Sequence[WorkerResult],
        synthesize: Callable[[str], Any],
    ) -> ResponseOutcome:
        """Combine ordered worker results into one public outcome.

        Evidence keeps its server-owned provenance through synthesis, and a
        synthesis that carries any declares the grounding policy so the public
        validator revalidates it. A synthesis is exactly where an otherwise
        grounded worker result can be distorted.
        """
        payload = self._synthesis_payload(objective, results)
        content = synthesize(payload)
        if hasattr(content, "__await__"):
            content = await content

        evidence = tuple(
            record for result in results for record in (result.evidence or ())
        )
        artifacts = tuple(
            artifact for result in results for artifact in (result.artifacts or ())
        )
        policies: tuple[str, ...] = ("public_content",)
        if evidence:
            policies = (*policies, "rag_grounding")
        if artifacts:
            policies = (*policies, "artifact_provenance")

        return ResponseOutcome(
            agent_id=PLANNING_AGENT_ID,
            response=AgentResponse(
                agent_type=AgentType.PLANNING,
                agent_id=PLANNING_AGENT_ID,
                message=AgentMessage(role=MessageRole.ASSISTANT, content=str(content)),
                metadata={},
                tool_artifacts=list(artifacts) or None,
            ),
            provenance=OutcomeProvenance(
                output_policy_ids=policies,
                evidence=evidence,
                artifacts=artifacts,
            ),
        )

    def _synthesis_payload(self, objective: str, results: Sequence[WorkerResult]) -> str:
        """Render results as clearly delimited untrusted data.

        Worker output is a document to read, not an instruction to obey; the
        delimiters are what make that distinction visible to the model.
        """
        blocks = [f"Objective: {objective[: self._limits.objective_max_chars]}"]
        for result in results:
            blocks.append(
                f"{_UNTRUSTED_OPEN} task={result.task_id} agent={result.agent_id} "
                f"status={result.status}\n"
                f"{result.content[: self._limits.parent_context_max_chars]}\n"
                f"{_UNTRUSTED_CLOSE}"
            )
        payload = "\n\n".join(blocks)
        return payload[: self._limits.parent_context_max_chars]
