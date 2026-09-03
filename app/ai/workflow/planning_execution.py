"""Planning as an orchestrator over isolated workers.

Planning owns delegation and synthesis. Independent tasks fan out as real
outer-graph topology; each worker runs one specialist path and returns a typed
private result.

Three things a worker deliberately cannot do: publish a public assistant
message, perform a parent-level handoff, or recurse into Planning. All three
would take a decision that belongs to the orchestrator and hide it inside a
branch.

Dispatch validation here is all-or-nothing. A proposal is checked in full
before a single task is constructed, so an invalid ninth task cannot leave
eight workers already running. Nothing is truncated: an oversized objective is
rejected and reported back to the model, because a shortened objective is
different work than the one that was asked for.

Worker objectives and results are delimited as untrusted data. A worker result
is content the orchestrator reads, never instructions it follows.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import NotRequired, TypedDict

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.contracts import (
    DispatchSubagentsInput,
    OutcomeProvenance,
    PlanningDispatch,
    ResponseOutcome,
    WorkerResult,
    WorkerTask,
)
from app.ai.workflow.inventory import RoutingInventory
from app.ai.workflow.specialists import UnavailableSpecialist

logger = logging.getLogger(__name__)

__all__ = [
    "DISPATCH_CONTROL_TOOL_NAME",
    "HANDOFF_TOOL_NAME",
    "PLANNING_AGENT_ID",
    "InvalidPlanningDispatch",
    "PlanningLimits",
    "PlanningState",
    "PlanningWorkerRuntime",
    "build_planning_outcome",
    "collect_worker_results",
    "failed_worker",
    "planning_control_message_id",
    "render_worker_results",
    "validate_dispatch_call",
]

PLANNING_AGENT_ID = "planning_agent"
RAG_AGENT_ID = "rag_agent"
DISPATCH_CONTROL_TOOL_NAME = "dispatch_subagents"
HANDOFF_TOOL_NAME = "hand_off"

_UNTRUSTED_OPEN = "BEGIN UNTRUSTED WORKER RESULT"
_UNTRUSTED_CLOSE = "END UNTRUSTED WORKER RESULT"


class PlanningLimits(BaseModel):
    """Bounds on one Planning turn. Every value is validated positive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tasks: int = Field(gt=0, le=64)
    max_concurrency: int = Field(gt=0, le=32)
    max_dispatch_waves: int = Field(default=2, gt=0, le=8)
    objective_max_chars: int = Field(gt=0, le=64_000)
    parent_context_max_chars: int = Field(gt=0, le=200_000)

    @classmethod
    def from_settings(cls, settings: Any) -> PlanningLimits:
        return cls(
            max_tasks=int(getattr(settings, "planning_worker_max_tasks", 8)),
            max_concurrency=int(getattr(settings, "planning_worker_max_concurrency", 4)),
            max_dispatch_waves=int(getattr(settings, "planning_worker_max_dispatch_waves", 2)),
            objective_max_chars=int(getattr(settings, "planning_worker_objective_max_chars", 4000)),
            parent_context_max_chars=int(
                getattr(settings, "planning_parent_context_max_chars", 12000)
            ),
        )


class InvalidPlanningDispatch(ValueError):
    """One rejected dispatch proposal, paired to the call that made it.

    ``tool_call_id`` travels with the error so the Planning model node can
    answer the exact call. An unpaired rejection would leave the model with a
    tool call that never received a result.
    """

    def __init__(self, code: str, tool_call_id: str) -> None:
        super().__init__(code)
        self.code = code
        self.tool_call_id = tool_call_id


def planning_control_message_id(tool_call_id: str) -> str:
    """Deterministic ID for the control ``ToolMessage`` answering one call."""
    return f"planning-control:{tool_call_id}"


class PlanningState(TypedDict):
    """Planning-graph state. Worker messages never appear here."""

    worker_tasks: list[WorkerTask]
    limits: PlanningLimits
    runtime_request: dict[str, Any]
    worker_results: NotRequired[list[WorkerResult]]
    planning_result: NotRequired[Any]


def _dispatch_identity(tool_call_id: str, wave: int) -> str:
    """Derive the dispatch ID from the call and wave, not from a counter.

    Deriving it makes the same call in the same wave produce the same ID on
    replay, which is what lets a resumed turn recognize results it already
    has instead of scheduling the work twice.
    """
    raw = f"{tool_call_id}:{wave}".encode()
    return hashlib.sha256(raw).hexdigest()[:32]


def _sibling_tool_call_names(state: Mapping[str, Any]) -> tuple[str, ...]:
    """Names of every tool call on the message that produced this dispatch."""
    messages = state.get("messages") or []
    for message in reversed(messages):
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        names: list[str] = []
        for call in tool_calls:
            name = call.get("name") if isinstance(call, Mapping) else getattr(call, "name", None)
            if name:
                names.append(str(name))
        return tuple(names)
    return ()


def _build_parent_context(
    state: Mapping[str, Any], limits: PlanningLimits, tool_call_id: str
) -> dict[str, Any]:
    """Project the bounded parent scope a worker is allowed to see.

    The objective is deliberately absent: it belongs to one task and travels
    on that task. Everything here is plan-level context shared by the wave.
    """
    todos = state.get("todos")
    projected_todos: list[dict[str, Any]] = []
    if isinstance(todos, list):
        for todo in todos:
            if not isinstance(todo, Mapping):
                continue
            projected_todos.append(
                {
                    "id": str(todo.get("id") or ""),
                    "status": str(todo.get("status") or ""),
                    "content": str(todo.get("content") or ""),
                }
            )

    context: dict[str, Any] = {
        "task_plan_id": state.get("task_plan_id"),
        "plan_summary": _plan_summary(projected_todos),
        "todos": projected_todos,
    }

    encoded = json.dumps(context, ensure_ascii=False, default=str)
    if len(encoded) > limits.parent_context_max_chars:
        raise InvalidPlanningDispatch("parent_context_too_long", tool_call_id)
    return context


def _plan_summary(todos: Sequence[Mapping[str, Any]]) -> str:
    if not todos:
        return ""
    open_count = sum(1 for todo in todos if todo.get("status") != "completed")
    return f"{len(todos)} plan items, {open_count} not completed"


def validate_dispatch_call(
    *,
    tool_call: Mapping[str, Any],
    state: Mapping[str, Any],
    inventory: RoutingInventory,
    limits: PlanningLimits,
    resolve_allowed_tools: Callable[[str, Mapping[str, Any]], tuple[str, ...]],
) -> PlanningDispatch:
    """Validate the entire proposal before returning server-owned tasks.

    Raises :class:`InvalidPlanningDispatch` and constructs nothing on any
    failure. Every ``Send`` the caller later builds comes from the returned
    dispatch, so there is no window in which some tasks are scheduled and
    others are still being checked.
    """
    tool_call_id = str(tool_call.get("id") or "")
    if not tool_call_id:
        raise InvalidPlanningDispatch("missing_tool_call_id", "unknown")

    # A dispatch and a handoff in one message are two different decisions
    # about who continues the turn. Running both would fan out work and then
    # abandon it mid-flight.
    sibling_names = _sibling_tool_call_names(state)
    if HANDOFF_TOOL_NAME in sibling_names:
        raise InvalidPlanningDispatch("dispatch_with_handoff", tool_call_id)

    wave = int(state.get("planning_dispatch_waves") or 0) + 1
    if wave > limits.max_dispatch_waves:
        raise InvalidPlanningDispatch("dispatch_wave_limit", tool_call_id)

    try:
        proposal = DispatchSubagentsInput.model_validate(tool_call.get("args") or {})
    except ValidationError as exc:
        logger.info("Planning dispatch %s rejected as invalid input: %s", tool_call_id, exc)
        raise InvalidPlanningDispatch("invalid_dispatch_input", tool_call_id) from exc

    seen: set[str] = set()
    for task in proposal.tasks:
        if task.task_id in seen:
            raise InvalidPlanningDispatch("duplicate_task_id", tool_call_id)
        seen.add(task.task_id)

    already_dispatched = int(state.get("planning_dispatched_task_count") or 0)
    if already_dispatched + len(proposal.tasks) > limits.max_tasks:
        raise InvalidPlanningDispatch("dispatch_task_limit", tool_call_id)

    for task in proposal.tasks:
        if task.agent_id == PLANNING_AGENT_ID:
            raise InvalidPlanningDispatch("recursive_planning", tool_call_id)
        descriptor = inventory.get(task.agent_id)
        if descriptor is None:
            raise InvalidPlanningDispatch("unknown_agent", tool_call_id)
        if not inventory.is_routable(task.agent_id):
            raise InvalidPlanningDispatch("agent_unavailable", tool_call_id)
        if len(task.objective) > limits.objective_max_chars:
            raise InvalidPlanningDispatch("objective_too_long", tool_call_id)

    parent_context = _build_parent_context(state, limits, tool_call_id)
    dispatch_id = _dispatch_identity(tool_call_id, wave)
    model_request = state.get("model_request")

    tasks = tuple(
        WorkerTask(
            dispatch_id=dispatch_id,
            task_id=proposed.task_id,
            position=position,
            objective=proposed.objective,
            agent_id=proposed.agent_id,
            parent_context=parent_context,
            allowed_tool_ids=tuple(resolve_allowed_tools(proposed.agent_id, state)),
            model_request=model_request if isinstance(model_request, dict) else None,
            related_todo_ids=proposed.related_todo_ids,
        )
        for position, proposed in enumerate(proposal.tasks)
    )

    return PlanningDispatch(
        dispatch_id=dispatch_id,
        tool_call_id=tool_call_id,
        wave=wave,
        tasks=tasks,
    )


def collect_worker_results(
    dispatch: PlanningDispatch, results: Sequence[WorkerResult]
) -> list[WorkerResult]:
    """Order this dispatch's results by server-owned ``position``.

    Synthesis reads the plan in the order it was written; completion order is
    an accident of latency and would make the same plan synthesize differently
    on different runs. Results from another wave are not this dispatch's to
    collect.
    """
    mine = [result for result in results if result.dispatch_id == dispatch.dispatch_id]
    return sorted(mine, key=lambda result: result.position)


class PlanningWorkerRuntime:
    """The one path a dispatched task takes to a typed private result.

    RAG workers and standard specialists diverge only in *which* runtime runs
    them; everything that decides authority — the objective, the bounded parent
    context, the approval policy, the custom-agent snapshot, the attachments,
    the model request, and the restricted tool set — is assembled once in
    ``build_worker_request`` so neither path can quietly get more than the
    dispatch granted.
    """

    def __init__(
        self,
        *,
        specialist_factory: Any,
        rag_execution_graph: Any,
        limits: PlanningLimits,
    ) -> None:
        self.specialist_factory = specialist_factory
        self.rag_execution_graph = rag_execution_graph
        self._limits = limits

    async def run(
        self,
        task: WorkerTask,
        state: Mapping[str, Any],
        writer: Callable[[dict[str, Any]], None] | None = None,
    ) -> WorkerResult:
        """Execute one dispatched task.

        The ``except`` order is the contract. ``GraphBubbleUp`` is re-raised
        first: a worker that paused for approval has not failed, and turning
        that pause into a failed result would answer the turn without the human
        who was asked. Everything after it is a genuine failure, narrowed
        before the generic case so a limit or a timeout keeps its own code.
        """
        _emit(
            writer,
            {
                "type": "planning_worker",
                "phase": "start",
                "dispatch_id": task.dispatch_id,
                "task_id": task.task_id,
                "agent_id": task.agent_id,
            },
        )

        if task.agent_id == PLANNING_AGENT_ID:
            return self._finish(writer, failed_worker(task, "recursive_planning"))

        try:
            if task.agent_id == RAG_AGENT_ID:
                result = await self._run_rag_worker(task, state)
            else:
                result = await self._run_specialist_worker(task, state)
        except GraphBubbleUp:
            raise
        except (ModelCallLimitExceededError, ToolCallLimitExceededError):
            return self._finish(writer, failed_worker(task, "agent_execution_limit"))
        except TimeoutError:
            return self._finish(writer, failed_worker(task, "worker_timeout"))
        except UnavailableSpecialist:
            return self._finish(writer, failed_worker(task, "agent_unavailable"))
        except Exception as exc:  # noqa: BLE001 - normalized into a typed result
            logger.warning("Planning worker %s failed: %s", task.task_id, exc)
            return self._finish(writer, failed_worker(task, "tool_execution_failed"))

        return self._finish(writer, result)

    @staticmethod
    def _finish(
        writer: Callable[[dict[str, Any]], None] | None, result: WorkerResult
    ) -> WorkerResult:
        """Report the outcome without leaking the objective or the content."""
        _emit(
            writer,
            {
                "type": "planning_worker",
                "phase": "end",
                "dispatch_id": result.dispatch_id,
                "task_id": result.task_id,
                "agent_id": result.agent_id,
                "status": result.status,
                "error_code": result.error_code,
            },
        )
        return result

    async def _run_specialist_worker(
        self, task: WorkerTask, state: Mapping[str, Any]
    ) -> WorkerResult:
        from app.ai.workflow.specialists import build_worker_request

        request = build_worker_request(task, dict(state))
        return await self.specialist_factory.invoke_worker(request, task=task)

    async def _run_rag_worker(self, task: WorkerTask, state: Mapping[str, Any]) -> WorkerResult:
        """RAG workers use the same compiled graph and gate as top-level RAG."""
        from app.ai.workflow.rag_execution import RagExecutionRequest
        from app.ai.workflow.specialists import build_worker_request

        request = build_worker_request(task, dict(state))
        result = await self.rag_execution_graph.ainvoke(
            RagExecutionRequest(
                objective=task.objective,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                device_id=request.device_id,
                persona=request.persona,
                model_request=request.model_request,
                allowed_tool_ids=task.allowed_tool_ids,
                hitl_policy=request.hitl_policy or {},
                attachments=list(request.attachments),
                mode="worker",
                dispatch_id=task.dispatch_id,
                task_id=task.task_id,
            )
        )
        return WorkerResult(
            dispatch_id=task.dispatch_id,
            task_id=task.task_id,
            position=task.position,
            agent_id=task.agent_id,
            status="completed",
            content=str(getattr(result, "content", "")),
            evidence=tuple(getattr(result, "evidence", ()) or ()),
            artifacts=tuple(getattr(result, "artifacts", ()) or ()),
            images=tuple(getattr(result, "images", ()) or ()),
        )


def failed_worker(task: WorkerTask, error_code: str) -> WorkerResult:
    """A failed worker keeps full identity, so no dispatched task is orphaned."""
    return WorkerResult(
        dispatch_id=task.dispatch_id,
        task_id=task.task_id,
        position=task.position,
        agent_id=task.agent_id,
        status="failed",
        content="",
        error_code=error_code,
    )


def _emit(writer: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    """Emit one typed custom event, if the node was given a writer.

    A writer failure must not fail the work it was describing.
    """
    if writer is None:
        return
    try:
        writer(event)
    except Exception as exc:  # noqa: BLE001 - telemetry never fails the turn
        logger.debug("Dropped planning worker event: %s", exc)


def render_worker_results(
    objective: str, results: Sequence[WorkerResult], limits: PlanningLimits
) -> str:
    """Render results as clearly delimited untrusted data.

    Worker output is a document to read, not an instruction to obey; the
    delimiters are what make that distinction visible to the model.
    """
    blocks = [f"Objective: {objective[: limits.objective_max_chars]}"]
    for result in results:
        blocks.append(
            f"{_UNTRUSTED_OPEN} task={result.task_id} agent={result.agent_id} "
            f"status={result.status}\n"
            f"{result.content[: limits.parent_context_max_chars]}\n"
            f"{_UNTRUSTED_CLOSE}"
        )
    payload = "\n\n".join(blocks)
    return payload[: limits.parent_context_max_chars]


def build_planning_outcome(*, content: str, results: Sequence[WorkerResult]) -> ResponseOutcome:
    """Aggregate server-owned worker provenance into one public outcome.

    Evidence keeps its server-owned provenance through synthesis, and a
    synthesis that carries any declares the grounding policy so the public
    validator revalidates it. A synthesis is exactly where an otherwise
    grounded worker result can be distorted.
    """
    evidence = tuple(record for result in results for record in (result.evidence or ()))
    artifacts = tuple(artifact for result in results for artifact in (result.artifacts or ()))
    images = tuple(image for result in results for image in (result.images or ()))

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
            metadata={"images": list(images)} if images else {},
            tool_artifacts=list(artifacts) or None,
        ),
        provenance=OutcomeProvenance(
            output_policy_ids=policies,
            evidence=evidence,
            artifacts=artifacts,
            images=images,
        ),
    )
