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
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command, Send
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import NotRequired, TypedDict

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.contracts import (
    DispatchSubagentsInput,
    OutcomeProvenance,
    PendingTransition,
    PlanningDispatch,
    ResponseOutcome,
    WorkerResult,
    WorkerTask,
)
from app.ai.workflow.inventory import RoutingInventory
from app.ai.workflow.specialists import UnavailableSpecialist
from app.observability.routing import get_routing_metrics_recorder

logger = logging.getLogger(__name__)

__all__ = [
    "DISPATCH_CONTROL_TOOL_NAME",
    "DispatchControlSchemaExecuted",
    "EXECUTION_ACTIONS",
    "HANDOFF_TOOL_NAME",
    "PLANNING_AGENT_ID",
    "PLANNING_NODE_NAMES",
    "PLAN_MODIFYING_ACTIONS",
    "WRITE_TODOS_TOOL_NAME",
    "InvalidPlanningDispatch",
    "PlanningLimits",
    "PlanningNodeFactory",
    "PlanningState",
    "PlanningWorkerRuntime",
    "TodoActionOutcome",
    "build_dispatch_control_tool",
    "build_planning_outcome",
    "collect_worker_results",
    "failed_worker",
    "planning_control_message_id",
    "render_worker_results",
    "validate_dispatch_call",
]

#: The exact set of parent-graph nodes Planning owns, in registration order.
#: A tool stage is deliberately absent: fan-out is topology, not a tool.
PLANNING_NODE_NAMES: tuple[str, ...] = (
    "planning_model",
    "planning_dispatch",
    "planning_worker",
    "planning_collect",
    "planning_actions",
    "planning_package",
)

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


class DispatchControlSchemaExecuted(RuntimeError):
    """Raised if anything ever tries to *run* the dispatch control schema.

    ``dispatch_subagents`` is bound to the Planning model so the model can
    propose a fan-out. It is not a tool: the server reads the proposal,
    validates it, and owns the tasks. Reaching this is a wiring bug in which
    the common tool pipeline resolved a control schema, and failing loudly
    beats fanning out from an unvalidated proposal.
    """


def build_dispatch_control_tool() -> Any:
    """The model-facing dispatch schema, with no executable body.

    ``returns_control_command`` keeps the specialist middleware from routing it
    through the product execution pipeline, and ``non_executable`` is what the
    execution-policy allowlist checks so it can never be resolved as a tool.
    """
    from langchain_core.tools import tool

    @tool(
        DISPATCH_CONTROL_TOOL_NAME,
        args_schema=DispatchSubagentsInput,
    )
    def dispatch_subagents(**kwargs: Any) -> str:
        """Delegate independent pieces of the plan to specialist subagents.

        Propose one task per independent piece of work: a short task_id, the
        objective in plain words, and which agent should do it. The server
        validates the whole proposal and runs the tasks; results come back as
        one paired result for this call.
        """
        raise DispatchControlSchemaExecuted(DISPATCH_CONTROL_TOOL_NAME)

    dispatch_subagents.metadata = {
        "non_executable": True,
        "returns_control_command": True,
        "tool_origin": "internal",
    }
    return dispatch_subagents


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
        agent_name = _worker_display_name(task.agent_id, state)
        _emit(
            writer,
            {
                "type": "planning_worker",
                "phase": "start",
                "dispatch_id": task.dispatch_id,
                "task_id": task.task_id,
                "agent_id": task.agent_id,
                "agent_name": agent_name,
            },
        )

        if task.agent_id == PLANNING_AGENT_ID:
            return self._finish(writer, failed_worker(task, "recursive_planning"), agent_name)

        try:
            if task.agent_id == RAG_AGENT_ID:
                result = await self._run_rag_worker(task, state)
            else:
                result = await self._run_specialist_worker(task, state)
        except GraphBubbleUp:
            # Control flow, not a failure. The event says the worker is waiting
            # on a human so the trace does not simply stop mid-worker; the
            # pause itself must still reach the parent untouched.
            _emit(
                writer,
                {
                    "type": "planning_worker",
                    "phase": "interrupt",
                    "dispatch_id": task.dispatch_id,
                    "task_id": task.task_id,
                    "agent_id": task.agent_id,
                    "agent_name": agent_name,
                },
            )
            raise
        except (ModelCallLimitExceededError, ToolCallLimitExceededError):
            return self._finish(writer, failed_worker(task, "agent_execution_limit"), agent_name)
        except TimeoutError:
            return self._finish(writer, failed_worker(task, "worker_timeout"), agent_name)
        except UnavailableSpecialist:
            return self._finish(writer, failed_worker(task, "agent_unavailable"), agent_name)
        except Exception as exc:  # noqa: BLE001 - normalized into a typed result
            logger.warning("Planning worker %s failed: %s", task.task_id, exc)
            return self._finish(writer, failed_worker(task, "tool_execution_failed"), agent_name)

        return self._finish(writer, result, agent_name)

    @staticmethod
    def _finish(
        writer: Callable[[dict[str, Any]], None] | None,
        result: WorkerResult,
        agent_name: str | None = None,
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
                "agent_name": agent_name,
                "status": result.status,
                "error_code": result.error_code,
            },
        )
        # `bounded_agent_label` collapses every custom agent to one bucket, so a
        # per-instance id can never become a metric label.
        get_routing_metrics_recorder().worker_completed(
            agent_id=result.agent_id,
            status=result.status,
            evidence_count=len(result.evidence or ()),
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


def _worker_display_name(agent_id: str, state: Mapping[str, Any]) -> str | None:
    """What a human should see this worker called.

    A custom agent's runtime id is ``custom_agent:<uuid>``, which is not a name.
    Resolving it here keeps the trace readable for a failed worker too -- the
    case where the id alone is least useful.
    """
    custom_agents = state.get("custom_agents")
    if not isinstance(custom_agents, Mapping):
        return None
    entry = custom_agents.get(agent_id)
    if not isinstance(entry, Mapping):
        return None
    name = entry.get("name")
    return str(name) if name else None


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


class TodoActionOutcome(BaseModel):
    """What one round of ``write_todos`` calls decided.

    Returned by the workflow's own todo applier rather than computed here: the
    plan-mutation rules, the todo cap, and the phase transitions are existing
    product behavior, and reimplementing them in the graph layer is how they
    would drift.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    todos: list[dict[str, Any]] = Field(default_factory=list)
    current_task_index: int | None = None
    tool_messages: tuple[Any, ...] = ()
    actions: tuple[str, ...] = ()
    had_error: bool = False
    planning_phase: str | None = None
    plan_just_modified: bool = False


#: Plan-mutating todo actions. These are the ones the rubric grades, because
#: they are the ones that change what the plan says.
PLAN_MODIFYING_ACTIONS = frozenset({"set_todos", "add_todo", "update_todo", "remove_todo"})

#: Todo actions that move the plan from writing into doing.
EXECUTION_ACTIONS = frozenset({"start_todo", "complete_todo"})

WRITE_TODOS_TOOL_NAME = "write_todos"


class PlanningNodeFactory:
    """The six real outer-graph nodes that make up Planning.

    Fan-out is topology here, not a tool. That is the whole point: a tool call
    is one unit of work to the checkpointer, so a worker pausing for approval
    left the entire call unfinished and the resume re-ran every sibling that
    had already completed. ``Send`` from a checkpointed parent node gives each
    worker its own recorded result, so a resume runs only what never finished.

    Nothing here invokes or compiles a child graph, and no node reaches
    ``END`` -- ``finalize`` owns termination.
    """

    def __init__(
        self,
        *,
        call_model: Callable[[Mapping[str, Any]], Any],
        worker_runtime: Any,
        limits: PlanningLimits,
        inventory_for: Callable[[Mapping[str, Any]], RoutingInventory],
        resolve_allowed_tools: Callable[[str, Mapping[str, Any]], tuple[str, ...]],
        apply_todo_actions: Callable[[Mapping[str, Any], Sequence[dict[str, Any]]], Any],
        review_rubric: Callable[[Mapping[str, Any], list[dict[str, Any]]], Any] | None = None,
    ) -> None:
        self._call_model = call_model
        self._worker_runtime = worker_runtime
        self.limits = limits
        self._inventory_for = inventory_for
        self._resolve_allowed_tools = resolve_allowed_tools
        self._apply_todo_actions = apply_todo_actions
        self._review_rubric = review_rubric

    # -- topology ---------------------------------------------------------

    def descriptors(self) -> tuple[tuple[str, Callable[..., Any], tuple[str, ...]], ...]:
        """One inspectable table of node name, callable, and destinations.

        The builder registers from this rather than naming nodes inline, so
        "which Planning nodes exist" is a single readable fact and a test can
        assert it without reconstructing the graph.
        """
        return (
            (
                "planning_model",
                self.planning_model,
                (
                    "planning_dispatch",
                    "planning_actions",
                    "planning_package",
                    "resolve_transition",
                    "finalize",
                ),
            ),
            ("planning_dispatch", self._dispatch_node, ("planning_worker",)),
            ("planning_worker", self._worker_node, ("planning_collect",)),
            ("planning_collect", self.planning_collect, ("planning_model", "finalize")),
            ("planning_actions", self.planning_actions, ("planning_model", "finalize")),
            ("planning_package", self.planning_package, ("validate_output", "finalize")),
        )

    # -- the model node ---------------------------------------------------

    async def planning_model(self, state: Mapping[str, Any], runtime: Any = None) -> Command:
        """One Planning model turn, and the one decision it produces.

        ``dispatch_subagents`` is bound to this model as a schema only. It is
        never executed by the common tool pipeline, so nothing it says can
        reach a provider or a credential -- the server reads the proposal,
        validates it in full, and owns every task that results.
        """
        response = await self._call_model(state)
        content = str(getattr(getattr(response, "message", None), "content", "") or "")
        tool_calls = [
            dict(call)
            for call in (getattr(getattr(response, "message", None), "tool_calls", None) or [])
        ]

        if not tool_calls:
            return Command(
                update={
                    "messages": [AIMessage(content=content)],
                    "execution_phase": "executing",
                },
                goto="planning_package",
            )

        ai_message = AIMessage(content=content, tool_calls=tool_calls)
        names = {str(call.get("name") or "") for call in tool_calls}

        if DISPATCH_CONTROL_TOOL_NAME in names:
            return self._route_dispatch(state, ai_message, tool_calls)

        if HANDOFF_TOOL_NAME in names and len(tool_calls) == 1:
            return self._route_handoff(state, ai_message, tool_calls[0])

        if WRITE_TODOS_TOOL_NAME in names:
            return Command(
                update={"messages": [ai_message], "execution_phase": "executing"},
                goto="planning_actions",
            )

        # A call Planning has no node for. Answering it with deterministic
        # feedback is what lets the model correct itself; leaving it unpaired
        # would strand a tool call the provider requires a result for.
        return Command(
            update={
                "messages": [
                    ai_message,
                    *(_control_message(call, "unsupported_planning_tool") for call in tool_calls),
                ]
            },
            goto="planning_model",
        )

    def _route_dispatch(
        self,
        state: Mapping[str, Any],
        ai_message: AIMessage,
        tool_calls: Sequence[dict[str, Any]],
    ) -> Command:
        """Validate the proposal in full, then either fan out or report back."""
        dispatch_call = next(
            call for call in tool_calls if call.get("name") == DISPATCH_CONTROL_TOOL_NAME
        )
        # Validation reads sibling calls off the message, so it has to see the
        # message this turn produced rather than the one already in state.
        validation_state = {**dict(state), "messages": [*(state.get("messages") or []), ai_message]}

        try:
            dispatch = validate_dispatch_call(
                tool_call=dispatch_call,
                state=validation_state,
                inventory=self._inventory_for(state),
                limits=self.limits,
                resolve_allowed_tools=self._resolve_allowed_tools,
            )
        except InvalidPlanningDispatch as invalid:
            return Command(
                update={
                    "messages": [
                        ai_message,
                        *(
                            _control_message(
                                call,
                                invalid.code
                                if call.get("id") == invalid.tool_call_id
                                else "dispatch_rejected",
                            )
                            for call in tool_calls
                        ),
                    ]
                },
                goto="planning_model",
            )

        return Command(
            update={
                "messages": [ai_message],
                "planning_dispatch": dispatch,
                "planning_control_call_id": dispatch.tool_call_id,
                "execution_phase": "executing",
            },
            goto="planning_dispatch",
        )

    def _route_handoff(
        self, state: Mapping[str, Any], ai_message: AIMessage, call: Mapping[str, Any]
    ) -> Command:
        """Record the requested transition; the resolver decides if it happens."""
        args = call.get("args") if isinstance(call.get("args"), Mapping) else {}
        tool_call_id = str(call.get("id") or "")
        target = str((args or {}).get("to_agent_id") or "")
        reason = str((args or {}).get("reason") or "planning delegated the turn")

        if not target:
            return Command(
                update={"messages": [ai_message, _control_message(call, "handoff_missing_target")]},
                goto="planning_model",
            )

        pending = PendingTransition(
            from_agent_id=PLANNING_AGENT_ID,
            to_agent_id=target,
            tool_call_id=tool_call_id,
            tool_message_id=f"handoff:{tool_call_id}",
            reason=reason[:500],
        )
        return Command(
            update={
                "messages": [
                    ai_message,
                    ToolMessage(
                        content=f"handoff requested: {target}",
                        tool_call_id=tool_call_id,
                        name=HANDOFF_TOOL_NAME,
                        id=pending.tool_message_id,
                    ),
                ],
                "pending_transition": pending,
            },
            goto="resolve_transition",
        )

    # -- fan-out ----------------------------------------------------------

    def _dispatch_node(self, state: Mapping[str, Any]) -> Command:
        """Register :meth:`planning_dispatch` as a node.

        A node must return a state update or a ``Command``; a bare list of
        ``Send`` is only valid from a conditional edge. Keeping the ``Send``
        construction in its own method leaves it directly assertable, and the
        ``Command`` is what makes the fan-out a real parent-graph step -- which
        is the property the whole task exists for.
        """
        sends = self.planning_dispatch(state)
        if not sends:
            return Command(goto="planning_model")
        return Command(goto=sends)

    def planning_dispatch(self, state: Mapping[str, Any]) -> list[Send]:
        """Fan the validated wave out as real parent-graph tasks.

        Every task here came from one already-completed validation, so there is
        no window in which some workers are running while others are still
        being checked. Nothing is invoked or compiled.
        """
        dispatch = state.get("planning_dispatch")
        if dispatch is None:
            logger.warning("planning_dispatch ran with no validated dispatch in state")
            return []

        _emit(
            _stream_writer(),
            {
                "type": "planning_dispatch",
                "phase": "validated",
                "dispatch_id": dispatch.dispatch_id,
                "wave": dispatch.wave,
                "task_count": len(dispatch.tasks),
            },
        )

        scope = _bounded_parent_scope(state)
        return [
            Send("planning_worker", {"worker_task": task, "worker_parent_state": scope})
            for task in dispatch.tasks
        ]

    async def _worker_node(self, payload: Mapping[str, Any]) -> Command:
        """Register :meth:`planning_worker` as a node.

        The ``goto`` is what fans back in. Every parallel worker names the same
        target, and LangGraph runs it once for the whole wave with all results
        accumulated -- so collection sees the complete set without the worker
        needing a static outgoing edge.
        """
        return Command(update=await self.planning_worker(payload), goto="planning_collect")

    async def planning_worker(
        self,
        payload: Mapping[str, Any],
        writer: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run exactly one dispatched task and return only its typed result.

        The writer is resolved from the live run when the caller does not pass
        one, so events flow in production without the node signature depending
        on framework injection.
        """
        task: WorkerTask = payload["worker_task"]
        parent_state = payload.get("worker_parent_state") or {}
        result = await self._worker_runtime.run(task, parent_state, writer or _stream_writer())
        return {"worker_results": [result]}

    def planning_collect(self, state: Mapping[str, Any]) -> Command:
        """Answer the dispatch call with this wave's ordered results.

        One ``ToolMessage`` paired to the original call: the model made one
        call and a provider requires exactly one result for it. Ordering is by
        server-owned ``position``, because completion order is an accident of
        latency and would make the same plan synthesize differently on
        different runs.
        """
        dispatch = state.get("planning_dispatch")
        if dispatch is None:
            logger.warning("planning_collect ran with no dispatch to collect")
            return Command(goto="planning_model")

        ordered = collect_worker_results(dispatch, state.get("worker_results") or [])
        _emit(
            _stream_writer(),
            {
                "type": "planning_dispatch",
                "phase": "collected",
                "dispatch_id": dispatch.dispatch_id,
                "wave": dispatch.wave,
                "task_count": len(ordered),
            },
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=render_worker_results(
                            _dispatch_objective(dispatch), ordered, self.limits
                        ),
                        tool_call_id=dispatch.tool_call_id,
                        name=DISPATCH_CONTROL_TOOL_NAME,
                        id=planning_control_message_id(dispatch.tool_call_id),
                    )
                ],
                # Cleared so a resumed turn cannot re-dispatch a wave it has
                # already collected, and so the wave counters are the only
                # thing deciding whether another dispatch is allowed.
                "planning_dispatch": None,
                "planning_control_call_id": None,
                "planning_dispatch_waves": dispatch.wave,
                "planning_dispatched_task_count": int(
                    state.get("planning_dispatched_task_count") or 0
                )
                + len(dispatch.tasks),
            },
            goto="planning_model",
        )

    # -- todos ------------------------------------------------------------

    async def planning_actions(self, state: Mapping[str, Any]) -> Command:
        """Apply this turn's ``write_todos`` calls and grade the result.

        The rubric runs only on plan-mutating actions. Grading a
        ``start_todo`` would score the plan for something that did not change
        it, and ``needs_revision`` routes back to the model with feedback
        instead of emitting the normal plan-summary pass.
        """
        calls = [
            call for call in _last_tool_calls(state) if call.get("name") == WRITE_TODOS_TOOL_NAME
        ]
        outcome = _as_todo_outcome(await _maybe_await(self._apply_todo_actions(state, calls)))

        update: dict[str, Any] = {
            "messages": list(outcome.tool_messages),
            "todos": list(outcome.todos),
            "current_task_index": outcome.current_task_index,
        }
        if outcome.planning_phase:
            update["planning_phase"] = outcome.planning_phase
        elif any(action in EXECUTION_ACTIONS for action in outcome.actions):
            update["planning_phase"] = "executing"

        context = dict(state.get("context") or {})
        if outcome.plan_just_modified or any(
            action in PLAN_MODIFYING_ACTIONS for action in outcome.actions
        ):
            context["plan_just_modified"] = True
            context = await self._graded_context(state, context, outcome)
        update["context"] = context

        return Command(update=update, goto="planning_model")

    async def _graded_context(
        self,
        state: Mapping[str, Any],
        context: dict[str, Any],
        outcome: TodoActionOutcome,
    ) -> dict[str, Any]:
        if self._review_rubric is None:
            return context

        attempt = await _maybe_await(self._review_rubric(state, list(outcome.todos)))
        status = getattr(attempt, "status", None)
        if status is None or status == "disabled":
            return context

        metadata = getattr(attempt, "metadata", None)
        context["planning_rubric"] = metadata() if callable(metadata) else metadata
        feedback = getattr(attempt, "feedback", None)
        if status == "needs_revision" and feedback:
            context["planning_rubric_feedback"] = feedback
            context["plan_just_modified"] = False
            context.pop("generate_plan_response", None)
        elif status in {"satisfied", "max_iterations_reached"}:
            context.pop("planning_rubric_feedback", None)
        return context

    # -- packaging --------------------------------------------------------

    def planning_package(self, state: Mapping[str, Any]) -> Command:
        """Turn the finished turn into one candidate public outcome.

        Provenance is aggregated from server-owned worker records only, and it
        goes to ``validate_output`` rather than to ``finalize``: Planning
        writes the public answer itself, so it can introduce a citation no
        worker retrieved, and the synthesis has to be revalidated before
        publication.
        """
        results = list(state.get("worker_results") or [])
        outcome = build_planning_outcome(content=_last_ai_text(state), results=results)
        return Command(
            update={
                "agent_outcome": outcome,
                "final_agent_id": PLANNING_AGENT_ID,
                "execution_phase": "validating",
            },
            goto="validate_output",
        )


# ----------------------------------------------------------------------
# node helpers
# ----------------------------------------------------------------------


def _control_message(call: Mapping[str, Any], code: str) -> ToolMessage:
    """Deterministic paired feedback for one rejected control call.

    The text is a stable code, not prose: the model has to be able to tell the
    same rejection apart from a different one across turns and locales.
    """
    tool_call_id = str(call.get("id") or "")
    return ToolMessage(
        content=code,
        tool_call_id=tool_call_id,
        name=str(call.get("name") or DISPATCH_CONTROL_TOOL_NAME),
        status="error",
        id=planning_control_message_id(tool_call_id),
    )


def _bounded_parent_scope(state: Mapping[str, Any]) -> dict[str, Any]:
    """The parent facts a worker is allowed to see.

    ``messages`` is deliberately absent. A worker given the public transcript
    could answer the user directly instead of doing the delegated work, and it
    would also carry every other worker's output into its context.
    """
    return {
        "conversation_id": state.get("conversation_id"),
        "user_id": state.get("user_id"),
        "device_id": state.get("device_id"),
        "persona": state.get("persona"),
        "attachments": list(state.get("attachments") or []),
        "custom_agents": dict(state.get("custom_agents") or {}),
        "model_request": state.get("model_request"),
        "context": dict(state.get("context") or {}),
        "turn_identity": state.get("turn_identity"),
        "task_plan_id": state.get("task_plan_id"),
        "todos": list(state.get("todos") or []),
    }


def _dispatch_objective(dispatch: PlanningDispatch) -> str:
    """A label for the wave, assembled from server-owned task objectives."""
    return "; ".join(task.objective for task in dispatch.tasks)


def _last_tool_calls(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    for message in reversed(state.get("messages") or []):
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            return [dict(call) for call in tool_calls]
    return []


def _last_ai_text(state: Mapping[str, Any]) -> str:
    """The last thing the Planning model actually said."""
    for message in reversed(state.get("messages") or []):
        if getattr(message, "type", None) != "ai":
            continue
        if getattr(message, "tool_calls", None):
            continue
        content = getattr(message, "content", "")
        text = content if isinstance(content, str) else ""
        if text.strip():
            return text
    return ""


def _stream_writer() -> Callable[[dict[str, Any]], None] | None:
    """The live custom-event writer, when there is a run to write into."""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except (RuntimeError, ImportError):  # pragma: no cover - outside a run
        return None


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _as_todo_outcome(value: Any) -> TodoActionOutcome:
    if isinstance(value, TodoActionOutcome):
        return value
    if value is None:
        return TodoActionOutcome()
    if isinstance(value, Mapping):
        return TodoActionOutcome(**value)
    raise TypeError(f"todo applier returned an unsupported outcome: {type(value).__name__}")
