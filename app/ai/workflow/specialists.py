"""Parent-level specialist wrappers for the routing-v2 graph.

A wrapper does three things and nothing else:

1. verifies the node it is running in matches ``state["active_agent_id"]``;
2. runs the specialist;
3. converts the result into a ``ResponseOutcome`` and a dynamic ``Command``.

Wrappers have no static outgoing edges, so the parent graph's dynamic routing
is the only thing that decides what runs next. No wrapper appends the terminal
public ``AIMessage`` and no wrapper reaches ``END`` — ``finalize`` owns both.

The execution *internals* behind these wrappers are the pre-v2 agent loops.
Task 5 of the cutover replaces those internals with per-invocation
``create_agent`` subgraphs without changing this contract.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.tool_context import rich_response_capable_from_context
from app.ai.workflow.contracts import (
    OutcomeProvenance,
    ResponseOutcome,
    WorkerResult,
    WorkerTask,
    WorkflowError,
)
from app.ai.workflow.execution_budget import (
    ExecutionBudgetAccountant,
    ExecutionBudgetLimits,
    ExecutionBudgetState,
)
from app.ai.workflow.execution_budget_middleware import SoftExecutionBudgetMiddleware
from app.ai.workflow.inventory import (
    BASE_AGENT_NODE_OVERRIDES,
    CUSTOM_AGENT_NODE,
    CUSTOM_AGENT_PREFIX,
)
from app.ai.workflow.middleware import (
    TOP_LEVEL_DISPATCH_ID,
    SpecialistToolScope,
    ToolApprovalMiddleware,
    ToolExecutionMiddleware,
    WorkerToolScopeMiddleware,
    build_specialist_middleware,
)
from app.observability.routing import get_routing_metrics_recorder

PLANNING_AGENT_ID = "planning_agent"

logger = logging.getLogger(__name__)

#: Stands in for the answer when the execution ceiling fires mid-loop. Written
#: as a statement about the turn rather than an apology: the user's next move is
#: to continue it, and a partial that pretends to be an answer is worse than one
#: that says what happened.
HARD_LIMIT_PARTIAL_TEXT = (
    "I reached this turn's execution limit before finishing. Anything I "
    "already gathered is kept, and continuing the turn resumes from there."
)

__all__ = [
    "FINALIZE_OWNS_TERMINAL_MESSAGE",
    "PLANNING_AGENT_ID",
    "ModelCallLimitExceededError",
    "SpecialistDefinition",
    "SpecialistFactory",
    "SpecialistRequest",
    "SpecialistRuntimeContext",
    "ToolCallLimitExceededError",
    "UnavailableSpecialist",
    "build_worker_request",
    "make_specialist_wrapper",
    "make_subgraph_specialist_wrapper",
    "make_tool_stage_wrapper",
    "resolve_node_for_agent_id",
]


class UnavailableSpecialist(KeyError):
    """No definition exists for the requested agent in this request's scope."""


# Turn-scoped context flag telling the pre-v2 ``_finalize_agent_response`` path
# that the parent finalizer owns the terminal public message.
FINALIZE_OWNS_TERMINAL_MESSAGE = "v2_finalize_owns_terminal_message"

SpecialistCallable = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
StageRouter = Callable[[dict[str, Any]], Any]


def resolve_node_for_agent_id(agent_id: str | None) -> str | None:
    """Map an agent ID onto its graph node without consulting the inventory.

    Must agree with ``RoutingInventory.resolve_node``; both read the same
    override table so a Planning handoff lands on ``planning_model`` whichever
    path resolved it.
    """
    if not agent_id:
        return None
    if agent_id.startswith(CUSTOM_AGENT_PREFIX):
        return CUSTOM_AGENT_NODE
    return BASE_AGENT_NODE_OVERRIDES.get(agent_id, agent_id)


def _with_finalizer_ownership(state: dict[str, Any]) -> dict[str, Any]:
    working = dict(state)
    context = dict(working.get("context") or {})
    context[FINALIZE_OWNS_TERMINAL_MESSAGE] = True
    working["context"] = context
    return working


def _response_outcome(
    agent_id: str, response: AgentResponse, state: dict[str, Any] | None = None
) -> ResponseOutcome:
    """Build a server-owned outcome from a specialist response.

    Provenance is assembled from runtime records only. Model text can never
    declare its own evidence, artifacts, images, or validation authority.
    """
    artifacts = tuple(
        artifact for artifact in (response.tool_artifacts or []) if isinstance(artifact, dict)
    )
    metadata = response.metadata or {}
    images = tuple(image for image in (metadata.get("images") or []) if isinstance(image, dict))
    return ResponseOutcome(
        agent_id=agent_id,
        response=response,
        provenance=OutcomeProvenance(
            artifacts=artifacts, images=images, evidence=_recorded_evidence(state)
        ),
    )


def _recorded_evidence(state: dict[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """The evidence records this turn's own tool calls produced.

    A citation selects the grounding policy, so an answer that legitimately
    cites ``[E1]`` is rejected as invented unless the ids the runtime actually
    retrieved travel with it. They live on the tool artifacts, which is the
    same place the grounded-answer gate reads them from.
    """
    context = (state or {}).get("context")
    if not isinstance(context, dict):
        return ()
    records: list[dict[str, Any]] = []
    for artifact in context.get("tool_artifacts") or ():
        if not isinstance(artifact, dict):
            continue
        evidence = artifact.get("rag_evidence")
        if not isinstance(evidence, dict):
            continue
        for record in evidence.get("records") or ():
            if isinstance(record, dict) and record.get("evidence_id"):
                records.append(record)
    return tuple(records)


def _carry_forward(state: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Project a legacy node's mutated state back onto the v2 update.

    Only keys the v2 state schema declares survive; everything else the legacy
    path scribbled on the dict is discarded rather than silently persisted.
    """
    update: dict[str, Any] = {}
    for key in (
        "context",
        "todos",
        "current_task_index",
        "planning_call_count",
        "planning_phase",
        "plan_lifecycle",
        "task_plan_id",
        "current_task",
        "all_tasks",
        "custom_agents",
    ):
        if key in result and result[key] is not state.get(key):
            update[key] = result[key]

    appended = _appended_messages(state.get("messages") or [], result.get("messages") or [])
    if appended:
        update["messages"] = appended
    return update


def _appended_messages(before: list[Any], after: list[Any]) -> list[Any]:
    if len(after) <= len(before):
        return []
    return list(after[len(before) :])


def make_specialist_wrapper(
    node_name: str,
    specialist: SpecialistCallable,
    *,
    stage_router: StageRouter,
    stage_targets: dict[str, str],
) -> Callable[..., Awaitable[Command]]:
    """Build the parent wrapper node for one specialist.

    ``stage_router`` is the pre-v2 decision function ("tools", "approval",
    "end", ...). ``stage_targets`` maps those decisions onto v2 node names;
    ``"end"`` always maps to ``validate_output``, never to ``END``.
    """

    async def wrapper(state: dict[str, Any], runtime: Any = None) -> Command:
        active_agent_id = state.get("active_agent_id")
        expected_node = resolve_node_for_agent_id(active_agent_id)
        if expected_node != node_name:
            # Reaching a specialist that is not the active agent means the
            # control plane and the topology disagree. Fail closed.
            logger.error(
                "Specialist node %s reached while active agent is %s",
                node_name,
                active_agent_id,
            )
            return Command(
                update={
                    "execution_phase": "failed",
                    "workflow_error": WorkflowError(
                        code="response_validation_failed",
                        retriable=False,
                        request_id=_request_id(state),
                        details={"reason": "active_agent_node_mismatch", "node": node_name},
                    ),
                },
                goto="finalize",
            )

        working = _with_finalizer_ownership(state)
        result = await specialist(working)
        result = result if isinstance(result, dict) else working

        update = _carry_forward(state, result)
        decision = stage_router(result)
        if _is_awaitable(decision):
            decision = await decision

        target = stage_targets.get(str(decision), "validate_output")
        if target == "validate_output":
            response = result.get("response")
            if not isinstance(response, AgentResponse):
                return Command(
                    update={
                        **update,
                        "execution_phase": "failed",
                        "workflow_error": WorkflowError(
                            code="response_validation_failed",
                            retriable=False,
                            request_id=_request_id(state),
                            details={"reason": "specialist_produced_no_response"},
                        ),
                    },
                    goto="finalize",
                )
            update["agent_outcome"] = _response_outcome(
                active_agent_id or response.agent_id, response, result
            )
            update["execution_phase"] = "validating"

        return Command(update=update, goto=target)

    wrapper.__name__ = f"{node_name}_wrapper"
    return wrapper


def make_subgraph_specialist_wrapper(
    node_name: str,
    invoke: Callable[[dict[str, Any]], Awaitable[Any]],
) -> Callable[..., Awaitable[Command]]:
    """Parent wrapper for a specialist that runs its whole loop in a subgraph.

    Because the model/tool loop lives inside the compiled subgraph, there is no
    tool stage to return to: the specialist either produced a candidate answer
    (go validate it) or hit a typed execution limit (go fail through the
    finalizer).

    A handoff never arrives here. The ``hand_off`` tool returns
    ``Command(graph=PARENT, goto="resolve_transition")``, which propagates out
    of the subgraph as a ``ParentCommand`` the parent loop applies directly —
    this wrapper is bypassed entirely.
    """

    async def wrapper(state: dict[str, Any], runtime: Any = None) -> Command:
        active_agent_id = state.get("active_agent_id")
        if resolve_node_for_agent_id(active_agent_id) != node_name:
            logger.error(
                "Specialist node %s reached while active agent is %s",
                node_name,
                active_agent_id,
            )
            return Command(
                update={
                    "execution_phase": "failed",
                    "workflow_error": WorkflowError(
                        code="response_validation_failed",
                        retriable=False,
                        request_id=_request_id(state),
                        details={"reason": "active_agent_node_mismatch", "node": node_name},
                    ),
                },
                goto="finalize",
            )

        try:
            outcome = await invoke(state)
        except (ModelCallLimitExceededError, ToolCallLimitExceededError) as exc:
            # Only the framework's own limit errors map to this code; every
            # other failure keeps its own typed mapping.
            get_routing_metrics_recorder().agent_execution_limit(
                agent_id=active_agent_id, limit_kind=type(exc).__name__
            )
            return Command(
                update={
                    "execution_phase": "failed",
                    "workflow_error": WorkflowError(
                        code="agent_execution_limit",
                        retriable=False,
                        request_id=_request_id(state),
                        details={"reason": type(exc).__name__, "agent": str(active_agent_id)},
                    ),
                },
                goto="finalize",
            )

        update: dict[str, Any] = {
            "agent_outcome": outcome,
            "execution_phase": "validating",
        }
        budget = _outcome_budget(outcome)
        if budget is not None:
            update["execution_budget"] = budget
        return Command(update=update, goto="validate_output")

    wrapper.__name__ = f"{node_name}_subgraph_wrapper"
    return wrapper


def make_tool_stage_wrapper(
    node_name: str,
    stage: SpecialistCallable,
    *,
    stage_router: StageRouter,
) -> Callable[..., Awaitable[Command]]:
    """Wrap a pre-v2 tool/approval stage so it routes dynamically.

    These stages hand control back to a specialist or, when the turn is done,
    to ``validate_output``. They never reach ``END``.
    """

    async def wrapper(state: dict[str, Any], runtime: Any = None) -> Command:
        result = await stage(dict(state))
        result = result if isinstance(result, dict) else state
        update = _carry_forward(state, result)

        decision = stage_router(result)
        if _is_awaitable(decision):
            decision = await decision
        decision = str(decision)

        if decision == "end":
            response = result.get("response")
            if isinstance(response, AgentResponse):
                update["agent_outcome"] = _response_outcome(
                    str(result.get("active_agent_id") or response.agent_id), response, result
                )
                update["execution_phase"] = "validating"
                return Command(update=update, goto="validate_output")
            return Command(
                update={
                    **update,
                    "execution_phase": "failed",
                    "workflow_error": WorkflowError(
                        code="tool_execution_failed",
                        retriable=False,
                        request_id=_request_id(state),
                        details={"reason": "stage_ended_without_response", "stage": node_name},
                    ),
                },
                goto="finalize",
            )

        if decision == "approval":
            return Command(update=update, goto="approval")
        if decision == "tools":
            return Command(update=update, goto="tools")

        target = resolve_node_for_agent_id(decision) or "validate_output"
        if decision != (result.get("active_agent_id") or decision):
            update["active_agent_id"] = decision
        return Command(update=update, goto=target)

    wrapper.__name__ = f"{node_name}_wrapper"
    return wrapper


def _is_awaitable(value: Any) -> bool:
    return hasattr(value, "__await__")


def _request_id(state: dict[str, Any]) -> str:
    identity = state.get("turn_identity")
    request_id = getattr(identity, "request_id", None)
    return str(request_id) if request_id else "unknown"


# ======================================================================
# Per-invocation specialist subgraphs
# ======================================================================


@dataclass(frozen=True)
class SpecialistDefinition:
    """Everything that makes one specialist different from another.

    Deliberately configuration, not behavior: the model/tool loop belongs to
    the framework, so a specialist declares its prompt, its tools, and the
    output contracts its answers must satisfy — nothing else.

    ``agent`` is the object those factories already close over. It is named
    here because tool *execution* needs it too: the execution map, the
    deferred-tool state key, and the mid-turn tool refresh are all keyed off
    the agent that owns the tools.
    """

    agent_id: str
    agent_type: AgentType
    model_config_key: str
    system_prompt_factory: Callable[[SpecialistRequest], Awaitable[str] | str]
    tool_factory: Callable[[SpecialistRequest], Awaitable[list[Any]] | list[Any]]
    agent: Any = None
    output_policy_ids: tuple[str, ...] = ()


@dataclass
class SpecialistRequest:
    """One invocation's authenticated scope and inputs.

    Every field is per-request. Nothing derived from it may be cached across
    users or devices.
    """

    agent_id: str
    conversation_id: str | None
    user_id: str | None
    device_id: str | None
    persona: str | None
    model_request: dict[str, Any] | None
    messages: list[Any]
    history: list[Any] = field(default_factory=list)
    # What an earlier epoch of this same turn already gathered. Empty for a
    # first epoch. Carried explicitly rather than recovered from the checkpoint
    # because the produced messages are sliced into the outcome's provenance
    # and never return to the request on their own.
    carried_messages: list[Any] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    hitl_policy: dict[str, Any] | None = None
    attachments: list[Any] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def build_worker_request(task: WorkerTask, state: dict[str, Any]) -> SpecialistRequest:
    """The one way a dispatched task becomes a specialist invocation.

    The objective travels as the worker's ``HumanMessage``, not as part of a
    system instruction: a worker that reads its objective from its own prompt
    cannot distinguish the task from its standing rules, and a plan that says
    "ignore your instructions" would then be indistinguishable from one.

    History is deliberately empty. A worker is given a bounded parent context
    and its own objective; replaying the public conversation would let it
    answer the user directly instead of doing the delegated work.
    """
    context = state.get("context") or {}
    identity = state.get("turn_identity")
    return SpecialistRequest(
        agent_id=task.agent_id,
        conversation_id=state.get("conversation_id"),
        user_id=state.get("user_id"),
        device_id=state.get("device_id"),
        persona=state.get("persona"),
        model_request=task.model_request or state.get("model_request"),
        messages=[HumanMessage(content=task.objective)],
        history=[],
        state={"attachments": state.get("attachments") or [], "context": context},
        hitl_policy=_hitl_policy(context),
        attachments=list(state.get("attachments") or []),
        extras={
            "worker": True,
            "dispatch_id": task.dispatch_id,
            "task_id": task.task_id,
            "position": task.position,
            "parent_context": task.parent_context,
            "allowed_tool_ids": task.allowed_tool_ids,
            "hitl_policy": _hitl_policy(context),
            "custom_agents": state.get("custom_agents") or {},
            # Receipt identity. Runtime-owned, so a mutation this worker makes
            # is keyed to this turn and this task and nothing else.
            "thread_id": getattr(identity, "checkpoint_thread_id", None),
            "turn_id": getattr(identity, "turn_id", None),
        },
    )


def _hitl_policy(context: Any) -> dict[str, Any]:
    """The request's approval policy. A worker inherits it, never a default.

    Falling back to a permissive default would make a delegated mutation
    unapproved on exactly the path the user cannot see.
    """
    if not isinstance(context, dict):
        return {}
    policy = context.get("hitl_policy")
    return dict(policy) if isinstance(policy, dict) else {}


@dataclass(frozen=True)
class SpecialistRuntimeContext:
    """Typed runtime context handed to a compiled specialist subgraph."""

    agent_id: str
    conversation_id: str | None
    user_id: str | None
    device_id: str | None
    persona: str | None


class SpecialistFactory:
    """Compiles and runs one specialist subgraph per invocation.

    A compiled agent is never reused across invocations. That is not caution
    for its own sake: the prompt, tool set, model, and credentials are all
    scoped to one authenticated caller, so a shared instance would leak one
    user's execution scope into another's turn.
    """

    def __init__(
        self,
        *,
        definitions: dict[str, SpecialistDefinition],
        runtime_model_resolver: Any,
        model_factory: Any,
        agent_builder: Callable[..., Any] | None = None,
        usage_recorder: Any = None,
        settings: Any = None,
        receipt_service: Any = None,
    ) -> None:
        self._definitions = dict(definitions)
        self._runtime_model_resolver = runtime_model_resolver
        self._model_factory = model_factory
        self._agent_builder = agent_builder or _default_agent_builder
        self._usage_recorder = usage_recorder
        self._settings = settings
        self._receipt_service = receipt_service

    # -- registry --------------------------------------------------------

    def register(self, definition: SpecialistDefinition) -> None:
        self._definitions[definition.agent_id] = definition

    def definition_for(self, agent_id: str) -> SpecialistDefinition:
        definition = self._definitions.get(agent_id)
        if definition is None and agent_id.startswith(CUSTOM_AGENT_PREFIX):
            definition = self._definitions.get(CUSTOM_AGENT_NODE)
        if definition is None:
            raise UnavailableSpecialist(f"no specialist definition for {agent_id!r}")
        return definition

    # -- public invocation -----------------------------------------------

    async def invoke(self, request: SpecialistRequest) -> ResponseOutcome:
        """Run a specialist for a public turn and return a server-owned outcome."""
        definition = self.definition_for(request.agent_id)
        agent, tool_execution, accountant = await self._build(definition, request)

        try:
            result = await agent.ainvoke(
                {"messages": self._invocation_messages(request)},
                context=self._runtime_context(request),
                config=self._run_config(request),
            )
        except (ModelCallLimitExceededError, ToolCallLimitExceededError) as exc:
            # The soft budget was supposed to reserve an answer one call
            # earlier. It did not, so this is the last honest thing that can be
            # said -- and saying it beats reporting agent_execution_limit, which
            # discards the artifacts the pipeline already recorded and gives the
            # user nothing to act on.
            return self._hard_limit_outcome(definition, request, tool_execution, accountant, exc)
        _reject_swallowed_interrupt(result, request.agent_id)
        produced = self._produced_messages(request, result)
        return self._to_outcome(definition, request, produced, tool_execution, accountant)

    async def invoke_worker(self, request: SpecialistRequest, *, task: WorkerTask) -> WorkerResult:
        """Run a specialist as a Planning worker.

        A worker returns a typed private result. It never appends a public
        assistant message and never performs a parent-level handoff.

        The ``except`` order below is the contract. ``GraphBubbleUp`` is
        re-raised first and unwrapped: an approval interrupt travelling out of
        a worker is control flow, and normalizing it into a failed result would
        answer the turn without the human who was asked. Everything after it is
        a genuine failure, narrowed before the generic case so a limit or a
        timeout keeps its own code.
        """
        if request.agent_id == PLANNING_AGENT_ID:
            return _failed_worker(task, "recursive_planning")

        # Bound outside the `try` so the limit handler can still reach the
        # records the tool pipeline wrote before the ceiling fired. Reporting a
        # bare failure there discarded them, which is the loss R1 objects to.
        tool_execution: ToolExecutionMiddleware | None = None
        try:
            definition = self.definition_for(request.agent_id)
            # A delegated worker gets the same budget as a top-level turn: it
            # runs the same specialists through the same builder, so leaving it
            # out would have been the R1 gap one level down.
            agent, tool_execution, _accountant = await self._build(definition, request)
            result = await agent.ainvoke(
                {"messages": self._invocation_messages(request)},
                context=self._runtime_context(request),
                config=self._run_config(request),
            )
        except GraphBubbleUp:
            raise
        except (ModelCallLimitExceededError, ToolCallLimitExceededError) as exc:
            get_routing_metrics_recorder().agent_execution_limit(
                agent_id=task.agent_id, limit_kind=type(exc).__name__
            )
            return _partial_worker(task, tool_execution)
        except TimeoutError:
            return _failed_worker(task, "worker_timeout")
        except UnavailableSpecialist:
            return _failed_worker(task, "agent_unavailable")
        except Exception as exc:  # noqa: BLE001 - normalized into a typed result
            logger.warning("Worker %s failed for task %s: %s", request.agent_id, task.task_id, exc)
            return _failed_worker(task, "tool_execution_failed")

        produced = self._produced_messages(request, result)
        return WorkerResult(
            dispatch_id=task.dispatch_id,
            task_id=task.task_id,
            position=task.position,
            agent_id=request.agent_id,
            status="completed",
            content=_final_text(produced),
            artifacts=tuple(tool_execution.artifacts),
            images=tuple(tool_execution.images),
        )

    # -- construction ----------------------------------------------------

    async def _build(
        self,
        definition: SpecialistDefinition,
        request: SpecialistRequest,
    ) -> tuple[Any, ToolExecutionMiddleware, ExecutionBudgetAccountant]:
        system_prompt = await _resolve(definition.system_prompt_factory, request)
        tools = await _resolve(definition.tool_factory, request) or []

        worker_scope = _worker_tool_scope(request)
        if worker_scope is not None:
            tools = worker_scope.filter_tools(tools)

        scope = SpecialistToolScope(
            agent=definition.agent,
            agent_key=_tool_state_key(definition),
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            device_id=request.device_id,
            rich_response_capable=rich_response_capable_from_context(request.state.get("context")),
            internal_tools=request.extras.get("internal_tools"),
            receipt_service=self._receipt_service,
            thread_id=_extra_str(request, "thread_id"),
            turn_id=_extra_str(request, "turn_id"),
            # A top-level call's task is the active specialist, so it can never
            # collide on an execution key with a delegated one.
            dispatch_id=_extra_str(request, "dispatch_id") or TOP_LEVEL_DISPATCH_ID,
            task_id=_extra_str(request, "task_id") or request.agent_id,
        )
        # Seed the scope with the tools bound at build time. A resumed run
        # re-enters after the model call, so the refresh in the execution
        # middleware never fires and this is the only offer it gets.
        scope.offer(tools)
        accountant = _budget_accountant(self._settings, request)

        async def _live_tools() -> list[Any]:
            """The tool set for the next model call, or none once out of room.

            This is where forced synthesis actually takes effect. The execution
            middleware re-consults this factory on every model call, so
            returning nothing here is what makes the reserved answer call
            tool-free -- and it holds through provider retries and fallbacks,
            which re-enter the same chain.
            """
            if accountant.state.forced_synthesis:
                return []
            return await _resolve(definition.tool_factory, request) or []

        tool_execution = ToolExecutionMiddleware(scope=scope, tool_factory=_live_tools)

        middleware = build_specialist_middleware(
            runtime_model_resolver=self._runtime_model_resolver,
            model_factory=self._model_factory,
            agent_key=definition.model_config_key,
            agent_id=definition.agent_id,
            user_id=request.user_id,
            model_request=request.model_request,
            usage_recorder=self._usage_recorder,
            hitl_policy=request.hitl_policy,
            # The framework ceilings are the hard rungs of one ladder with the
            # soft budget. Reading them from anywhere else is how the framework
            # comes to raise on the very call the budget reserved.
            max_model_calls=accountant.limits.hard_model_calls,
            max_tool_calls=accountant.limits.hard_tool_calls,
            tool_execution=tool_execution,
            budget=SoftExecutionBudgetMiddleware(accountant=accountant),
            approval=ToolApprovalMiddleware(scope=scope, hitl_policy=request.hitl_policy or {}),
            worker_tool_scope=worker_scope,
            preflight=_preflight_for(definition, request),
        )

        # The model is resolved inside RuntimeModelMiddleware per attempt; the
        # placeholder here only satisfies create_agent's constructor.
        agent = self._agent_builder(
            model=None,
            tools=tools,
            system_prompt=system_prompt,
            middleware=middleware,
            context_schema=SpecialistRuntimeContext,
        )
        return agent, tool_execution, accountant

    def _limit(self, name: str, default: int) -> int:
        return int(getattr(self._settings, name, default) or default)

    def _runtime_context(self, request: SpecialistRequest) -> SpecialistRuntimeContext:
        return SpecialistRuntimeContext(
            agent_id=request.agent_id,
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            device_id=request.device_id,
            persona=request.persona,
        )

    @staticmethod
    def _run_config(request: SpecialistRequest) -> dict[str, Any]:
        return {"tags": [f"specialist:{request.agent_id}"]}

    @staticmethod
    def _invocation_messages(request: SpecialistRequest) -> list[Any]:
        """History, then what this turn already learned, then what was asked.

        The carried evidence sits before the question so the model reads it as
        established context rather than as a fresh turn to react to. It is
        pair-complete by construction (see ``continuation.carry_messages``);
        putting it after the question would separate a tool call from its
        result with a human message, which some providers reject.
        """
        return [*request.history, *request.carried_messages, *request.messages]

    @staticmethod
    def _produced_messages(request: SpecialistRequest, result: Any) -> list[Any]:
        messages = result.get("messages") if isinstance(result, dict) else None
        if not isinstance(messages, list):
            return []
        sent = len(request.history) + len(request.messages)
        return messages[sent:] if len(messages) > sent else []

    def _hard_limit_outcome(
        self,
        definition: SpecialistDefinition,
        request: SpecialistRequest,
        tool_execution: ToolExecutionMiddleware,
        accountant: ExecutionBudgetAccountant,
        exc: BaseException,
    ) -> ResponseOutcome:
        """A validated partial built from what survived the exception.

        The model's own text did not: it was mid-loop when the ceiling fired.
        What did survive is the artifact and image records the tool pipeline
        wrote, and the counters, so the message is server-owned and says
        exactly that rather than inventing an answer.
        """
        accountant.note_hard_limit()
        get_routing_metrics_recorder().agent_execution_limit(
            agent_id=request.agent_id, limit_kind=type(exc).__name__
        )
        logger.warning(
            "Specialist %s hit the execution ceiling (%s); returning a server-owned partial",
            request.agent_id,
            type(exc).__name__,
        )
        return self._to_outcome(
            definition,
            request,
            [AIMessage(content=HARD_LIMIT_PARTIAL_TEXT)],
            tool_execution,
            accountant,
        )

    def _to_outcome(
        self,
        definition: SpecialistDefinition,
        request: SpecialistRequest,
        produced: list[Any],
        tool_execution: ToolExecutionMiddleware,
        accountant: ExecutionBudgetAccountant | None = None,
    ) -> ResponseOutcome:
        artifacts = list(tool_execution.artifacts)
        images = list(tool_execution.images)
        metadata: dict[str, Any] = {"images": images} if images else {}
        if accountant is not None:
            # Carried on the response so the graph can read why the turn
            # stopped without reaching back into middleware that has gone.
            metadata["execution_budget"] = accountant.state.model_dump(mode="json")
        response = AgentResponse(
            agent_type=definition.agent_type,
            agent_id=request.agent_id,
            message=AgentMessage(role=MessageRole.ASSISTANT, content=_final_text(produced)),
            metadata=metadata,
            tool_artifacts=artifacts or None,
        )
        return ResponseOutcome(
            agent_id=request.agent_id,
            response=response,
            provenance=OutcomeProvenance(
                output_policy_ids=definition.output_policy_ids,
                artifacts=tuple(artifacts),
                images=tuple(images),
                private_messages=tuple(produced),
            ),
        )


def _outcome_budget(outcome: Any) -> dict[str, Any] | None:
    """The budget snapshot an outcome carries, wherever it produced one.

    Specialists put it on the response metadata; the RAG graph returns it on
    its result and ``_invoke_rag_specialist`` copies it to the same place. One
    reader means the graph does not have to know which path answered.
    """
    metadata = getattr(getattr(outcome, "response", None), "metadata", None)
    if not isinstance(metadata, dict):
        return None
    budget = metadata.get("execution_budget")
    return dict(budget) if isinstance(budget, dict) and budget else None


def _budget_accountant(
    settings: Any, request: SpecialistRequest
) -> ExecutionBudgetAccountant:
    """Build this invocation's accountant, resuming a carried epoch if there is one.

    A Continue may be served by a worker that never ran the previous epoch, so
    the state arrives on the request rather than from process memory. An absent
    one is a first epoch, not an empty quota.
    """
    carried = request.extras.get("execution_budget")
    state: ExecutionBudgetState | None = None
    if isinstance(carried, dict):
        try:
            state = ExecutionBudgetState.model_validate(carried)
        except Exception:
            logger.warning("Ignored an unreadable carried execution budget")
    return ExecutionBudgetAccountant(
        limits=ExecutionBudgetLimits.from_settings(settings), state=state
    )


def _extra_str(request: SpecialistRequest, key: str) -> str | None:
    """One runtime-supplied identity field, or nothing."""
    value = request.extras.get(key)
    text = str(value).strip() if value is not None else ""
    return text or None


def _worker_tool_scope(request: SpecialistRequest) -> WorkerToolScopeMiddleware | None:
    """The scope guard for a worker invocation, or nothing when unrestricted.

    An *empty* ``allowed_tool_ids`` means "this agent's own scope", not "no
    tools". The restriction a dispatch applies is the agent identity it chose —
    a chat worker already cannot reach canvas tools — and per-task narrowing is
    server-derived, not yet computed. Reading empty as fail-closed would leave
    every worker toolless while looking like a security property.

    A non-empty set is a real narrowing and is enforced: the model is offered
    only those tools, and any call outside them is refused before approval.
    """
    if not request.extras.get("worker"):
        return None
    allowed = tuple(request.extras.get("allowed_tool_ids") or ())
    if not allowed:
        return None
    return WorkerToolScopeMiddleware(allowed_tool_ids=allowed)


def _failed_worker(task: WorkerTask, error_code: str) -> WorkerResult:
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


def _partial_worker(
    task: WorkerTask, tool_execution: ToolExecutionMiddleware | None
) -> WorkerResult:
    """A worker that ran out of budget with evidence already in hand.

    The model's own text died with the exception, so the content is the same
    server-owned statement a top-level hard limit produces -- ``partial`` is
    what tells the synthesizing parent to read it as unfinished rather than as
    an answer. ``error_code`` stays unset: a partial is not an error, and
    populating it would render as one wherever a worker end is displayed.

    ``tool_execution`` is ``None`` only when the ceiling fired before the stack
    was assembled, which cannot happen through ``_build`` but is cheap to allow.
    """
    return WorkerResult(
        dispatch_id=task.dispatch_id,
        task_id=task.task_id,
        position=task.position,
        agent_id=task.agent_id,
        status="partial",
        content=HARD_LIMIT_PARTIAL_TEXT,
        artifacts=tuple(getattr(tool_execution, "artifacts", ()) or ()),
        images=tuple(getattr(tool_execution, "images", ()) or ()),
    )


def _preflight_for(definition: SpecialistDefinition, request: SpecialistRequest):
    """The token-budget preflight for one specialist invocation.

    The budget is a property of the resolved provider and the assembled
    request, so it runs per model attempt rather than once per turn: a
    fallback to a different provider is a different budget.
    """
    agent = definition.agent
    if agent is None or not hasattr(agent, "_preflight_model_request"):
        return None

    async def preflight(model_request: Any, runtime_config: Any) -> list[Any] | None:
        if runtime_config is None:
            return None
        messages = list(model_request.messages)
        turn_start = min(len(request.history), len(messages))
        system_message = getattr(model_request, "system_message", None)
        result = await agent._preflight_model_request(
            runtime_config,
            system_messages=[system_message] if system_message is not None else [],
            history_messages=messages[:turn_start],
            current_messages=messages[turn_start:],
            tools=list(getattr(model_request, "tools", None) or ()),
            attachments=request.attachments,
            conversation_id=request.conversation_id,
            user_id=request.user_id,
        )
        if result is None:
            return None
        envelope = result.envelope
        return [*envelope.history_messages, *envelope.current_messages]

    return preflight


def _reject_swallowed_interrupt(result: Any, agent_id: str) -> None:
    """Refuse to answer for a run that actually paused for approval.

    Running inside a parent graph, ``interrupt()`` propagates and the parent
    pauses. Running standalone there is no loop to pause, so the subgraph
    returns a state carrying ``__interrupt__`` and an empty answer — which
    would publish silence as if the specialist had nothing to say.
    """
    if isinstance(result, dict) and result.get("__interrupt__"):
        raise RuntimeError(f"{agent_id} paused for approval outside a resumable graph")


def _tool_state_key(definition: SpecialistDefinition) -> str:
    """The key deferred tool state and the execution context are filed under."""
    agent = definition.agent
    return str(
        getattr(agent, "tool_state_key", None)
        or getattr(agent, "agent_config_key", None)
        or definition.model_config_key
    )


def _default_agent_builder(**kwargs: Any) -> Any:
    from langchain.agents import create_agent

    return create_agent(**kwargs)


async def _resolve(factory: Callable[[Any], Any], request: SpecialistRequest) -> Any:
    value = factory(request)
    if hasattr(value, "__await__"):
        return await value
    return value


def _final_text(messages: list[Any]) -> str:
    """Last assistant text produced by the subgraph.

    Intermediate tool-calling turns carry no answer, so they are skipped
    rather than concatenated: the answer is the last message that actually
    said something.
    """
    for message in reversed(messages):
        if getattr(message, "type", None) != "ai":
            continue
        content = getattr(message, "content", "")
        text = content if isinstance(content, str) else _coerce_blocks(content)
        if text.strip():
            return text
    return ""


def _coerce_blocks(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)
