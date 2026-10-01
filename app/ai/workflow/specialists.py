"""Parent-level specialist wrappers for the routing-v2 graph.

A wrapper does three things and nothing else:

1. verifies the node it is running in matches ``state["active_agent_id"]``;
2. runs the specialist;
3. converts the result into a ``ResponseOutcome`` and a dynamic ``Command``.

Wrappers have no static outgoing edges, so the parent graph's dynamic routing
is the only thing that decides what runs next. No wrapper appends the terminal
public ``AIMessage`` and no wrapper reaches ``END`` — ``finalize`` owns both.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from app.ai.hitl_config import policy_from_context
from app.ai.image_generation import current_media_delivery_service
from app.ai.research_budget import get_research_budget
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.tool_context import rich_response_capable_from_context
from app.ai.utils import coerce_response_text, extract_inline_images_from_content
from app.ai.web_research.contracts import ResearchScope
from app.ai.web_research.grounding import GroundingParser, GroundingResolution
from app.ai.web_research.providers import (
    BraveImageSearchProvider,
    ProviderResolver,
    TavilyPageOpenProvider,
    TavilyTextSearchProvider,
)
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
    "PLANNING_AGENT_ID",
    "ModelCallLimitExceededError",
    "SpecialistBuild",
    "SpecialistDefinition",
    "SpecialistFactory",
    "SpecialistRequest",
    "SpecialistRuntimeContext",
    "ToolCallLimitExceededError",
    "UnavailableSpecialist",
    "build_worker_request",
    "make_subgraph_specialist_wrapper",
    "planning_worker_run_config",
    "resolve_node_for_agent_id",
]


class UnavailableSpecialist(KeyError):
    """No definition exists for the requested agent in this request's scope."""


def planning_worker_run_config(
    *, agent_id: str, dispatch_id: str | None, task_id: str | None
) -> dict[str, Any]:
    """Tag a private worker model run so its stream can be attributed."""
    return {
        "tags": [f"specialist:{agent_id}", "internal", "planning_subagent"],
        "metadata": {
            "purpose": "planning_subagent",
            "subagent_dispatch_id": dispatch_id,
            "subagent_task_id": task_id,
            "subagent_agent": agent_id,
        },
    }


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
    unapproved on exactly the path the user cannot see. A turn without a
    per-user policy gets the process-wide one, as every other gate does.
    """
    return dict(policy_from_context(context if isinstance(context, dict) else None))


def _effective_hitl_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    """The policy a specialist's approval gate enforces.

    ``None`` means "no per-user policy", not "approval off": leaving the gate
    out would skip the global tool list and the mutation floor. Only an
    explicit ``master_enabled: False`` turns approval off.
    """
    return _hitl_policy({"hitl_policy": policy})


@dataclass(frozen=True)
class SpecialistRuntimeContext:
    """Typed runtime context handed to a compiled specialist subgraph."""

    agent_id: str
    conversation_id: str | None
    user_id: str | None
    device_id: str | None
    persona: str | None


@dataclass(frozen=True)
class SpecialistBuild:
    agent: Any
    tool_execution: ToolExecutionMiddleware
    accountant: ExecutionBudgetAccountant
    web_research_session: Any = None


#: Builds the definition one custom-agent invocation runs from that request's
#: own roster, or returns ``None`` when the agent is not attached to it.
CustomDefinitionResolver = Callable[[SpecialistRequest], SpecialistDefinition | None]


class SpecialistFactory:
    """Compiles and runs one specialist subgraph per invocation.

    A compiled agent is never reused across invocations. That is not caution
    for its own sake: the prompt, tool set, model, and credentials are all
    scoped to one authenticated caller, so a shared instance would leak one
    user's execution scope into another's turn.

    The same holds for definitions. The fixed ones are the base specialists;
    a custom agent's definition belongs to one turn's roster and is resolved
    from the request every time, never stored here. Storing it made every turn
    share one mutable registry: a later turn ran an earlier turn's tools,
    handoff targets, and instructions, and a Planning worker (which never
    stored one) found none.
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
        web_research_service: Any = None,
        custom_definition_resolver: CustomDefinitionResolver | None = None,
    ) -> None:
        self._definitions = dict(definitions)
        self._runtime_model_resolver = runtime_model_resolver
        self._model_factory = model_factory
        self._agent_builder = agent_builder or _default_agent_builder
        self._usage_recorder = usage_recorder
        self._settings = settings
        self._receipt_service = receipt_service
        self._web_research_service = web_research_service
        self._custom_definition_resolver = custom_definition_resolver

    # -- resolution ------------------------------------------------------

    def definition_for(self, request: SpecialistRequest) -> SpecialistDefinition:
        """The definition this request runs.

        A custom agent's comes from the request's own roster, so an edited or
        detached agent takes effect on the turn that sees it, and a resolver
        answering for a different id is refused rather than trusted.
        """
        agent_id = request.agent_id
        if agent_id.startswith(CUSTOM_AGENT_PREFIX):
            resolver = self._custom_definition_resolver
            definition = resolver(request) if resolver is not None else None
            if definition is None or definition.agent_id != agent_id:
                raise UnavailableSpecialist(f"custom agent {agent_id!r} is not attached")
            return definition
        definition = self._definitions.get(agent_id)
        if definition is None:
            raise UnavailableSpecialist(f"no specialist definition for {agent_id!r}")
        return definition

    # -- public invocation -----------------------------------------------

    async def invoke(self, request: SpecialistRequest) -> ResponseOutcome:
        """Run a specialist for a public turn and return a server-owned outcome."""
        definition = self.definition_for(request)
        build = await self._build(definition, request)

        try:
            invocation_messages = self._invocation_messages(request)
            result = await build.agent.ainvoke(
                {"messages": invocation_messages},
                context=self._runtime_context(request),
                config=self._run_config(request),
            )
            _reject_swallowed_interrupt(result, request.agent_id)
            produced = self._produced_messages(request, result)
            grounding = None
            if build.web_research_session is not None:
                parser = GroundingParser(build.web_research_session)
                grounding = parser.resolve(_final_text(produced))
                if build.web_research_session.answer_sources and not grounding.source_ids:
                    correction = HumanMessage(
                        content=(
                            "Your previous draft cannot be published because it did not cite an "
                            "admitted web source. Rewrite the complete answer now using at least "
                            "one [[source:S#]] token from WEB EVIDENCE. If you show an image, use "
                            "[[image:I#]] only for an image you inspected and include its "
                            "supporting [[source:S#]]. Do not call tools or mention this "
                            "correction."
                        )
                    )
                    repair_input = [*invocation_messages, *produced, correction]
                    previous_disable_tools = request.extras.get("disable_tools")
                    request.extras["disable_tools"] = True
                    try:
                        repaired = await build.agent.ainvoke(
                            {"messages": repair_input},
                            context=self._runtime_context(request),
                            config=self._run_config(request),
                        )
                    finally:
                        if previous_disable_tools is None:
                            request.extras.pop("disable_tools", None)
                        else:
                            request.extras["disable_tools"] = previous_disable_tools
                    _reject_swallowed_interrupt(repaired, request.agent_id)
                    repaired_messages = (
                        repaired.get("messages") if isinstance(repaired, dict) else []
                    )
                    repaired_produced = (
                        repaired_messages[len(repair_input) :]
                        if isinstance(repaired_messages, list)
                        else []
                    )
                    produced = [*produced, correction, *repaired_produced]
                    grounding = parser.resolve(_final_text(repaired_produced))
                produced = _replace_final_text(produced, grounding.text)
                await build.web_research_session.finish(grounding.selected_image_ids)
        except GraphBubbleUp:
            if build.web_research_session is not None:
                await build.web_research_session.abort()
            raise
        except (ModelCallLimitExceededError, ToolCallLimitExceededError):
            if build.web_research_session is not None:
                await build.web_research_session.abort()
            raise
        except BaseException:
            if build.web_research_session is not None:
                await build.web_research_session.abort()
            raise
        outcome = self._to_outcome(
            definition,
            request,
            produced,
            build.tool_execution,
            build.accountant,
            grounding=grounding,
            web_research_session=build.web_research_session,
        )
        if not outcome.response.message.content.strip() and not outcome.provenance.images:
            # Why the model returned no text is not established, and a recovery
            # here would hide the evidence needed to find out. The turn fails,
            # but retriably (see `PublicContentPolicy`), and this line carries
            # what the old `details={"reason": ...}` never did: one occurrence
            # was otherwise indistinguishable from any other.
            logger.warning(
                "Specialist %s produced %d message(s) and no answer text. Final content: %.300r",
                request.agent_id,
                len(produced),
                getattr(produced[-1], "content", None) if produced else None,
            )
        return outcome

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
        web_research_session = None
        try:
            definition = self.definition_for(request)
            # A delegated worker gets the same budget as a top-level turn: it
            # runs the same specialists through the same builder, so leaving it
            # out would have been the R1 gap one level down.
            build = await self._build(definition, request)
            tool_execution = build.tool_execution
            web_research_session = build.web_research_session
            result = await build.agent.ainvoke(
                {"messages": self._invocation_messages(request)},
                context=self._runtime_context(request),
                config=self._run_config(request),
            )
        except GraphBubbleUp:
            if web_research_session is not None:
                await web_research_session.abort()
            raise
        except (ModelCallLimitExceededError, ToolCallLimitExceededError) as exc:
            if web_research_session is not None:
                await web_research_session.abort()
            get_routing_metrics_recorder().agent_execution_limit(
                agent_id=task.agent_id, limit_kind=type(exc).__name__
            )
            return _partial_worker(task, tool_execution)
        except TimeoutError:
            if web_research_session is not None:
                await web_research_session.abort()
            return _failed_worker(task, "worker_timeout")
        except UnavailableSpecialist:
            if web_research_session is not None:
                await web_research_session.abort()
            return _failed_worker(task, "agent_unavailable")
        except Exception as exc:  # noqa: BLE001 - normalized into a typed result
            if web_research_session is not None:
                await web_research_session.abort()
            logger.warning("Worker %s failed for task %s: %s", request.agent_id, task.task_id, exc)
            return _failed_worker(task, "tool_execution_failed")

        produced = self._produced_messages(request, result)
        grounded_images = tuple(
            [*tool_execution.images, *self._inline_generated_images(definition, request, produced)]
        )
        grounded_sources: tuple[dict[str, Any], ...] = ()
        if web_research_session is not None:
            resolution = GroundingParser(web_research_session).resolve(_final_text(produced))
            produced = _replace_final_text(produced, resolution.text)
            # A worker's answer is private synthesis input. Its image markers
            # must never become a parent selection the Planning model did not
            # inspect, so release worker web images at this boundary.
            await web_research_session.finish(())
            grounded_images = ()
            grounded_sources = tuple(
                source.model_dump(mode="json")
                for source in web_research_session.published_sources(resolution.source_ids)
            )
        return WorkerResult(
            dispatch_id=task.dispatch_id,
            task_id=task.task_id,
            position=task.position,
            agent_id=request.agent_id,
            status="completed",
            content=_final_text(produced) or ("Here is your image." if grounded_images else ""),
            artifacts=tuple(tool_execution.artifacts),
            evidence=grounded_sources,
            images=grounded_images,
        )

    # -- construction ----------------------------------------------------

    async def _build(
        self,
        definition: SpecialistDefinition,
        request: SpecialistRequest,
    ) -> SpecialistBuild:
        system_prompt = await _resolve(definition.system_prompt_factory, request)
        tools = await _resolve(definition.tool_factory, request) or []
        web_research_enabled = getattr(self._settings, "web_research_enabled", True)
        if not web_research_enabled:
            tools = [
                tool
                for tool in tools
                if getattr(tool, "name", None) not in {"web_search", "web_open"}
            ]

        worker_scope = _worker_tool_scope(request)
        if worker_scope is not None:
            tools = worker_scope.filter_tools(tools)

        web_research_session = None
        if (
            self._web_research_service is not None
            and web_research_enabled
            and definition.model_config_key in {"chat", "search"}
            and request.conversation_id
            and request.user_id
        ):
            provider_tools = {
                tool.name: tool
                for tool in (getattr(definition.agent, "tools", None) or ())
                if getattr(tool, "name", None)
            }
            text_tool = provider_tools.get("tavily_search")
            image_tool = provider_tools.get("brave_image_search")
            open_tool = provider_tools.get("tavily_extract")
            health_partition = hashlib.sha256(
                f"{request.user_id}:{request.device_id or 'server'}".encode()
            ).hexdigest()[:16]
            resolver = ProviderResolver(
                text=(TavilyTextSearchProvider(text_tool, health_key=f"tavily:{health_partition}"),)
                if text_tool
                else (),
                images=(
                    BraveImageSearchProvider(image_tool, health_key=f"brave:{health_partition}"),
                )
                if image_tool
                else (),
                openers=(
                    TavilyPageOpenProvider(open_tool, health_key=f"tavily:{health_partition}"),
                )
                if open_tool
                else (),
            )
            logical_turn_id = _extra_str(request, "turn_id") or request.conversation_id
            routing_decision = request.state.get("routing_decision")
            requested_mode = getattr(routing_decision, "research_mode", "none")
            session_mode = (
                requested_mode
                if requested_mode in {"quick", "agentic"}
                else "agentic"
                if definition.model_config_key == "search"
                else "quick"
            )
            web_research_session = self._web_research_service.new_session(
                ResearchScope(
                    conversation_id=request.conversation_id,
                    user_id=request.user_id,
                    logical_turn_id=logical_turn_id,
                    device_id=request.device_id,
                ),
                get_research_budget(
                    logical_turn_id=logical_turn_id,
                    conversation_id=request.conversation_id,
                ),
                mode=session_mode,
                resolver=resolver,
            )
            carried_sources = request.state.get("carried_web_sources")
            if isinstance(carried_sources, list):
                from app.ai.web_research.contracts import SourceRecord

                parsed_sources = []
                for record in carried_sources:
                    try:
                        parsed_sources.append(SourceRecord.model_validate(record))
                    except Exception:
                        logger.warning("Ignored invalid carried web source")
                web_research_session.source_registry.import_records(parsed_sources)

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
            web_research_session=web_research_session,
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
            if accountant.state.forced_synthesis or request.extras.get("disable_tools"):
                return []
            live_tools = await _resolve(definition.tool_factory, request) or []
            if not web_research_enabled:
                live_tools = [
                    tool
                    for tool in live_tools
                    if getattr(tool, "name", None) not in {"web_search", "web_open"}
                ]
            return live_tools

        tool_execution = ToolExecutionMiddleware(scope=scope, tool_factory=_live_tools)
        hitl_policy = _effective_hitl_policy(request.hitl_policy)

        middleware = build_specialist_middleware(
            runtime_model_resolver=self._runtime_model_resolver,
            model_factory=self._model_factory,
            agent_key=definition.model_config_key,
            agent_id=definition.agent_id,
            user_id=request.user_id,
            model_request=request.model_request,
            usage_recorder=self._usage_recorder,
            hitl_policy=hitl_policy,
            # The framework ceilings are the hard rungs of one ladder with the
            # soft budget. Reading them from anywhere else is how the framework
            # comes to raise on the very call the budget reserved.
            max_model_calls=accountant.limits.hard_model_calls,
            max_tool_calls=accountant.limits.hard_tool_calls,
            tool_execution=tool_execution,
            budget=SoftExecutionBudgetMiddleware(accountant=accountant),
            approval=ToolApprovalMiddleware(scope=scope, hitl_policy=hitl_policy),
            worker_tool_scope=worker_scope,
            preflight=_preflight_for(definition, request),
            web_research_session=web_research_session,
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
        return SpecialistBuild(
            agent=agent,
            tool_execution=tool_execution,
            accountant=accountant,
            web_research_session=web_research_session,
        )

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
        tags = [f"specialist:{request.agent_id}"]
        if not request.extras.get("worker"):
            return {"tags": tags}
        return planning_worker_run_config(
            agent_id=request.agent_id,
            dispatch_id=_extra_str(request, "dispatch_id"),
            task_id=_extra_str(request, "task_id"),
        )

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
        """The messages this invocation added, by position.

        Counted against everything ``_invocation_messages`` sent, the carried
        epoch included. Omitting it made the slice start inside the carried
        evidence on a Continue, so input messages were recorded as this
        epoch's own production and re-entered the transcript.
        """
        messages = result.get("messages") if isinstance(result, dict) else None
        if not isinstance(messages, list):
            return []
        sent = len(request.history) + len(request.carried_messages) + len(request.messages)
        return messages[sent:] if len(messages) > sent else []

    def _to_outcome(
        self,
        definition: SpecialistDefinition,
        request: SpecialistRequest,
        produced: list[Any],
        tool_execution: ToolExecutionMiddleware,
        accountant: ExecutionBudgetAccountant | None = None,
        *,
        grounding: GroundingResolution | None = None,
        web_research_session: Any = None,
    ) -> ResponseOutcome:
        artifacts = list(tool_execution.artifacts)
        images = [
            *tool_execution.images,
            *self._inline_generated_images(definition, request, produced),
        ]
        metadata: dict[str, Any] = {"images": images} if images else {}
        if getattr(tool_execution, "mutation_outcome_unknown", False):
            # Travels on the response so the continuation decision can read it
            # without reaching back into middleware that has gone.
            metadata["mutation_outcome_unknown"] = True
        if accountant is not None:
            # Carried on the response so the graph can read why the turn
            # stopped without reaching back into middleware that has gone.
            metadata["execution_budget"] = accountant.state.model_dump(mode="json")
        web_sources: tuple[dict[str, Any], ...] = ()
        rich_items: tuple[dict[str, Any], ...] = ()
        if web_research_session is not None:
            web_sources = tuple(
                source.model_dump(mode="json")
                for source in web_research_session.published_sources(
                    grounding.source_ids if grounding is not None else ()
                )
            )
            if web_sources:
                metadata["web_sources"] = list(web_sources)
                metadata["web_sources_version"] = 1
        if grounding is not None:
            rich_items = grounding.rich_items
            if rich_items:
                metadata["_rich_item_candidates"] = list(rich_items)
                metadata["_inline_rich_response_v1"] = True
            if grounding.warnings:
                metadata["web_grounding_warnings"] = list(grounding.warnings)
        response = AgentResponse(
            agent_type=definition.agent_type,
            agent_id=request.agent_id,
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content=_final_text(produced) or ("Here is your image." if images else ""),
            ),
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
                web_sources=web_sources,
                rich_items=rich_items,
                private_messages=tuple(produced),
            ),
        )

    @staticmethod
    def _inline_generated_images(
        definition: SpecialistDefinition, request: SpecialistRequest, produced: list[Any]
    ) -> list[dict[str, Any]]:
        """Capture images from the final image specialist reply, with delivery provenance."""
        if definition.agent_type == AgentType.IMAGE_GENERATOR:
            final_message = next(
                (
                    message
                    for message in reversed(produced)
                    if getattr(message, "type", None) == "ai"
                ),
                None,
            )
            if final_message is not None and not getattr(final_message, "tool_calls", None):
                prompt = next(
                    (
                        coerce_response_text(message.content).strip()
                        for message in reversed(request.messages)
                        if isinstance(message, HumanMessage)
                    ),
                    "",
                )
                media = current_media_delivery_service()
                max_images = max(1, getattr(definition.agent, "max_images", 1))
                images: list[dict[str, Any]] = []
                for index, inline in enumerate(
                    extract_inline_images_from_content(final_message.content)[:max_images]
                ):
                    image = {
                        "data": inline["data"],
                        "mime": inline["mime"],
                        "prompt": prompt,
                        "model": getattr(definition.agent, "model_name", ""),
                        "aspect_ratio": getattr(definition.agent, "default_aspect_ratio", "1:1"),
                    }
                    if media is not None:
                        descriptor = media.persist_final(
                            image_index=index,
                            mime=inline["mime"],
                            data_b64=inline["data"],
                        )
                        if descriptor is not None:
                            image["stored_ref"] = descriptor
                    images.append(image)
                return images
        return []


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


def _budget_accountant(settings: Any, request: SpecialistRequest) -> ExecutionBudgetAccountant:
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

    ``BaseMessage.text`` is the extraction, not a hand-rolled one. The previous
    version collected any block carrying a ``text`` key, which published a
    Gemini thought part -- ``{"type": "thinking", "text": ...}`` -- as the
    answer. langchain-core already knows which blocks are reasoning; this
    module should not be relitigating that per block type.
    """
    for message in reversed(messages):
        if getattr(message, "type", None) != "ai":
            continue
        text = getattr(message, "text", "") or ""
        if text.strip():
            return text
    return ""


def _replace_final_text(messages: list[Any], text: str) -> list[Any]:
    """Replace only the final answering message, preserving tool history."""

    revised = list(messages)
    for index in range(len(revised) - 1, -1, -1):
        message = revised[index]
        if getattr(message, "type", None) == "ai" and (getattr(message, "text", "") or "").strip():
            revised[index] = message.model_copy(update={"content": text})
            break
    return revised
