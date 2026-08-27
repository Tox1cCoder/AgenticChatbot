"""Parent-level specialist wrappers for the routing-v2 graph.

A wrapper does three things and nothing else:

1. verifies the node it is running in matches ``state["active_agent_id"]``;
2. runs the specialist;
3. converts the result into an ``AgentOutcome`` and a dynamic ``Command``.

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
from langgraph.types import Command

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.contracts import (
    HandoffOutcome,
    OutcomeProvenance,
    ResponseOutcome,
    WorkerResult,
    WorkflowError,
)
from app.ai.workflow.inventory import CUSTOM_AGENT_NODE, CUSTOM_AGENT_PREFIX
from app.ai.workflow.middleware import ArtifactCaptureMiddleware, build_specialist_middleware

PLANNING_AGENT_ID = "planning_agent"

logger = logging.getLogger(__name__)

__all__ = [
    "FINALIZE_OWNS_TERMINAL_MESSAGE",
    "PLANNING_AGENT_ID",
    "ModelCallLimitExceededError",
    "SpecialistDefinition",
    "SpecialistFactory",
    "SpecialistRequest",
    "SpecialistRuntimeContext",
    "ToolCallLimitExceededError",
    "make_specialist_wrapper",
    "make_subgraph_specialist_wrapper",
    "make_tool_stage_wrapper",
    "resolve_node_for_agent_id",
]

# Turn-scoped context flag telling the pre-v2 ``_finalize_agent_response`` path
# that the parent finalizer owns the terminal public message.
FINALIZE_OWNS_TERMINAL_MESSAGE = "v2_finalize_owns_terminal_message"

SpecialistCallable = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
StageRouter = Callable[[dict[str, Any]], Any]


def resolve_node_for_agent_id(agent_id: str | None) -> str | None:
    """Map an agent ID onto its graph node without consulting the inventory."""
    if not agent_id:
        return None
    return CUSTOM_AGENT_NODE if agent_id.startswith(CUSTOM_AGENT_PREFIX) else agent_id


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

        if isinstance(outcome, HandoffOutcome):
            return Command(
                update={"agent_outcome": outcome, "execution_phase": "executing"},
                goto="resolve_transition",
            )

        return Command(
            update={"agent_outcome": outcome, "execution_phase": "validating"},
            goto="validate_output",
        )

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
    """

    agent_id: str
    agent_type: AgentType
    model_config_key: str
    system_prompt_factory: Callable[[SpecialistRequest], Awaitable[str] | str]
    tool_factory: Callable[[SpecialistRequest], Awaitable[list[Any]] | list[Any]]
    output_policy_ids: tuple[str, ...] = ()
    # Set for specialists whose streamed tokens are internal rather than an
    # answer (an enhanced image prompt, for example).
    tokens_are_internal: bool = False


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
    state: dict[str, Any] = field(default_factory=dict)
    hitl_policy: dict[str, Any] | None = None
    attachments: list[Any] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


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
        authorize: Callable[..., bool] | None = None,
        preflight: Any = None,
    ) -> None:
        self._definitions = dict(definitions)
        self._runtime_model_resolver = runtime_model_resolver
        self._model_factory = model_factory
        self._agent_builder = agent_builder or _default_agent_builder
        self._usage_recorder = usage_recorder
        self._settings = settings
        self._authorize = authorize or (lambda *args, **kwargs: True)
        self._preflight = preflight

    # -- registry --------------------------------------------------------

    def register(self, definition: SpecialistDefinition) -> None:
        self._definitions[definition.agent_id] = definition

    def definition_for(self, agent_id: str) -> SpecialistDefinition:
        definition = self._definitions.get(agent_id)
        if definition is None and agent_id.startswith(CUSTOM_AGENT_PREFIX):
            definition = self._definitions.get(CUSTOM_AGENT_NODE)
        if definition is None:
            raise KeyError(f"no specialist definition for {agent_id!r}")
        return definition

    # -- public invocation -----------------------------------------------

    async def invoke(self, request: SpecialistRequest) -> ResponseOutcome:
        """Run a specialist for a public turn and return a server-owned outcome."""
        definition = self.definition_for(request.agent_id)
        artifact_sink = ArtifactCaptureMiddleware()
        agent = await self._build(definition, request, artifact_sink)

        result = await agent.ainvoke(
            {"messages": self._invocation_messages(request)},
            context=self._runtime_context(request),
            config=self._run_config(request),
        )
        produced = self._produced_messages(request, result)
        return self._to_outcome(definition, request, produced, artifact_sink)

    async def invoke_worker(self, request: SpecialistRequest, *, task_id: str) -> WorkerResult:
        """Run a specialist as a Planning worker.

        A worker returns a typed private result. It never appends a public
        assistant message and never performs a parent-level handoff.
        """
        if request.agent_id == PLANNING_AGENT_ID:
            return WorkerResult(
                task_id=task_id,
                agent_id=request.agent_id,
                status="failed",
                error_code="recursive_planning",
            )

        try:
            definition = self.definition_for(request.agent_id)
            artifact_sink = ArtifactCaptureMiddleware()
            agent = await self._build(definition, request, artifact_sink)
            result = await agent.ainvoke(
                {"messages": self._invocation_messages(request)},
                context=self._runtime_context(request),
                config=self._run_config(request),
            )
        except (ModelCallLimitExceededError, ToolCallLimitExceededError):
            return WorkerResult(
                task_id=task_id,
                agent_id=request.agent_id,
                status="failed",
                error_code="agent_execution_limit",
            )
        except TimeoutError:
            return WorkerResult(
                task_id=task_id,
                agent_id=request.agent_id,
                status="failed",
                error_code="worker_timeout",
            )
        except Exception as exc:  # noqa: BLE001 - normalized into a typed result
            logger.warning("Worker %s failed for task %s: %s", request.agent_id, task_id, exc)
            return WorkerResult(
                task_id=task_id,
                agent_id=request.agent_id,
                status="failed",
                error_code="tool_execution_failed",
            )

        produced = self._produced_messages(request, result)
        return WorkerResult(
            task_id=task_id,
            agent_id=request.agent_id,
            status="completed",
            content=_final_text(produced),
            artifacts=tuple(artifact_sink.artifacts),
        )

    # -- construction ----------------------------------------------------

    async def _build(
        self,
        definition: SpecialistDefinition,
        request: SpecialistRequest,
        artifact_sink: ArtifactCaptureMiddleware,
    ) -> Any:
        system_prompt = await _resolve(definition.system_prompt_factory, request)
        tools = await _resolve(definition.tool_factory, request) or []

        middleware = build_specialist_middleware(
            runtime_model_resolver=self._runtime_model_resolver,
            model_factory=self._model_factory,
            agent_key=definition.model_config_key,
            agent_id=definition.agent_id,
            user_id=request.user_id,
            device_id=request.device_id,
            model_request=request.model_request,
            usage_recorder=self._usage_recorder,
            authorize=self._authorize,
            hitl_policy=request.hitl_policy,
            max_model_calls=self._limit("specialist_max_model_calls", 8),
            max_tool_calls=self._limit("specialist_max_tool_calls", 16),
            preflight=self._preflight,
            artifact_sink=artifact_sink,
        )

        # The model is resolved inside RuntimeModelMiddleware per attempt; the
        # placeholder here only satisfies create_agent's constructor.
        return self._agent_builder(
            model=None,
            tools=tools,
            system_prompt=system_prompt,
            middleware=middleware,
            context_schema=SpecialistRuntimeContext,
        )

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
        return [*request.history, *request.messages]

    @staticmethod
    def _produced_messages(request: SpecialistRequest, result: Any) -> list[Any]:
        messages = result.get("messages") if isinstance(result, dict) else None
        if not isinstance(messages, list):
            return []
        sent = len(request.history) + len(request.messages)
        return messages[sent:] if len(messages) > sent else []

    def _to_outcome(
        self,
        definition: SpecialistDefinition,
        request: SpecialistRequest,
        produced: list[Any],
        artifact_sink: ArtifactCaptureMiddleware,
    ) -> ResponseOutcome:
        content = _final_text(produced)
        response = AgentResponse(
            agent_type=definition.agent_type,
            agent_id=request.agent_id,
            message=AgentMessage(role=MessageRole.ASSISTANT, content=content),
            metadata={},
            tool_artifacts=list(artifact_sink.artifacts) or None,
        )
        return ResponseOutcome(
            agent_id=request.agent_id,
            response=response,
            provenance=OutcomeProvenance(
                output_policy_ids=definition.output_policy_ids,
                artifacts=tuple(artifact_sink.artifacts),
                images=tuple(artifact_sink.images),
                private_messages=tuple(produced),
            ),
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
