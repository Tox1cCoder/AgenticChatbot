"""Typed control-plane contracts for the routing-v2 production workflow.

This module owns the immutable vocabulary the graph uses to describe *who*
handled a turn and *why*. It deliberately contains no message interpretation,
no agent defaults, and no provider knowledge: every value here is either
produced by a schema-constrained model call or constructed by server-owned
code and validated before it reaches state.

Dependency rule: this module may import leaf response types from
``app.ai.schemas`` but ``app.ai.schemas`` must never import this module. That
keeps ``contracts`` loadable from every workflow component without a cycle.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from app.ai.schemas import AgentResponse

__all__ = [
    "EXECUTION_PHASES",
    "WORKFLOW_ERROR_CODES",
    "AgentTransition",
    "DispatchSubagentsInput",
    "ExecutionPhase",
    "InvalidWorkflowStateUpdate",
    "OutcomeProvenance",
    "PendingTransition",
    "PlanningDispatch",
    "ResponseOutcome",
    "RoutingDecision",
    "TransitionSource",
    "TurnIdentity",
    "WorkerResult",
    "WorkerStatus",
    "WorkerTask",
    "WorkerTaskProposal",
    "WorkflowError",
    "WorkflowErrorCode",
    "WorkflowRoutingException",
]


ExecutionPhase = Literal[
    "routing",
    "executing",
    "awaiting_approval",
    "validating",
    "finalizing",
    "completed",
    "failed",
]

EXECUTION_PHASES: tuple[ExecutionPhase, ...] = (
    "routing",
    "executing",
    "awaiting_approval",
    "validating",
    "finalizing",
    "completed",
    "failed",
)

TransitionSource = Literal["router", "handoff", "resume"]

WorkflowErrorCode = Literal[
    "routing_timeout",
    "routing_provider_unavailable",
    "routing_invalid_output",
    "routing_target_unavailable",
    "agent_execution_limit",
    "tool_execution_failed",
    "response_validation_failed",
    "finalization_failed",
    "response_persistence_failed",
    "conversation_turn_conflict",
]

WORKFLOW_ERROR_CODES: tuple[WorkflowErrorCode, ...] = (
    "routing_timeout",
    "routing_provider_unavailable",
    "routing_invalid_output",
    "routing_target_unavailable",
    "agent_execution_limit",
    "tool_execution_failed",
    "response_validation_failed",
    "finalization_failed",
    "response_persistence_failed",
    "conversation_turn_conflict",
)

#: ``partial`` is a worker that hit its execution ceiling with real evidence in
#: hand. It is deliberately neither of the others: ``failed`` tells the
#: synthesizing parent to disregard the result and discards the artifacts, while
#: ``completed`` would claim the objective was met. Only the top-level turn
#: pauses for a Continue decision; a delegated worker reports what it has and
#: lets its parent decide (R1).
WorkerStatus = Literal["completed", "partial", "failed"]


class RoutingDecision(BaseModel):
    """Schema-constrained output of one router model call.

    ``confidence`` is telemetry. No branch in the runtime may read it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1, max_length=160)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=500)


class TurnIdentity(BaseModel):
    """Stable identifiers for exactly one user turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1, max_length=160)
    turn_id: str = Field(min_length=1, max_length=160)
    checkpoint_thread_id: str = Field(min_length=1, max_length=320)


class AgentTransition(BaseModel):
    """One accepted movement of execution authority within a turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_agent_id: str | None = None
    to_agent_id: str = Field(min_length=1, max_length=160)
    source: TransitionSource
    tool_call_id: str | None = None


class PendingTransition(BaseModel):
    """A handoff request awaiting control-plane validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_agent_id: str = Field(min_length=1, max_length=160)
    to_agent_id: str = Field(min_length=1, max_length=160)
    tool_call_id: str = Field(min_length=1, max_length=160)
    tool_message_id: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=500)


class OutcomeProvenance(BaseModel):
    """Server-owned record of what produced a response.

    Model text can never declare its own evidence, artifacts, images, or
    validation authority; every field here is written by runtime code.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_policy_ids: tuple[str, ...] = ()
    evidence: tuple[dict[str, JsonValue], ...] = ()
    artifacts: tuple[dict[str, JsonValue], ...] = ()
    images: tuple[dict[str, JsonValue], ...] = ()
    private_messages: tuple[AnyMessage, ...] = ()


class ResponseOutcome(BaseModel):
    """A specialist finished its work and produced a candidate public answer.

    There is no sibling handoff outcome: a handoff leaves the subgraph as a
    parent ``Command`` and never passes through the specialist wrapper.
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1, max_length=160)
    response: AgentResponse
    provenance: OutcomeProvenance = Field(default_factory=OutcomeProvenance)


class WorkerTaskProposal(BaseModel):
    """One task exactly as the Planning model is allowed to propose it.

    The model owns *what* to do and *who* should do it. It owns nothing else:
    there is no field here for tool scope, dispatch identity, position, or
    parent context, so a proposal can never widen its own authority. Bounds
    are rejections rather than truncations — a shortened objective is
    different work than the one that was asked for.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1, max_length=160)
    objective: str = Field(min_length=1, max_length=4000)
    agent_id: str = Field(min_length=1, max_length=160)
    related_todo_ids: tuple[str, ...] = ()


class DispatchSubagentsInput(BaseModel):
    """The model-facing control schema for one dispatch call.

    This type is bound to the Planning model as a schema only. It is never
    executed by the common tool pipeline, so nothing here reaches a provider
    or a credential.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: tuple[WorkerTaskProposal, ...] = Field(min_length=1)
    rationale: str | None = Field(default=None, max_length=1000)


class WorkerTask(BaseModel):
    """One validated, server-owned unit of delegated work.

    ``position`` is the proposal's original index and is what collection
    orders by: completion order is an accident of latency and would make the
    same plan synthesize differently on different runs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dispatch_id: str = Field(min_length=1, max_length=64)
    task_id: str = Field(min_length=1, max_length=160)
    position: int = Field(ge=0)
    objective: str = Field(min_length=1, max_length=4000)
    agent_id: str = Field(min_length=1, max_length=160)
    parent_context: dict[str, JsonValue] = Field(default_factory=dict)
    allowed_tool_ids: tuple[str, ...] = ()
    model_request: dict[str, JsonValue] | None = None
    related_todo_ids: tuple[str, ...] = ()


class WorkerResult(BaseModel):
    """Private result of one Planning worker task.

    Identity is ``(dispatch_id, task_id)``, not ``task_id`` alone: the same
    task ID legitimately reappears in a later dispatch wave, and keying on the
    ID alone would reject that as a collision.

    There is no ``awaiting_approval`` status: a worker that needs human
    approval interrupts the graph instead of fabricating a result.
    """

    model_config = ConfigDict(extra="forbid")

    dispatch_id: str = Field(min_length=1, max_length=64)
    task_id: str = Field(min_length=1, max_length=160)
    position: int = Field(ge=0)
    agent_id: str = Field(min_length=1, max_length=160)
    status: WorkerStatus
    content: str = ""
    artifacts: tuple[dict[str, JsonValue], ...] = ()
    evidence: tuple[dict[str, JsonValue], ...] = ()
    images: tuple[dict[str, JsonValue], ...] = ()
    error_code: str | None = None


class PlanningDispatch(BaseModel):
    """One accepted fan-out: a validated wave paired to its originating call.

    ``tool_call_id`` is retained so collection can answer the exact call the
    model made. An unpaired dispatch would leave the model with a tool call
    that never received a result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dispatch_id: str = Field(min_length=1, max_length=64)
    tool_call_id: str = Field(min_length=1, max_length=160)
    wave: int = Field(ge=1)
    tasks: tuple[WorkerTask, ...] = Field(min_length=1)


class WorkflowError(BaseModel):
    """Stable, localization-free terminal failure payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: WorkflowErrorCode
    retriable: bool
    request_id: str = Field(min_length=1, max_length=160)
    details: dict[str, JsonValue] = Field(default_factory=dict)


class InvalidWorkflowStateUpdate(RuntimeError):
    """Raised when a state update violates a set-once or uniqueness invariant."""


class WorkflowRoutingException(RuntimeError):
    """Carries a typed :class:`WorkflowError` across service boundaries.

    Boundaries translate ``.error`` directly; they never parse the message
    text of this exception.
    """

    def __init__(self, error: WorkflowError) -> None:
        super().__init__(f"workflow_error:{error.code}")
        self._error = error

    @property
    def error(self) -> WorkflowError:
        return self._error
