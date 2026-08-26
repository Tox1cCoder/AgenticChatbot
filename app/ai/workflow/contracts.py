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
    "AgentOutcome",
    "AgentTransition",
    "ExecutionPhase",
    "HandoffOutcome",
    "InvalidWorkflowStateUpdate",
    "OutcomeProvenance",
    "PendingTransition",
    "ResponseOutcome",
    "RoutingDecision",
    "TransitionSource",
    "TurnIdentity",
    "WorkerResult",
    "WorkerStatus",
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

WorkerStatus = Literal["completed", "failed"]


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
    """A specialist finished its work and produced a candidate public answer."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["response"] = "response"
    agent_id: str = Field(min_length=1, max_length=160)
    response: AgentResponse
    provenance: OutcomeProvenance = Field(default_factory=OutcomeProvenance)


class HandoffOutcome(BaseModel):
    """A specialist asked the parent graph to move to another specialist."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["handoff"] = "handoff"
    agent_id: str = Field(min_length=1, max_length=160)
    handoff: AgentTransition


AgentOutcome = ResponseOutcome | HandoffOutcome


class WorkerResult(BaseModel):
    """Private result of one Planning worker task.

    There is no ``awaiting_approval`` status: a worker that needs human
    approval interrupts the Planning graph instead of fabricating a result.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=160)
    agent_id: str = Field(min_length=1, max_length=160)
    status: WorkerStatus
    content: str = ""
    artifacts: tuple[dict[str, JsonValue], ...] = ()
    evidence: tuple[dict[str, JsonValue], ...] = ()
    error_code: str | None = None


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
