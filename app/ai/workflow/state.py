"""Graph state and reducers for the routing-v2 production workflow.

``WorkflowState`` replaces the pre-v2 mutable routing field with three
separate concerns:

* ``routing_decision`` — the immutable, model-owned decision for the new turn.
* ``active_agent_id`` — who is executing right now.
* ``agent_history`` — the append-only audit trail of accepted transitions.

Every turn runs on its own checkpoint thread
(``routing-v2:{conversation_id}:{turn_id}``), so the append reducers below are
turn-local by construction and never accumulate across user turns.
"""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.ai.schemas import AgentResponse, GraphContext
from app.ai.workflow.contracts import (
    AgentTransition,
    ExecutionPhase,
    InvalidWorkflowStateUpdate,
    PendingTransition,
    ResponseOutcome,
    RoutingDecision,
    TurnIdentity,
    WorkerResult,
    WorkflowError,
)

__all__ = [
    "CHECKPOINT_THREAD_PREFIX",
    "WorkflowState",
    "append_transitions",
    "append_worker_results",
    "build_checkpoint_thread_id",
    "parse_checkpoint_thread_id",
    "set_routing_decision_once",
]

CHECKPOINT_THREAD_PREFIX = "routing-v2"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def append_transitions(
    existing: list[AgentTransition] | AgentTransition | None,
    update: list[AgentTransition] | AgentTransition | None,
) -> list[AgentTransition]:
    """Append-only reducer for the ordered transition audit trail."""
    return [*_as_list(existing), *_as_list(update)]


def append_worker_results(
    existing: list[WorkerResult] | WorkerResult | None,
    update: list[WorkerResult] | WorkerResult | None,
) -> list[WorkerResult]:
    """Append worker results, rejecting duplicate task IDs.

    A duplicate task ID means two branches claimed the same dispatched task;
    silently overwriting one would hide a lost result.
    """
    merged = _as_list(existing)
    seen = {result.task_id for result in merged}
    for result in _as_list(update):
        if result.task_id in seen:
            raise InvalidWorkflowStateUpdate(
                f"duplicate worker result for task_id={result.task_id!r}"
            )
        seen.add(result.task_id)
        merged.append(result)
    return merged


def set_routing_decision_once(
    existing: RoutingDecision | None,
    update: RoutingDecision | None,
) -> RoutingDecision | None:
    """Set-once reducer for the immutable initial routing decision.

    ``None -> decision`` and an identical replay of the accepted frozen value
    are valid. Replacing an accepted decision is a state error, not a silent
    reroute.
    """
    if update is None:
        return existing
    if existing is None:
        return update
    if existing == update:
        return existing
    raise InvalidWorkflowStateUpdate(
        "routing_decision is immutable for a turn; "
        f"cannot replace {existing.agent_id!r} with {update.agent_id!r}"
    )


def build_checkpoint_thread_id(conversation_id: str, turn_id: str) -> str:
    """Build the one canonical per-turn checkpoint thread ID."""
    conversation = str(conversation_id or "").strip()
    turn = str(turn_id or "").strip()
    if not conversation:
        raise ValueError("conversation_id is required to build a checkpoint thread id")
    if not turn:
        raise ValueError("turn_id is required to build a checkpoint thread id")
    if ":" in conversation or ":" in turn:
        raise ValueError("conversation_id and turn_id must not contain ':'")
    return f"{CHECKPOINT_THREAD_PREFIX}:{conversation}:{turn}"


def parse_checkpoint_thread_id(thread_id: str) -> tuple[str, str]:
    """Validate a v2 thread ID and return ``(conversation_id, turn_id)``.

    Retention and resume paths use this so they can never act on a
    reconstructed, prefix-matched, or v1 thread ID.
    """
    parts = str(thread_id or "").split(":")
    if len(parts) != 3 or parts[0] != CHECKPOINT_THREAD_PREFIX or not parts[1] or not parts[2]:
        raise ValueError(f"not a routing-v2 checkpoint thread id: {thread_id!r}")
    return parts[1], parts[2]


class WorkflowState(TypedDict):
    """The sole graph state type for the routing-v2 workflow."""

    messages: Annotated[list[BaseMessage], add_messages]

    # --- turn identity and routing control plane -------------------------
    turn_identity: NotRequired[TurnIdentity | None]
    routing_decision: NotRequired[Annotated[RoutingDecision | None, set_routing_decision_once]]
    routing_inventory_version: NotRequired[str | None]
    active_agent_id: NotRequired[str | None]
    final_agent_id: NotRequired[str | None]
    agent_history: NotRequired[Annotated[list[AgentTransition], append_transitions]]
    pending_transition: NotRequired[PendingTransition | None]
    agent_outcome: NotRequired[ResponseOutcome | None]
    worker_results: NotRequired[Annotated[list[WorkerResult], append_worker_results]]
    execution_phase: NotRequired[ExecutionPhase]
    workflow_error: NotRequired[WorkflowError | None]

    # --- request scope ----------------------------------------------------
    conversation_id: NotRequired[str | None]
    user_id: NotRequired[str | None]
    device_id: NotRequired[str | None]
    persona: NotRequired[str | None]
    attachments: NotRequired[list[Any] | None]
    model_request: NotRequired[dict[str, Any] | None]
    custom_agents: NotRequired[dict[str, Any] | None]
    user_message_id: NotRequired[str | None]
    assistant_message_id: NotRequired[str | None]

    # --- planning data ----------------------------------------------------
    task_plan_id: NotRequired[str | None]
    current_task: NotRequired[dict[str, Any] | None]
    all_tasks: NotRequired[list[dict[str, Any]] | None]
    planning_mode_enabled: NotRequired[bool | None]
    has_existing_plan: NotRequired[bool | None]
    todos: NotRequired[list[dict[str, Any]] | None]
    current_task_index: NotRequired[int | None]
    planning_call_count: NotRequired[int | None]
    planning_phase: NotRequired[str | None]
    plan_lifecycle: NotRequired[Any]

    # --- execution scratch space -----------------------------------------
    context: NotRequired[GraphContext]
    response: NotRequired[AgentResponse | None]
