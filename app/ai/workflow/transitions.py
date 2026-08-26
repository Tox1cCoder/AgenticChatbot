"""The sole resolver of execution transitions.

One node decides whether a requested handoff happens. It validates only
control-plane invariants — identity, reachability, attachment, pairing, and
depth — and never interprets what the user or the model said. A refusal is
model-visible feedback on the same paired ``ToolMessage``, so the model can
choose differently instead of retrying blindly.

The turn's ``routing_decision`` is never rewritten here: who was chosen at the
start of the turn and who is executing now are separate facts.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.ai.hand_off_tool import HAND_OFF_TOOL_NAME
from app.ai.workflow.contracts import AgentTransition, PendingTransition, WorkflowError
from app.ai.workflow.inventory import RoutingInventory

logger = logging.getLogger(__name__)

__all__ = ["TransitionResolver", "TransitionRejection"]


class TransitionRejection(RuntimeError):
    """A pending transition failed a control-plane invariant."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


class TransitionResolver:
    """Accept or refuse one pending transition per invocation."""

    def __init__(self, *, inventory: RoutingInventory, max_delegation_depth: int) -> None:
        self._inventory = inventory
        self._max_delegation_depth = max(0, int(max_delegation_depth))

    async def __call__(self, state: dict[str, Any], runtime: Any = None) -> Command:
        # Attached custom agents are per-request, so the live turn inventory
        # wins over the one captured when the graph was compiled. Judging a
        # handoff against a stale inventory would let a detached agent through.
        inventory = getattr(getattr(runtime, "context", None), "inventory", None)
        if isinstance(inventory, RoutingInventory):
            self._inventory = inventory

        pending = state.get("pending_transition")
        if not isinstance(pending, PendingTransition):
            # Reaching the resolver with nothing pending means the control
            # plane and the topology disagree; fail closed rather than guess.
            logger.error("resolve_transition reached with no pending transition")
            return Command(
                update={
                    "execution_phase": "failed",
                    "workflow_error": WorkflowError(
                        code="response_validation_failed",
                        retriable=False,
                        request_id=_request_id(state),
                        details={"reason": "missing_pending_transition"},
                    ),
                },
                goto="finalize",
            )

        active_agent_id = state.get("active_agent_id")
        try:
            self._validate(pending, state, active_agent_id)
        except TransitionRejection as rejection:
            return self._reject(pending, active_agent_id, rejection)

        return Command(
            update={
                "active_agent_id": pending.to_agent_id,
                "agent_history": [
                    AgentTransition(
                        from_agent_id=pending.from_agent_id,
                        to_agent_id=pending.to_agent_id,
                        source="handoff",
                        tool_call_id=pending.tool_call_id,
                    )
                ],
                "pending_transition": None,
                "execution_phase": "executing",
            },
            goto=self._inventory.resolve_node(pending.to_agent_id),
        )

    # -- validation ------------------------------------------------------

    def _validate(
        self,
        pending: PendingTransition,
        state: dict[str, Any],
        active_agent_id: Any,
    ) -> None:
        if not pending.tool_call_id.strip():
            raise TransitionRejection(
                "bad_call_id", "the originating tool call could not be identified."
            )

        if not isinstance(active_agent_id, str) or not active_agent_id:
            raise TransitionRejection("no_active_agent", "the active agent is unavailable.")

        if pending.from_agent_id != active_agent_id:
            raise TransitionRejection(
                "source_mismatch",
                f"'{pending.from_agent_id}' is not the agent currently executing.",
            )

        if pending.to_agent_id == active_agent_id:
            raise TransitionRejection(
                "self_target", f"'{pending.to_agent_id}' is already the active agent."
            )

        if not self._inventory.is_routable(pending.to_agent_id):
            raise TransitionRejection(
                "unreachable_target",
                f"'{pending.to_agent_id}' is not a reachable target for {active_agent_id}.",
            )

        history = [
            transition
            for transition in (state.get("agent_history") or [])
            if isinstance(transition, AgentTransition)
        ]
        visited = {transition.to_agent_id for transition in history}
        if pending.to_agent_id in visited:
            raise TransitionRejection(
                "revisited_target",
                f"'{pending.to_agent_id}' has already handled this turn.",
            )

        # Only accepted handoffs count. A resume transition is audit history,
        # not a delegation the user asked for.
        accepted_handoffs = sum(1 for transition in history if transition.source == "handoff")
        if accepted_handoffs >= self._max_delegation_depth:
            raise TransitionRejection(
                "over_depth",
                f"this turn reached the maximum delegation depth of "
                f"{self._max_delegation_depth}.",
            )

    # -- rejection -------------------------------------------------------

    @staticmethod
    def _reject(
        pending: PendingTransition,
        active_agent_id: Any,
        rejection: TransitionRejection,
    ) -> Command:
        logger.info(
            "Hand-off refused (%s): %s -> %s",
            rejection.reason,
            pending.from_agent_id,
            pending.to_agent_id,
        )
        feedback = ToolMessage(
            # Same deterministic id as the request marker, so the reducer
            # replaces it rather than pairing two messages to one call.
            id=pending.tool_message_id,
            content=f"Hand-off refused: {rejection.message}",
            name=HAND_OFF_TOOL_NAME,
            tool_call_id=pending.tool_call_id,
            status="error",
        )
        source_node = _source_node(active_agent_id, pending.from_agent_id)
        return Command(
            update={
                "messages": [feedback],
                "pending_transition": None,
                "execution_phase": "executing",
            },
            goto=source_node,
        )


def _source_node(active_agent_id: Any, from_agent_id: str) -> str:
    from app.ai.workflow.specialists import resolve_node_for_agent_id

    candidate = active_agent_id if isinstance(active_agent_id, str) else from_agent_id
    return resolve_node_for_agent_id(candidate) or "finalize"


def _request_id(state: dict[str, Any]) -> str:
    identity = state.get("turn_identity")
    request_id = getattr(identity, "request_id", None)
    return str(request_id) if request_id else "unknown"
