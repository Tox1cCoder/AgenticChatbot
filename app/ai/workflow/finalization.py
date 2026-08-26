"""Output validation and universal public finalization.

Two invariants live here and nowhere else:

* every public response passes ``validate_output`` before it can be published;
* only ``finalize`` appends the terminal public ``AIMessage`` and reaches
  ``END`` — on success *and* on failure.

The validator in this module is deliberately small but real: it checks that the
outcome was constructed by server-owned code and carries publishable content or
a typed error. Provenance policies (grounding, artifacts, images, canvas,
tool-message pairing) plug into the same interface without changing the graph
contract.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.ai.agent_metadata import attach_agent_metadata
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.contracts import (
    AgentTransition,
    ResponseOutcome,
    WorkflowError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OutputValidationError",
    "PublicResponseFinalizer",
    "make_finalize_node",
    "make_validate_output_node",
]


class OutputValidationError(RuntimeError):
    """A candidate public response failed a mandatory output contract."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


class OutputValidator:
    """Minimal-but-real validation boundary.

    Task 9 registers the full provenance policy registry behind this same
    interface; the graph contract does not change when it does.
    """

    async def validate(self, outcome: ResponseOutcome, state: dict[str, Any]) -> ResponseOutcome:
        if not isinstance(outcome, ResponseOutcome):
            raise OutputValidationError(
                "outcome_not_server_owned", "validate_output requires a ResponseOutcome"
            )
        if not outcome.agent_id:
            raise OutputValidationError("missing_agent_identity", "outcome has no agent_id")

        response = outcome.response
        if response.error is not None and not isinstance(response.error, (str, dict)):
            raise OutputValidationError("invalid_error_shape", "response.error is not renderable")

        has_content = bool((response.message.content or "").strip())
        has_artifacts = bool(response.tool_artifacts) or bool(outcome.provenance.artifacts)
        has_images = bool(outcome.provenance.images)
        has_interrupt = bool((response.metadata or {}).get("interrupt"))
        if not (has_content or has_artifacts or has_images or has_interrupt or response.error):
            raise OutputValidationError(
                "empty_public_content", "no publishable content, artifact, image, or error"
            )
        return outcome


def make_validate_output_node(validator: OutputValidator | None = None):
    """Build the mandatory validation node.

    Success routes to ``finalize``; failure also routes to ``finalize`` with a
    typed error, because a failed turn must still terminate through the one
    finalization boundary.
    """
    validator = validator or OutputValidator()

    async def validate_output(state: dict[str, Any], runtime: Any = None) -> Command:
        outcome = state.get("agent_outcome")
        request_id = _request_id(state)
        try:
            validated = await validator.validate(outcome, state)
        except OutputValidationError as exc:
            logger.warning("Output validation failed: %s", exc.reason)
            return Command(
                update={
                    "execution_phase": "failed",
                    "workflow_error": WorkflowError(
                        code="response_validation_failed",
                        retriable=False,
                        request_id=request_id,
                        details={"reason": exc.reason},
                    ),
                },
                goto="finalize",
            )

        return Command(
            update={"agent_outcome": validated, "execution_phase": "finalizing"},
            goto="finalize",
        )

    return validate_output


class PublicResponseFinalizer:
    """The sole owner of the terminal public message and response.

    It guarantees graph-level validation, response construction, and
    serialization. It does **not** claim the database write has committed;
    ``MessageService`` owns that and publishes only after it does.
    """

    def finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        workflow_error = state.get("workflow_error")
        if workflow_error is not None or state.get("execution_phase") == "failed":
            return self._finalize_failure(state, workflow_error)
        return self._finalize_success(state)

    # -- success ---------------------------------------------------------

    def _finalize_success(self, state: dict[str, Any]) -> dict[str, Any]:
        outcome = state.get("agent_outcome")
        request_id = _request_id(state)
        if not isinstance(outcome, ResponseOutcome):
            return self._finalize_failure(
                state,
                WorkflowError(
                    code="finalization_failed",
                    retriable=False,
                    request_id=request_id,
                    details={"reason": "missing_validated_outcome"},
                ),
            )

        try:
            response = self._build_response(state, outcome)
        except Exception as exc:  # noqa: BLE001 - normalized into a typed error
            logger.warning("Finalization could not normalize the response: %s", exc)
            return self._finalize_failure(
                state,
                WorkflowError(
                    code="finalization_failed",
                    retriable=False,
                    request_id=request_id,
                    details={"reason": "normalization_failed"},
                ),
            )

        assistant_message_id = state.get("assistant_message_id")
        message_kwargs: dict[str, Any] = {"content": response.message.content}
        if assistant_message_id:
            message_kwargs["id"] = assistant_message_id

        # Build the whole update locally before returning it: a half-applied
        # metadata normalization must never reach state.
        return {
            "messages": [AIMessage(**message_kwargs)],
            "response": response,
            "final_agent_id": outcome.agent_id,
            "execution_phase": "completed",
            "validated_public_content": response.message.content,
        }

    def _build_response(self, state: dict[str, Any], outcome: ResponseOutcome) -> AgentResponse:
        response = outcome.response.model_copy(deep=True)
        metadata = dict(response.metadata or {})

        routing_decision = state.get("routing_decision")
        history: list[AgentTransition] = list(state.get("agent_history") or [])
        initial_agent_id = routing_decision.agent_id if routing_decision else None
        active_agent_id = state.get("active_agent_id") or outcome.agent_id

        workflow_metadata = {
            "graph_version": "routing-v2",
            "initial_agent_id": initial_agent_id,
            "active_agent_id": active_agent_id,
            "final_agent_id": outcome.agent_id,
            "execution_phase": "completed",
            "transitions": [transition.model_dump(mode="json") for transition in history],
        }
        if routing_decision is not None:
            workflow_metadata["routing"] = {
                "agent_id": routing_decision.agent_id,
                "confidence": routing_decision.confidence,
                "inventory_version": state.get("routing_inventory_version"),
            }
        metadata["workflow"] = workflow_metadata

        if outcome.provenance.output_policy_ids:
            metadata["validation"] = {
                "passed": True,
                "policy_ids": list(outcome.provenance.output_policy_ids),
            }
        else:
            metadata["validation"] = {"passed": True, "policy_ids": []}

        attach_agent_metadata(
            metadata,
            response_agent_id=outcome.agent_id,
            selected_agent_id=active_agent_id,
            custom_agents=state.get("custom_agents") if isinstance(state, dict) else None,
        )

        response.metadata = metadata
        return response

    # -- failure ---------------------------------------------------------

    def _finalize_failure(
        self, state: dict[str, Any], workflow_error: WorkflowError | None
    ) -> dict[str, Any]:
        error = workflow_error or WorkflowError(
            code="finalization_failed",
            retriable=False,
            request_id=_request_id(state),
            details={"reason": "unspecified"},
        )
        logger.warning("Workflow turn failed: code=%s retriable=%s", error.code, error.retriable)

        # A failed turn publishes no assistant text and appends no message.
        return {
            "response": AgentResponse(
                agent_type=AgentType.CHAT,
                agent_id=state.get("active_agent_id") or "unknown",
                message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
                metadata={
                    "workflow": {
                        "graph_version": "routing-v2",
                        "execution_phase": "failed",
                        "error": error.model_dump(mode="json"),
                    }
                },
                error=error.code,
            ),
            "workflow_error": error,
            "execution_phase": "failed",
            "validated_public_content": "",
        }


def make_finalize_node(finalizer: PublicResponseFinalizer | None = None):
    finalizer = finalizer or PublicResponseFinalizer()

    async def finalize(state: dict[str, Any], runtime: Any = None) -> dict[str, Any]:
        return finalizer.finalize(state)

    return finalize


def _request_id(state: dict[str, Any]) -> str:
    identity = state.get("turn_identity")
    request_id = getattr(identity, "request_id", None)
    return str(request_id) if request_id else "unknown"
