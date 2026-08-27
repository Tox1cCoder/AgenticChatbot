"""Output validation and universal public finalization.

Two invariants live here and nowhere else:

* every public response passes ``validate_output`` before it can be published;
* only ``finalize`` appends the terminal public ``AIMessage`` and reaches
  ``END`` — on success *and* on failure.

Which contracts a response must satisfy is chosen from server-owned
provenance — the evidence, artifacts, and images runtime code actually
recorded — plus the policies the producing component declared. It is not chosen
from the final agent's name: a name changes through a handoff, but what
produced the answer does not.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.ai.agent_metadata import attach_agent_metadata
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.contracts import (
    AgentTransition,
    OutcomeProvenance,
    ResponseOutcome,
    WorkerResult,
    WorkflowError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "POLICY_REGISTRY",
    "OutputPolicy",
    "OutputValidationError",
    "OutputValidator",
    "PublicResponseFinalizer",
    "WorkerOutputFinalizer",
    "make_finalize_node",
    "make_validate_output_node",
    "select_policies",
]

GRAPH_VERSION = "routing-v2"
POLICY_IMPLEMENTATION_VERSION = "1"

# Citations the grounding policy can recognise, e.g. ``[E12]``.
_CITATION_PATTERN = re.compile(r"\[(E\d+)\]")


class OutputValidationError(RuntimeError):
    """A candidate public response failed a mandatory output contract."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


class OutputPolicy(Protocol):
    policy_id: str

    async def validate(self, outcome: ResponseOutcome, context: dict[str, Any]) -> None: ...


class PublicContentPolicy:
    """A published response must actually carry something."""

    policy_id = "public_content"

    async def validate(self, outcome: ResponseOutcome, context: dict[str, Any]) -> None:
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


class ArtifactProvenancePolicy:
    """Every published artifact must be one runtime code recorded."""

    policy_id = "artifact_provenance"

    async def validate(self, outcome: ResponseOutcome, context: dict[str, Any]) -> None:
        recorded = {
            str(artifact.get("artifact_id"))
            for artifact in outcome.provenance.artifacts
            if isinstance(artifact, dict) and artifact.get("artifact_id")
        }
        published = [
            str(artifact.get("artifact_id"))
            for artifact in (outcome.response.tool_artifacts or [])
            if isinstance(artifact, dict) and artifact.get("artifact_id")
        ]
        unrecorded = [artifact_id for artifact_id in published if artifact_id not in recorded]
        if unrecorded:
            raise OutputValidationError(
                "unrecorded_artifact",
                f"artifact ids not captured by the runtime: {sorted(unrecorded)}",
            )


class ImageDeliveryPolicy:
    """Every published image must trace back to a captured image record."""

    policy_id = "image_delivery"

    async def validate(self, outcome: ResponseOutcome, context: dict[str, Any]) -> None:
        recorded = {
            str(image.get("image_id"))
            for image in outcome.provenance.images
            if isinstance(image, dict) and image.get("image_id")
        }
        metadata = outcome.response.metadata or {}
        published = [
            str(image.get("image_id"))
            for image in (metadata.get("images") or [])
            if isinstance(image, dict) and image.get("image_id")
        ]
        unrecorded = [image_id for image_id in published if image_id not in recorded]
        if unrecorded:
            raise OutputValidationError(
                "unrecorded_image", f"image ids not captured by the runtime: {sorted(unrecorded)}"
            )


class RagGroundingPolicy:
    """Citations must resolve to evidence this turn actually retrieved.

    Runs whenever evidence is present *and* whenever a component declared it,
    so an empty-evidence RAG result is still validated rather than waved
    through for having nothing to check.
    """

    policy_id = "rag_grounding"

    async def validate(self, outcome: ResponseOutcome, context: dict[str, Any]) -> None:
        known = {
            str(record.get("evidence_id"))
            for record in outcome.provenance.evidence
            if isinstance(record, dict) and record.get("evidence_id")
        }
        cited = set(_CITATION_PATTERN.findall(outcome.response.message.content or ""))
        unknown = sorted(cited - known)
        if unknown:
            raise OutputValidationError(
                "unknown_evidence_id", f"answer cites unretrieved evidence: {unknown}"
            )


class ToolMessagePairingPolicy:
    """Every tool result must answer a tool call the model actually made."""

    policy_id = "tool_message_pairing"

    async def validate(self, outcome: ResponseOutcome, context: dict[str, Any]) -> None:
        requested: set[str] = set()
        answered: list[str] = []
        for message in outcome.provenance.private_messages:
            for tool_call in getattr(message, "tool_calls", None) or []:
                call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
                if call_id:
                    requested.add(str(call_id))
            if getattr(message, "type", None) == "tool":
                answered.append(str(getattr(message, "tool_call_id", "")))

        orphans = sorted({call_id for call_id in answered if call_id not in requested})
        if orphans:
            raise OutputValidationError(
                "unpaired_tool_message", f"tool results without a matching call: {orphans}"
            )


class CanvasOutputPolicy:
    """A canvas response must carry the artifact it claims to have built."""

    policy_id = "canvas_contract"

    async def validate(self, outcome: ResponseOutcome, context: dict[str, Any]) -> None:
        metadata = outcome.response.metadata or {}
        artifact = metadata.get("canvas_artifact")
        update = metadata.get("canvas_update")
        if artifact is None and update is None:
            return
        if artifact is not None and not isinstance(artifact, dict):
            raise OutputValidationError(
                "invalid_canvas_artifact", "canvas_artifact must be a mapping"
            )
        if isinstance(artifact, dict) and not str(artifact.get("content") or "").strip():
            raise OutputValidationError(
                "empty_canvas_artifact", "canvas_artifact carries no content"
            )


POLICY_REGISTRY: dict[str, OutputPolicy] = {
    policy.policy_id: policy
    for policy in (
        PublicContentPolicy(),
        ArtifactProvenancePolicy(),
        ImageDeliveryPolicy(),
        RagGroundingPolicy(),
        ToolMessagePairingPolicy(),
        CanvasOutputPolicy(),
    )
}


def select_policies(outcome: ResponseOutcome) -> tuple[str, ...]:
    """Choose the contracts this response must satisfy.

    Selection reads provenance first and declarations second. Evidence,
    artifacts, and images each activate their own validator whether or not the
    producing component remembered to declare it.
    """
    if not isinstance(outcome, ResponseOutcome):
        raise OutputValidationError(
            "outcome_not_server_owned", "validation requires a ResponseOutcome"
        )

    selected: list[str] = ["public_content"]
    provenance = outcome.provenance

    for policy_id in provenance.output_policy_ids:
        if policy_id not in POLICY_REGISTRY:
            raise OutputValidationError(
                "unknown_output_policy", f"no registered policy named {policy_id!r}"
            )
        if policy_id not in selected:
            selected.append(policy_id)

    # A policy is selected when the runtime recorded something of its kind, or
    # when the response *claims* something of its kind. Selecting only on what
    # the runtime recorded left the worst case unchecked: a response publishing
    # an artifact, image, or citation with nothing recorded at all skipped the
    # policy written to catch exactly that. Empty provenance is not "nothing to
    # verify" — against a response that claims something, it is the strongest
    # evidence there is that the claim was invented.
    response = outcome.response
    metadata = response.metadata or {}
    for present, policy_id in (
        (
            provenance.evidence or _CITATION_PATTERN.search(response.message.content or ""),
            "rag_grounding",
        ),
        (provenance.artifacts or response.tool_artifacts, "artifact_provenance"),
        (provenance.images or metadata.get("images"), "image_delivery"),
    ):
        if present and policy_id not in selected:
            selected.append(policy_id)

    return tuple(selected)


class OutputValidator:
    """Runs every selected policy before a response may be published."""

    async def validate(self, outcome: Any, state: dict[str, Any]) -> ResponseOutcome:
        if not isinstance(outcome, ResponseOutcome):
            raise OutputValidationError(
                "outcome_not_server_owned", "validate_output requires a ResponseOutcome"
            )
        if not outcome.agent_id:
            raise OutputValidationError("missing_agent_identity", "outcome has no agent_id")

        policy_ids = select_policies(outcome)
        context = {"state": state}
        for policy_id in policy_ids:
            await POLICY_REGISTRY[policy_id].validate(outcome, context)

        # Record what actually ran, so the finalized response is auditable.
        return outcome.model_copy(
            update={
                "provenance": outcome.provenance.model_copy(
                    update={"output_policy_ids": policy_ids}
                )
            }
        )


class WorkerOutputFinalizer:
    """Validates a worker's output against the same policy registry.

    A worker result is private: it never gets a public message id and never
    enters conversation history, but the evidence and artifact contracts that
    protect a public answer protect it too.
    """

    def __init__(self, validator: OutputValidator | None = None) -> None:
        self._validator = validator or OutputValidator()

    async def finalize(self, outcome: ResponseOutcome, *, task_id: str) -> WorkerResult:
        try:
            validated = await self._validator.validate(outcome, {})
        except OutputValidationError as exc:
            logger.warning("Worker output failed validation (%s)", exc.reason)
            return WorkerResult(
                task_id=task_id,
                agent_id=outcome.agent_id if isinstance(outcome, ResponseOutcome) else "unknown",
                status="failed",
                error_code="response_validation_failed",
            )
        return WorkerResult(
            task_id=task_id,
            agent_id=validated.agent_id,
            status="completed",
            content=validated.response.message.content or "",
            artifacts=validated.provenance.artifacts,
            evidence=validated.provenance.evidence,
        )


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
            return self._failure_update(state, request_id, "missing_validated_outcome")

        assistant_message_id = state.get("assistant_message_id")
        if not assistant_message_id:
            # Without the reserved id the response cannot be persisted under a
            # known key, so publishing it would create an unreferenceable turn.
            return self._failure_update(state, request_id, "missing_assistant_message_id")

        try:
            response = self._build_response(state, outcome)
        except Exception as exc:  # noqa: BLE001 - normalized into a typed error
            logger.warning("Finalization could not normalize the response: %s", exc)
            return self._failure_update(state, request_id, "normalization_failed")

        # The whole update is built locally before returning, so a failure part
        # way through leaves nothing half-applied in state.
        return {
            "messages": [AIMessage(content=response.message.content, id=str(assistant_message_id))],
            "response": response,
            "final_agent_id": outcome.agent_id,
            "execution_phase": "completed",
            "validated_public_content": response.message.content,
        }

    def _build_response(self, state: dict[str, Any], outcome: ResponseOutcome) -> AgentResponse:
        response = outcome.response.model_copy(deep=True)
        metadata = dict(response.metadata or {})

        routing_decision = state.get("routing_decision")
        history: list[AgentTransition] = [
            transition
            for transition in (state.get("agent_history") or [])
            if isinstance(transition, AgentTransition)
        ]
        active_agent_id = state.get("active_agent_id") or outcome.agent_id

        workflow_metadata: dict[str, Any] = {
            "graph_version": GRAPH_VERSION,
            "initial_agent_id": routing_decision.agent_id if routing_decision else None,
            "active_agent_id": active_agent_id,
            "final_agent_id": outcome.agent_id,
            "execution_phase": "completed",
            "transitions": [transition.model_dump(mode="json") for transition in history],
        }
        if routing_decision is not None:
            # ``confidence`` is telemetry. It is recorded, never branched on.
            workflow_metadata["routing"] = {
                "agent_id": routing_decision.agent_id,
                "confidence": routing_decision.confidence,
                "inventory_version": state.get("routing_inventory_version"),
            }
        metadata["workflow"] = workflow_metadata

        policy_ids = list(outcome.provenance.output_policy_ids)
        metadata["validation"] = {
            "passed": True,
            "policy_ids": policy_ids,
            "policy_versions": {
                policy_id: POLICY_IMPLEMENTATION_VERSION for policy_id in policy_ids
            },
        }
        metadata["provenance"] = _provenance_metadata(outcome.provenance)

        attach_agent_metadata(
            metadata,
            response_agent_id=outcome.agent_id,
            active_agent_id=active_agent_id,
            custom_agents=state.get("custom_agents"),
        )

        response.metadata = metadata
        return response

    # -- failure ---------------------------------------------------------

    def _failure_update(
        self, state: dict[str, Any], request_id: str, reason: str
    ) -> dict[str, Any]:
        return self._finalize_failure(
            state,
            WorkflowError(
                code="finalization_failed",
                retriable=False,
                request_id=request_id,
                details={"reason": reason},
            ),
        )

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
                        "graph_version": GRAPH_VERSION,
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


def _provenance_metadata(provenance: OutcomeProvenance) -> dict[str, Any]:
    """Record what produced this answer, by id only."""
    return {
        "artifact_ids": [
            str(artifact.get("artifact_id"))
            for artifact in provenance.artifacts
            if isinstance(artifact, dict) and artifact.get("artifact_id")
        ],
        "image_ids": [
            str(image.get("image_id"))
            for image in provenance.images
            if isinstance(image, dict) and image.get("image_id")
        ],
        "evidence_ids": [
            str(record.get("evidence_id"))
            for record in provenance.evidence
            if isinstance(record, dict) and record.get("evidence_id")
        ],
    }


def _request_id(state: dict[str, Any]) -> str:
    identity = state.get("turn_identity")
    request_id = getattr(identity, "request_id", None)
    return str(request_id) if request_id else "unknown"
