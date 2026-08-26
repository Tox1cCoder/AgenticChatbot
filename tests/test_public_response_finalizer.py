"""One finalization boundary for every public response.

The finalizer is the only component that appends the terminal assistant
message, and it is the only node with an edge to END — on success and on
failure alike. It guarantees graph-level validation and response construction;
it does not claim the database write has committed.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.contracts import (
    AgentTransition,
    OutcomeProvenance,
    ResponseOutcome,
    RoutingDecision,
    TurnIdentity,
    WorkflowError,
)
from app.ai.workflow.finalization import PublicResponseFinalizer

ASSISTANT_ID = "assistant-message-1"


def _identity() -> TurnIdentity:
    return TurnIdentity(
        request_id="request-1",
        turn_id="message-1",
        checkpoint_thread_id="routing-v2:conversation-1:message-1",
    )


def _outcome(agent_id="search_agent", content="final answer", **provenance) -> ResponseOutcome:
    return ResponseOutcome(
        agent_id=agent_id,
        response=AgentResponse(
            agent_type=AgentType.SEARCH,
            agent_id=agent_id,
            message=AgentMessage(role=MessageRole.ASSISTANT, content=content),
        ),
        provenance=OutcomeProvenance(**provenance),
    )


def _validated_state(**overrides) -> dict:
    state = {
        "turn_identity": _identity(),
        "assistant_message_id": ASSISTANT_ID,
        "routing_decision": RoutingDecision(
            agent_id="chat_agent", confidence=0.8, reason="general help"
        ),
        "routing_inventory_version": "abc123",
        "active_agent_id": "search_agent",
        "agent_history": [
            AgentTransition(from_agent_id=None, to_agent_id="chat_agent", source="router"),
            AgentTransition(
                from_agent_id="chat_agent",
                to_agent_id="search_agent",
                source="handoff",
                tool_call_id="call-1",
            ),
        ],
        "agent_outcome": _outcome(),
        "messages": [],
        "execution_phase": "finalizing",
    }
    state.update(overrides)
    return state


def _finalizer() -> PublicResponseFinalizer:
    return PublicResponseFinalizer()


# ----------------------------------------------------------------------
# terminal message
# ----------------------------------------------------------------------


def test_finalizer_appends_exactly_one_terminal_ai_message():
    update = _finalizer().finalize(_validated_state())
    terminal = [
        message
        for message in update["messages"]
        if isinstance(message, AIMessage) and message.id == ASSISTANT_ID
    ]
    assert len(terminal) == 1


def test_terminal_message_carries_the_validated_content():
    update = _finalizer().finalize(_validated_state())
    assert update["messages"][0].content == "final answer"
    assert update["validated_public_content"] == "final answer"


def test_finalizer_completes_the_execution_phase():
    update = _finalizer().finalize(_validated_state())
    assert update["execution_phase"] == "completed"


# ----------------------------------------------------------------------
# identity and history
# ----------------------------------------------------------------------


def test_public_metadata_preserves_the_initial_decision_after_a_handoff():
    """Who was routed to and who answered are different facts; both survive."""
    workflow = _finalizer().finalize(_validated_state())["response"].metadata["workflow"]

    assert workflow["initial_agent_id"] == "chat_agent"
    assert workflow["final_agent_id"] == "search_agent"
    assert workflow["active_agent_id"] == "search_agent"


def test_final_agent_id_is_set_from_the_active_agent():
    update = _finalizer().finalize(_validated_state())
    assert update["final_agent_id"] == "search_agent"


def test_transition_history_is_ordered_and_complete():
    workflow = _finalizer().finalize(_validated_state())["response"].metadata["workflow"]
    assert [t["to_agent_id"] for t in workflow["transitions"]] == [
        "chat_agent",
        "search_agent",
    ]
    assert [t["source"] for t in workflow["transitions"]] == ["router", "handoff"]


def test_routing_metadata_records_confidence_as_telemetry():
    routing = _finalizer().finalize(_validated_state())["response"].metadata["workflow"]["routing"]
    assert routing["agent_id"] == "chat_agent"
    assert routing["confidence"] == 0.8
    assert routing["inventory_version"] == "abc123"


# ----------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------


def test_outcome_provenance_is_server_owned_and_complete():
    state = _validated_state(
        agent_outcome=_outcome(
            output_policy_ids=("public_content", "artifact_provenance"),
            artifacts=({"artifact_id": "artifact-1"},),
        )
    )
    metadata = _finalizer().finalize(state)["response"].metadata

    assert metadata["validation"]["passed"] is True
    assert metadata["validation"]["policy_ids"] == ["public_content", "artifact_provenance"]
    assert metadata["validation"]["policy_versions"]
    assert metadata["provenance"]["artifact_ids"] == ["artifact-1"]


def test_provenance_records_evidence_and_image_ids():
    state = _validated_state(
        agent_outcome=_outcome(
            evidence=({"evidence_id": "E1"},), images=({"image_id": "img-1"},)
        )
    )
    provenance = _finalizer().finalize(state)["response"].metadata["provenance"]
    assert provenance["evidence_ids"] == ["E1"]
    assert provenance["image_ids"] == ["img-1"]


# ----------------------------------------------------------------------
# failure
# ----------------------------------------------------------------------


def test_a_failed_turn_publishes_no_assistant_message():
    state = _validated_state(
        execution_phase="failed",
        workflow_error=WorkflowError(
            code="routing_timeout", retriable=True, request_id="request-1"
        ),
        agent_outcome=None,
    )
    update = _finalizer().finalize(state)

    assert update.get("messages", []) == []
    assert update["validated_public_content"] == ""
    assert update["execution_phase"] == "failed"


def test_a_failed_turn_preserves_the_typed_error():
    error = WorkflowError(
        code="routing_provider_unavailable", retriable=True, request_id="request-1"
    )
    update = _finalizer().finalize(
        _validated_state(execution_phase="failed", workflow_error=error, agent_outcome=None)
    )
    assert update["workflow_error"] is error
    assert update["response"].error == "routing_provider_unavailable"


def test_a_missing_validated_outcome_fails_rather_than_publishing():
    update = _finalizer().finalize(_validated_state(agent_outcome=None))
    assert update["execution_phase"] == "failed"
    assert update["workflow_error"].code == "finalization_failed"
    assert update.get("messages", []) == []


def test_a_missing_assistant_message_id_fails_closed():
    update = _finalizer().finalize(_validated_state(assistant_message_id=None))
    assert update["execution_phase"] == "failed"
    assert update["workflow_error"].code == "finalization_failed"


def test_state_is_not_partially_mutated_when_normalization_fails():
    """The update is built locally, so a failure leaves nothing half-applied."""
    state = _validated_state(assistant_message_id=None)
    before = dict(state)

    _finalizer().finalize(state)

    assert state["messages"] == before["messages"]
    assert state.get("response") is before.get("response")


# ----------------------------------------------------------------------
# durability boundary
# ----------------------------------------------------------------------


def test_the_finalizer_does_not_claim_the_database_write_committed():
    """Persistence is MessageService's job; the finalizer only prepares."""
    update = _finalizer().finalize(_validated_state())
    assert "persisted" not in update
    assert update["execution_phase"] == "completed"
    # The answer is buffered for the stream projector, not yet published.
    assert update["validated_public_content"] == "final answer"
