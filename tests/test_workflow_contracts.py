from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.ai.workflow.contracts import (
    EXECUTION_PHASES,
    AgentTransition,
    DispatchSubagentsInput,
    OutcomeProvenance,
    PendingTransition,
    PlanningDispatch,
    ResponseOutcome,
    RoutingDecision,
    TurnIdentity,
    WorkerResult,
    WorkerTask,
    WorkerTaskProposal,
    WorkflowError,
    WorkflowRoutingException,
)


def _agent_response(content: str = "answer") -> AgentResponse:
    return AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content=content),
    )


def test_routing_decision_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        RoutingDecision.model_validate(
            {
                "agent_id": "chat_agent",
                "confidence": 0.8,
                "reason": "best capability",
                "selected_agent": "legacy",
            }
        )


def test_routing_decision_is_frozen_and_bounded():
    decision = RoutingDecision(agent_id="chat_agent", confidence=0.8, reason="general help")
    with pytest.raises(ValidationError):
        decision.agent_id = "search_agent"

    with pytest.raises(ValidationError):
        RoutingDecision(agent_id="chat_agent", confidence=1.5, reason="out of range")
    with pytest.raises(ValidationError):
        RoutingDecision(agent_id="", confidence=0.5, reason="empty target")
    with pytest.raises(ValidationError):
        RoutingDecision(agent_id="chat_agent", confidence=0.5, reason="")
    with pytest.raises(ValidationError):
        RoutingDecision(agent_id="chat_agent", confidence=0.5, reason="x" * 501)


def test_turn_identity_requires_all_identifiers():
    identity = TurnIdentity(
        request_id="request-1",
        turn_id="message-1",
        checkpoint_thread_id="routing-v2:conversation-1:message-1",
    )
    assert identity.checkpoint_thread_id == "routing-v2:conversation-1:message-1"

    with pytest.raises(ValidationError):
        TurnIdentity(request_id="", turn_id="message-1", checkpoint_thread_id="thread")


def test_agent_transition_sources_are_bounded():
    transition = AgentTransition(
        from_agent_id=None, to_agent_id="chat_agent", source="router", tool_call_id=None
    )
    assert transition.tool_call_id is None

    with pytest.raises(ValidationError):
        AgentTransition(from_agent_id=None, to_agent_id="chat_agent", source="stickiness")


def test_pending_transition_requires_paired_tool_identifiers():
    pending = PendingTransition(
        from_agent_id="chat_agent",
        to_agent_id="search_agent",
        tool_call_id="call-1",
        tool_message_id="handoff:call-1",
        reason="needs current sources",
    )
    assert pending.tool_message_id == "handoff:call-1"

    with pytest.raises(ValidationError):
        PendingTransition(
            from_agent_id="chat_agent",
            to_agent_id="search_agent",
            tool_call_id="call-1",
            tool_message_id="handoff:call-1",
            reason="",
        )


def test_outcome_provenance_defaults_are_empty_tuples():
    provenance = OutcomeProvenance()
    assert provenance.evidence == ()
    assert provenance.artifacts == ()
    assert provenance.images == ()
    assert provenance.private_messages == ()
    assert provenance.output_policy_ids == ()


def test_outcome_provenance_preserves_message_subclasses():
    provenance = OutcomeProvenance(
        private_messages=(
            AIMessage(content="thinking", id="private-1"),
            ToolMessage(content="ok", tool_call_id="call-1", id="handoff:call-1"),
        )
    )
    restored = OutcomeProvenance.model_validate_json(provenance.model_dump_json())
    assert [type(message).__name__ for message in restored.private_messages] == [
        "AIMessage",
        "ToolMessage",
    ]


def test_a_specialist_outcome_carries_only_server_owned_fields():
    """There is one outcome shape. A handoff leaves as a parent command."""
    outcome = ResponseOutcome(
        agent_id="chat_agent", response=_agent_response(), provenance=OutcomeProvenance()
    )
    assert outcome.agent_id == "chat_agent"

    with pytest.raises(ValidationError):
        ResponseOutcome.model_validate(
            {
                "agent_id": "chat_agent",
                "response": _agent_response().model_dump(),
                "provenance": {},
                "handoff": {"to_agent_id": "search_agent"},
            }
        )


def test_worker_result_status_has_no_awaiting_approval():
    result = WorkerResult(
        dispatch_id="d1",
        task_id="t1",
        position=0,
        agent_id="search_agent",
        status="completed",
        content="worker output",
    )
    assert result.error_code is None
    assert "public_messages" not in WorkerResult.model_fields

    with pytest.raises(ValidationError):
        WorkerResult(
            dispatch_id="d1",
            task_id="t1",
            position=0,
            agent_id="search_agent",
            status="awaiting_approval",
            content="",
        )


def test_workflow_error_codes_are_closed_and_details_are_json_safe():
    error = WorkflowError(
        code="routing_timeout",
        retriable=True,
        request_id="request-1",
        details={"attempts": 2},
    )
    assert error.details == {"attempts": 2}

    with pytest.raises(ValidationError):
        WorkflowError(code="chat_fallback", retriable=False, request_id="request-1")


def test_workflow_routing_exception_carries_immutable_error():
    error = WorkflowError(code="routing_invalid_output", retriable=True, request_id="request-1")
    exception = WorkflowRoutingException(error)
    assert exception.error is error
    assert exception.error.code == "routing_invalid_output"
    assert isinstance(exception, RuntimeError)


def test_execution_phases_cover_the_approved_lifecycle():
    assert EXECUTION_PHASES == (
        "routing",
        "executing",
        "awaiting_approval",
        "validating",
        "finalizing",
        "completed",
        "failed",
    )


# ----------------------------------------------------------------------
# planning dispatch control plane
# ----------------------------------------------------------------------


def test_model_proposal_cannot_widen_tool_scope():
    """The model-facing proposal has no tool, dispatch, or position authority."""
    with pytest.raises(ValidationError):
        WorkerTaskProposal.model_validate(
            {
                "task_id": "t1",
                "objective": "inspect",
                "agent_id": "chat_agent",
                "allowed_tool_ids": ["admin::delete"],
            }
        )
    for forbidden in ("allowed_tool_ids", "dispatch_id", "position", "parent_context"):
        assert forbidden not in WorkerTaskProposal.model_fields, forbidden


def test_model_proposal_rejects_oversized_objective_instead_of_truncating():
    with pytest.raises(ValidationError):
        WorkerTaskProposal(task_id="t1", objective="x" * 4001, agent_id="chat_agent")
    accepted = WorkerTaskProposal(task_id="t1", objective="x" * 4000, agent_id="chat_agent")
    assert len(accepted.objective) == 4000


def test_dispatch_subagents_input_requires_at_least_one_task():
    with pytest.raises(ValidationError):
        DispatchSubagentsInput(tasks=())
    payload = DispatchSubagentsInput(
        tasks=(WorkerTaskProposal(task_id="t1", objective="do it", agent_id="chat_agent"),)
    )
    assert payload.rationale is None
    with pytest.raises(ValidationError):
        DispatchSubagentsInput.model_validate({"tasks": [], "dispatch_id": "forged"})


def test_worker_task_is_server_owned_and_positioned():
    task = WorkerTask(
        dispatch_id="d1",
        task_id="t1",
        position=0,
        objective="do it",
        agent_id="chat_agent",
        allowed_tool_ids=("calc::tax",),
    )
    assert task.position == 0
    assert task.allowed_tool_ids == ("calc::tax",)
    assert task.parent_context == {}
    with pytest.raises(ValidationError):
        task.allowed_tool_ids = ("admin::delete",)


def test_worker_task_rejects_negative_position_and_extra_fields():
    base = {
        "dispatch_id": "d1",
        "task_id": "t1",
        "position": 0,
        "objective": "do it",
        "agent_id": "chat_agent",
    }
    with pytest.raises(ValidationError):
        WorkerTask.model_validate({**base, "position": -1})
    with pytest.raises(ValidationError):
        WorkerTask.model_validate({**base, "execution_key": "leaked"})


def test_worker_result_identity_is_dispatch_and_task():
    result = WorkerResult(
        dispatch_id="d1",
        task_id="t1",
        position=0,
        agent_id="chat_agent",
        status="completed",
        content="one",
    )
    assert (result.dispatch_id, result.task_id, result.position) == ("d1", "t1", 0)
    assert result.images == ()


def test_planning_dispatch_pairs_one_tool_call_with_one_wave():
    dispatch = PlanningDispatch(
        dispatch_id="d1",
        tool_call_id="call-1",
        wave=1,
        tasks=(
            WorkerTask(
                dispatch_id="d1",
                task_id="t1",
                position=0,
                objective="do it",
                agent_id="chat_agent",
            ),
        ),
    )
    assert dispatch.wave == 1
    with pytest.raises(ValidationError):
        dispatch.model_validate({**dispatch.model_dump(), "wave": 0})
