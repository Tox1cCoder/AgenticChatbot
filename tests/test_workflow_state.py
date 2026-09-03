from __future__ import annotations

import pytest

from app.ai.workflow.contracts import (
    AgentTransition,
    InvalidWorkflowStateUpdate,
    RoutingDecision,
    TurnIdentity,
    WorkerResult,
)
from app.ai.workflow.state import (
    WorkflowState,
    append_transitions,
    append_worker_results,
    build_checkpoint_thread_id,
    set_routing_decision_once,
)


def _transition(to_agent_id: str, tool_call_id: str) -> AgentTransition:
    return AgentTransition(
        from_agent_id="chat_agent",
        to_agent_id=to_agent_id,
        source="handoff",
        tool_call_id=tool_call_id,
    )


def test_transition_reducer_is_append_only():
    first = _transition("search_agent", "call-1")
    second = AgentTransition(
        from_agent_id="search_agent",
        to_agent_id="chat_agent",
        source="handoff",
        tool_call_id="call-2",
    )
    assert append_transitions([first], [second]) == [first, second]


def test_transition_reducer_accepts_none_and_single_values():
    first = _transition("search_agent", "call-1")
    assert append_transitions(None, [first]) == [first]
    assert append_transitions([], first) == [first]
    assert append_transitions([first], None) == [first]


def _worker_result(dispatch_id: str, task_id: str, position: int = 0, **overrides) -> WorkerResult:
    payload = {
        "dispatch_id": dispatch_id,
        "task_id": task_id,
        "position": position,
        "agent_id": "search_agent",
        "status": "completed",
        "content": "a",
    }
    payload.update(overrides)
    return WorkerResult(**payload)


def test_worker_result_identity_is_dispatch_and_task():
    """The same task_id in two waves is two results, not a collision."""
    first = _worker_result("d1", "t1")
    second = first.model_copy(update={"dispatch_id": "d2", "content": "two"})

    assert append_worker_results([], [first, second]) == [first, second]

    with pytest.raises(InvalidWorkflowStateUpdate, match="d1.*t1"):
        append_worker_results([first], [first])


def test_worker_result_reducer_rejects_duplicate_dispatch_task_pairs():
    first = _worker_result("d1", "t1")
    duplicate = _worker_result("d1", "t1", agent_id="chat_agent", content="b")
    with pytest.raises(InvalidWorkflowStateUpdate):
        append_worker_results([first], [duplicate])


def test_worker_result_reducer_preserves_dispatch_order():
    first = _worker_result("d1", "t1", 0)
    second = _worker_result("d1", "t2", 1, agent_id="rag_agent", status="failed", content="")
    assert append_worker_results([first], [second]) == [first, second]


def test_graph_state_has_no_selected_agent_field():
    assert "selected_agent" not in WorkflowState.__annotations__
    assert "last_agent" not in WorkflowState.__annotations__
    assert "delegation_count" not in WorkflowState.__annotations__


def test_graph_state_declares_v2_routing_identity_fields():
    annotations = WorkflowState.__annotations__
    for field in (
        "turn_identity",
        "routing_decision",
        "routing_inventory_version",
        "active_agent_id",
        "final_agent_id",
        "agent_history",
        "pending_transition",
        "agent_outcome",
        "worker_results",
        "execution_phase",
        "workflow_error",
    ):
        assert field in annotations, field


def test_graph_state_declares_planning_dispatch_checkpoint_fields():
    """Dispatch bookkeeping is checkpointed state, not node-local memory."""
    annotations = WorkflowState.__annotations__
    for field in (
        "planning_dispatch",
        "planning_dispatch_waves",
        "planning_dispatched_task_count",
        "planning_control_call_id",
    ):
        assert field in annotations, field


def test_routing_decision_reducer_rejects_replacement_within_turn():
    accepted = RoutingDecision(agent_id="chat_agent", confidence=0.8, reason="general help")
    replacement = RoutingDecision(agent_id="search_agent", confidence=0.9, reason="changed")
    with pytest.raises(InvalidWorkflowStateUpdate):
        set_routing_decision_once(accepted, replacement)


def test_routing_decision_reducer_allows_initial_set_and_identical_replay():
    accepted = RoutingDecision(agent_id="chat_agent", confidence=0.8, reason="general help")
    replay = RoutingDecision(agent_id="chat_agent", confidence=0.8, reason="general help")
    assert set_routing_decision_once(None, accepted) is accepted
    assert set_routing_decision_once(accepted, replay) == accepted
    assert set_routing_decision_once(accepted, None) == accepted


def test_checkpoint_thread_id_is_turn_scoped():
    assert (
        build_checkpoint_thread_id("conversation-1", "message-1")
        == "routing-v2:conversation-1:message-1"
    )
    with pytest.raises(ValueError):
        build_checkpoint_thread_id("", "message-1")
    with pytest.raises(ValueError):
        build_checkpoint_thread_id("conversation-1", "")


def test_two_turns_do_not_accumulate_transition_history_across_turns():
    """Append reducers are turn-local because every turn gets its own thread."""
    first_turn: list[AgentTransition] = []
    first_identity = TurnIdentity(
        request_id="request-1",
        turn_id="message-1",
        checkpoint_thread_id=build_checkpoint_thread_id("conversation-1", "message-1"),
    )
    first_turn = append_transitions(
        first_turn,
        [AgentTransition(from_agent_id=None, to_agent_id="chat_agent", source="router")],
    )
    first_turn = append_transitions(first_turn, [_transition("search_agent", "call-1")])

    second_identity = TurnIdentity(
        request_id="request-2",
        turn_id="message-2",
        checkpoint_thread_id=build_checkpoint_thread_id("conversation-1", "message-2"),
    )
    second_turn = append_transitions(
        [],
        [AgentTransition(from_agent_id=None, to_agent_id="canvas_agent", source="router")],
    )

    assert first_identity.checkpoint_thread_id != second_identity.checkpoint_thread_id
    assert len(first_turn) == 2
    assert len(second_turn) == 1


def test_both_workflow_request_schemas_carry_turn_identity_fields():
    """The service->AI request conversion drops any field the AI schema lacks."""
    from app.ai.schemas import WorkflowExecutionRequest as AIWorkflowExecutionRequest
    from app.schemas.workflow import WorkflowExecutionRequest as ServiceWorkflowExecutionRequest

    service_request = ServiceWorkflowExecutionRequest(
        message="hello",
        conversation_id="conversation-1",
        user_message_id="message-1",
        request_id="request-1",
        turn_id="message-1",
    )
    ai_request = AIWorkflowExecutionRequest.model_validate(
        service_request.model_dump(mode="python")
    )

    assert ai_request.request_id == "request-1"
    assert ai_request.turn_id == "message-1"
    assert (
        build_checkpoint_thread_id(service_request.conversation_id or "", ai_request.turn_id or "")
        == "routing-v2:conversation-1:message-1"
    )
