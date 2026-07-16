"""The per-turn HITL policy reaches graph context via the workflow request."""

from app.schemas.workflow import WorkflowExecutionRequest


def test_request_carries_hitl_policy_field():
    policy = {"master_enabled": True, "servers": {"excel": True}, "tools": {}, "global_tools": []}
    req = WorkflowExecutionRequest(
        message="hi", conversation_id="c1", user_id="u1", hitl_policy=policy
    )
    assert req.hitl_policy == policy


def test_initial_state_injects_hitl_policy_into_context():
    from app.ai.graph import MultiAgentWorkflow

    wf = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    policy = {"master_enabled": True, "servers": {"excel": True}, "tools": {}, "global_tools": []}
    req = WorkflowExecutionRequest(
        message="hi", conversation_id="c1", user_id="u1", hitl_policy=policy
    )
    state = wf._build_initial_state_from_request(req)
    assert state["context"]["hitl_policy"] == policy


def test_ai_service_conversion_preserves_hitl_policy():
    from app.services.ai_service import AIService

    policy = {"master_enabled": True, "servers": {"excel": True}, "tools": {}, "global_tools": []}
    req = WorkflowExecutionRequest(
        message="hi", conversation_id="c1", user_id="u1", hitl_policy=policy
    )
    ai_req = AIService._to_ai_request(req)
    assert ai_req.hitl_policy == policy
