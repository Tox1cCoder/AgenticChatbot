"""The device-scoped HITL policy reaches graph context via the workflow request."""

import inspect
from types import SimpleNamespace
from uuid import uuid4

from app.schemas.workflow import WorkflowExecutionRequest
from app.services.message_service import MessageService


def _policy():
    return {
        "master_enabled": True,
        "client_rules": {
            "client_mcp": {"servers": {"excel": True}, "tools": {}},
            "client_skill": {"servers": {}, "tools": {}},
        },
        "global_tools": [],
    }


def test_request_carries_hitl_policy_field():
    policy = _policy()
    req = WorkflowExecutionRequest(
        message="hi", conversation_id="c1", user_id="u1", hitl_policy=policy
    )
    assert req.hitl_policy == policy


def test_initial_state_injects_hitl_policy_into_context():
    from app.ai.graph import MultiAgentWorkflow

    wf = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    policy = _policy()
    req = WorkflowExecutionRequest(
        message="hi", conversation_id="c1", user_id="u1", hitl_policy=policy
    )
    state = wf._build_initial_state_from_request(req)
    assert state["context"]["hitl_policy"] == policy


def test_ai_service_conversion_preserves_hitl_policy():
    from app.services.ai_service import AIService

    policy = _policy()
    req = WorkflowExecutionRequest(
        message="hi", conversation_id="c1", user_id="u1", hitl_policy=policy
    )
    ai_req = AIService._to_ai_request(req)
    assert ai_req.hitl_policy == policy


def test_message_service_loads_policy_for_the_validated_device():
    user_id = uuid4()
    device_id = uuid4()
    calls = []
    grouped = _policy()["client_rules"]
    service = MessageService.__new__(MessageService)
    service.tool_approval_setting_repository = SimpleNamespace(
        build_policy=lambda requested_user, requested_device: (
            calls.append((requested_user, requested_device)) or grouped
        )
    )

    policy = service._resolve_hitl_policy(user_id, str(device_id))

    assert calls == [(user_id, device_id)]
    assert policy["client_rules"] == grouped


def test_message_service_validates_device_before_resolving_policy():
    source = inspect.getsource(MessageService._build_user_message_workflow_request)

    assert source.index("_validate_request_device_id") < source.index("_resolve_hitl_policy")
