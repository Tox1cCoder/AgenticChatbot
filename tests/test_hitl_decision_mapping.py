from datetime import datetime, timedelta, timezone
from importlib import import_module
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ai.schemas import InterruptDecision as AIInterruptDecision
from app.ai.utils import apply_hitl_decisions
from app.core.exceptions import CustomHTTPException
from app.models.hitl_interrupt import HITLInterruptStatus
from app.schemas.message import InterruptResumeRequest
from app.schemas.workflow import InterruptDecision, InterruptDecisionType
from app.services.message_service import MessageService


def test_interrupt_resume_request_preserves_tool_call_id():
    request = InterruptResumeRequest.model_validate(
        {
            "threadId": "thread-1",
            "conversationId": str(uuid4()),
            "interruptId": "interrupt-1",
            "decisions": [{"type": "approve", "toolCallId": "tool-1"}],
        }
    )

    assert request.decisions[0].tool_call_id == "tool-1"


def test_interrupt_resume_request_requires_interrupt_id():
    with pytest.raises(ValidationError):
        InterruptResumeRequest.model_validate(
            {
                "threadId": "thread-1",
                "conversationId": str(uuid4()),
                "decisions": [{"type": "approve", "toolCallId": "tool-1"}],
            }
        )


def test_service_rejects_resume_without_interrupt_id_before_graph_resume():
    service = MessageService(
        message_repository=SimpleNamespace(),
        conversation_validation_utils=SimpleNamespace(
            validate_conversation_access=lambda _user_id, _conversation_id: None,
        ),
        message_validation_utils=SimpleNamespace(),
        ai_service=SimpleNamespace(),
    )

    with pytest.raises(CustomHTTPException) as exc_info:
        service._validate_and_claim_interrupt_resume(
            thread_id="thread-1",
            conversation_id=uuid4(),
            user_id=uuid4(),
            interrupt_id=None,
            device_id=None,
            decisions=[],
        )

    assert exc_info.value.error_code == "INTERRUPT_ID_REQUIRED"


def test_per_user_hitl_policy_load_failure_fails_closed():
    service = MessageService(
        message_repository=SimpleNamespace(),
        conversation_validation_utils=SimpleNamespace(),
        message_validation_utils=SimpleNamespace(),
        ai_service=SimpleNamespace(),
    )
    service.tool_approval_setting_repository = SimpleNamespace(
        build_policy=lambda _user_id: (_ for _ in ()).throw(RuntimeError("database unavailable"))
    )

    with pytest.raises(RuntimeError, match="Unable to load"):
        service._resolve_hitl_policy(uuid4())


def test_interrupt_validation_rejects_disallowed_and_duplicate_decisions():
    record = SimpleNamespace(
        action_requests_json=[
            {
                "tool_call_id": "tool-1",
                "action": "client__write",
                "allowed_decisions": ["reject"],
            }
        ]
    )

    with pytest.raises(CustomHTTPException) as disallowed:
        MessageService._validate_complete_interrupt_decisions(
            record=record,
            decisions=[
                InterruptDecision(
                    type=InterruptDecisionType.APPROVE,
                    tool_call_id="tool-1",
                )
            ],
        )
    assert disallowed.value.error_code == "INTERRUPT_DECISION_NOT_ALLOWED"

    record.action_requests_json[0]["allowed_decisions"] = ["approve", "reject"]
    with pytest.raises(CustomHTTPException) as duplicate:
        MessageService._validate_complete_interrupt_decisions(
            record=record,
            decisions=[
                InterruptDecision(type=InterruptDecisionType.APPROVE, tool_call_id="tool-1"),
                InterruptDecision(type=InterruptDecisionType.REJECT, tool_call_id="tool-1"),
            ],
        )
    assert duplicate.value.error_code == "INTERRUPT_DUPLICATE_DECISION"


def test_service_to_ai_decision_conversion_preserves_tool_call_id():
    service_decision = InterruptDecision.model_validate(
        {"type": "approve", "tool_call_id": "tool-1"}
    )

    ai_decision = AIInterruptDecision.model_validate(service_decision.model_dump(mode="python"))

    assert ai_decision.tool_call_id == "tool-1"


def test_runtime_validation_provenance_matches_tool_call_id():
    record = SimpleNamespace(
        interrupt_metadata_json={
            "tool_provenance": {
                "tool-1": {
                    "tool_origin": "client_mcp",
                    "device_id": str(uuid4()),
                    "qualified_tool_id": "desktop_commander::list_directory",
                }
            }
        },
        session_id=None,
        catalog_version=None,
        tool_instance_id=None,
    )
    decision = InterruptDecision.model_validate({"type": "approve", "tool_call_id": "tool-1"})

    matched = MessageService._get_runtime_validation_provenance(record, [decision])

    assert matched == [("tool-1", record.interrupt_metadata_json["tool_provenance"]["tool-1"])]


def test_apply_hitl_decisions_prefers_explicit_tool_call_id_over_task_id():
    tool_calls = [{"id": "tool-1", "name": "client__tool", "args": {"path": "."}}]
    decisions = [
        {
            "type": "approve",
            "task_id": "legacy-task-1",
            "tool_call_id": "tool-1",
        }
    ]

    approved, rejected = apply_hitl_decisions(tool_calls, decisions)

    assert approved == tool_calls
    assert rejected == {}


def test_apply_hitl_decisions_rejects_partial_explicit_decision_set():
    tool_calls = [
        {"id": "tool-1", "name": "client__first", "args": {}},
        {"id": "tool-2", "name": "client__second", "args": {}},
    ]
    decisions = [{"type": "approve", "tool_call_id": "tool-1"}]

    with pytest.raises(ValueError, match="Missing HITL decision"):
        apply_hitl_decisions(tool_calls, decisions)


def test_utils_exposes_decision_target_helper():
    utils = import_module("app.ai.utils")

    assert callable(getattr(utils, "resolve_interrupt_decision_id", None))


def test_resume_payload_builder_preserves_explicit_tool_call_id():
    utils = import_module("app.ai.utils")
    decision = InterruptDecision.model_validate(
        {
            "type": "approve",
            "task_id": "legacy-task-1",
            "tool_call_id": "tool-1",
        }
    )

    payload = utils.build_interrupt_resume_payload([decision])

    assert payload == [
        {
            "task_id": "legacy-task-1",
            "tool_call_id": "tool-1",
            "type": "approve",
            "args": None,
        }
    ]


def test_interrupt_resume_rejects_incomplete_multi_tool_decisions():
    conversation_id = uuid4()
    user_id = uuid4()
    interrupt_record = SimpleNamespace(
        interrupt_id="interrupt-1",
        conversation_id=conversation_id,
        thread_id="thread-1",
        user_id=user_id,
        device_id=None,
        status=HITLInterruptStatus.PENDING,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        action_requests_json=[
            {
                "task_id": "tool-1",
                "tool_call_id": "tool-1",
                "action": "client__first",
                "args": {},
            },
            {
                "task_id": "tool-2",
                "tool_call_id": "tool-2",
                "action": "client__second",
                "args": {},
            },
        ],
        interrupt_metadata_json={},
        session_id=None,
        catalog_version=None,
        tool_instance_id=None,
    )
    transitions: list[str] = []
    service = MessageService(
        message_repository=SimpleNamespace(),
        conversation_validation_utils=SimpleNamespace(
            validate_conversation_access=lambda _user_id, _conversation_id: None,
        ),
        message_validation_utils=SimpleNamespace(),
        ai_service=SimpleNamespace(),
        hitl_interrupt_repository=SimpleNamespace(
            get_by_id=lambda _interrupt_id: interrupt_record,
            try_transition_to_resolving=lambda **_kwargs: transitions.append("claimed") or True,
        ),
    )
    service.redis_client = None

    with pytest.raises(CustomHTTPException) as exc_info:
        service._validate_and_claim_interrupt_resume(
            thread_id="thread-1",
            conversation_id=conversation_id,
            user_id=user_id,
            interrupt_id="interrupt-1",
            device_id=None,
            decisions=[
                InterruptDecision(
                    type=InterruptDecisionType.APPROVE,
                    tool_call_id="tool-1",
                )
            ],
        )

    assert exc_info.value.error_code == "INTERRUPT_INCOMPLETE_DECISIONS"
    assert transitions == []


def test_interrupt_resume_requires_tool_call_id_when_task_id_differs():
    conversation_id = uuid4()
    user_id = uuid4()
    interrupt_record = SimpleNamespace(
        interrupt_id="interrupt-1",
        conversation_id=conversation_id,
        thread_id="thread-1",
        user_id=user_id,
        device_id=None,
        status=HITLInterruptStatus.PENDING,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        action_requests_json=[
            {
                "task_id": "approval-row-1",
                "tool_call_id": "tool-1",
                "action": "client__first",
                "args": {},
            },
        ],
        interrupt_metadata_json={},
        session_id=None,
        catalog_version=None,
        tool_instance_id=None,
    )
    transitions: list[str] = []
    service = MessageService(
        message_repository=SimpleNamespace(),
        conversation_validation_utils=SimpleNamespace(
            validate_conversation_access=lambda _user_id, _conversation_id: None,
        ),
        message_validation_utils=SimpleNamespace(),
        ai_service=SimpleNamespace(),
        hitl_interrupt_repository=SimpleNamespace(
            get_by_id=lambda _interrupt_id: interrupt_record,
            try_transition_to_resolving=lambda **_kwargs: transitions.append("claimed") or True,
        ),
    )
    service.redis_client = None

    with pytest.raises(CustomHTTPException) as exc_info:
        service._validate_and_claim_interrupt_resume(
            thread_id="thread-1",
            conversation_id=conversation_id,
            user_id=user_id,
            interrupt_id="interrupt-1",
            device_id=None,
            decisions=[
                InterruptDecision(
                    type=InterruptDecisionType.APPROVE,
                    task_id="approval-row-1",
                )
            ],
        )

    assert exc_info.value.error_code == "INTERRUPT_INCOMPLETE_DECISIONS"
    assert transitions == []
