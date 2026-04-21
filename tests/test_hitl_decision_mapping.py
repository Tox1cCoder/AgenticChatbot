from importlib import import_module
from types import SimpleNamespace
from uuid import uuid4

from app.ai.schemas import InterruptDecision as AIInterruptDecision
from app.schemas.message import InterruptResumeRequest
from app.schemas.workflow import InterruptDecision
from app.services.message_service import MessageService
from app.ai.utils import build_interrupt_resume_payload


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


def test_utils_exposes_decision_target_helper():
    utils = import_module("app.ai.utils")

    assert callable(getattr(utils, "resolve_interrupt_decision_id", None))


def test_resume_payload_builder_preserves_explicit_tool_call_id():
    decision = InterruptDecision.model_validate(
        {
            "type": "approve",
            "task_id": "legacy-task-1",
            "tool_call_id": "tool-1",
        }
    )

    payload = build_interrupt_resume_payload([decision])

    assert payload == [
        {
            "task_id": "legacy-task-1",
            "tool_call_id": "tool-1",
            "type": "approve",
            "args": None,
        }
    ]
