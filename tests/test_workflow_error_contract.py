"""The public failure contract is a typed payload, not an English string.

A caller needs to know *what* failed and *whether retrying helps*. Both are
machine-readable fields; the display text is the API layer's job. Structured
details are allowlisted so a failure can never leak a credential, a prompt, a
provider payload, or document content.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.ai.workflow.contracts import WORKFLOW_ERROR_CODES, WorkflowError
from app.ai.workflow.errors import (
    ALLOWED_ERROR_DETAIL_KEYS,
    RETRIABLE_BY_DEFAULT,
    sanitize_error_details,
    workflow_error_from_exception,
    workflow_error_payload,
)


def _error(code="routing_timeout", **overrides) -> WorkflowError:
    payload = {"code": code, "retriable": True, "request_id": "request-1"}
    payload.update(overrides)
    return WorkflowError(**payload)


# ----------------------------------------------------------------------
# the payload
# ----------------------------------------------------------------------


def test_every_declared_code_is_constructible():
    for code in WORKFLOW_ERROR_CODES:
        error = WorkflowError(
            code=code, retriable=RETRIABLE_BY_DEFAULT[code], request_id="request-1"
        )
        assert error.code == code


def test_an_undeclared_code_is_rejected():
    with pytest.raises(ValidationError):
        WorkflowError(code="something_went_wrong", retriable=True, request_id="request-1")


def test_request_id_is_required_and_never_optional():
    with pytest.raises(ValidationError):
        WorkflowError(code="routing_timeout", retriable=True, request_id="")


def test_payload_is_json_serializable_for_the_api_boundary():
    payload = workflow_error_payload(_error(details={"attempts": 2}))
    assert json.loads(json.dumps(payload)) == payload
    assert payload["code"] == "routing_timeout"
    assert payload["retriable"] is True
    assert payload["request_id"] == "request-1"


def test_payload_carries_no_english_display_copy():
    """Localization belongs to the API layer, not the workflow contract."""
    payload = workflow_error_payload(_error())
    assert "message" not in payload
    assert "detail" not in payload


# ----------------------------------------------------------------------
# detail allowlist
# ----------------------------------------------------------------------


def test_details_outside_the_allowlist_are_dropped():
    sanitized = sanitize_error_details(
        {
            "attempts": 2,
            "api_key": "sk-secret",
            "prompt": "the whole user prompt",
            "provider_response": {"raw": "..."},
            "document_content": "confidential body",
            "traceback": "Traceback (most recent call last)...",
        }
    )
    assert sanitized == {"attempts": 2}


def test_allowlist_covers_only_bounded_diagnostic_keys():
    assert frozenset(
        {
            "attempts",
            "cause",
            "reason",
            "agent",
            "node",
            "provider",
            "model",
            "limit_kind",
            "stage",
            "inventory_version",
        }
    ) == ALLOWED_ERROR_DETAIL_KEYS


def test_sanitized_details_stay_json_safe():
    sanitized = sanitize_error_details({"attempts": 2, "cause": "target_race"})
    assert json.loads(json.dumps(sanitized)) == sanitized


def test_unserializable_allowlisted_values_are_dropped():
    sanitized = sanitize_error_details({"attempts": object(), "cause": "ok"})
    assert sanitized == {"cause": "ok"}


def test_no_secret_shaped_value_survives_even_under_an_allowlisted_key():
    """An allowlisted key still cannot smuggle a whole payload through."""
    sanitized = sanitize_error_details({"reason": {"api_key": "sk-secret"}})
    assert "sk-secret" not in json.dumps(sanitized)


# ----------------------------------------------------------------------
# translation
# ----------------------------------------------------------------------


def test_a_routing_exception_translates_without_parsing_its_text():
    from app.ai.workflow.contracts import WorkflowRoutingException

    original = _error(code="routing_provider_unavailable")
    translated = workflow_error_from_exception(
        WorkflowRoutingException(original), request_id="request-9"
    )
    assert translated is original


def test_an_unexpected_exception_becomes_a_terminal_typed_error():
    translated = workflow_error_from_exception(RuntimeError("boom"), request_id="request-1")
    assert translated.code == "finalization_failed"
    assert translated.retriable is False
    assert "boom" not in json.dumps(translated.model_dump(mode="json"))


def test_retriability_is_declared_per_code_not_guessed():
    assert RETRIABLE_BY_DEFAULT["routing_timeout"] is True
    assert RETRIABLE_BY_DEFAULT["routing_provider_unavailable"] is True
    assert RETRIABLE_BY_DEFAULT["conversation_turn_conflict"] is True
    assert RETRIABLE_BY_DEFAULT["response_validation_failed"] is False
    assert RETRIABLE_BY_DEFAULT["agent_execution_limit"] is False


def test_every_code_has_a_declared_retriability():
    assert set(RETRIABLE_BY_DEFAULT) == set(WORKFLOW_ERROR_CODES)


# ----------------------------------------------------------------------
# schema boundary
# ----------------------------------------------------------------------


def test_the_service_boundary_carries_the_typed_error_payload():
    """The public contract is the service response, not every agent response.

    ``AgentResponse.error`` reports the failing code an agent saw; the service
    boundary is where a caller reads ``code`` / ``retriable`` / ``request_id``
    as structured data.
    """
    from app.schemas.workflow import WorkflowResponse

    response = WorkflowResponse(error=workflow_error_payload(_error()))
    assert response.error["code"] == "routing_timeout"
    assert response.error["retriable"] is True
    assert response.error["request_id"] == "request-1"


def test_the_agent_response_error_is_a_stable_code_not_prose():
    from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole

    response = AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content=""),
        error="routing_timeout",
    )
    assert response.error in WORKFLOW_ERROR_CODES


def test_service_response_error_stays_json_safe():
    from app.schemas.workflow import WorkflowResponse

    response = WorkflowResponse(error=workflow_error_payload(_error(details={"attempts": 2})))
    assert json.loads(json.dumps(response.error)) == response.error
