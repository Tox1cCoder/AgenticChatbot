from __future__ import annotations

import json

import pytest
from anyio import BrokenResourceError, ClosedResourceError

from app.ai.rag_tool_actions import compact_rag_tool_error
from app.ai.tool_error_policy import (
    ToolErrorKind,
    build_tool_error_payloads,
    classify_tool_error,
)
from app.schemas.runtime_protocol import RuntimeErrorContext


def test_compact_rag_tool_error_preserves_retryable_parameter_semantics():
    for retryable in (False, True):
        payload = compact_rag_tool_error(
            error_type="validation",
            message="x",
            hint="y",
            retryable=retryable,
        )

        assert json.loads(payload) == {
            "status": "error",
            "error_type": "validation",
            "retryable": retryable,
            "message": "x",
            "hint": "y",
        }


def test_timeout_error_is_compact_and_retryable():
    summary = classify_tool_error(
        TimeoutError("raw provider timeout with verbose internals"),
        tool_name="example_tool",
        timeout_seconds=30,
        attempts=1,
    )

    assert summary.error_type == ToolErrorKind.TIMEOUT.value
    assert summary.failure_retryable is True
    assert "30s" in summary.message
    assert "raw provider" not in summary.message


def test_argument_error_is_not_retryable_by_code():
    summary = classify_tool_error(
        TypeError("unexpected keyword argument 'pathh'"),
        tool_name="read_file",
        timeout_seconds=30,
        attempts=1,
    )

    assert summary.error_type == ToolErrorKind.ARGUMENT.value
    assert summary.failure_retryable is False
    assert "argument" in summary.hint.lower()


def test_session_errors_are_retryable_transport_failures():
    for exc in (ClosedResourceError(), BrokenResourceError()):
        summary = classify_tool_error(
            exc,
            tool_name="mcp_tool",
            timeout_seconds=30,
            attempts=1,
        )
        assert summary.error_type == ToolErrorKind.SESSION.value
        assert summary.failure_retryable is True


def test_model_payload_stays_small_and_structured():
    summary = classify_tool_error(
        PermissionError("permission denied for /secret/token"),
        tool_name="read_file",
        timeout_seconds=30,
        attempts=1,
    )
    model_content, artifact_detail = build_tool_error_payloads(
        summary,
        tool_name="read_file",
        exception=PermissionError("permission denied for /secret/token"),
        policy_retry_allowed=False,
    )

    assert '"status":"error"' in model_content
    assert '"error_type":"permission"' in model_content
    assert len(model_content) < 500
    assert artifact_detail["exception_type"] == "PermissionError"
    assert artifact_detail["attempts"] == 1


def test_transient_failure_on_unsafe_tool_is_not_model_retryable():
    summary = classify_tool_error(
        ConnectionError("network unavailable"),
        tool_name="send_message",
        timeout_seconds=30,
        attempts=1,
    )

    model_content, _ = build_tool_error_payloads(
        summary,
        tool_name="send_message",
        exception=ConnectionError("network unavailable"),
        policy_retry_allowed=False,
    )

    assert '"retryable":false' in model_content


def test_failure_retryable_is_preserved_for_artifact_diagnostics():
    summary = classify_tool_error(
        ConnectionError("network unavailable"),
        tool_name="send_message",
        timeout_seconds=30,
        attempts=1,
    )

    _, artifact_detail = build_tool_error_payloads(
        summary,
        tool_name="send_message",
        exception=ConnectionError("network unavailable"),
        policy_retry_allowed=False,
    )

    assert artifact_detail["failure_retryable"] is True
    assert artifact_detail["policy_retry_allowed"] is False


@pytest.mark.parametrize(
    ("code", "expected_type", "failure_retryable"),
    [
        ("INVALID_ARGUMENT", "validation", False),
        ("VALIDATION_FAILED", "validation", False),
        ("PERMISSION_DENIED", "permission", False),
        ("DENIED_BY_POLICY", "permission", False),
        ("DEVICE_DISCONNECTED", "session", True),
        ("SESSION_EXPIRED", "session", True),
        ("CATALOG_STALE", "session", True),
        ("NETWORK_RESET", "network", True),
        ("UNAVAILABLE_SERVICE", "network", True),
        ("TIMEOUT_LOCAL_TOOL", "timeout", True),
        ("SOMETHING_NEW", "unknown", False),
        (None, "unknown", False),
    ],
)
def test_structured_runtime_error_code_families(
    code,
    expected_type,
    failure_retryable,
):
    from app.ai.client_runtime_errors import ClientRuntimeToolError

    error = ClientRuntimeToolError(
        RuntimeErrorContext(
            message="raw sidecar message C:/private/secret.txt",
            code=code,
            detail={"path": "C:/private/secret.txt"},
        )
    )

    summary = classify_tool_error(
        error,
        tool_name="read_file",
        timeout_seconds=30,
        attempts=1,
    )

    assert summary.error_type == expected_type
    assert summary.failure_retryable is failure_retryable
    assert "secret.txt" not in summary.message


def test_structured_runtime_error_detail_is_artifact_only():
    from app.ai.client_runtime_errors import ClientRuntimeToolError

    context = RuntimeErrorContext(
        message="local path was denied",
        code="PERMISSION_DENIED",
        detail={"path": "C:/private/secret.txt"},
    )
    error = ClientRuntimeToolError(context)
    summary = classify_tool_error(
        error,
        tool_name="read_file",
        timeout_seconds=30,
        attempts=1,
    )

    model_content, artifact_detail = build_tool_error_payloads(
        summary,
        tool_name="read_file",
        exception=error,
        policy_retry_allowed=False,
    )

    assert "secret.txt" not in model_content
    assert artifact_detail["runtime_error_context"]["detail"]["path"].endswith("secret.txt")


def test_missing_sidecar_error_context_falls_back_to_typed_unknown_error():
    from app.ai.client_runtime_errors import client_runtime_error_from_response

    error = client_runtime_error_from_response(
        {"success": False, "error": "sidecar returned no structured context"}
    )

    assert error.context.message == "sidecar returned no structured context"
    assert error.context.code == "UNKNOWN_RUNTIME_ERROR"
    assert error.context.detail is None
