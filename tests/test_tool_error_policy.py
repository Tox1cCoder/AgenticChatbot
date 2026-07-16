from __future__ import annotations

import json

from anyio import BrokenResourceError, ClosedResourceError

from app.ai.rag_tool_actions import compact_rag_tool_error
from app.ai.tool_error_policy import (
    ToolErrorKind,
    build_tool_error_payloads,
    classify_tool_error,
    should_auto_retry_tool,
)


class _Tool:
    name = "example_tool"
    metadata = {}


class _RetrySafeTool:
    name = "safe_reader"
    metadata = {"retry_safe": True}


class _StringMetadataTool:
    name = "unsafe_string_flag"
    metadata = {"retry_safe": "true"}


class _ToolSearchTool:
    name = "tool_search"
    metadata = {}


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


def test_auto_retry_requires_retryable_error_and_safe_tool():
    retryable_summary = classify_tool_error(
        ConnectionError("connection reset"),
        tool_name="safe_reader",
        timeout_seconds=30,
        attempts=1,
    )
    argument_summary = classify_tool_error(
        ValueError("bad date"),
        tool_name="safe_reader",
        timeout_seconds=30,
        attempts=1,
    )

    assert (
        should_auto_retry_tool(
            _RetrySafeTool(),
            retryable_summary,
            tool_name="safe_reader",
        )
        is True
    )
    assert (
        should_auto_retry_tool(
            _Tool(),
            retryable_summary,
            tool_name="example_tool",
        )
        is False
    )
    assert (
        should_auto_retry_tool(
            _ToolSearchTool(),
            retryable_summary,
            tool_name="tool_search",
            retry_safe_tool_names={"tool_search"},
        )
        is True
    )
    assert (
        should_auto_retry_tool(
            _StringMetadataTool(),
            retryable_summary,
            tool_name="unsafe_string_flag",
        )
        is False
    )
    assert (
        should_auto_retry_tool(
            _RetrySafeTool(),
            argument_summary,
            tool_name="safe_reader",
        )
        is False
    )
