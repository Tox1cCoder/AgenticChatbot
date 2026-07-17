from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from anyio import BrokenResourceError, ClosedResourceError

from .client_runtime_errors import (
    ClientRuntimeToolError,
    classify_client_runtime_error_code,
)


class ToolErrorKind(str, Enum):
    TIMEOUT = "timeout"
    SESSION = "session"
    ARGUMENT = "argument"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    NETWORK = "network"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ToolErrorSummary:
    error_type: str
    failure_retryable: bool
    message: str
    hint: str
    attempts: int

    def model_dict(self, *, policy_retry_allowed: bool) -> dict[str, Any]:
        return {
            "status": "error",
            "error_type": self.error_type,
            "retryable": self.failure_retryable and policy_retry_allowed,
            "message": self.message,
            "hint": self.hint,
        }


def _clean_text(value: Any, *, max_chars: int = 180) -> str:
    text = str(value or "").strip().replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


_SKILL_COMMAND_CAPABILITY = "run_skill_command"
_SKILL_TERMINAL_HINT = (
    "Read untrusted_terminal_output and address the reported problem. "
    "If it reports a missing credential or environment variable, ask the user to "
    "add that secret binding in the skill's secret settings; it is injected into "
    "the command environment automatically on the next run. Never pass secret "
    "values as command arguments, set environment variables through another "
    "tool, or repeat secret values in conversation."
)


def skill_terminal_output(exception: BaseException) -> str | None:
    """Return the complete redacted terminal error for a skill command failure.

    Skill command failures are the one class of client-runtime errors whose raw
    message must reach the model verbatim (uncapped-terminal-errors design).
    Returns ``None`` unless the exception is a structured client-runtime
    failure whose ``qualified_tool_id`` has the canonical
    ``skill::<name>::run_skill_command`` shape and whose message is nonempty.
    """
    if not isinstance(exception, ClientRuntimeToolError):
        return None
    detail = exception.context.detail
    if not isinstance(detail, dict):
        return None
    parts = str(detail.get("qualified_tool_id") or "").split("::")
    if len(parts) != 3 or parts[0] != "skill" or not parts[1]:
        return None
    if parts[2] != _SKILL_COMMAND_CAPABILITY:
        return None
    message = exception.context.message
    if not message or not message.strip():
        return None
    return message


def classify_tool_error(
    exc: BaseException,
    *,
    tool_name: str,
    timeout_seconds: int | float | None,
    attempts: int,
) -> ToolErrorSummary:
    raw = str(exc).lower()

    if isinstance(exc, ClientRuntimeToolError):
        error_type = classify_client_runtime_error_code(exc.context.code)
        failure_retryable = error_type in {
            ToolErrorKind.SESSION.value,
            ToolErrorKind.NETWORK.value,
            ToolErrorKind.TIMEOUT.value,
        }
        messages = {
            ToolErrorKind.VALIDATION.value: "Client runtime rejected the provided values.",
            ToolErrorKind.PERMISSION.value: "Client runtime denied the requested operation.",
            ToolErrorKind.SESSION.value: "Client runtime session was interrupted.",
            ToolErrorKind.NETWORK.value: ("Client runtime could not reach the requested service."),
            ToolErrorKind.TIMEOUT.value: "Client runtime operation timed out.",
            ToolErrorKind.UNKNOWN.value: "Client runtime tool failed with an unknown error.",
        }
        hints = {
            ToolErrorKind.VALIDATION.value: (
                "Correct the values before retrying. Do not repeat the same arguments."
            ),
            ToolErrorKind.PERMISSION.value: (
                "Ask the user for access, choose a permitted alternative, or explain the blocker."
            ),
            ToolErrorKind.SESSION.value: (
                "Use the active client session or ask the user to reconnect before retrying."
            ),
            ToolErrorKind.NETWORK.value: (
                "Retry only if the operation is safe to repeat; otherwise use another tool."
            ),
            ToolErrorKind.TIMEOUT.value: (
                "Retry only if the operation is safe to repeat; otherwise adjust the request."
            ),
            ToolErrorKind.UNKNOWN.value: (
                "Do not repeat the same call without changing inputs or choosing another tool."
            ),
        }
        return ToolErrorSummary(
            error_type=error_type,
            failure_retryable=failure_retryable,
            message=messages[error_type],
            hint=hints[error_type],
            attempts=attempts,
        )

    if isinstance(exc, TimeoutError):
        if timeout_seconds is None:
            timeout_message = "Tool timed out in an underlying operation."
        else:
            timeout_value = (
                int(timeout_seconds) if float(timeout_seconds).is_integer() else timeout_seconds
            )
            timeout_message = f"Tool timed out after {timeout_value}s."
        return ToolErrorSummary(
            error_type=ToolErrorKind.TIMEOUT.value,
            failure_retryable=True,
            message=timeout_message,
            hint=(
                "Retry only if the operation is likely safe; otherwise adjust inputs, "
                "use another available tool, discover a better tool, or ask the user."
            ),
            attempts=attempts,
        )

    if isinstance(exc, (ClosedResourceError, BrokenResourceError)):
        return ToolErrorSummary(
            error_type=ToolErrorKind.SESSION.value,
            failure_retryable=True,
            message="Tool session was interrupted.",
            hint=(
                "The runtime may reconnect. If it still fails, use another "
                "available tool or ask the user."
            ),
            attempts=attempts,
        )

    if isinstance(exc, TypeError):
        return ToolErrorSummary(
            error_type=ToolErrorKind.ARGUMENT.value,
            failure_retryable=False,
            message="Tool arguments did not match the expected schema.",
            hint=(
                "Read the tool schema or prior error, then call the tool again "
                "only with corrected arguments."
            ),
            attempts=attempts,
        )

    if isinstance(exc, (ValueError, KeyError)):
        return ToolErrorSummary(
            error_type=ToolErrorKind.VALIDATION.value,
            failure_retryable=False,
            message=_clean_text(exc) or "Tool rejected the provided values.",
            hint="Correct the values before retrying. Do not repeat the same arguments.",
            attempts=attempts,
        )

    if isinstance(exc, (PermissionError,)):
        return ToolErrorSummary(
            error_type=ToolErrorKind.PERMISSION.value,
            failure_retryable=False,
            message="Tool lacks permission for that operation.",
            hint="Ask the user for access, choose a permitted alternative, or explain the blocker.",
            attempts=attempts,
        )

    if isinstance(exc, FileNotFoundError) or "not found" in raw:
        return ToolErrorSummary(
            error_type=ToolErrorKind.NOT_FOUND.value,
            failure_retryable=False,
            message=_clean_text(exc) or "Requested resource was not found.",
            hint=(
                "Verify the target exists, adjust the query/path, or ask the "
                "user for the correct target."
            ),
            attempts=attempts,
        )

    if any(
        token in raw for token in ("connection", "network", "temporarily", "reset", "unavailable")
    ):
        return ToolErrorSummary(
            error_type=ToolErrorKind.NETWORK.value,
            failure_retryable=True,
            message="Tool failed because the connection or service was unavailable.",
            hint=(
                "Retry only if the tool is safe to repeat; otherwise use "
                "another tool or ask the user."
            ),
            attempts=attempts,
        )

    return ToolErrorSummary(
        error_type=ToolErrorKind.UNKNOWN.value,
        failure_retryable=False,
        message=_clean_text(exc) or "Tool failed with an unknown error.",
        hint="Do not repeat the same call without changing inputs or choosing a better tool.",
        attempts=attempts,
    )


def build_tool_error_payloads(
    summary: ToolErrorSummary,
    *,
    tool_name: str,
    exception: BaseException,
    policy_retry_allowed: bool,
) -> tuple[str, dict[str, Any]]:
    model_retryable = summary.failure_retryable and policy_retry_allowed
    model_payload = summary.model_dict(policy_retry_allowed=policy_retry_allowed)
    terminal_output = skill_terminal_output(exception)
    if terminal_output is not None:
        if summary.error_type == ToolErrorKind.UNKNOWN.value:
            model_payload["hint"] = _SKILL_TERMINAL_HINT
        model_payload["untrusted_terminal_output"] = terminal_output
    model_content = json.dumps(
        model_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    artifact_detail = {
        "status": "error",
        "tool": tool_name,
        "error_type": summary.error_type,
        "retryable": model_retryable,
        "failure_retryable": summary.failure_retryable,
        "policy_retry_allowed": policy_retry_allowed,
        "attempts": summary.attempts,
        "exception_type": type(exception).__name__,
        "diagnostic": _clean_text(exception, max_chars=1000),
    }
    if isinstance(exception, ClientRuntimeToolError):
        artifact_detail["runtime_error_context"] = exception.context.model_dump(mode="json")
    if terminal_output is not None:
        artifact_detail["skill_terminal_error"] = True
    return model_content, artifact_detail
