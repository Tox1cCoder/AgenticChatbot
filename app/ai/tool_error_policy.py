from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from anyio import BrokenResourceError, ClosedResourceError


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
    retryable: bool
    message: str
    hint: str
    attempts: int

    def model_dict(self) -> dict[str, Any]:
        return {
            "status": "error",
            "error_type": self.error_type,
            "retryable": self.retryable,
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


def classify_tool_error(
    exc: BaseException,
    *,
    tool_name: str,
    timeout_seconds: int | float,
    attempts: int,
) -> ToolErrorSummary:
    raw = str(exc).lower()
    timeout_value = int(timeout_seconds) if float(timeout_seconds).is_integer() else timeout_seconds

    if isinstance(exc, TimeoutError):
        return ToolErrorSummary(
            error_type=ToolErrorKind.TIMEOUT.value,
            retryable=True,
            message=f"Tool timed out after {timeout_value}s.",
            hint=(
                "Retry only if the operation is likely safe; otherwise adjust inputs, "
                "use another available tool, discover a better tool, or ask the user."
            ),
            attempts=attempts,
        )

    if isinstance(exc, (ClosedResourceError, BrokenResourceError)):
        return ToolErrorSummary(
            error_type=ToolErrorKind.SESSION.value,
            retryable=True,
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
            retryable=False,
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
            retryable=False,
            message=_clean_text(exc) or "Tool rejected the provided values.",
            hint="Correct the values before retrying. Do not repeat the same arguments.",
            attempts=attempts,
        )

    if isinstance(exc, (PermissionError,)):
        return ToolErrorSummary(
            error_type=ToolErrorKind.PERMISSION.value,
            retryable=False,
            message="Tool lacks permission for that operation.",
            hint="Ask the user for access, choose a permitted alternative, or explain the blocker.",
            attempts=attempts,
        )

    if isinstance(exc, FileNotFoundError) or "not found" in raw:
        return ToolErrorSummary(
            error_type=ToolErrorKind.NOT_FOUND.value,
            retryable=False,
            message=_clean_text(exc) or "Requested resource was not found.",
            hint=(
                "Verify the target exists, adjust the query/path, or ask the "
                "user for the correct target."
            ),
            attempts=attempts,
        )

    if any(
        token in raw
        for token in ("connection", "network", "temporarily", "reset", "unavailable")
    ):
        return ToolErrorSummary(
            error_type=ToolErrorKind.NETWORK.value,
            retryable=True,
            message="Tool failed because the connection or service was unavailable.",
            hint=(
                "Retry only if the tool is safe to repeat; otherwise use "
                "another tool or ask the user."
            ),
            attempts=attempts,
        )

    return ToolErrorSummary(
        error_type=ToolErrorKind.UNKNOWN.value,
        retryable=False,
        message=_clean_text(exc) or "Tool failed with an unknown error.",
        hint="Do not repeat the same call without changing inputs or choosing a better tool.",
        attempts=attempts,
    )


def should_auto_retry_tool(
    tool: Any,
    summary: ToolErrorSummary,
    *,
    tool_name: str,
    retry_safe_tool_names: set[str] | None = None,
) -> bool:
    if not summary.retryable:
        return False
    if tool_name in (retry_safe_tool_names or set()):
        return True
    metadata = getattr(tool, "metadata", None) or {}
    if not isinstance(metadata, dict):
        return False
    return metadata.get("retry_safe") is True or metadata.get("idempotent") is True


def build_tool_error_payloads(
    summary: ToolErrorSummary,
    *,
    tool_name: str,
    exception: BaseException,
) -> tuple[str, dict[str, Any]]:
    model_content = json.dumps(summary.model_dict(), ensure_ascii=False, separators=(",", ":"))
    artifact_detail = {
        "status": "error",
        "tool": tool_name,
        "error_type": summary.error_type,
        "retryable": summary.retryable,
        "attempts": summary.attempts,
        "exception_type": type(exception).__name__,
        "diagnostic": _clean_text(exception, max_chars=1000),
    }
    return model_content, artifact_detail
