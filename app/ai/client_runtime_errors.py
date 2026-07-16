from __future__ import annotations

from typing import Any, Literal

from pydantic import ValidationError

from app.schemas.runtime_protocol import RuntimeErrorContext

UNKNOWN_RUNTIME_ERROR = "UNKNOWN_RUNTIME_ERROR"

ClientRuntimeErrorKind = Literal[
    "validation",
    "permission",
    "session",
    "network",
    "timeout",
    "unknown",
]


class ClientRuntimeToolError(RuntimeError):
    """Typed sidecar failure that preserves structured runtime context."""

    def __init__(self, context: RuntimeErrorContext):
        super().__init__(context.message)
        self.context = context


def runtime_error_context_from_response(response: dict[str, Any]) -> RuntimeErrorContext:
    """Safely recover structured context from an unsuccessful sidecar response."""
    raw_context = response.get("error_context")
    context: RuntimeErrorContext | None = None
    if isinstance(raw_context, RuntimeErrorContext):
        context = raw_context
    elif isinstance(raw_context, dict):
        try:
            context = RuntimeErrorContext.model_validate(raw_context)
        except ValidationError:
            context = None

    if context is not None:
        if context.code:
            return context
        return context.model_copy(update={"code": UNKNOWN_RUNTIME_ERROR})

    raw_message = response.get("error")
    message = (
        str(raw_message).strip()
        if raw_message not in (None, "")
        else "Client runtime tool failed without structured error context."
    )
    return RuntimeErrorContext(message=message, code=UNKNOWN_RUNTIME_ERROR)


def client_runtime_error_from_response(response: dict[str, Any]) -> ClientRuntimeToolError:
    return ClientRuntimeToolError(runtime_error_context_from_response(response))


def classify_client_runtime_error_code(code: str | None) -> ClientRuntimeErrorKind:
    normalized = str(code or "").strip().upper()
    if normalized.startswith(("INVALID_", "VALIDATION_")):
        return "validation"
    if normalized.startswith(("PERMISSION_", "DENIED_")):
        return "permission"
    if normalized.startswith(("DEVICE_", "SESSION_", "CATALOG_")):
        return "session"
    if normalized.startswith(("NETWORK_", "UNAVAILABLE_")):
        return "network"
    if normalized.startswith("TIMEOUT_"):
        return "timeout"
    return "unknown"
