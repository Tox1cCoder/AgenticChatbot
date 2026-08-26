"""Translation and sanitization for the public workflow failure contract.

A caller needs two machine-readable facts: what failed, and whether retrying
helps. Both are fields. Display copy belongs to the API/localization layer, so
nothing here produces English prose.

Structured details are allowlisted rather than filtered. A denylist has to
anticipate every shape a secret can take; an allowlist only has to name the
handful of bounded diagnostics that are safe to return.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.ai.workflow.contracts import (
    WORKFLOW_ERROR_CODES,
    WorkflowError,
    WorkflowErrorCode,
    WorkflowRoutingException,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ALLOWED_ERROR_DETAIL_KEYS",
    "RETRIABLE_BY_DEFAULT",
    "sanitize_error_details",
    "workflow_error",
    "workflow_error_from_exception",
    "workflow_error_payload",
]

# Whether retrying the same request can plausibly succeed. Declared per code so
# no call site has to guess, and so a caller's retry policy is a property of the
# failure rather than of who reported it.
RETRIABLE_BY_DEFAULT: dict[WorkflowErrorCode, bool] = {
    "routing_timeout": True,
    "routing_provider_unavailable": True,
    "routing_invalid_output": True,
    "routing_target_unavailable": True,
    "agent_execution_limit": False,
    "tool_execution_failed": False,
    "response_validation_failed": False,
    "finalization_failed": False,
    "response_persistence_failed": True,
    "conversation_turn_conflict": True,
}

# Bounded diagnostics that are safe to return to a caller. Anything else —
# prompts, provider payloads, document content, credentials, stack traces — is
# dropped rather than redacted, because dropping cannot be defeated by an
# unanticipated shape.
ALLOWED_ERROR_DETAIL_KEYS = frozenset(
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
)

# Substrings that mark a value as unsafe to return even under an allowlisted
# key: an allowlisted key must not become a channel for a whole payload.
_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)

_MAX_DETAIL_VALUE_CHARS = 200


def sanitize_error_details(details: Any) -> dict[str, Any]:
    """Keep only bounded, JSON-safe, allowlisted diagnostics."""
    if not isinstance(details, dict):
        return {}

    sanitized: dict[str, Any] = {}
    for key, value in details.items():
        if key not in ALLOWED_ERROR_DETAIL_KEYS:
            continue
        try:
            encoded = json.dumps(value)
        except (TypeError, ValueError):
            continue
        if any(marker in encoded.lower() for marker in _SECRET_MARKERS):
            continue
        if len(encoded) > _MAX_DETAIL_VALUE_CHARS:
            continue
        sanitized[key] = value
    return sanitized


def workflow_error(
    code: WorkflowErrorCode,
    *,
    request_id: str,
    details: Any = None,
    retriable: bool | None = None,
) -> WorkflowError:
    """Build a sanitized workflow error with the code's declared retriability."""
    if code not in WORKFLOW_ERROR_CODES:  # pragma: no cover - defensive
        raise ValueError(f"unknown workflow error code: {code!r}")
    return WorkflowError(
        code=code,
        retriable=RETRIABLE_BY_DEFAULT[code] if retriable is None else bool(retriable),
        request_id=request_id or "unknown",
        details=sanitize_error_details(details),
    )


def workflow_error_from_exception(exception: BaseException, *, request_id: str) -> WorkflowError:
    """Translate an exception into the typed contract without parsing its text.

    A typed exception carries its own error; anything else is a terminal
    finalization failure. The exception message is deliberately not copied into
    the payload — it is uncontrolled text that may contain anything.
    """
    if isinstance(exception, WorkflowRoutingException):
        return exception.error

    logger.warning(
        "Unexpected workflow failure translated to finalization_failed: %s",
        type(exception).__name__,
    )
    return workflow_error(
        "finalization_failed",
        request_id=request_id,
        details={"reason": "unexpected_error"},
    )


def workflow_error_payload(error: WorkflowError) -> dict[str, Any]:
    """The JSON-safe body the API boundary returns.

    Display copy is added by the API/localization layer; ``code``,
    ``retriable``, ``request_id``, and the allowlisted details stay intact.
    """
    return {
        "code": error.code,
        "retriable": error.retriable,
        "request_id": error.request_id,
        "details": sanitize_error_details(error.details),
    }
