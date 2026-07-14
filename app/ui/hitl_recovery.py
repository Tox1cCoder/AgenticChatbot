"""Pure lifecycle recovery policy shared by HITL user interfaces."""

from collections.abc import Mapping
from typing import Any, Literal

_DUPLICATE_CODES = frozenset({"INTERRUPT_ALREADY_RESOLVED", "INTERRUPT_CONFLICT"})


def extract_error_code(payload: Mapping[str, Any]) -> str | None:
    """Extract a canonical error code from supported response envelopes."""
    for key in ("error_code", "errorCode", "code"):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    detail = payload.get("detail")
    return extract_error_code(detail) if isinstance(detail, Mapping) else None


def is_recoverable_resume_conflict(payload: Mapping[str, Any]) -> bool:
    """Return whether a resume response should reconcile durable lifecycle state."""
    return extract_error_code(payload) in _DUPLICATE_CODES


def should_suppress_pending_interrupt(
    interrupt_id: str | None, reconciling_interrupt_id: str | None
) -> bool:
    """Suppress only the exact interrupt currently being reconciled."""
    return bool(interrupt_id and interrupt_id == reconciling_interrupt_id)


def reconciliation_action(
    status: str | None,
) -> Literal["restore_form", "show_processing", "refresh_history", "require_new_message"]:
    """Map a durable interrupt status to the UI's next action."""
    return {
        "pending": "restore_form",
        "resolving": "show_processing",
        "resolved": "refresh_history",
    }.get(status, "require_new_message")
