"""Pure helpers for the demo's "Fork conversation" action.

Kept free of Streamlit so the request-id, visibility, lineage and cleanup rules
are unit-testable. ``demo.py`` owns only the HTTP call and the rerun.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import Any

FORK_ATTEMPTS_KEY = "fork_attempts"

# A retry with the same requestId is only meaningful while the server could still
# answer it differently; these outcomes are final for that id.
_TERMINAL_ERROR_CODES = frozenset({"FORK_RESULT_DELETED", "FORK_REQUEST_CONFLICT"})

_ERROR_MESSAGES = {
    "FORK_MESSAGE_NOT_FOUND": "That reply no longer exists in this conversation.",
    "FORK_POINT_NOT_COMPLETED": "Only a completed assistant reply can be forked.",
    "FORK_REQUEST_CONFLICT": "This fork request was already used for a different reply.",
    "FORK_RESULT_DELETED": "The conversation created by this fork was deleted.",
    "FORK_DOCUMENT_CONTEXT_UNSUPPORTED": (
        "Conversations with attached documents cannot be forked yet."
    ),
    "FORK_HISTORY_TOO_LARGE": "The history up to this reply is too large to fork.",
    "conversation_turn_conflict": "A reply is still being generated. Try again in a moment.",
}

_FORK_NAVIGATION_POPPED_KEYS = (
    "generation_snapshot",
    "pending_interrupt",
    "interrupt_conversation_id",
)


def fork_error_message(code: str | None, fallback: str) -> str:
    """Return the user-facing text for a fork failure code."""
    return _ERROR_MESSAGES.get(code or "", fallback)


def can_fork_message(message: dict[str, Any], metadata: dict[str, Any], *, is_user: bool) -> bool:
    """A fork point is a persisted assistant reply, never a user or placeholder row.

    The server stays authoritative (it rejects unfinished or continued replies).
    """
    if is_user or not message.get("id") or metadata.get("paused"):
        return False
    return bool(str(message.get("content") or "").strip())


def acquire_fork_request(
    attempts: MutableMapping[str, Any],
    source_id: str,
    message_id: str,
    *,
    new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
) -> str | None:
    """Return the request id for this action, or None while one is already pending.

    The id is minted once per (source, message) and kept across failed attempts so
    a retry replays the same server-side fork instead of creating a second one.
    """
    key = f"{source_id}:{message_id}"
    attempt = attempts.get(key)
    if attempt is None:
        attempt = {"request_id": new_id(), "pending": False}
        attempts[key] = attempt
    if attempt["pending"]:
        return None
    attempt["pending"] = True
    return str(attempt["request_id"])


def release_fork_request(
    attempts: MutableMapping[str, Any],
    source_id: str,
    message_id: str,
    *,
    succeeded: bool,
    error_code: str | None = None,
) -> None:
    """Re-enable the control; drop the id on success or a terminal server answer."""
    key = f"{source_id}:{message_id}"
    attempt = attempts.get(key)
    if attempt is None:
        return
    if succeeded or error_code in _TERMINAL_ERROR_CODES:
        del attempts[key]
        return
    attempt["pending"] = False


def is_fork_pending(attempts: MutableMapping[str, Any], source_id: str, message_id: str) -> bool:
    attempt = attempts.get(f"{source_id}:{message_id}")
    return bool(attempt and attempt["pending"])


def build_fork_request(message_id: str, request_id: str) -> dict[str, str]:
    return {"messageId": message_id, "requestId": request_id}


def clear_source_state_for_fork_navigation(state: MutableMapping[str, Any]) -> None:
    """Drop source-only UI state the shared reset deliberately retains.

    ``generation_snapshot`` survives ``_clear_inflight_state`` so a paused turn stays
    continuable in the SAME conversation; navigating to a child must not inherit it.
    """
    for key in _FORK_NAVIGATION_POPPED_KEYS:
        state.pop(key, None)
    state["live_widget_mounts"] = {}


@dataclass(frozen=True)
class ForkLineage:
    source_id: str
    source_title: str | None
    linked: bool


def fork_lineage(
    conversation: dict[str, Any] | None, conversations: list[dict[str, Any]]
) -> ForkLineage | None:
    """Describe where a forked conversation came from, or None if it is not a fork.

    The server nulls ``forkedFromConversationId`` when the source is deleted, but a
    ``forkedThroughSequence`` remains; a stale list can also lack a live source.
    Both cases render as a non-linked historical label.
    """
    if not conversation or conversation.get("forkedThroughSequence") is None:
        return None
    source_id = str(conversation.get("forkedFromConversationId") or "")
    source = next((c for c in conversations if source_id and c.get("id") == source_id), None)
    if source is None:
        return ForkLineage(source_id=source_id, source_title=None, linked=False)
    return ForkLineage(source_id=source_id, source_title=source.get("title"), linked=True)


def lineage_label(lineage: ForkLineage) -> str:
    if not lineage.linked:
        return "Forked from a deleted conversation"
    return f"Forked from {lineage.source_title or 'Untitled conversation'}"
