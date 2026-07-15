"""Small HITL decision helpers shared by UI surfaces."""

from collections.abc import Mapping
from typing import Any

_DEFAULT_ALLOWED_DECISIONS = frozenset({"approve", "edit", "reject"})


def _first_present(data: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def interrupt_request_target_ids(
    action_request: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Return the UI task id and LangChain tool-call id for an interrupt request."""
    tool_call_id = _first_present(action_request, "tool_call_id", "toolCallId", "id")
    task_id = _first_present(action_request, "task_id", "taskId") or tool_call_id
    return task_id, tool_call_id


def interrupt_request_action(action_request: Mapping[str, Any]) -> str | None:
    """Return the display/action tool name from an interrupt request."""
    return _first_present(action_request, "action", "tool", "name")


def interrupt_allowed_decisions(action_request: Mapping[str, Any]) -> frozenset[str]:
    """Return the normalized decision types the request explicitly permits.

    Older interrupt payloads did not include this field, so they retain the
    original approve/edit/reject behavior. A string is treated as one decision
    instead of an iterable of characters, which also makes recovery tolerant of
    hand-authored and legacy payloads.
    """
    raw = action_request.get("allowed_decisions")
    if raw in (None, ""):
        raw = action_request.get("allowedDecisions")
    if raw in (None, "", [], (), set(), frozenset()):
        return _DEFAULT_ALLOWED_DECISIONS

    values = [raw] if isinstance(raw, str) else raw
    if not isinstance(values, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(
        str(value).strip().lower()
        for value in values
        if isinstance(value, str) and str(value).strip()
    )


def build_interrupt_decision(
    decision_type: str,
    action_request: Mapping[str, Any],
    *,
    args: dict[str, Any] | None,
    action: str | None = None,
) -> dict[str, Any]:
    """Build a backend resume decision preserving both legacy and tool-call ids."""
    task_id, tool_call_id = interrupt_request_target_ids(action_request)
    return {
        "type": decision_type,
        "task_id": task_id,
        "tool_call_id": tool_call_id,
        "action": action or interrupt_request_action(action_request),
        "args": args,
    }


# Keys under which the agent's pre-interrupt reasoning/partial answer are stashed
# on the interrupt payload so the approval UI can show them. Without this the
# streamed thinking/content is lost when Streamlit reruns into the approval view
# (an interrupt turn never persists a visible assistant message).
_STREAM_THINKING_KEY = "agent_thinking"
_STREAM_CONTENT_KEY = "agent_partial_content"


def attach_stream_context(
    interrupt_data: Any,
    *,
    thinking: str | None = None,
    content: str | None = None,
) -> Any:
    """Stash the agent's streamed reasoning/partial text onto a pending interrupt.

    Returns ``interrupt_data`` unchanged when it is not a mutable mapping. Empty
    or whitespace-only values are ignored so a quiet tool-only turn does not add
    blank panels to the approval UI.
    """
    if not isinstance(interrupt_data, dict):
        return interrupt_data
    thinking = (thinking or "").strip()
    content = (content or "").strip()
    if thinking:
        interrupt_data[_STREAM_THINKING_KEY] = thinking
    if content:
        interrupt_data[_STREAM_CONTENT_KEY] = content
    return interrupt_data


def interrupt_stream_context(interrupt_data: Any) -> tuple[str, str]:
    """Return ``(thinking, partial_content)`` previously attached to an interrupt."""
    if not isinstance(interrupt_data, Mapping):
        return "", ""
    return (
        str(interrupt_data.get(_STREAM_THINKING_KEY) or ""),
        str(interrupt_data.get(_STREAM_CONTENT_KEY) or ""),
    )


def approval_tool_label(step_base: int, idx: int, tool_name: str) -> str:
    """Cumulative ``Tool N`` label across a multi-turn approval sequence.

    Tools surface one interrupt at a time, so a per-interrupt ``idx`` alone keeps
    restarting at "Tool 1". ``step_base`` carries the count already resolved this
    turn so the numbering reflects real progress (Tool 1, Tool 2, ...).
    """
    try:
        base = max(0, int(step_base))
    except (TypeError, ValueError):
        base = 0
    return f"Tool {base + idx + 1}: `{tool_name}`"
