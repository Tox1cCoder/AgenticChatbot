from __future__ import annotations

from typing import Any

CANONICAL_STREAM_EVENT_TYPES = (
    "agent_selected",
    "thinking",
    "token",
    "tool",
    "interrupt",
    "complete",
    "error",
    "continuation_start",
    "node_complete",
)

TOOL_PHASE_START = "start"
TOOL_PHASE_END = "end"

_ERROR_PREFIXES = (
    "error:",
    "error executing tool:",
    "tool not found:",
)


def normalize_tool_phase(value: Any) -> str | None:
    phase = str(value or "").strip().lower()
    if phase in {TOOL_PHASE_START, TOOL_PHASE_END}:
        return phase
    return None


def _looks_like_tool_error(value: Any) -> bool:
    if isinstance(value, dict):
        for key in ("error", "errors", "exception"):
            candidate = value.get(key)
            if candidate not in (None, "", [], {}):
                return True

    if isinstance(value, str):
        stripped = value.strip().lower()
        return any(stripped.startswith(prefix) for prefix in _ERROR_PREFIXES)

    return False


def infer_tool_state(
    *,
    phase: Any,
    result: Any = None,
) -> str:
    normalized_phase = normalize_tool_phase(phase)
    if normalized_phase == TOOL_PHASE_START:
        return "running"
    if normalized_phase != TOOL_PHASE_END:
        return "unknown"
    return "error" if _looks_like_tool_error(result) else "completed"


def build_canonical_tool_event(
    *,
    phase: Any,
    name: str | None,
    tool_call_id: Any,
    args: Any = None,
    result: Any = None,
    duration_ms: int | None = None,
    state: str | None = None,
    render: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_phase = normalize_tool_phase(phase) or "unknown"
    payload: dict[str, Any] = {
        "type": "tool",
        "name": name or "unknown",
        "status": normalized_phase,
        "phase": normalized_phase,
        "state": state or infer_tool_state(phase=normalized_phase, result=result),
        "tool_call_id": tool_call_id,
    }

    if args is not None:
        payload["args"] = args

    if result is not None:
        payload["result"] = result

    if duration_ms is not None:
        payload["duration_ms"] = duration_ms

    if render is not None:
        payload["render"] = render

    return payload
