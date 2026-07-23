"""Tool execution phase/state inference.

Shared, dependency-free helpers used to classify a tool's lifecycle phase and
whether a completed tool result represents an error. Kept independent of the
legacy stream-coercion bridge so they survive its removal.
"""

from __future__ import annotations

from typing import Any

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
