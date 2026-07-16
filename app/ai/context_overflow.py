from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage

from app.ai.request_budget import _atomic_history_groups

_CONTEXT_ERROR_MARKERS = (
    "maximum context length",
    "context length exceeded",
    "context window",
    "input token limit",
    "token limit exceeded",
    "too many tokens",
)


def is_context_overflow_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _CONTEXT_ERROR_MARKERS)


def _compact_tool_results_for_retry(messages: list[Any], *, max_chars: int) -> list[Any]:
    compacted: list[Any] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            compacted.append(message)
            continue
        content = str(message.content or "")
        if len(content) <= max_chars:
            compacted.append(message)
            continue
        preview = content[:max_chars].rstrip()
        tool_name = getattr(message, "name", "") or "unknown"
        compacted.append(
            ToolMessage(
                content=(
                    "Tool output compacted for context retry "
                    f"({message.tool_call_id} {tool_name}):\n"
                    f"{preview}"
                ),
                tool_call_id=message.tool_call_id,
                name=getattr(message, "name", None),
            )
        )
    return compacted


def prepare_aggressive_context_retry(
    messages: list[Any],
    *,
    tool_preview_chars: int,
) -> list[Any]:
    """Drop oldest complete history turns and compact remaining tool results."""
    if len(messages) <= 2:
        return _compact_tool_results_for_retry(messages, max_chars=tool_preview_chars)
    system_prefix = [messages[0]] if _message_role(messages[0]) == "system" else []
    current_suffix = [messages[-1]] if _message_role(messages[-1]) == "user" else []
    start = len(system_prefix)
    end = len(messages) - len(current_suffix)
    history = messages[start:end]
    groups = _atomic_history_groups(history)
    complete_groups = [group for group in groups if group.complete]
    remove_count = max(1, len(complete_groups) // 2) if complete_groups else 0
    removed = 0
    retained_history: list[Any] = []
    for group in groups:
        if group.complete and removed < remove_count:
            removed += 1
            continue
        retained_history.extend(group.messages)
    reduced = [*system_prefix, *retained_history, *current_suffix]
    return _compact_tool_results_for_retry(reduced, max_chars=tool_preview_chars)


async def invoke_with_context_overflow_retry(
    invoke,
    messages: list[Any],
    *,
    enabled: bool,
    tool_preview_chars: int,
):
    """Invoke once and perform at most one aggressive overflow retry."""
    try:
        return await invoke(messages)
    except Exception as exc:
        if not enabled or not is_context_overflow_error(exc):
            raise
    retry_messages = prepare_aggressive_context_retry(
        messages,
        tool_preview_chars=tool_preview_chars,
    )
    return await invoke(retry_messages)


def _message_role(message: Any) -> str:
    role = getattr(message, "type", None) or getattr(message, "role", None)
    aliases = {"human": "user", "ai": "assistant"}
    normalized = str(getattr(role, "value", role) or "").lower()
    return aliases.get(normalized, normalized)
