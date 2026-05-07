from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage


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


def compact_tool_messages_for_retry(messages: list[Any], *, max_chars: int) -> list[Any]:
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
                    f"Tool output compacted for context retry ({message.tool_call_id} {tool_name}):\n"
                    f"{preview}"
                ),
                tool_call_id=message.tool_call_id,
                name=getattr(message, "name", None),
            )
        )
    return compacted
