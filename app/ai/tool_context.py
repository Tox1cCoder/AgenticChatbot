"""
Tool Execution Context - Provides context variables for tool execution.

The context is set by the graph during tool execution and includes:
- conversation_id: The current conversation
- user_id: The user making the request
- agent_key: The agent processing the request (e.g., "chat", "rag", "search")
"""

import logging
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from .tool_scope import ToolScope, resolve_tool_scope

logger = logging.getLogger(__name__)


def rich_response_capable_from_context(context: Any) -> bool:
    """Read the per-request inline rich-response capability off graph context.

    Kept here rather than in the graph so every ``tool_execution_context``
    call site reads the flag the same way, and so this module stays free of
    any graph-state import.
    """

    return bool(isinstance(context, Mapping) and context.get("inline_rich_response_v1"))


@dataclass(frozen=True)
class ToolContext:
    """
    Immutable context available during tool execution.

    Attributes:
        conversation_id: The ID of the current conversation (may be None)
        user_id: The ID of the user making the request (may be None)
        agent_key: The key of the agent (e.g., "chat", "rag", "search")
        rich_response_capable: Whether this request advertised the inline
            rich-response capability. Tools that can only produce rich items
            use it to skip work whose output would be discarded downstream.
            Defaults to True so a caller that never sets it behaves exactly as
            it did before the field existed — this is a waste-avoidance hint,
            not the correctness gate, which lives in the graph.
    """

    conversation_id: str | None = None
    user_id: str | None = None
    agent_key: str | None = None
    device_id: str | None = None
    tool_scope: str | None = None
    rich_response_capable: bool = True

    def __bool__(self) -> bool:
        """Return True if any context field is set."""
        return bool(
            self.conversation_id
            or self.user_id
            or self.agent_key
            or self.device_id
            or self.tool_scope
        )


# Context variable for the current tool execution context
_tool_context: ContextVar[ToolContext | None] = ContextVar(
    "tool_context",
    default=None,
)


def get_tool_context() -> ToolContext:
    """
    Get the current tool execution context.

    Returns:
        ToolContext with current execution context, or empty ToolContext if not set.

    Note:
        This is safe to call from any async task. The context is automatically
        propagated to child tasks created within the same context manager.
    """
    ctx = _tool_context.get()
    if ctx is None:
        return ToolContext()
    return ctx


def set_tool_context(ctx: ToolContext) -> None:
    """
    Set the tool execution context.

    This is primarily for testing. In production, use tool_execution_context().

    Args:
        ctx: The ToolContext to set
    """
    _tool_context.set(ctx)


def clear_tool_context() -> None:
    """
    Clear the tool execution context.

    This is primarily for testing to ensure clean state between tests.
    """
    _tool_context.set(None)


@contextmanager
def tool_execution_context(
    conversation_id: str | None = None,
    user_id: str | None = None,
    agent_key: str | None = None,
    device_id: str | None = None,
    tool_scope: str | ToolScope | None = None,
    rich_response_capable: bool = True,
):
    """
    Context manager that sets tool execution context for the duration of a block.

    This should be used in the graph's tool node to set context before executing
    tool calls. The context is automatically available to all tools executed
    within the block.

    Args:
        conversation_id: The current conversation ID
        user_id: The current user ID
        agent_key: The agent key (e.g., "chat", "rag", "search")

    Yields:
        The ToolContext that was set

    Example:
        with tool_execution_context(conv_id, user_id, "rag"):
            results = await execute_tool_calls(tool_calls, tool_map)
    """
    ctx = ToolContext(
        conversation_id=conversation_id,
        user_id=user_id,
        agent_key=agent_key,
        device_id=device_id,
        tool_scope=resolve_tool_scope(device_id=device_id, tool_scope=tool_scope).value,
        rich_response_capable=bool(rich_response_capable),
    )

    # Save previous context (for nested contexts, though unlikely)
    previous = _tool_context.get()

    try:
        _tool_context.set(ctx)
        yield ctx
    finally:
        # Restore previous context
        _tool_context.set(previous)
