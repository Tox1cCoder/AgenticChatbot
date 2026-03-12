"""
Tool Execution Context - Provides context variables for tool execution.

The context is set by the graph during tool execution and includes:
- conversation_id: The current conversation
- user_id: The user making the request
- agent_key: The agent processing the request (e.g., "chat", "rag", "search")
"""

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolContext:
    """
    Immutable context available during tool execution.

    Attributes:
        conversation_id: The ID of the current conversation (may be None)
        user_id: The ID of the user making the request (may be None)
        agent_key: The key of the agent (e.g., "chat", "rag", "search")
    """

    conversation_id: str | None = None
    user_id: str | None = None
    agent_key: str | None = None

    def __bool__(self) -> bool:
        """Return True if any context field is set."""
        return bool(self.conversation_id or self.user_id or self.agent_key)


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
    )

    # Save previous context (for nested contexts, though unlikely)
    previous = _tool_context.get()

    try:
        _tool_context.set(ctx)
        yield ctx
    finally:
        # Restore previous context
        _tool_context.set(previous)
