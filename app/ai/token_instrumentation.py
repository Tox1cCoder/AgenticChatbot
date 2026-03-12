"""
Token Instrumentation Module.

Provides utilities for estimating and logging token usage across the AI pipeline.
This helps identify prompt bloat and optimize token budgets.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)


# Approximate characters per token ratio (varies by model/content)
# Using conservative estimate - actual varies between 3-4 for English text
CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass
class TokenBudgetBreakdown:
    """Breakdown of estimated token usage for a request."""

    system_prompt_tokens: int = 0
    history_tokens: int = 0
    current_turn_tokens: int = 0
    tool_message_tokens: int = 0
    tool_schema_tokens: int = 0
    total_tokens: int = 0

    # Actual usage from provider (when available)
    actual_input_tokens: Optional[int] = None
    actual_output_tokens: Optional[int] = None

    # Metadata
    history_message_count: int = 0
    tool_message_count: int = 0
    bound_tool_count: int = 0
    bound_tool_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "estimated": {
                "system_prompt_tokens": self.system_prompt_tokens,
                "history_tokens": self.history_tokens,
                "current_turn_tokens": self.current_turn_tokens,
                "tool_message_tokens": self.tool_message_tokens,
                "tool_schema_tokens": self.tool_schema_tokens,
                "total_tokens": self.total_tokens,
            },
            "actual": {
                "input_tokens": self.actual_input_tokens,
                "output_tokens": self.actual_output_tokens,
            },
            "counts": {
                "history_messages": self.history_message_count,
                "tool_messages": self.tool_message_count,
                "bound_tools": self.bound_tool_count,
            },
            "bound_tool_names": self.bound_tool_names,
        }


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.

    Uses a simple character-based heuristic. For more accuracy,
    you would use the model's actual tokenizer.
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def estimate_message_tokens(message: BaseMessage) -> int:
    """Estimate tokens for a LangChain message."""
    content = (
        message.content if isinstance(message.content, str) else str(message.content)
    )
    tokens = estimate_tokens(content)

    # Add overhead for message structure
    tokens += 4  # role + formatting overhead

    # Add tool call tokens if present
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            tokens += estimate_tokens(tc.get("name", ""))
            args = tc.get("args", {})
            if isinstance(args, dict):
                tokens += estimate_tokens(str(args))
            tokens += 10  # tool call structure overhead

    return tokens


def estimate_agent_message_tokens(message: Any) -> int:
    """
    Estimate tokens for an AgentMessage (or similar object with role/content).

    Used for trimming conversation history before converting to LangChain format.
    """
    if not message:
        return 0

    content = ""
    if hasattr(message, "content"):
        content = message.content or ""
    elif isinstance(message, dict):
        content = message.get("content", "")

    return estimate_tokens(content) + 4  # content + role overhead


def estimate_tool_schema_tokens(tools: List[Any]) -> int:
    """
    Estimate tokens for tool schemas that get sent to the model.

    Tool schemas include: name, description, and parameter definitions.
    """
    total_tokens = 0

    for tool in tools:
        # Tool name
        name = getattr(tool, "name", "") or ""
        total_tokens += estimate_tokens(name)

        # Tool description
        description = getattr(tool, "description", "") or ""
        total_tokens += estimate_tokens(description)

        # Args schema
        args_schema = getattr(tool, "args_schema", None)
        if args_schema:
            if isinstance(args_schema, dict):
                total_tokens += estimate_tokens(str(args_schema))
            elif hasattr(args_schema, "model_json_schema"):
                try:
                    schema_dict = args_schema.model_json_schema()
                    total_tokens += estimate_tokens(str(schema_dict))
                except Exception:
                    total_tokens += 50  # Default estimate for schema
            else:
                total_tokens += 50  # Default estimate

        # Overhead per tool
        total_tokens += 20

    return total_tokens


def compute_token_breakdown(
    system_prompt: str,
    history_messages: List[BaseMessage],
    current_turn_messages: List[BaseMessage],
    tools: Optional[List[Any]] = None,
) -> TokenBudgetBreakdown:
    """
    Compute a full token breakdown for a model request.

    Args:
        system_prompt: The system prompt text
        history_messages: Conversation history messages
        current_turn_messages: Messages for the current turn
        tools: List of tools to be bound to the model

    Returns:
        TokenBudgetBreakdown with estimated token counts
    """
    breakdown = TokenBudgetBreakdown()

    # System prompt
    breakdown.system_prompt_tokens = estimate_tokens(system_prompt)

    # History messages
    breakdown.history_message_count = len(history_messages)
    for msg in history_messages:
        tokens = estimate_message_tokens(msg)
        breakdown.history_tokens += tokens
        if isinstance(msg, ToolMessage):
            breakdown.tool_message_count += 1
            breakdown.tool_message_tokens += tokens

    # Current turn messages
    for msg in current_turn_messages:
        tokens = estimate_message_tokens(msg)
        breakdown.current_turn_tokens += tokens
        if isinstance(msg, ToolMessage):
            breakdown.tool_message_count += 1
            breakdown.tool_message_tokens += tokens

    # Tool schemas
    if tools:
        breakdown.bound_tool_count = len(tools)
        breakdown.bound_tool_names = [getattr(t, "name", "unknown") for t in tools]
        breakdown.tool_schema_tokens = estimate_tool_schema_tokens(tools)

    # Total
    breakdown.total_tokens = (
        breakdown.system_prompt_tokens
        + breakdown.history_tokens
        + breakdown.current_turn_tokens
        + breakdown.tool_schema_tokens
    )

    return breakdown


def extract_actual_usage(response: Any) -> Dict[str, Optional[int]]:
    """
    Extract actual token usage from model response if available.

    Works with OpenAI responses that include usage_metadata.
    """
    result = {"input_tokens": None, "output_tokens": None}

    if response is None:
        return result

    # Try response_metadata (LangChain convention)
    if hasattr(response, "response_metadata"):
        metadata = response.response_metadata
        if isinstance(metadata, dict):
            # OpenAI format
            if "usage" in metadata:
                usage = metadata["usage"]
                result["input_tokens"] = usage.get("prompt_tokens")
                result["output_tokens"] = usage.get("completion_tokens")
            # Token usage directly in metadata
            if "token_usage" in metadata:
                usage = metadata["token_usage"]
                result["input_tokens"] = usage.get("prompt_tokens")
                result["output_tokens"] = usage.get("completion_tokens")

    # Try usage_metadata (some providers)
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        if hasattr(usage, "input_tokens"):
            result["input_tokens"] = usage.input_tokens
        if hasattr(usage, "output_tokens"):
            result["output_tokens"] = usage.output_tokens

    return result


def trim_history_to_budget(
    history: List[Any],
    max_messages: int = 0,
    max_tokens: int = 0,
) -> List[Any]:
    """
    Trim conversation history to fit within configured budgets.

    Trims from the beginning (oldest messages first) to preserve
    recent context which is typically more relevant.

    Args:
        history: List of messages (AgentMessage or similar objects)
        max_messages: Maximum number of messages to keep (0 = no limit)
        max_tokens: Maximum estimated tokens to keep (0 = no limit)

    Returns:
        Trimmed list of messages (may be same list if no trimming needed)
    """
    if not history:
        return history

    # No limits configured - return as-is
    if max_messages <= 0 and max_tokens <= 0:
        return history

    result = history
    original_count = len(history)

    # Apply message count limit first (simple trim)
    if max_messages > 0 and len(result) > max_messages:
        result = result[-max_messages:]
        logger.debug(
            "Trimmed history from %d to %d messages (max_messages=%d)",
            original_count,
            len(result),
            max_messages,
        )

    # Apply token budget limit if configured
    if max_tokens > 0:
        # Calculate tokens from most recent (end) to oldest (start)
        # Keep messages until we exceed budget
        total_tokens = 0
        keep_from_idx = 0

        for i in range(len(result) - 1, -1, -1):
            msg_tokens = estimate_agent_message_tokens(result[i])
            if total_tokens + msg_tokens > max_tokens:
                keep_from_idx = i + 1
                break
            total_tokens += msg_tokens

        if keep_from_idx > 0:
            trimmed_count = len(result)
            result = result[keep_from_idx:]
            logger.debug(
                "Trimmed history from %d to %d messages (token budget %d, ~%d tokens kept)",
                trimmed_count,
                len(result),
                max_tokens,
                total_tokens,
            )

    return result


@dataclass
class HistoryBudgetConfig:
    """Configuration for history budget limits by agent type."""

    max_messages: int = 0
    max_tokens: int = 0

    @classmethod
    def for_agent(cls, agent_key: str, settings: Any) -> "HistoryBudgetConfig":
        """
        Get history budget config for a specific agent type.

        Looks up agent-specific settings (e.g., chat_history_max_messages)
        falling back to general settings.
        """
        max_messages = 0
        max_tokens = 0

        # Try agent-specific settings first
        agent_messages_key = f"{agent_key}_history_max_messages"
        agent_tokens_key = f"{agent_key}_history_max_tokens"

        if hasattr(settings, agent_messages_key):
            max_messages = getattr(settings, agent_messages_key, 0) or 0
        elif hasattr(settings, "chat_history_max_messages"):
            max_messages = settings.chat_history_max_messages or 0

        if hasattr(settings, agent_tokens_key):
            max_tokens = getattr(settings, agent_tokens_key, 0) or 0
        elif hasattr(settings, "chat_history_max_tokens"):
            max_tokens = settings.chat_history_max_tokens or 0

        return cls(max_messages=max_messages, max_tokens=max_tokens)


def truncate_tool_result(
    content: str,
    max_chars: int = 0,
    truncation_suffix: str = "\n\n[Output truncated - full result available in tool artifacts]",
) -> tuple[str, bool]:
    """
    Truncate tool result content to fit within character budget.

    Args:
        content: The full tool result content
        max_chars: Maximum characters to keep (0 = no limit)
        truncation_suffix: Text to append when truncating

    Returns:
        Tuple of (possibly truncated content, was_truncated)
    """
    if not content or max_chars <= 0:
        return content, False

    if len(content) <= max_chars:
        return content, False

    # Truncate with room for suffix
    suffix_len = len(truncation_suffix)
    truncate_at = max(0, max_chars - suffix_len)

    # Try to truncate at a natural boundary (newline or space)
    truncated = content[:truncate_at]

    # Look for a good break point in the last 200 chars
    last_newline = truncated.rfind("\n", max(0, truncate_at - 200))
    if last_newline > truncate_at - 200:
        truncated = truncated[:last_newline]

    return truncated + truncation_suffix, True
