"""
Token Instrumentation Module.

Provides utilities for estimating and logging token usage across the AI pipeline.
This helps identify prompt bloat and optimize token budgets.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
)

from app.ai.request_budget import _atomic_history_groups
from app.ai.token_counter import TokenCounter

logger = logging.getLogger(__name__)


_TOKEN_COUNTER = TokenCounter()

# Per-image prompt cost used when trimming history to a token budget. Vision
# providers price an image far above the few tokens of its text reference
# (Anthropic ~= (w*h)/750 up to ~1600; OpenAI high-detail ~= 765+). A single
# conservative constant keeps image-bearing turns from silently evading the
# trim guardrail without needing image dimensions here. This is a trimming
# estimate only -- not an app-level cap and not the provider's real bill.
IMAGE_ATTACHMENT_TOKEN_ESTIMATE = 1_200


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
    actual_input_tokens: int | None = None
    actual_output_tokens: int | None = None
    actual_total_tokens: int | None = None
    actual_reasoning_tokens: int | None = None

    # Metadata
    history_message_count: int = 0
    tool_message_count: int = 0
    bound_tool_count: int = 0
    bound_tool_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
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
                "total_tokens": self.actual_total_tokens,
                "reasoning_tokens": self.actual_reasoning_tokens,
            },
            "counts": {
                "history_messages": self.history_message_count,
                "tool_messages": self.tool_message_count,
                "bound_tools": self.bound_tool_count,
            },
            "bound_tool_names": self.bound_tool_names,
        }


def _count_text_tokens(text: str) -> int:
    return _TOKEN_COUNTER.count_text(
        provider="gemini",
        model="gemini-2.5-flash",
        text=text,
    ).tokens


def estimate_message_tokens(message: BaseMessage) -> int:
    """Estimate tokens for a LangChain message."""
    content = message.content if isinstance(message.content, str) else str(message.content)
    tokens = _count_text_tokens(content)

    # Add overhead for message structure
    tokens += 4  # role + formatting overhead

    # Add tool call tokens if present
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            tokens += _count_text_tokens(tc.get("name", ""))
            args = tc.get("args", {})
            if isinstance(args, dict):
                tokens += _count_text_tokens(str(args))
            tokens += 10  # tool call structure overhead

    # Tool result messages carry identity metadata alongside the visible body.
    if isinstance(message, ToolMessage):
        tool_name = getattr(message, "name", "") or ""
        tool_call_id = getattr(message, "tool_call_id", "") or ""
        if tool_name or tool_call_id:
            tokens += _count_text_tokens(str(tool_name))
            tokens += _count_text_tokens(str(tool_call_id))
            tokens += 8  # tool result envelope overhead

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

    tokens = _count_text_tokens(content) + 4  # content + role overhead
    attachments = getattr(message, "attachments", None)
    if isinstance(attachments, list):
        tokens += IMAGE_ATTACHMENT_TOKEN_ESTIMATE * len(attachments)
    return tokens


def estimate_tool_schema_tokens(tools: list[Any]) -> int:
    """
    Estimate tokens for tool schemas that get sent to the model.

    Tool schemas include: name, description, and parameter definitions.
    """
    total_tokens = 0

    for tool in tools:
        # Tool name
        name = getattr(tool, "name", "") or ""
        total_tokens += _count_text_tokens(name)

        # Tool description
        description = getattr(tool, "description", "") or ""
        total_tokens += _count_text_tokens(description)

        # Args schema
        args_schema = getattr(tool, "args_schema", None)
        if args_schema:
            if isinstance(args_schema, dict):
                total_tokens += _count_text_tokens(str(args_schema))
            elif hasattr(args_schema, "model_json_schema"):
                try:
                    schema_dict = args_schema.model_json_schema()
                    total_tokens += _count_text_tokens(str(schema_dict))
                except Exception:
                    total_tokens += 50  # Default estimate for schema
            else:
                total_tokens += 50  # Default estimate

        # Overhead per tool
        total_tokens += 20

    return total_tokens


def compute_token_breakdown(
    system_prompt: str,
    history_messages: list[BaseMessage],
    current_turn_messages: list[BaseMessage],
    tools: list[Any] | None = None,
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
    breakdown.system_prompt_tokens = _count_text_tokens(system_prompt)

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


def extract_actual_usage(response: Any) -> dict[str, int | None]:
    """Compatibility adapter over the canonical provider-usage extractor."""
    usage = _TOKEN_COUNTER.extract_reported_usage(provider="unknown", response=response)
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
        }
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
    }


def estimate_output_tokens(text: str, *, provider: str, model: str) -> int | None:
    """Estimate output tokens for a completion when the provider reports none.

    Used only to fill an absent output count so the context gauge can show a
    figure; the caller labels the resulting source ``mixed_reported_estimated``
    or ``locally_estimated`` and must never overwrite a provider-reported count.
    """
    if not text:
        return None
    return _TOKEN_COUNTER.count_text(provider=provider, model=model, text=text).tokens


def trim_history_to_budget(
    history: list[Any],
    max_messages: int = 0,
    max_tokens: int = 0,
) -> list[Any]:
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

    original_count = len(history)
    selected_groups = []
    selected_messages = 0
    selected_tokens = 0
    for group in reversed(_atomic_history_groups(history)):
        if not group.complete:
            continue
        group_message_count = len(group.messages)
        group_tokens = sum(estimate_agent_message_tokens(message) for message in group.messages)
        if max_messages > 0 and selected_messages + group_message_count > max_messages:
            break
        if max_tokens > 0 and selected_tokens + group_tokens > max_tokens:
            break
        selected_groups.append(group)
        selected_messages += group_message_count
        selected_tokens += group_tokens

    result = [message for group in reversed(selected_groups) for message in group.messages]
    if len(result) != original_count:
        logger.debug(
            "Trimmed history from %d to %d messages as complete groups "
            "(max_messages=%d, max_tokens=%d, ~%d tokens kept)",
            original_count,
            len(result),
            max_messages,
            max_tokens,
            selected_tokens,
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
