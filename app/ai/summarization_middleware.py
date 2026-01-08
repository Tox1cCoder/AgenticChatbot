import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.config import settings
from .agent_config import get_api_key

logger = logging.getLogger(__name__)


@dataclass
class SummarizationConfig:
    """Configuration for summarization behavior."""

    # Trigger thresholds (ANY condition triggers summarization)
    trigger_tokens: int = 20000  # Trigger when estimated tokens exceed this
    trigger_messages: int = 50  # OR when message count exceeds this
    trigger_fraction: float = 0.8  # OR when context usage exceeds this fraction

    # What to keep after summarization
    keep_messages: int = 20  # Keep the last N messages (most recent context)

    # Model context window size (for fraction calculation)
    model_context_size: int = 1000000  # Gemini 3 has 1M context

    # Summarization model (use cheaper/faster model)
    model: str = "gemini-3-flash-preview"
    temperature: float = 0.7


# Default summarization prompt
SUMMARIZATION_PROMPT = """You are a conversation summarizer. Create a concise summary of this conversation that preserves:
- Key facts, names, and decisions made
- Important context for continuing the conversation
- Tool calls and their results (summarized)

Conversation:
{messages}

Provide a clear, structured summary in bullet points:"""


def _get_config() -> SummarizationConfig:
    """Get summarization config from settings."""
    return SummarizationConfig(
        trigger_tokens=getattr(settings, "summarization_trigger_tokens", 20000),
        trigger_messages=getattr(settings, "summarization_trigger_messages", 50),
        trigger_fraction=getattr(settings, "summarization_trigger_fraction", 0.8),
        keep_messages=getattr(settings, "summarization_keep_messages", 20),
        model_context_size=getattr(settings, "summarization_model_context_size", 1000000),
        model=getattr(settings, "summarization_model", "gemini-3-flash-preview"),
        temperature=0.7,
    )


def _estimate_tokens(messages: List[BaseMessage]) -> int:
    """
    Estimate token count for a list of messages.
    Uses a simple heuristic: ~4 characters per token for English text.
    """
    total_chars = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total_chars += len(content)
    return total_chars // 4


def should_summarize(
    messages: List[BaseMessage],
    config: Optional[SummarizationConfig] = None,
    already_summarized: bool = False,
) -> bool:
    """
    Check if summarization should be triggered.

    Returns False if:
    - Summarization is disabled
    - Already summarized in this request
    - Below all thresholds

    Returns True if ANY threshold is exceeded.
    """
    if not getattr(settings, "enable_summarization", True):
        return False

    # Don't re-summarize if already done in this request
    if already_summarized:
        return False

    if config is None:
        config = _get_config()

    # Filter out system messages for threshold checks
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]

    # Don't summarize if we have fewer messages than we'd keep
    if len(non_system_messages) <= config.keep_messages:
        return False

    # Check token threshold
    estimated_tokens = _estimate_tokens(non_system_messages)
    if estimated_tokens >= config.trigger_tokens:
        logger.info(
            f"Summarization triggered: {estimated_tokens} tokens >= {config.trigger_tokens}"
        )
        return True

    # Check message count threshold
    if len(non_system_messages) >= config.trigger_messages:
        logger.info(
            f"Summarization triggered: {len(non_system_messages)} messages >= {config.trigger_messages}"
        )
        return True

    # Check fraction threshold
    fraction_threshold = int(config.model_context_size * config.trigger_fraction)
    if estimated_tokens >= fraction_threshold:
        logger.info(
            f"Summarization triggered: {estimated_tokens} tokens >= {config.trigger_fraction * 100}% of context"
        )
        return True

    return False


def _format_messages_for_summary(messages: List[BaseMessage]) -> str:
    """Format messages into a string for the summarization prompt."""
    formatted_parts = []

    for msg in messages:
        role = msg.__class__.__name__.replace("Message", "")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Handle tool calls in AI messages
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_info = ", ".join(
                f"{tc.get('name', 'unknown')}" for tc in msg.tool_calls
            )
            content = f"{content} [Called tools: {tool_info}]"

        # Truncate very long messages
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"

        formatted_parts.append(f"**{role}**: {content}")

    return "\n\n".join(formatted_parts)


def _get_summarization_model(config: SummarizationConfig) -> ChatGoogleGenerativeAI:
    """Create a LangChain model for summarization."""
    return ChatGoogleGenerativeAI(
        model=config.model,
        google_api_key=get_api_key(),
        temperature=config.temperature,
    )


async def generate_summary(
    messages_to_summarize: List[BaseMessage],
    config: Optional[SummarizationConfig] = None,
) -> str:
    """Generate a summary of the given messages."""
    if config is None:
        config = _get_config()

    try:
        model = _get_summarization_model(config)

        formatted_messages = _format_messages_for_summary(messages_to_summarize)
        prompt = SUMMARIZATION_PROMPT.format(messages=formatted_messages)

        response = await model.ainvoke([HumanMessage(content=prompt)])

        summary = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        logger.info(
            f"Generated summary for {len(messages_to_summarize)} messages "
            f"({len(summary)} chars)"
        )
        return summary

    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        # Return a simple fallback
        return f"[Previous conversation with {len(messages_to_summarize)} messages]"


def apply_summarization_to_state(
    state: Dict[str, Any],
    summary: str,
    config: Optional[SummarizationConfig] = None,
) -> Dict[str, Any]:
    """
    Apply summarization by REPLACING old messages in state.

    This is the key difference from the old approach:
    - Old: Prepend summary as new message (bloats context over iterations)
    - New: Replace old messages with summary (clean context)
    """
    if config is None:
        config = _get_config()

    messages = state.get("messages", [])

    # Separate system messages (keep all)
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    # Keep only recent messages
    messages_to_keep = non_system[-config.keep_messages:]

    # Create summary message as SystemMessage
    # Using SystemMessage because:
    # 1. It's context, not user input
    # 2. Won't be confused as part of the conversation flow
    # 3. Semantically correct
    summary_message = SystemMessage(
        content=f"[Summary of previous conversation]\n{summary}\n[End of summary]"
    )

    # REPLACE state messages (not append)
    # Order: System prompts -> Summary -> Recent messages
    state["messages"] = system_messages + [summary_message] + messages_to_keep

    # Mark that summarization happened in context
    context = state.get("context", {})
    context["conversation_summarized"] = True
    context["summary_text"] = summary
    context["messages_summarized_count"] = len(non_system) - len(messages_to_keep)
    state["context"] = context

    logger.info(
        f"Applied summarization: replaced {len(non_system) - len(messages_to_keep)} "
        f"messages with summary, keeping {len(messages_to_keep)} recent messages"
    )

    return state


async def summarize_if_needed_for_state(
    state: Dict[str, Any],
    config: Optional[SummarizationConfig] = None,
) -> Dict[str, Any]:
    """
    Main entry point for graph-level summarization.

    This function should be called from a graph node at the START of each request.
    It checks if summarization is needed and applies it by mutating state.

    Args:
        state: The LangGraph state dictionary
        config: Optional configuration override

    Returns:
        Modified state (may have messages replaced with summary)
    """
    if config is None:
        config = _get_config()

    context = state.get("context", {})
    already_summarized = context.get("conversation_summarized", False)

    messages = state.get("messages", [])

    # Check if summarization is needed
    if not should_summarize(messages, config, already_summarized):
        return state

    # Separate messages
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    # Calculate split point
    split_point = len(non_system) - config.keep_messages
    messages_to_summarize = non_system[:split_point]

    if not messages_to_summarize:
        return state

    # Generate summary
    summary = await generate_summary(messages_to_summarize, config)

    # Apply to state (REPLACE, not append)
    return apply_summarization_to_state(state, summary, config)


# =============================================================================
# DEPRECATED: Old per-call middleware (kept for backward compatibility)
# =============================================================================

async def summarize_messages_if_needed(
    messages: List[BaseMessage],
    config: Optional[SummarizationConfig] = None,
    conversation_id: Optional[str] = None,
) -> List[BaseMessage]:
    """
    DEPRECATED: Use summarize_if_needed_for_state() instead.

    This function is kept for backward compatibility but should not be used
    in new code. It operates at the wrong abstraction level (per-LLM-call)
    which causes context bloat during ReAct loops.

    The new approach uses graph-level summarization via the summarize node.
    """
    logger.warning(
        "summarize_messages_if_needed() is deprecated. "
        "Use graph-level summarization via summarize_if_needed_for_state() instead."
    )

    # For backward compatibility, just return messages unchanged
    # The new graph-level summarization handles this properly
    return messages
