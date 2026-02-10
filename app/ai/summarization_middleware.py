import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import RemoveMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.config import settings
from .agent_config import get_api_key
from .utils import coerce_response_text

logger = logging.getLogger(__name__)


@dataclass
class SummarizationConfig:
    """Configuration for summarization behavior."""

    trigger_tokens: int = 18000
    trigger_messages: int = 60
    trigger_fraction: float = 0.55

    keep_messages: int = 8  # Keep the last N messages (most recent context)

    # Model context window size (for fraction calculation)
    model_context_size: int = 128000

    # Summarization model
    model: str = "gemini-3-flash-preview"
    temperature: float = 0.2

    # Hard cap on the rolling summary (estimated tokens ≈ chars / 4).
    max_summary_tokens: int = 1500


# Default summarization prompt
SUMMARIZATION_PROMPT = """You are a conversation summarizer. Create a concise summary of this conversation that preserves:
- Key facts, names, and decisions made
- Important context for continuing the conversation
- Tool calls and their results (summarized)

Conversation:
{messages}

Provide a clear, structured summary in bullet points:"""

# Rolling / incremental summarization prompt (merges existing summary with new messages)
ROLLING_SUMMARIZATION_PROMPT = """You are a conversation summarizer. You have an existing summary and new messages to incorporate.

EXISTING SUMMARY:
{existing_summary}

NEW MESSAGES TO INCORPORATE:
{messages}

Merge the existing summary with the new information. Produce a single, concise, updated summary in bullet points that preserves:
- Key facts, names, and decisions made
- Important context for continuing the conversation
- Tool calls and their results (summarized)
- Drop details that are superseded or no longer relevant

Updated summary:"""


def _get_config() -> SummarizationConfig:
    """Get summarization config from settings."""
    return SummarizationConfig(
        trigger_tokens=getattr(settings, "summarization_trigger_tokens", 18000),
        trigger_messages=getattr(settings, "summarization_trigger_messages", 60),
        trigger_fraction=getattr(settings, "summarization_trigger_fraction", 0.55),
        keep_messages=getattr(settings, "summarization_keep_messages", 8),
        model_context_size=getattr(settings, "summarization_model_context_size", 128000),
        model=getattr(settings, "summarization_model", "gemini-3-flash-preview"),
        temperature=0.2,
        max_summary_tokens=getattr(settings, "summarization_max_summary_tokens", 1500),
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

    if already_summarized:
        return False

    if config is None:
        config = _get_config()

    # Filter out system messages for threshold checks
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_system_messages) <= config.keep_messages:
        return False

    # Check token threshold
    estimated_tokens = _estimate_tokens(non_system_messages)
    if estimated_tokens >= config.trigger_tokens:
        return True

    # Check message count threshold
    if len(non_system_messages) >= config.trigger_messages:
        return True

    # Check fraction threshold
    fraction_threshold = int(config.model_context_size * config.trigger_fraction)
    if estimated_tokens >= fraction_threshold:
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
    existing_summary: Optional[str] = None,
) -> str:
    """Generate a summary of the given messages, optionally merging with an existing summary."""
    if config is None:
        config = _get_config()

    try:
        model = _get_summarization_model(config)

        formatted_messages = _format_messages_for_summary(messages_to_summarize)

        if existing_summary:
            prompt = ROLLING_SUMMARIZATION_PROMPT.format(
                existing_summary=existing_summary,
                messages=formatted_messages,
            )
        else:
            prompt = SUMMARIZATION_PROMPT.format(messages=formatted_messages)

        response = await model.ainvoke([HumanMessage(content=prompt)])

        summary = coerce_response_text(response.content)

        # Hard-cap: truncate if the summary exceeds the configured budget.
        # A max_summary_tokens of 0 means unlimited (no truncation).
        max_chars = config.max_summary_tokens * 4  # rough token-to-char ratio
        if max_chars > 0 and len(summary) > max_chars:
            summary = summary[:max_chars].rsplit("\n", 1)[0] + "\n[...truncated]"
            logger.info(
                "Truncated summary from %d to %d chars (max_summary_tokens=%d)",
                len(coerce_response_text(response.content)),
                len(summary),
                config.max_summary_tokens,
            )

        logger.debug(
            "Generated %ssummary for %d messages (%d chars)",
            "rolling " if existing_summary else "",
            len(messages_to_summarize),
            len(summary),
        )
        return summary

    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        # Return a simple fallback
        return f"[Previous conversation with {len(messages_to_summarize)} messages]"


def apply_summarization_to_state(
    state: Dict[str, Any],
    summary: str,
    messages_to_remove: List[BaseMessage],
    config: Optional[SummarizationConfig] = None,
) -> Dict[str, Any]:
    """
    Apply summarization results to graph state using explicit RemoveMessage
    entries so the ``add_messages`` reducer correctly deletes covered messages
    from the checkpoint instead of list-overwrite (which doesn't work with the
    reducer).

    Updates the rolling ``history_summary`` state key instead of injecting a
    SystemMessage into the messages list — the summary is injected into agent
    prompts at invocation time (Phase 3).
    """
    if config is None:
        config = _get_config()

    # Build RemoveMessage entries for every message that was summarized.
    # The add_messages reducer will delete messages with these IDs.
    removals: List[RemoveMessage] = []
    for msg in messages_to_remove:
        msg_id = getattr(msg, "id", None)
        if msg_id:
            removals.append(RemoveMessage(id=msg_id))

    if removals:
        state.setdefault("messages", []).extend(removals)

    # Record the cursor: the ID of the last message that was summarized.
    # Future summarization runs can use this to avoid re-summarizing.
    last_summarized = messages_to_remove[-1] if messages_to_remove else None
    cursor_id = getattr(last_summarized, "id", None) if last_summarized else None

    # Update rolling summary state keys
    state["history_summary"] = summary
    state["history_summary_updated_at"] = datetime.now(timezone.utc).isoformat()
    if cursor_id:
        state["summary_cursor_message_id"] = str(cursor_id)

    # Legacy context flags (kept for backward compatibility)
    context = state.get("context", {})
    context["conversation_summarized"] = True
    context["summary_text"] = summary
    context["messages_summarized_count"] = len(messages_to_remove)
    state["context"] = context

    logger.info(
        "Applied summarization: removed %d messages, summary %d chars, cursor=%s",
        len(removals),
        len(summary),
        cursor_id,
    )

    return state


async def summarize_for_state(
    state: Dict[str, Any],
    config: Optional[SummarizationConfig] = None,
) -> Dict[str, Any]:
    """
    Main entry point for graph-level summarization.

    This function should be called from a graph node at the START of each request.
    It checks if summarization is needed and applies it using explicit
    ``RemoveMessage`` entries (compatible with the ``add_messages`` reducer).

    The rolling summary is stored in ``state["history_summary"]`` and merged
    incrementally on subsequent triggers.

    Args:
        state: The LangGraph state dictionary
        config: Optional configuration override

    Returns:
        Modified state (old messages removed, summary updated)
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

    # Calculate split point — keep the most recent `keep_messages`
    split_point = len(non_system) - config.keep_messages
    messages_to_summarize = non_system[:split_point]

    if not messages_to_summarize:
        return state

    # Get existing rolling summary for incremental merge
    existing_summary = state.get("history_summary")

    # Generate (or merge) summary
    summary = await generate_summary(
        messages_to_summarize, config, existing_summary=existing_summary
    )

    # Apply to state using RemoveMessage (not list replacement)
    return apply_summarization_to_state(state, summary, messages_to_summarize, config)