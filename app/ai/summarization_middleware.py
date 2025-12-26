"""
Summarization Middleware for Long Conversations.

This module provides conversation summarization to handle long conversations
that may exceed the LLM's context window. It only triggers when the conversation
exceeds configured token or message thresholds.

Usage:
    from app.ai.summarization_middleware import summarize_messages_if_needed

    # In agent's invoke_model_with_history:
    messages = await summarize_messages_if_needed(messages)
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.config import settings

from .agent_config import get_api_key


@dataclass
class SummarizationConfig:
    """Configuration for summarization behavior."""

    # Trigger thresholds
    trigger_tokens: int = 8000  # Trigger when estimated tokens exceed this
    trigger_messages: int = 20  # OR when message count exceeds this

    # What to keep after summarization
    keep_messages: int = 10  # Keep the last N messages (most recent context)

    # Summarization model
    model: str = "gemini-3-flash-preview"
    temperature: float = 1


# Default summarization prompt
SUMMARIZATION_PROMPT = """You are a conversation summarizer. Your task is to create a concise summary of the conversation history that preserves the key context, facts, and information exchanged.

Guidelines:
- Preserve important facts, names, preferences, and decisions made
- Maintain the chronological flow of the conversation  
- Keep tool calls and their results summarized (e.g., "User searched for X and found Y")
- Focus on information that would be relevant for continuing the conversation
- Use bullet points for clarity
- Be concise but comprehensive

Summarize the following conversation:

{messages}

Provide a clear, structured summary:"""


def _get_config() -> SummarizationConfig:
    """Get summarization config from settings."""
    return SummarizationConfig(
        trigger_tokens=getattr(settings, "summarization_trigger_tokens", 4000),
        trigger_messages=getattr(settings, "summarization_trigger_messages", 20),
        keep_messages=getattr(settings, "summarization_keep_messages", 10),
        model=getattr(settings, "summarization_model"),
        temperature=1,
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


def _should_summarize(
    messages: List[BaseMessage],
    config: SummarizationConfig,
) -> bool:
    """
    Check if summarization should be triggered.

    Returns True if EITHER:
    - Estimated tokens exceed trigger_tokens threshold
    - Message count exceeds trigger_messages threshold
    """
    if not getattr(settings, "enable_summarization", True):
        return False

    # Don't summarize if we have fewer messages than we'd keep
    if len(messages) <= config.keep_messages:
        return False

    # Check token threshold
    estimated_tokens = _estimate_tokens(messages)
    if estimated_tokens >= config.trigger_tokens:
        return True

    # Check message threshold
    if len(messages) >= config.trigger_messages:
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

        formatted_parts.append(f"**{role}**: {content}")

    return "\n\n".join(formatted_parts)


def _get_summarization_model(config: SummarizationConfig) -> ChatGoogleGenerativeAI:
    """Create a LangChain model for summarization."""

    return ChatGoogleGenerativeAI(
        model=config.model,
        google_api_key=get_api_key(),
        temperature=config.temperature,
    )


async def _generate_summary(
    messages_to_summarize: List[BaseMessage],
    config: SummarizationConfig,
) -> str:
    """Generate a summary of the given messages."""
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

        return summary

    except Exception as e:
        # Return a simple fallback
        return f"[Previous conversation with {len(messages_to_summarize)} messages]"


async def summarize_messages_if_needed(
    messages: List[BaseMessage],
    config: Optional[SummarizationConfig] = None,
) -> List[BaseMessage]:
    """
    Summarize older messages if the conversation exceeds thresholds.

    This function:
    1. Checks if summarization is needed based on token/message thresholds
    2. If needed, summarizes older messages while keeping recent ones
    3. Returns a new message list with the summary prepended

    Args:
        messages: List of LangChain messages (excluding system message)
        config: Optional configuration override

    Returns:
        Potentially modified list of messages with older context summarized
    """
    if config is None:
        config = _get_config()

    # Filter out system messages for summarization check
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]

    if not _should_summarize(non_system_messages, config):
        return messages

    # Split messages: older ones to summarize, recent ones to keep
    split_point = len(non_system_messages) - config.keep_messages
    messages_to_summarize = non_system_messages[:split_point]
    messages_to_keep = non_system_messages[split_point:]

    if not messages_to_summarize:
        return messages

    # Generate summary
    summary = await _generate_summary(messages_to_summarize, config)

    # Create a system-style message with the summary
    summary_message = SystemMessage(
        content=f"[Summary of previous conversation]\n{summary}\n[End of summary]"
    )

    # Reconstruct message list: system messages + summary + recent messages
    result = system_messages + [summary_message] + messages_to_keep

    return result
