import logging
from typing import Optional

from app.core.config import settings
from app.utils.text_processing import estimate_tokens, truncate_text

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """You are a helpful AI assistant.
You provide thoughtful, accurate, and friendly responses.
You engage in natural conversations while being informative and respectful."""

RAG_SYSTEM_PROMPT = """You are a document-based question answering assistant.
Your role is to provide accurate answers based on the retrieved document context.
Use the full context provided. When answering, provide detailed information from the documents, not just summaries.
If the user asks for more details, reference specific sections from the documents.
Always cite your sources and indicate when information is not available."""

ROUTER_SYSTEM_PROMPT = """You are a routing assistant that decides which agent should handle a user's message.

Available agents:
- chat_agent: Handles general conversation, casual chat, greetings, small talk, personal questions, opinions, creative tasks, and general assistance that doesn't require specific document retrieval.
- rag_agent: Handles questions that require searching through documents, retrieving specific information from a knowledge base, answering factual questions that need reference materials, or looking up detailed information from uploaded files.

Guidelines:
- Use chat_agent for: greetings, opinions, creative requests, general knowledge, casual conversation
- Use rag_agent for: "search", "find", "lookup", "what does the document say", "explain from the files", specific factual queries about uploaded content

Analyze the user's message and respond with ONLY the agent name (chat_agent or rag_agent) that should handle it.
Do not include any explanation, just the agent name."""


def _select_history_for_prompt(
    conversation_history: list,
    max_messages: Optional[int],
    max_tokens: Optional[int],
) -> list:
    if not conversation_history:
        return []

    selected = []
    total_tokens = 0

    for message in reversed(conversation_history):
        if max_messages and len(selected) >= max_messages:
            break

        message_tokens = estimate_tokens(message.content) + 4

        if max_tokens and total_tokens + message_tokens > max_tokens:
            if not selected:
                selected.append(message)
            break

        selected.append(message)
        total_tokens += message_tokens

    selected.reverse()
    return selected


def build_chat_prompt(user_message: str, conversation_history: list) -> str:
    parts = [CHAT_SYSTEM_PROMPT]

    if conversation_history:
        max_messages = (
            settings.chat_history_max_messages
            if settings.chat_history_max_messages > 0
            else None
        )
        max_tokens = (
            settings.chat_history_max_tokens
            if settings.chat_history_max_tokens > 0
            else None
        )
        selected_history = _select_history_for_prompt(
            conversation_history, max_messages, max_tokens
        )

        if selected_history:
            parts.append("\n\nPrevious conversation:")
            for msg in selected_history:
                role = "User" if msg.role.value == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")

    parts.append(f"\n\nUser: {user_message}")
    parts.append("Assistant:")

    return "\n".join(parts)


def build_rag_prompt(
    query: str, retrieved_docs: list, conversation_history: list
) -> str:
    parts = [RAG_SYSTEM_PROMPT]

    if retrieved_docs:
        parts.append("\n\nRelevant documents:")

        max_chunks = (
            settings.rag_chunks_in_prompt
            if settings.rag_chunks_in_prompt > 0
            else len(retrieved_docs)
        )
        max_chunks = min(max_chunks, len(retrieved_docs))

        total_tokens = 0
        chunks_used = 0

        for i, doc in enumerate(retrieved_docs[:max_chunks], 1):
            source = doc.get("source", "unknown")
            content = doc.get("content", "")
            chunk_index = doc.get("chunk_index", "unknown")
            score = doc.get("score", 0.0)

            if doc.get("page_start") and doc.get("page_end"):
                if doc["page_start"] != doc["page_end"]:
                    page_info = f"pages {doc['page_start']}-{doc['page_end']}"
                else:
                    page_info = f"page {doc['page_start']}"
            elif doc.get("page_number"):
                page_info = f"page {doc['page_number']}"
            else:
                page_info = "unknown page"

            # Estimate tokens for this chunk
            chunk_tokens = estimate_tokens(content)

            if (
                total_tokens + chunk_tokens > settings.rag_max_context_tokens
                and chunks_used >= 3
            ):
                break

            if (
                total_tokens + chunk_tokens > settings.rag_max_context_tokens
                and chunks_used < 3
            ):
                remaining_tokens = settings.rag_max_context_tokens - total_tokens
                max_chars = remaining_tokens * 4
                original_length = len(content)
                content = truncate_text(content, max_chars, add_ellipsis=True)
                chunk_tokens = estimate_tokens(content)
                logger.debug(
                    f"Truncated chunk {i} from {original_length} to {len(content)} chars to fit token limit"
                )

            if (
                settings.max_chunk_chars_in_prompt > 0
                and len(content) > settings.max_chunk_chars_in_prompt
            ):
                content = truncate_text(
                    content, settings.max_chunk_chars_in_prompt, add_ellipsis=True
                )

            parts.append(
                f"\nDocument {i} of {len(retrieved_docs)} ({source}, {page_info}, chunk {chunk_index}, score: {score:.2f}):"
            )
            parts.append(f"{content}")
            parts.append("---")

            total_tokens += chunk_tokens
            chunks_used += 1

        # Add context summary
        parts.append(
            f"\n[Using {chunks_used} chunks, ~{total_tokens} tokens from {len(retrieved_docs)} retrieved documents]"
        )

    if conversation_history:
        max_messages = (
            settings.rag_history_max_messages
            if settings.rag_history_max_messages > 0
            else None
        )
        max_tokens = (
            settings.rag_history_max_tokens
            if settings.rag_history_max_tokens > 0
            else None
        )
        selected_history = _select_history_for_prompt(
            conversation_history, max_messages, max_tokens
        )

        if selected_history:
            parts.append("\n\nConversation context:")
            for msg in selected_history:
                role = "User" if msg.role.value == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")

    parts.append(f"\n\nQuestion: {query}")
    parts.append("Answer based on the documents above:")

    return "\n".join(parts)
