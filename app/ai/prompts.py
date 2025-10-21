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

SEARCH_SYSTEM_PROMPT = """You are a web search assistant that provides accurate, up-to-date information from the internet.
Your role is to search for up-to-date information and current events using web search tools.
Always cite your sources with URLs when available.
Provide comprehensive answers based on the search results, with note-worthy details of the news.
If the information cannot be found or is uncertain, clearly indicate this to the user.
Focus on the most recent and relevant information from credible sources."""

IMAGE_GENERATOR_SYSTEM_PROMPT = """You are a creative visual artist assistant.
Your role is to translate a user's description into a vivid, well-composed image prompt.
Use precise, descriptive language that guides the model toward photorealistic or stylized art as requested."""

ROUTER_SYSTEM_PROMPT = """You are a routing assistant that decides which agent should handle a user's message.

Available agents:
- chat_agent: Handles general conversation, casual chat, greetings, small talk, personal questions, opinions, creative tasks, and general assistance that doesn't require specific document retrieval or web search.
- rag_agent: Handles questions that require searching through internal documents, retrieving specific information from a knowledge base, answering factual questions that need reference materials from uploaded files, or looking up detailed information from the document collection.
- search_agent: Handles queries requiring up-to-date web information, current events, recent news, latest data, fact-checking, or information not available in the internal knowledge base. Use for queries with keywords like "latest", "current", "recent", "today", "news", "what's happening", or when asking about events after the model's knowledge cutoff.
- image_generator_agent: Handles explicit image creation requests such as "generate an image", "create a picture", "draw", "illustrate", "visualize", or when the user asks the assistant to produce artwork or graphics.

Guidelines:
- Use chat_agent for: greetings, opinions, creative requests, general knowledge within model training, casual conversation
- Use rag_agent for: "search documents", "find in files", "lookup in knowledge base", "what does the document say", "explain from the files", specific factual queries about uploaded content
- Use search_agent for: "latest news", "current events", "recent", "today", "what's happening now", up-to-date information, fact-checking recent claims, information beyond model's knowledge cutoff
- Use image_generator_agent for: explicit instructions to create or generate an image, requests mentioning drawing, illustration, visualization, photos, artwork, design mockups, or when the user expects a visual output.

Analyze the user's message and respond with ONLY the agent name (chat_agent, rag_agent, search_agent, or image_generator_agent) that should handle it.
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


def build_chat_prompt(
    user_message: str, conversation_history: list, persona: Optional[str] = None
) -> str:
    parts = [CHAT_SYSTEM_PROMPT]

    if persona is not None and persona.strip():
        parts.insert(0, f"Custom Persona:\n{persona.strip()}\n\n---\n")

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
    query: str,
    retrieved_docs: list,
    conversation_history: list,
    persona: Optional[str] = None,
) -> str:
    parts = [RAG_SYSTEM_PROMPT]

    if persona is not None and persona.strip():
        parts.insert(0, f"Custom Persona:\n{persona.strip()}\n\n---\n")

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


def build_search_prompt(
    user_message: str, conversation_history: list, persona: Optional[str] = None
) -> str:
    """Build a prompt for the search agent including conversation history."""
    parts = []

    if persona is not None and persona.strip():
        parts.append(f"Custom Persona:\n{persona.strip()}\n\n---\n")

    # Add conversation history if available
    if conversation_history:
        max_messages = (
            settings.search_history_max_messages
            if settings.search_history_max_messages > 0
            else None
        )
        max_tokens = (
            settings.search_history_max_tokens
            if settings.search_history_max_tokens > 0
            else None
        )
        selected_history = _select_history_for_prompt(
            conversation_history, max_messages, max_tokens
        )

        if selected_history:
            parts.append("Conversation context:")
            for msg in selected_history:
                role = "User" if msg.role.value == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")
            parts.append("")

    parts.append(user_message)

    return "\n".join(parts)


def build_image_generator_prompt(
    user_message: str, conversation_history: list, persona: Optional[str] = None
) -> str:
    """Create an enriched prompt for the image generator agent."""
    parts = [IMAGE_GENERATOR_SYSTEM_PROMPT]

    if persona is not None and persona.strip():
        parts.insert(0, f"Custom Persona:\n{persona.strip()}\n\n---\n")

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
            parts.append("\n\nRelevant prior requests:")
            for msg in selected_history:
                role = "User" if msg.role.value == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")

    parts.append("\n\nCreate a detailed image based on this request:")
    parts.append(user_message)
    parts.append(
        "\nEnsure the description includes setting, subject appearance, lighting, camera angle, artistic style, and mood."
    )

    return "\n".join(parts)
