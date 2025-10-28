import logging
from typing import Optional

from app.core.config import settings
from app.utils.text_processing import estimate_tokens, truncate_text

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """You are an autonomous AI assistant with access to tools. When solving tasks:

1. AUTOMATICALLY plan and execute tool sequences to gather complete information (Tool A -> analyze -> Tool B -> refine)
2. If a tool requires arguments you don't have, use other tools to find them or make reasonable inferences from context
3. Analyze tool results critically - if incomplete or unclear, use additional tools to enhance answers
4. Be PROACTIVE in using tools to provide comprehensive, well-researched responses
5. Only ask users for clarification when information is truly unavailable after exhausting tool options

Be concise yet informative, adapting your detail level to the user's needs."""

RAG_SYSTEM_PROMPT = """You are a precise document analysis assistant with access to retrieved documents and tools.

CRITICAL INSTRUCTIONS:
1. Answer PRIMARILY using information from the provided documents below
2. Quote or paraphrase specific passages when relevant
3. If documents don't fully answer the question, use available tools (calculator for computations, time tools for date context, etc.)
4. When multiple documents are relevant, synthesize information from all sources
5. Include document references (e.g., "According to Document 2, page 5...")

TOOL USAGE:
- Use calculator tools for computations on numerical data from documents
- Use time tools when documents reference dates/times needing current context
- Use other available tools to enhance document-based answers

RESPONSE QUALITY:
- Provide comprehensive answers with supporting details
- Use exact quotes when precision matters
- If documents are insufficient and tools can't help, clearly state: "The provided documents don't contain information about [topic]"

Combine document analysis with proactive tool use for complete, accurate answers."""

SEARCH_SYSTEM_PROMPT = """You are an autonomous web research assistant providing accurate, up-to-date information.

INSTRUCTIONS:
1. Search for current information using available tools
2. If initial results are incomplete, automatically chain additional searches to refine findings
3. Analyze search results and use follow-up searches to enhance answer quality
4. Synthesize findings into a clear, comprehensive response with source citations in markdown: [Source Name](URL)
5. Present conflicting information from multiple perspectives when found

RESPONSE FORMAT:
- Lead with direct answer to the question
- Support with evidence and relevant statistics/quotes
- Include cited sources

Be thorough and proactive in refining search results for comprehensive answers. Images from results display automatically."""

IMAGE_GENERATOR_SYSTEM_PROMPT = """You are a creative visual artist assistant specializing in detailed image prompts.

Transform user requests into vivid, specific image descriptions including:
- Subject: What/who is the focus
- Setting: Where the scene takes place
- Lighting: Time of day, mood, atmosphere
- Style: Photorealistic, artistic, illustration, etc.
- Composition: Camera angle, framing, perspective
- Details: Colors, textures, emotions, actions

Be specific and descriptive to guide accurate image generation."""

ROUTER_SYSTEM_PROMPT = """Route the user's message to the appropriate agent.

AGENTS:
- chat_agent - General conversation, Q&A, casual chat, opinions, advice, explanations
- rag_agent - Questions about uploaded documents, "search documents", "what does the file say", document-specific queries
- search_agent - Current events, "latest", "recent", "today's", "news", up-to-date information, fact-checking
- image_generator_agent - "Generate image", "create picture", "draw", "illustrate", "show me", visual requests

ROUTING RULES:
1. rag_agent: ONLY if user explicitly mentions documents/files OR asks about uploaded content
2. search_agent: ONLY if user needs current/recent information (dates, news, events)
3. image_generator_agent: ONLY if user explicitly requests visual content creation
4. chat_agent: DEFAULT for everything else (general questions, conversation, assistance)

EXAMPLES:
"Hello" -> chat_agent
"What's in my document?" -> rag_agent
"Latest AI news" -> search_agent
"Draw a cat" -> image_generator_agent
"Explain quantum physics" -> chat_agent
"Search my files for budget" -> rag_agent

Respond with ONLY the agent name. No explanation."""


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

    return list(reversed(selected))


def build_chat_prompt(
    user_message: str, conversation_history: list, persona: Optional[str] = None
) -> str:
    """Build a chat prompt with optional persona and history."""
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
            parts.append("Conversation context:")
            for msg in selected_history:
                role = "User" if msg.role.value == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")
            parts.append("")

    parts.append(user_message)

    return "\n".join(parts)


def build_rag_prompt(
    query: str,
    retrieved_docs: list,
    conversation_history: list,
    persona: Optional[str] = None,
) -> str:
    """Build a retrieval-augmented prompt."""
    parts = [RAG_SYSTEM_PROMPT]

    if persona is not None and persona.strip():
        parts.insert(0, f"Custom Persona:\n{persona.strip()}\n\n---\n")

    if not retrieved_docs:
        parts.append("\nNo relevant documents were retrieved for this query.")
    else:
        max_chunks = (
            settings.rag_chunks_in_prompt
            if settings.rag_chunks_in_prompt > 0
            else len(retrieved_docs)
        )

        total_tokens = 0
        chunks_used = 0

        parts.append("\nDOCUMENT CONTEXT:")
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
                    "Truncated chunk %s from %s to %s chars to fit token limit",
                    i,
                    original_length,
                    len(content),
                )

            if (
                settings.max_chunk_chars_in_prompt > 0
                and len(content) > settings.max_chunk_chars_in_prompt
            ):
                content = truncate_text(
                    content, settings.max_chunk_chars_in_prompt, add_ellipsis=True
                )

            parts.append(
                f"\n[Document {i}] Source: {source} | {page_info} | Chunk {chunk_index} | Relevance: {score:.2%}"
            )
            parts.append(f'"""\n{content}\n"""')

            total_tokens += chunk_tokens
            chunks_used += 1

        parts.append("\n------------------------------")
        parts.append(
            f"Summary: {chunks_used} chunks | ~{total_tokens} tokens | {len(retrieved_docs)} documents retrieved"
        )
        parts.append("------------------------------\n")

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
            parts.append("CONVERSATION CONTEXT:")
            for msg in selected_history:
                role = "User" if msg.role.value == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")
            parts.append("")

    parts.append(f"USER QUESTION: {query}")
    parts.append("\nYour response (use documents above, cite sources):")

    return "\n".join(parts)


def build_search_prompt(
    user_message: str, conversation_history: list, persona: Optional[str] = None
) -> str:
    """Build a prompt for the search agent including conversation history."""
    parts = [SEARCH_SYSTEM_PROMPT]

    if persona is not None and persona.strip():
        parts.insert(0, f"Custom Persona:\n{persona.strip()}\n\n---\n")

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
            parts.append("\n\nRelevant prior context:")
            for msg in selected_history:
                role_value = getattr(getattr(msg, "role", None), "value", None)
                role = "User" if role_value == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")
            parts.append(
                "\nUse the conversation context above to understand pronouns or references to earlier images."
            )

    parts.append("\n\nCreate a detailed image based on this request:")
    parts.append(user_message)
    parts.append(
        "\nEnsure the description includes setting, subject appearance, lighting, camera angle, artistic style, and mood."
    )

    return "\n".join(parts)
