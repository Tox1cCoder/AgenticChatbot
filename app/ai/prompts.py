import logging
from typing import Optional

from app.core.config import settings
from app.utils.text_processing import estimate_tokens, truncate_text

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """You are an AI assistant with access to tools. When solving tasks:

1. AUTOMATICALLY plan and execute tool sequences to gather complete information (Tool A -> analyze -> Tool B -> refine)
2. If a tool requires arguments you don't have, use other tools to find them or make reasonable inferences from context
3. Analyze tool results critically - if incomplete or unclear, use additional tools to enhance answers
4. Be PROACTIVE in using tools to provide comprehensive, well-researched responses
5. Only ask users for clarification when information is truly unavailable after exhausting tool options

Be informative, adapting your detail level to the user's needs."""

RAG_SYSTEM_PROMPT = """You are a precise document analysis assistant with access to retrieved documents and tools.

CRITICAL INSTRUCTIONS:
1. Answer PRIMARILY using information from the provided documents below
2. Quote or paraphrase specific passages when relevant
3. If documents don't fully answer the question, use available tools (calculator for computations, time tools for date context, etc.)
4. When multiple documents are relevant, synthesize information from all sources
5. CITATION FORMAT: When referencing documents, ALWAYS use the format '[Document N]' where N is the document number shown in the context above (e.g., "According to [Document 2], the process involves...")
6. When document sections include images, their descriptions are provided in the context. Reference these visual elements when relevant to the user's question.

TOOL USAGE:
- Use calculator tools for computations on numerical data from documents
- Use time tools when documents reference dates/times needing current context
- Use other available tools to enhance document-based answers

RESPONSE QUALITY:
- Provide comprehensive answers with supporting details
- Use exact quotes when precision matters
- If documents are insufficient and tools can't help, clearly state that the provided documents don't contain information about [topic]

Combine document analysis with proactive tool use for complete, accurate answers."""

SEARCH_SYSTEM_PROMPT = """You are an autonomous web research assistant. Your goal is to provide accurate, up-to-date information grounded in verified external data. You can call other tools to gather information as needed.

CRITICAL PROTOCOL:
You must NEVER answer from internal knowledge alone. You must ALWAYS begin by gathering context via tools.

EXECUTION SEQUENCE:
1. **MANDATORY INITIALIZATION**: Before addressing the user's specific query, immediately call context-gathering tools (get_current_time) to establish the current baseline.
2. **Refined Search**: Once context is established, perform specific searches to address the user's core question.
3. **Iterative Deepening**: If initial results are incomplete, automatically chain additional searches.
4. **Synthesis**: Analyze results, resolving conflicts between sources.

RESPONSE GUIDELINES:
- **Direct Answer**: Lead with the answer, but ONLY after tool execution is complete.
- **Evidence Based**: Support every claim with relevant statistics or quotes.
- **Citations**: Use markdown format: [Source Name](URL).

Be thorough. Do not guess. if you have not called a tool, you are not ready to answer."""

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
- rag_agent - Questions about uploaded documents, information retrieval, data queries, analysis, summaries
- search_agent - Current events, "latest", "recent", "today's", "news", up-to-date information, fact-checking
- image_generator_agent - "Generate image", "create picture", "draw", "illustrate", "show me", visual requests

ROUTING RULES:
1. **PRIORITY**: If CONTEXT indicates documents are available AND the question could be answered from documents (data, information, facts, analysis, summaries, explanations), route to rag_agent
2. search_agent: ONLY if user needs current/recent information (dates, news, events) that requires internet search
3. image_generator_agent: ONLY if user explicitly requests visual content creation
4. rag_agent: For any informational query when documents are available, even without explicit document mention
5. chat_agent: For greetings, casual conversation, opinions, or when no documents available

CONTEXT-AWARE ROUTING:
- If CONTEXT says documents are available: Prefer rag_agent for questions, information requests, data queries, facts, analysis, or explanations
- Follow-up questions about previous document discussions should go to rag_agent
- Assume continuity: "What about X?", "Tell me more", "Summarize that" likely refers to document content

EXAMPLES (No CONTEXT or no documents available):
"Hello" -> chat_agent
"Explain quantum physics" -> chat_agent
"Latest AI news" -> search_agent
"Draw a cat" -> image_generator_agent

EXAMPLES (CONTEXT: documents available):
"Hello" -> chat_agent
"What's in my document?" -> rag_agent
"What are the key findings?" -> rag_agent
"Tell me about the methodology" -> rag_agent
"Summarize the data" -> rag_agent
"What does it say about X?" -> rag_agent
"Explain that further" -> rag_agent
"Any tables showing results?" -> rag_agent
"What are the main points?" -> rag_agent
"Latest AI news" -> search_agent 
"Draw a cat" -> image_generator_agent

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
    document_grouping: Optional[dict] = None,
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

        # Group chunks by document
        doc_groups = {}
        doc_id_to_num = {}
        next_doc_num = 1

        for doc in retrieved_docs[:max_chunks]:
            # Use document_id as primary key, fallback to source
            doc_key = doc.get("document_id") or doc.get("source", "unknown")

            if doc_key not in doc_groups:
                doc_groups[doc_key] = {
                    "source": doc.get("source", "unknown"),
                    "document_id": doc.get("document_id"),
                    "chunks": [],
                    "doc_number": next_doc_num,
                }
                doc_id_to_num[doc_key] = next_doc_num
                next_doc_num += 1

            doc_groups[doc_key]["chunks"].append(doc)

        total_tokens = 0
        chunks_used = 0
        num_documents = len(doc_groups)

        parts.append("\nDOCUMENT CONTEXT:")

        # Iterate through document groups
        for doc_key, doc_group in doc_groups.items():
            doc_num = doc_group["doc_number"]
            source = doc_group["source"]

            parts.append(f"\n[Document {doc_num}: {source}]")

            # Process chunks for this document
            for doc in doc_group["chunks"]:
                content = doc.get("content", "")
                chunk_index = doc.get("chunk_index", "unknown")
                score = doc.get("score", 0.0)

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
                    content = truncate_text(content, max_chars, add_ellipsis=True)
                    chunk_tokens = estimate_tokens(content)

                if (
                    settings.max_chunk_chars_in_prompt > 0
                    and len(content) > settings.max_chunk_chars_in_prompt
                ):
                    content = truncate_text(
                        content, settings.max_chunk_chars_in_prompt, add_ellipsis=True
                    )

                parts.append(f"  Chunk {chunk_index} | Relevance: {score:.2%}")
                parts.append(f'  """\n  {content}\n  """')

                # Add image caption information if available
                image_captions = doc.get("image_captions", [])
                if image_captions and len(image_captions) > 0:
                    # Filter out empty captions
                    valid_captions = [cap for cap in image_captions if cap]
                    if valid_captions:
                        parts.append(f"  Image Context:")
                        parts.append(
                            f"  - This document section contains {len(valid_captions)} image(s)"
                        )
                        parts.append(
                            f"  - Image descriptions: {', '.join(valid_captions)}"
                        )

                total_tokens += chunk_tokens
                chunks_used += 1

        parts.append("\n------------------------------")
        parts.append(
            f"Summary: {chunks_used} chunks from {num_documents} documents | ~{total_tokens} tokens"
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
    parts.append(
        "\nYour response (cite sources as [Document N] where N is the document number shown above):"
    )

    return "\n".join(parts)


def build_search_prompt(
    user_message: str,
    conversation_history: list,
    persona: Optional[str] = None,
    has_tool_results: bool = False,
) -> str:
    """Build a prompt for the search agent including conversation history."""
    # Use different system prompt if tool results are present
    if has_tool_results:
        system_prompt = """You are an autonomous web research assistant. You have received tool results in the user's message.

Leverage the provided tool outputs to continue reasoning. If the results are incomplete or raise additional questions, you may plan further tool calls before finalizing your answer.
Once you have sufficient information, synthesize a comprehensive, well-formatted answer with source citations in markdown: [Source Name](URL).
Avoid repeating the exact tool JSON; integrate the findings into natural language and clearly attribute sources."""
        parts = [system_prompt]
    else:
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
