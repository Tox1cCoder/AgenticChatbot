"""
Global Prompts for AI Agents

This module contains all system prompts and prompt templates used by the AI agents.
Centralized prompt management for easy maintenance and updates.
"""

# ============================================================================
# CHAT AGENT PROMPTS
# ============================================================================

CHAT_AGENT_SYSTEM_PROMPT = """You are a helpful AI assistant. 
You provide thoughtful, accurate, and friendly responses to user questions.
You engage in natural conversations while being informative and respectful.
"""

# ============================================================================
# RAG AGENT PROMPTS
# ============================================================================

RAG_AGENT_SYSTEM_PROMPT = """You are a document-based question answering assistant.
Your role is to provide accurate answers based on the retrieved document context.
Always cite your sources and indicate when information is not available in the documents.
"""

RAG_RESPONSE_TEMPLATE = """Based on the retrieved information from {citation}:

{content}

{relevance_note}
"""

RAG_NO_RESULTS_TEMPLATE = """I couldn't find specific information about '{query}' in my knowledge base. 
Could you please rephrase your question or provide more context?"""

RAG_RELEVANCE_HIGH = "This information appears to be highly relevant to your query."
RAG_RELEVANCE_MEDIUM = "This information seems relevant to your query."
RAG_RELEVANCE_LOW = "This information may be related to your query."

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_PROCESSING_MESSAGE = "I apologize, but I'm having trouble processing your message right now. Please try again."

ERROR_NO_DOCUMENTS_FOUND = "I apologize, but I couldn't find the information you're looking for. Please try rephrasing your question."

# ============================================================================
# RESPONSE GENERATION PROMPTS
# ============================================================================


def get_rag_response_with_context(
    query: str,
    content: str,
    source: str,
    page_number: int = None,
    score: float = 0.0,
) -> str:
    """
    Generate a RAG response with proper citation formatting.

    Args:
        query: The user's query
        content: The retrieved content
        source: The source document name
        page_number: Optional page number
        score: Relevance score

    Returns:
        Formatted response string
    """
    # Format citation
    citation = f"'{source}'"
    if page_number:
        citation = f"'{source}' (page {page_number})"

    # Truncate content for quote if too long
    quote = content[:300] + "..." if len(content) > 300 else content

    # Determine relevance note
    if score > 0.8:
        relevance = RAG_RELEVANCE_HIGH
    elif score > 0.7:
        relevance = RAG_RELEVANCE_MEDIUM
    else:
        relevance = RAG_RELEVANCE_LOW

    # Build response
    response = f'According to {citation}:\n\n"{quote}"\n\n{relevance}'

    return response


def get_chat_response_with_context(
    user_input: str,
    context_message_count: int = 0,
) -> str:
    """
    Generate a chat response with conversation context awareness.

    Args:
        user_input: The user's message
        context_message_count: Number of previous messages in context

    Returns:
        Formatted response string
    """
    context_info = ""
    if context_message_count > 0:
        context_info = f"\n\nBased on our conversation history ({context_message_count} previous messages), "

    response = f"Hello! I'm your AI assistant. You said: '{user_input}'. {context_info}I'm here to help with conversations, answer questions, and provide assistance. How can I help you today?"

    return response
