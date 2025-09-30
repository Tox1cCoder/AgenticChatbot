CHAT_SYSTEM_PROMPT = """You are a helpful AI assistant.
You provide thoughtful, accurate, and friendly responses.
You engage in natural conversations while being informative and respectful."""

RAG_SYSTEM_PROMPT = """You are a document-based question answering assistant.
Your role is to provide accurate answers based on the retrieved document context.
Always cite your sources and indicate when information is not available."""


def build_chat_prompt(user_message: str, conversation_history: list) -> str:
    parts = [CHAT_SYSTEM_PROMPT]

    if conversation_history:
        parts.append("\n\nPrevious conversation:")
        for msg in conversation_history[-5:]:
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
        for i, doc in enumerate(retrieved_docs[:3], 1):
            source = doc.get("source", "unknown")
            content = doc.get("content", "")[:500]
            page = doc.get("page_number")
            parts.append(
                f"\nDocument {i} ({source}{f', page {page}' if page else ''}):"
            )
            parts.append(f"{content}...")

    if conversation_history:
        parts.append("\n\nConversation context:")
        for msg in conversation_history[-3:]:
            role = "User" if msg.role.value == "user" else "Assistant"
            parts.append(f"{role}: {msg.content}")

    parts.append(f"\n\nQuestion: {query}")
    parts.append("Answer based on the documents above:")

    return "\n".join(parts)
