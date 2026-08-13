from langchain_core.tools import tool

from .schemas import DocumentAction, SearchDocumentsInput


def create_search_documents_tool():
    """Create the search_documents tool for RAG document exploration."""

    @tool(args_schema=SearchDocumentsInput)
    def search_documents(
        action: DocumentAction,
        document_id: str | None = None,
        query: str | None = None,
        pattern: str | None = None,
        page: int = 1,
        page_size: int = 10,
        start_chunk: int = 0,
        max_chunks: int = 8,
        reason: str | None = None,
    ) -> str:
        """Explore documents with scan, read, search, grep, list, or image actions."""
        # Stub: actual execution in graph._rag_tools_node
        return f"Tool call recorded: {action}"

    return search_documents
