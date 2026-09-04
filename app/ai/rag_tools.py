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
        # A schema, not an implementation: the RAG subgraph's own ``rag_tools``
        # node executes the call against retrieval. This body exists only so the
        # model has something to bind to, and returning the action keeps a
        # misrouted call legible instead of silently empty.
        return f"Tool call recorded: {action}"

    return search_documents
