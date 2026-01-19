from typing import Optional

from langchain_core.tools import tool

from .schemas import SearchDocumentsInput, DocumentAction


def create_search_documents_tool():
    """Create the search_documents tool for RAG document exploration."""

    @tool(args_schema=SearchDocumentsInput)
    def search_documents(
        action: DocumentAction,
        document_id: Optional[str] = None,
        query: Optional[str] = None,
        pattern: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> str:
        """Explore documents using SCAN_ALL, READ_DOCUMENT, SEARCH_CHUNKS, GREP_DOCUMENT, LIST_DOCUMENTS, or VIEW_IMAGES."""
        # Stub: actual execution in graph._rag_tools_node
        return f"Tool call recorded: {action}"

    return search_documents
