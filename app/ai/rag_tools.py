"""
RAG document exploration tools for agentic search.

This module provides the search_documents tool that enables three-phase
document exploration: scan, deep dive, and backtracking.
"""

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
        """
        Explore documents using a three-phase strategy to answer questions.

        ## Actions:
        - **SCAN_ALL**: Preview ALL documents at once (do this FIRST)
        - **READ_DOCUMENT**: Get full content of a specific document
        - **SEARCH_CHUNKS**: Semantic search across document chunks
        - **GREP_DOCUMENT**: Regex search in a specific document
        - **LIST_DOCUMENTS**: List available documents
        - **VIEW_IMAGES**: Get images from a specific document

        ## Three-Phase Strategy:

        ### Phase 1: SCAN_ALL
        1. Use SCAN_ALL to preview all documents at once
        2. In your `reason`, categorize each document:
           - RELEVANT: Clearly related to the query
           - MAYBE: Might contain relevant info
           - SKIP: Not relevant to this query

        ### Phase 2: READ_DOCUMENT
        1. Use READ_DOCUMENT on documents categorized as RELEVANT
        2. Look for cross-references: "See Exhibit A", "Refer to Section X"
        3. In your `reason`, note any cross-references found

        ### Phase 3: Backtracking
        If a document references another document you SKIPPED:
        1. In your `reason`, explain: "Found cross-reference to [doc] - backtracking"
        2. Use READ_DOCUMENT on the referenced document

        The actual execution happens in the graph's _rag_tools_node.
        This function only defines the schema for the LLM.
        """
        return f"Tool call recorded: {action}"

    return search_documents
