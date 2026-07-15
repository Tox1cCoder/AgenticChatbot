from app.ui.rag_artifacts import extract_rag_artifact_views


def test_extract_rag_artifact_views_returns_search_documents_outputs():
    metadata = {
        "tool_artifacts": [
            {
                "tool_call_id": "chunk-call",
                "tool": "search_documents",
                "args": {"action": "search_chunks", "query": "revenue"},
                "output": "SEARCH RESULTS:\n\n[1] report.pdf\nchunk evidence",
                "status": "success",
            },
            {
                "tool_call_id": "other-call",
                "tool": "widget_create",
                "args": {},
                "output": "widget",
                "status": "success",
            },
        ]
    }

    views = extract_rag_artifact_views(metadata)

    assert len(views) == 1
    assert views[0].tool_call_id == "chunk-call"
    assert views[0].action == "search_chunks"
    assert views[0].title == "Search Chunks"
    assert "chunk evidence" in views[0].output
    assert views[0].preview.endswith("chunk evidence")


def test_extract_rag_artifact_views_surfaces_structured_chunks_with_source_refs():
    metadata = {
        "tool_artifacts": [
            {
                "tool_call_id": "chunk-call",
                "tool": "search_documents",
                "args": {"action": "search_chunks", "query": "revenue"},
                "output": "SEARCH RESULTS:\n\n[1] report.pdf",
                "status": "success",
                "rag_evidence": {
                    "chunks": [
                        {
                            "rank": 1,
                            "source": "Q3-report.pdf",
                            "score": 0.87,
                            "document_id": "doc-1",
                            "chunk_id": "chunk-7",
                            "page_number": 12,
                            "image_ids": ["img-a"],
                            "image_captions": ["Revenue chart"],
                            "has_tables": True,
                            "table_count": 2,
                            "content": "Revenue increased 14% year-over-year.",
                        },
                        {
                            "rank": 2,
                            "source": "Q4-report.pdf",
                            "score": 0.71,
                            "document_id": "doc-2",
                            "page_start": 4,
                            "page_end": 5,
                            "content": "Recurring revenue grew faster than one-time.",
                        },
                    ]
                },
            }
        ]
    }

    views = extract_rag_artifact_views(metadata)

    assert len(views) == 1
    chunks = views[0].chunks
    assert len(chunks) == 2
    assert chunks[0].source == "Q3-report.pdf"
    assert chunks[0].score == 0.87
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].chunk_id == "chunk-7"
    assert chunks[0].page_label == "p. 12"
    assert chunks[0].image_count == 1
    assert chunks[0].image_captions == ("Revenue chart",)
    assert chunks[0].has_tables is True
    assert chunks[0].table_count == 2
    assert "14%" in chunks[0].content
    assert chunks[1].page_label == "pp. 4-5"


def test_extract_rag_artifact_views_surfaces_document_listings():
    metadata = {
        "tool_artifacts": [
            {
                "tool_call_id": "list-call",
                "tool": "search_documents",
                "args": {"action": "list_documents"},
                "output": "AVAILABLE DOCUMENTS:\n\n1. report.pdf",
                "status": "success",
                "rag_evidence": {
                    "documents": [
                        {
                            "rank": 1,
                            "filename": "report.pdf",
                            "document_id": "doc-1",
                            "chunk_count": 12,
                        }
                    ]
                },
            }
        ]
    }

    views = extract_rag_artifact_views(metadata)
    assert len(views[0].documents) == 1
    assert views[0].documents[0].filename == "report.pdf"
    assert views[0].documents[0].chunk_count == 12


def test_rag_chunk_counts_use_available_evidence_without_inventing_table_count():
    metadata = {
        "tool_artifacts": [
            {
                "tool_call_id": "chunk-call",
                "tool": "search_documents",
                "args": {"action": "search_chunks"},
                "rag_evidence": {
                    "chunks": [
                        {
                            "rank": "not-a-number",
                            "image_captions": ["A chart with no persisted image id"],
                            "has_tables": True,
                        }
                    ]
                },
            }
        ]
    }

    chunk = extract_rag_artifact_views(metadata)[0].chunks[0]
    assert chunk.rank == 1
    assert chunk.image_count == 1
    assert chunk.has_tables is True
    assert chunk.table_count is None
