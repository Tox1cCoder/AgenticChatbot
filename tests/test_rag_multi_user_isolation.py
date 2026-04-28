"""Phase 1 guards: sidecar / multi-user isolation for RAG retrieval.

These tests protect against:
  * Leaking ``device_id`` into the model-facing RAG tool schema.
  * RAG search that forgets to scope by ``user_id`` / ``conversation_id``.
  * Cross-user retrieval when two users happen to share a ``conversation_id``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.ai.agents.rag_agent import RAGAgent
from app.ai.schemas import SearchDocumentsInput


def _build_minimal_rag_agent(qdrant_stub, embedding_stub) -> RAGAgent:
    """Construct a RAGAgent bypassing BaseAgent.__init__.

    ``_search`` only touches a small subset of fields; we populate just those
    so the tests avoid pulling in LangChain, MCP, and model clients.
    """
    agent = object.__new__(RAGAgent)
    agent.settings = MagicMock()
    agent.settings.rerank_top_k = 5
    agent.settings.enable_citation_verification = False
    agent.qdrant_client = qdrant_stub
    agent.embedding_service = embedding_stub
    agent.collection_name = "documents_gemini_embedding_2_768"
    agent.embedding_dimension = 768
    agent.top_k = 5
    agent.score_threshold = 0.0
    agent.enable_reranking = False
    agent.reranker = None
    agent.agentic_mode = True
    agent.agentic_max_iterations = 5
    agent.agentic_preview_chars = 500
    agent._last_thinking_summary = None
    return agent


def _fake_embedding(dim: int = 768):
    fake = MagicMock()
    fake.embed_query.return_value = [0.0] * dim
    fake.embed_documents.return_value = [[0.0] * dim]
    fake.dimension = dim
    fake.model_name = "gemini-embedding-2"
    fake.provider = "gemini"
    return fake


def _fake_qdrant(points=None):
    points = points or []
    response = MagicMock()
    response.points = points
    fake = MagicMock()
    fake.query_points.return_value = response
    return fake


def _filter_keys(filter_arg) -> set[str]:
    if filter_arg is None or not getattr(filter_arg, "must", None):
        return set()
    return {getattr(cond, "key", None) for cond in filter_arg.must}


def test_search_documents_input_has_no_device_id():
    """The search_documents tool must not expose device_id to the model."""
    assert "device_id" not in SearchDocumentsInput.model_fields


def test_rag_search_filter_includes_user_id_and_conversation_id():
    qdrant = _fake_qdrant()
    agent = _build_minimal_rag_agent(qdrant, _fake_embedding())

    asyncio.run(
        agent._search(
            query="example",
            conversation_id="conv-1",
            user_id="user-1",
        )
    )

    call = qdrant.query_points.call_args
    filter_arg = call.kwargs.get("query_filter")
    assert filter_arg is not None, "Search must build a Qdrant filter"

    keys = _filter_keys(filter_arg)
    assert "conversation_id" in keys, f"conversation_id missing from filter: {keys}"
    assert "user_id" in keys, f"user_id missing from filter: {keys}"


def test_rag_search_still_scopes_conversation_when_user_id_absent():
    """When caller omits user_id, conversation_id scoping must remain intact."""
    qdrant = _fake_qdrant()
    agent = _build_minimal_rag_agent(qdrant, _fake_embedding())

    asyncio.run(agent._search(query="x", conversation_id="conv-1", user_id=None))

    call = qdrant.query_points.call_args
    filter_arg = call.kwargs.get("query_filter")
    keys = _filter_keys(filter_arg)
    assert "conversation_id" in keys


def test_rag_search_isolates_two_users_sharing_conversation_id():
    """Two users with the same ``conversation_id`` value must not see each other's chunks."""
    captured_calls = []
    user_a_chunk_id = uuid4()
    user_a_document_id = uuid4()
    user_b_chunk_id = uuid4()
    user_b_document_id = uuid4()

    def _fake_query_points(**kwargs):
        user_value = None
        conv_value = None
        filter_arg = kwargs.get("query_filter")
        if filter_arg and getattr(filter_arg, "must", None):
            for cond in filter_arg.must:
                if cond.key == "user_id":
                    user_value = cond.match.value
                elif cond.key == "conversation_id":
                    conv_value = cond.match.value
        captured_calls.append({"user_id": user_value, "conversation_id": conv_value})

        payload_map = {
            ("user-a", "shared-conv"): [
                {
                    "chunk_id": str(user_a_chunk_id),
                    "document_id": str(user_a_document_id),
                    "conversation_id": "shared-conv",
                    "user_id": "user-a",
                    "chunk_index": 0,
                }
            ],
            ("user-b", "shared-conv"): [
                {
                    "chunk_id": str(user_b_chunk_id),
                    "document_id": str(user_b_document_id),
                    "conversation_id": "shared-conv",
                    "user_id": "user-b",
                    "chunk_index": 0,
                }
            ],
        }
        payloads = payload_map.get((user_value, conv_value), [])
        points = []
        for p in payloads:
            point = MagicMock()
            point.payload = p
            point.score = 0.9
            points.append(point)
        response = MagicMock()
        response.points = points
        return response

    qdrant = MagicMock()
    qdrant.query_points.side_effect = _fake_query_points
    agent = _build_minimal_rag_agent(qdrant, _fake_embedding())

    chunk_map = {
        user_a_chunk_id: SimpleNamespace(
            id=user_a_chunk_id,
            document_id=user_a_document_id,
            chunk_index=0,
            content="alpha-content",
            page_start=1,
            page_end=1,
            section_path=[],
            document=SimpleNamespace(filename="alpha.pdf"),
        ),
        user_b_chunk_id: SimpleNamespace(
            id=user_b_chunk_id,
            document_id=user_b_document_id,
            chunk_index=0,
            content="beta-content",
            page_start=1,
            page_end=1,
            section_path=[],
            document=SimpleNamespace(filename="beta.pdf"),
        ),
    }
    chunk_repo = MagicMock()
    chunk_repo.get_by_ids.side_effect = lambda chunk_ids: [chunk_map[cid] for cid in chunk_ids]
    image_repo = MagicMock()
    image_repo.get_by_chunk_id.return_value = []

    with (
        patch("app.ai.agents.rag_agent.DocumentChunkRepository") as chunk_repo_cls,
        patch("app.ai.agents.rag_agent.DocumentImageRepository") as image_repo_cls,
    ):
        chunk_repo_cls.return_value = chunk_repo
        image_repo_cls.return_value = image_repo
        results_a = asyncio.run(
            agent._search(query="q", conversation_id="shared-conv", user_id="user-a")
        )
        results_b = asyncio.run(
            agent._search(query="q", conversation_id="shared-conv", user_id="user-b")
        )

    a_sources = {r["source"] for r in results_a}
    b_sources = {r["source"] for r in results_b}
    assert a_sources == {"alpha.pdf"}, f"User A result leaked: {a_sources}"
    assert b_sources == {"beta.pdf"}, f"User B result leaked: {b_sources}"
    assert not (a_sources & b_sources), "Cross-user leakage detected"


def test_rag_search_does_not_return_raw_qdrant_content_when_sql_chunk_is_missing():
    chunk_id = uuid4()
    qdrant_point = MagicMock()
    qdrant_point.payload = {
        "chunk_id": str(chunk_id),
        "document_id": str(uuid4()),
        "content": "raw qdrant content must not be served",
        "source": "legacy-payload.pdf",
    }
    qdrant_point.score = 0.9

    qdrant = _fake_qdrant(points=[qdrant_point])
    agent = _build_minimal_rag_agent(qdrant, _fake_embedding())
    chunk_repo = MagicMock()
    chunk_repo.get_by_ids.return_value = []

    with patch("app.ai.agents.rag_agent.DocumentChunkRepository") as chunk_repo_cls:
        chunk_repo_cls.return_value = chunk_repo
        results = asyncio.run(
            agent._search(query="q", conversation_id="conv-1", user_id="user-1")
        )

    assert results == []


def test_rag_search_hydrates_sql_chunk_content_and_images_from_lookup_payload():
    """Qdrant returns lookup IDs; canonical chunk text and images come from SQL."""
    chunk_id = uuid4()
    document_id = uuid4()
    image_id = uuid4()

    qdrant_point = MagicMock()
    qdrant_point.payload = {
        "chunk_id": str(chunk_id),
        "document_id": str(document_id),
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "chunk_index": 0,
    }
    qdrant_point.score = 0.87

    qdrant = _fake_qdrant(points=[qdrant_point])
    agent = _build_minimal_rag_agent(qdrant, _fake_embedding())

    sql_chunk = SimpleNamespace(
        id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        content="The embedded figure shows a red bar chart with revenue increasing each quarter.",
        page_start=1,
        page_end=1,
        section_path=["Financial Results"],
        document=SimpleNamespace(filename="quarterly-report.pdf"),
    )
    sql_image = SimpleNamespace(
        id=image_id,
        image_path="document_images/doc/chart.png",
        image_caption="A red bar chart showing revenue growth.",
        page_number=1,
    )

    chunk_repo = MagicMock()
    chunk_repo.get_by_ids.return_value = [sql_chunk]
    image_repo = MagicMock()
    image_repo.get_by_chunk_id.return_value = [sql_image]

    with (
        patch("app.ai.agents.rag_agent.DocumentChunkRepository", create=True) as chunk_repo_cls,
        patch("app.ai.agents.rag_agent.DocumentImageRepository") as image_repo_cls,
    ):
        chunk_repo_cls.return_value = chunk_repo
        image_repo_cls.return_value = image_repo
        results = asyncio.run(
            agent._search(query="what does the chart show", conversation_id="conv-1", user_id="user-1")
        )

    assert len(results) == 1
    assert results[0]["content"] == sql_chunk.content
    assert results[0]["source"] == "quarterly-report.pdf"
    assert results[0]["chunk_id"] == str(chunk_id)
    assert results[0]["image_ids"] == [str(image_id)]
    assert results[0]["image_captions"] == ["A red bar chart showing revenue growth."]


def test_get_document_full_content_reads_sql_chunks_not_qdrant_payloads():
    document_id = uuid4()
    qdrant = _fake_qdrant()
    qdrant.scroll.return_value = ([], None)
    agent = _build_minimal_rag_agent(qdrant, _fake_embedding())

    chunk_repo = MagicMock()
    chunk_repo.get_by_document_ordered.return_value = [
        SimpleNamespace(content="First SQL chunk."),
        SimpleNamespace(content="Second SQL chunk."),
    ]

    with patch("app.ai.agents.rag_agent.DocumentChunkRepository", create=True) as chunk_repo_cls:
        chunk_repo_cls.return_value = chunk_repo
        content = asyncio.run(agent.get_document_full_content(str(document_id)))

    assert content == "First SQL chunk.\n\nSecond SQL chunk."
    qdrant.scroll.assert_not_called()
