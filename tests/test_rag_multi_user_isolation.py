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

import pytest

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
    agent.collection_name = "documents_gemini_embedding_2_3072"
    agent.embedding_dimension = 3072
    agent.top_k = 5
    agent.score_threshold = 0.0
    agent.enable_reranking = False
    agent.reranker = None
    agent.agentic_mode = True
    agent.agentic_max_iterations = 5
    agent.agentic_preview_chars = 500
    agent._last_thinking_summary = None
    agent.retriever = None
    # The real constructor always builds an image_selector, regardless of
    # rag_multimodal_image_embeddings_enabled — only the native search call
    # itself is flag-gated (Task 11).
    agent.image_selector = SimpleNamespace(max_images=6)
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


def test_rag_search_delegates_to_typed_retriever_and_adapts_public_dict_contract():
    from app.services.rag_retrieval import RetrievalCandidate, RetrievalScope

    qdrant = _fake_qdrant()
    agent = _build_minimal_rag_agent(qdrant, _fake_embedding())
    conversation_id = uuid4()
    document_id = uuid4()
    chunk_id = uuid4()
    agent.retriever = MagicMock()
    agent.retriever.search.return_value = [
        RetrievalCandidate(
            document_id=document_id,
            chunk_id=chunk_id,
            image_id=None,
            modality="text",
            content="SQL-authorized content",
            filename="report.pdf",
            page_start=2,
            page_end=2,
            section_path=("Results",),
            dense_rank=1,
            dense_score=0.73,
            lexical_rank=2,
            lexical_score=0.12,
            fused_score=0.03,
            chunk_index=4,
            metadata={"has_tables": True, "table_count": 1},
        )
    ]

    with patch("app.ai.agents.rag_agent.DocumentImageRepository") as image_repo_cls:
        image_repo_cls.return_value.get_by_chunk_id_for_scope.return_value = []
        results = asyncio.run(
            agent._search(
                "revenue",
                top_k=3,
                conversation_id=str(conversation_id),
                user_id="server-user",
            )
        )

    agent.retriever.search.assert_called_once_with(
        "revenue",
        RetrievalScope(user_id="server-user", conversation_id=conversation_id),
        final_limit=3,
    )
    assert results == [
        {
            "content": "SQL-authorized content",
            "source": "report.pdf",
            "score": 0.73,
            "page_number": 2,
            "page_start": 2,
            "page_end": 2,
            "document_id": str(document_id),
            "conversation_id": str(conversation_id),
            "chunk_id": str(chunk_id),
            "chunk_index": 4,
            "has_tables": True,
            "table_count": 1,
            "image_ids": [],
            "image_paths": [],
            "image_captions": [],
            "dense_rank": 1,
            "dense_score": 0.73,
            "lexical_rank": 2,
            "lexical_score": 0.12,
            "fused_score": 0.03,
        }
    ]


@pytest.mark.parametrize(
    ("conversation_id", "user_id"),
    [(None, None), ("conv-1", None), (None, "user-1")],
)
def test_rag_search_fails_closed_when_either_server_scope_value_is_absent(
    conversation_id, user_id
):
    qdrant = _fake_qdrant()
    embedding = _fake_embedding()
    agent = _build_minimal_rag_agent(qdrant, embedding)

    with patch("app.ai.agents.rag_agent.DocumentChunkRepository") as repo_cls:
        results = asyncio.run(
            agent._search(
                query="x",
                conversation_id=conversation_id,
                user_id=user_id,
            )
        )

    assert results == []
    embedding.embed_query.assert_not_called()
    qdrant.query_points.assert_not_called()
    repo_cls.assert_not_called()


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
    chunk_repo.get_active_by_ids_for_scope.side_effect = lambda chunk_ids, **_scope: [
        chunk_map[cid] for cid in chunk_ids
    ]
    image_repo = MagicMock()
    image_repo.get_by_chunk_id_for_scope.return_value = []

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
    chunk_repo.get_by_ids.assert_not_called()


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
        results = asyncio.run(agent._search(query="q", conversation_id="conv-1", user_id="user-1"))

    assert results == []


def test_rag_search_rehydrates_chunks_with_server_scope():
    """Qdrant payload scope is not enough; SQL hydration must re-check scope."""
    foreign_chunk_id = uuid4()
    foreign_document_id = uuid4()

    qdrant_point = MagicMock()
    qdrant_point.payload = {
        "chunk_id": str(foreign_chunk_id),
        "document_id": str(foreign_document_id),
        "conversation_id": "victim-conv",
        "user_id": "victim-user",
        "source": "foreign.pdf",
    }
    qdrant_point.score = 0.9

    qdrant = _fake_qdrant(points=[qdrant_point])
    agent = _build_minimal_rag_agent(qdrant, _fake_embedding())

    chunk_repo = MagicMock()
    chunk_repo.get_by_ids.return_value = [
        SimpleNamespace(
            id=foreign_chunk_id,
            document_id=foreign_document_id,
            chunk_index=0,
            content="foreign content must not be returned",
            page_start=1,
            page_end=1,
            section_path=[],
            document=SimpleNamespace(filename="foreign.pdf"),
        )
    ]
    chunk_repo.get_active_by_ids_for_scope.return_value = []

    with patch("app.ai.agents.rag_agent.DocumentChunkRepository") as chunk_repo_cls:
        chunk_repo_cls.return_value = chunk_repo
        results = asyncio.run(
            agent._search(
                query="q",
                conversation_id="victim-conv",
                user_id="victim-user",
            )
        )

    chunk_repo.get_active_by_ids_for_scope.assert_called_once()
    call_kwargs = chunk_repo.get_active_by_ids_for_scope.call_args.kwargs
    assert call_kwargs.get("conversation_id") == "victim-conv"
    assert call_kwargs.get("user_id") == "victim-user"
    chunk_repo.get_by_ids.assert_not_called()
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
    chunk_repo.get_active_by_ids_for_scope.return_value = [sql_chunk]
    image_repo = MagicMock()
    image_repo.get_by_chunk_id_for_scope.return_value = [sql_image]

    with (
        patch("app.ai.agents.rag_agent.DocumentChunkRepository", create=True) as chunk_repo_cls,
        patch("app.ai.agents.rag_agent.DocumentImageRepository") as image_repo_cls,
    ):
        chunk_repo_cls.return_value = chunk_repo
        image_repo_cls.return_value = image_repo
        results = asyncio.run(
            agent._search(
                query="what does the chart show", conversation_id="conv-1", user_id="user-1"
            )
        )

    assert len(results) == 1
    assert results[0]["content"] == sql_chunk.content
    assert results[0]["source"] == "quarterly-report.pdf"
    assert results[0]["chunk_id"] == str(chunk_id)
    assert results[0]["image_ids"] == [str(image_id)]
    assert results[0]["image_captions"] == ["A red bar chart showing revenue growth."]
    chunk_repo.get_by_ids.assert_not_called()


def test_rag_search_reads_table_metadata_from_sql_chunk():
    """Table metadata is canonical SQL chunk metadata, not a trusted Qdrant payload field."""
    chunk_id = uuid4()
    document_id = uuid4()

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
        content="| Metric | Value |\n|---|---|\n| Accuracy | 91% |",
        page_start=1,
        page_end=1,
        section_path=["Results"],
        chunk_metadata={"has_tables": True, "table_count": 2},
        document=SimpleNamespace(filename="edgevit-results.pdf"),
    )

    chunk_repo = MagicMock()
    chunk_repo.get_active_by_ids_for_scope.return_value = [sql_chunk]
    image_repo = MagicMock()
    image_repo.get_by_chunk_id_for_scope.return_value = []

    with (
        patch("app.ai.agents.rag_agent.DocumentChunkRepository", create=True) as chunk_repo_cls,
        patch("app.ai.agents.rag_agent.DocumentImageRepository") as image_repo_cls,
    ):
        chunk_repo_cls.return_value = chunk_repo
        image_repo_cls.return_value = image_repo
        results = asyncio.run(
            agent._search(query="accuracy table", conversation_id="conv-1", user_id="user-1")
        )

    assert len(results) == 1
    assert results[0]["has_tables"] is True
    assert results[0]["table_count"] == 2


# ---------------------------------------------------------------------------
# Task 11: native multimodal image search — off by default, authorized when on
# ---------------------------------------------------------------------------


def test_native_image_search_is_off_by_default():
    """``rag_multimodal_image_embeddings_enabled`` is default-off: even with a
    retriever that supports it, ``_search`` must never call ``search_images``
    unless the flag is explicitly ``True``."""
    qdrant = _fake_qdrant()
    agent = _build_minimal_rag_agent(qdrant, _fake_embedding())
    # agent.settings is a bare MagicMock (no explicit flag set) — this pins
    # that an unconfigured mock attribute must not be treated as enabled.
    agent.retriever = MagicMock()
    agent.retriever.search.return_value = []

    asyncio.run(agent._search(query="q", conversation_id="conv-1", user_id="user-1"))

    agent.retriever.search_images.assert_not_called()


def test_native_image_search_scopes_by_user_and_conversation_when_enabled():
    qdrant = _fake_qdrant()
    agent = _build_minimal_rag_agent(qdrant, _fake_embedding())
    agent.settings.rag_multimodal_image_embeddings_enabled = True
    agent.image_selector = SimpleNamespace(max_images=4)
    agent.retriever = MagicMock()
    agent.retriever.search.return_value = []
    agent.retriever.search_images.return_value = []

    conversation_id = uuid4()
    asyncio.run(
        agent._search(
            query="compare charts",
            conversation_id=str(conversation_id),
            user_id="user-1",
        )
    )

    agent.retriever.search_images.assert_called_once()
    call = agent.retriever.search_images.call_args
    scope = call.args[1] if len(call.args) > 1 else call.kwargs["scope"]
    assert scope.user_id == "user-1"
    assert scope.conversation_id == conversation_id
    assert call.kwargs["limit"] == 4


def test_native_image_search_failure_falls_back_to_text_only_candidates():
    """A broken native-image path must never break caption-first retrieval."""
    from app.services.rag_retrieval import RetrievalCandidate

    qdrant = _fake_qdrant()
    agent = _build_minimal_rag_agent(qdrant, _fake_embedding())
    agent.settings.rag_multimodal_image_embeddings_enabled = True
    agent.image_selector = SimpleNamespace(max_images=4)
    text_document_id = uuid4()
    text_chunk_id = uuid4()
    agent.retriever = MagicMock()
    agent.retriever.search.return_value = [
        RetrievalCandidate(
            document_id=text_document_id,
            chunk_id=text_chunk_id,
            image_id=None,
            modality="text",
            content="caption-first text evidence",
            filename="report.pdf",
            page_start=1,
            page_end=1,
            section_path=(),
            dense_rank=1,
            dense_score=0.5,
            lexical_rank=None,
            lexical_score=None,
            fused_score=0.5,
        )
    ]
    agent.retriever.search_images.side_effect = RuntimeError("qdrant image collection down")

    with patch("app.ai.agents.rag_agent.DocumentImageRepository") as image_repo_cls:
        image_repo_cls.return_value.get_by_chunk_id_for_scope.return_value = []
        results = asyncio.run(
            agent._search(query="compare charts", conversation_id="conv-1", user_id="user-1")
        )

    assert len(results) == 1
    assert results[0]["content"] == "caption-first text evidence"


def _sqlite_image_repository():
    from sqlalchemy import create_engine
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.models.base import Base
    from app.models.conversation import Conversation
    from app.models.document import Document
    from app.models.document_image import DocumentImage
    from app.models.user import User
    from app.repositories.document_image import DocumentImageRepository

    @compiles(JSONB, "sqlite")
    def _compile_jsonb_for_sqlite(_type, _compiler, **_kwargs):
        return "JSON"

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(
        engine,
        tables=[
            User.__table__,
            Conversation.__table__,
            Document.__table__,
            DocumentImage.__table__,
        ],
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return engine, factory, DocumentImageRepository(factory)


def test_native_image_search_authorizes_document_image_through_parent_document():
    """RAGRetriever.search_images must re-check scope in SQL, not trust the
    Qdrant payload — a foreign tenant's image id must never hydrate."""
    from datetime import datetime, timezone

    from app.models.conversation import Conversation
    from app.models.document import Document
    from app.models.document_image import DocumentImage
    from app.models.user import User
    from app.services.rag_retrieval import RAGRetriever, RetrievalScope

    engine, factory, image_repo = _sqlite_image_repository()
    try:
        owner_id, other_owner_id = uuid4(), uuid4()
        conversation_id, other_conversation_id = uuid4(), uuid4()
        document_id, other_document_id = uuid4(), uuid4()
        image_id, other_image_id = uuid4(), uuid4()

        with factory.begin() as session:
            session.add_all(
                [
                    User(
                        id=owner_id,
                        username="owner",
                        email="owner@example.test",
                        password_hash="test",
                    ),
                    User(
                        id=other_owner_id,
                        username="other",
                        email="other@example.test",
                        password_hash="test",
                    ),
                    Conversation(id=conversation_id, owner_id=owner_id, title="mine"),
                    Conversation(
                        id=other_conversation_id, owner_id=other_owner_id, title="theirs"
                    ),
                ]
            )
            session.add_all(
                [
                    Document(
                        id=document_id,
                        conversation_id=conversation_id,
                        filename="mine.pdf",
                        filename_key="mine.pdf",
                        file_type="application/pdf",
                        status=2,
                        upload_time=datetime.now(timezone.utc),
                    ),
                    Document(
                        id=other_document_id,
                        conversation_id=other_conversation_id,
                        filename="theirs.pdf",
                        filename_key="theirs.pdf",
                        file_type="application/pdf",
                        status=2,
                        upload_time=datetime.now(timezone.utc),
                    ),
                ]
            )
            session.add_all(
                [
                    DocumentImage(
                        id=image_id,
                        document_id=document_id,
                        image_path="a.png",
                        mime_type="image/png",
                        page_number=1,
                    ),
                    DocumentImage(
                        id=other_image_id,
                        document_id=other_document_id,
                        image_path="b.png",
                        mime_type="image/png",
                        page_number=1,
                    ),
                ]
            )

        chunk_repo = MagicMock()
        chunk_repo.get_active_generation_ids_for_scope.return_value = [uuid4()]

        qdrant = MagicMock()
        response = MagicMock()
        # Qdrant is not the authorization authority: it returns points for
        # both tenants' images regardless of which scope is searching.
        response.points = [
            SimpleNamespace(score=0.9, payload={"image_id": str(image_id)}),
            SimpleNamespace(score=0.8, payload={"image_id": str(other_image_id)}),
        ]
        qdrant.query_points.return_value = response

        retriever = RAGRetriever(
            qdrant_client=qdrant,
            embedding_service=_fake_embedding(),
            chunk_repository=chunk_repo,
            collection_name="documents",
            document_image_repository=image_repo,
        )

        # DocumentImageRepository (like DocumentChunkRepository) compares
        # Conversation.owner_id directly against user_id without coercion, so
        # SQLite round-tripping needs a real UUID here — the same convention
        # test_rag_retrieval.py's SQLite scope tests already use.
        results = retriever.search_images(
            "chart",
            RetrievalScope(user_id=owner_id, conversation_id=conversation_id),
        )

        assert [result.image_id for result in results] == [image_id]
        assert results[0].document_id == document_id
        assert other_image_id not in [result.image_id for result in results]
    finally:
        engine.dispose()


def test_native_image_candidate_survives_truncation_when_text_page_is_full():
    """Review finding 6: a full page of text candidates must not evict every
    native image candidate. With reranking disabled (the failure mode from
    the finding), a plain ``candidates[:limit]`` slice always kept text
    first because images were appended after it."""
    from app.services.rag_retrieval import RetrievalCandidate

    qdrant = _fake_qdrant()
    agent = _build_minimal_rag_agent(qdrant, _fake_embedding())
    agent.settings.rag_multimodal_image_embeddings_enabled = True
    agent.top_k = 3
    agent.evidence_candidate_limit = 3
    agent.image_selector = SimpleNamespace(max_images=4)
    agent.reranker = None  # reranking disabled — the failure mode from the finding

    text_candidates = [
        RetrievalCandidate(
            document_id=uuid4(),
            chunk_id=uuid4(),
            image_id=None,
            modality="text",
            content=f"text evidence {i}",
            filename="report.pdf",
            page_start=i,
            page_end=i,
            section_path=(),
            dense_rank=i,
            dense_score=1.0 - i * 0.01,
            lexical_rank=None,
            lexical_score=None,
            fused_score=1.0 - i * 0.01,
        )
        for i in range(3)
    ]
    image_candidate = RetrievalCandidate(
        document_id=uuid4(),
        chunk_id=None,
        image_id=uuid4(),
        modality="image",
        content="a revenue chart",
        filename="report.pdf",
        page_start=2,
        page_end=2,
        section_path=(),
        dense_rank=1,
        dense_score=0.95,
        lexical_rank=None,
        lexical_score=None,
        fused_score=0.95,
    )

    agent.retriever = MagicMock()
    agent.retriever.search.return_value = text_candidates
    agent.retriever.search_images.return_value = [image_candidate]

    with patch("app.ai.agents.rag_agent.DocumentImageRepository") as image_repo_cls:
        image_repo_cls.return_value.get_by_chunk_id_for_scope.return_value = []
        results = asyncio.run(
            agent._search(
                query="revenue chart",
                conversation_id="conv-1",
                user_id="user-1",
                top_k=3,
            )
        )

    assert any(r.get("image_id") == str(image_candidate.image_id) for r in results), (
        f"native image candidate was truncated away: {results}"
    )


def test_image_reservation_never_evicts_all_text_when_limit_is_small():
    """Round 2 finding A: ``reserved`` was not bounded away from ``limit``,
    so whenever the evidence limit was <= max_images the reservation wiped
    out text evidence entirely, regardless of score — a 0.05-scoring image
    evicted a 0.9-scoring chunk. At least one text candidate must survive
    whenever any exist."""
    from app.services.rag_retrieval import RetrievalCandidate

    def _make(modality: str, index: int, score: float) -> RetrievalCandidate:
        return RetrievalCandidate(
            document_id=uuid4(),
            chunk_id=uuid4() if modality == "text" else None,
            image_id=uuid4() if modality == "image" else None,
            modality=modality,
            content=f"{modality}-{index}",
            filename="report.pdf",
            page_start=index,
            page_end=index,
            section_path=(),
            dense_rank=index,
            dense_score=score,
            lexical_rank=None,
            lexical_score=None,
            fused_score=score,
        )

    texts = [_make("text", i, 0.9 - i * 0.01) for i in range(10)]
    images = [_make("image", i, 0.05) for i in range(6)]
    candidates = texts + images

    for limit in (6, 3, 1):
        result = RAGAgent._cap_candidates_with_image_reservation(
            candidates, limit, max_reserved_images=6
        )
        assert len(result) == limit
        assert any(candidate.modality == "text" for candidate in result), (
            f"limit={limit}: text evidence was wiped out entirely: {result}"
        )
