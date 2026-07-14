"""Phase 8 + Phase 12 guards: RAGAgent has a single agentic execution path.

Prompt-built RAG, the ``agentic_rag_enabled`` toggle, and the traditional
streaming branch are all gone. Every code path funnels through the
search_documents tool. Phase 12 adds: READ_DOCUMENT / GREP_DOCUMENT /
LIST_DOCUMENTS hydrate from SQL with server-context auth filters.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.ai.agents import rag_agent as rag_agent_module
from app.ai.agents.rag_agent import RAGAgent
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


def test_rag_agent_module_does_not_import_build_rag_prompt():
    source = inspect.getsource(rag_agent_module)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "build_rag_prompt" not in stripped, (
            f"build_rag_prompt must not be referenced in rag_agent.py: {stripped!r}"
        )


def test_rag_agent_does_not_read_agentic_rag_enabled_flag():
    source = inspect.getsource(rag_agent_module)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "agentic_rag_enabled" not in stripped, (
            "agentic mode is the only mode — settings.agentic_rag_enabled must go"
        )


def test_process_message_has_no_traditional_rag_branch():
    source = inspect.getsource(RAGAgent.process_message)
    assert "Traditional RAG" not in source
    assert "build_rag_prompt" not in source
    # Single agentic path means no branch on agentic_mode.
    assert "self.agentic_mode" not in source, (
        "process_message must not branch on agentic_mode — it's the only path"
    )


def test_graph_document_aware_chat_does_not_split_on_traditional_rag():
    import app.ai.graph as graph_module

    source = inspect.getsource(graph_module)
    assert "Traditional RAG streaming" not in source, (
        "Traditional RAG streaming branch must be removed from graph.py"
    )
    assert "build_rag_prompt" not in source, (
        "graph.py must not reference build_rag_prompt after Phase 8"
    )


def test_agentic_rag_forwards_run_config_to_model_invocation():
    agent = object.__new__(RAGAgent)
    agent.tools = []
    agent._build_skills_suffix = lambda **_kwargs: ""
    agent._init_tools = AsyncMock()
    agent._get_tools_for_binding = MagicMock(return_value=[])
    agent._resolve_runtime_model_config = MagicMock(
        return_value=SimpleNamespace(
            capabilities={"supports_vision": True},
            fallback_config=None,
            provider="gemini",
            warnings=[],
        )
    )

    captured: dict[str, object] = {}

    async def fake_invoke_agentic_rag_model(**kwargs):
        captured.update(kwargs)
        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="done"),
            metadata={},
        )

    agent._invoke_agentic_rag_model = fake_invoke_agentic_rag_model
    run_config = {
        "tags": ["internal", "planning_subagent"],
        "metadata": {"internal": True, "purpose": "planning_subagent"},
    }

    message = AgentMessage(
        role=MessageRole.USER,
        content="search docs",
        metadata={"run_config": run_config},
    )

    asyncio.run(agent._process_message_agentic(message, "conv-1"))

    assert captured["run_config"] is run_config


def test_search_tool_schema_does_not_carry_server_context():
    """server-owned context (user_id, conversation_id) must live outside the model-facing schema."""
    from app.ai.schemas import SearchDocumentsInput

    forbidden = {"user_id", "conversation_id", "device_id"}
    leaked = forbidden & set(SearchDocumentsInput.model_fields)
    assert not leaked, f"Server context must not be model-facing: {leaked}"


# ---------------------------------------------------------------------------
# Phase 12: SQL hydration with server-context auth filters.
# ---------------------------------------------------------------------------


def _build_minimal_agent() -> RAGAgent:
    """Construct a RAGAgent without invoking BaseAgent.__init__.

    Tests of the public document-helper methods only need ``settings`` and
    a stable agentic_preview_chars; everything else stays as MagicMock.
    """
    agent = object.__new__(RAGAgent)
    agent.settings = MagicMock()
    agent.agentic_preview_chars = 500
    return agent


def test_read_document_helper_accepts_server_context_filters():
    """get_document_full_content must accept user_id/conversation_id filters."""
    sig = inspect.signature(RAGAgent.get_document_full_content)
    params = sig.parameters
    assert "user_id" in params, "get_document_full_content must accept server-context user_id"
    assert "conversation_id" in params, (
        "get_document_full_content must accept server-context conversation_id"
    )


def test_grep_document_helper_accepts_server_context_filters():
    sig = inspect.signature(RAGAgent.grep_document)
    params = sig.parameters
    assert "user_id" in params, "grep_document must accept user_id"
    assert "conversation_id" in params, "grep_document must accept conversation_id"


def test_list_conversation_documents_helper_accepts_user_id_filter():
    sig = inspect.signature(RAGAgent.list_conversation_documents)
    params = sig.parameters
    assert "user_id" in params, "list_conversation_documents must accept user_id"


def test_get_document_full_content_returns_none_when_user_or_conversation_does_not_match():
    """Auth filters must short-circuit before any chunk content is hydrated."""
    agent = _build_minimal_agent()

    chunk_repo = MagicMock()
    # Repository returns no rows when the auth filters don't match.
    chunk_repo.get_by_document_for_scope.return_value = []

    document_id = uuid4()

    with patch("app.ai.agents.rag_agent.DocumentChunkRepository") as repo_cls:
        repo_cls.return_value = chunk_repo
        result = asyncio.run(
            agent.get_document_full_content(
                str(document_id),
                user_id="other-user",
                conversation_id="other-conv",
            )
        )

    assert result is None
    # Auth filters must reach the SQL layer, not be applied after the fact.
    chunk_repo.get_by_document_for_scope.assert_called_once()
    call_kwargs = chunk_repo.get_by_document_for_scope.call_args.kwargs
    assert call_kwargs.get("user_id") == "other-user"
    assert call_kwargs.get("conversation_id") == "other-conv"


def test_get_document_full_content_hydrates_when_filters_match():
    agent = _build_minimal_agent()

    chunk_repo = MagicMock()
    chunk_repo.get_by_document_for_scope.return_value = [
        SimpleNamespace(content="alpha"),
        SimpleNamespace(content="beta"),
    ]

    with patch("app.ai.agents.rag_agent.DocumentChunkRepository") as repo_cls:
        repo_cls.return_value = chunk_repo
        result = asyncio.run(
            agent.get_document_full_content(str(uuid4()), user_id="u", conversation_id="c")
        )

    assert result == "alpha\n\nbeta"


def test_list_conversation_documents_filters_by_user_when_provided():
    agent = _build_minimal_agent()

    fake_db = MagicMock()
    fake_query = MagicMock()
    fake_query.outerjoin.return_value = fake_query
    fake_query.join.return_value = fake_query
    fake_query.filter.return_value = fake_query
    fake_query.group_by.return_value = fake_query
    fake_query.order_by.return_value = fake_query
    fake_query.all.return_value = []
    fake_db.__enter__ = MagicMock(return_value=fake_db)
    fake_db.__exit__ = MagicMock(return_value=False)
    fake_db.query.return_value = fake_query

    with patch("app.ai.agents.rag_agent.SessionLocal", return_value=fake_db):
        asyncio.run(agent.list_conversation_documents(str(uuid4()), user_id=str(uuid4())))

    # The filter must be applied at the SQL layer — that means at least one
    # ``filter(...)`` call applies a user-id condition. We don't introspect the
    # SQLAlchemy expression — we just assert the filter happened more than the
    # baseline conversation_id-only path does, and that the user-id auth path
    # additionally joined the Conversation table.
    assert fake_query.filter.call_count >= 2, (
        "user_id auth filter must be added to the SQL filter chain"
    )
    assert fake_query.join.called, (
        "user_id auth filter must reach Conversation.owner_id via a SQL join"
    )


def test_rag_tool_actions_threads_server_context_into_helpers():
    """rag_tool_actions must forward user_id / conversation_id to RAGAgent helpers."""
    import app.ai.rag_tool_actions as actions_module

    source = inspect.getsource(actions_module)
    # Each helper invocation in the action handlers must carry user_id and
    # conversation_id forward — guard against silent regressions.
    assert "user_id=user_id" in source, (
        "rag_tool_actions must forward server-owned user_id to RAGAgent helpers"
    )
    assert "conversation_id=conversation_id" in source, (
        "rag_tool_actions must forward server-owned conversation_id to RAGAgent helpers"
    )


# ---------------------------------------------------------------------------
# Phase 2 / Task 2.1: scope propagation through scan_all_documents,
# get_document_preview, and get_document_images, plus the rag_tool_actions
# entry points that fan out to them.
# ---------------------------------------------------------------------------


def test_scan_all_documents_signature_accepts_user_id():
    sig = inspect.signature(RAGAgent.scan_all_documents)
    assert "user_id" in sig.parameters, "scan_all_documents must accept user_id"


def test_get_document_preview_signature_accepts_scope():
    sig = inspect.signature(RAGAgent.get_document_preview)
    assert "user_id" in sig.parameters, "get_document_preview must accept user_id"
    assert "conversation_id" in sig.parameters, "get_document_preview must accept conversation_id"


def test_get_document_images_signature_accepts_scope():
    sig = inspect.signature(RAGAgent.get_document_images)
    assert "user_id" in sig.parameters, "get_document_images must accept user_id"
    assert "conversation_id" in sig.parameters, "get_document_images must accept conversation_id"


def test_scan_all_documents_threads_user_scope_into_helpers():
    agent = _build_minimal_agent()

    async def fake_list(conv_id, *, user_id=None):
        fake_list.calls.append({"conv_id": conv_id, "user_id": user_id})
        return [{"document_id": "doc-1", "filename": "a.pdf", "chunk_count": 1}]

    fake_list.calls = []

    async def fake_preview(doc_id, *, user_id=None, conversation_id=None, max_chars=None):
        fake_preview.calls.append(
            {
                "doc_id": doc_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
            }
        )
        return "preview text"

    fake_preview.calls = []

    agent.list_conversation_documents = fake_list
    agent.get_document_preview = fake_preview

    asyncio.run(agent.scan_all_documents("conv-1", user_id="user-1"))

    assert fake_list.calls == [{"conv_id": "conv-1", "user_id": "user-1"}], (
        f"list_conversation_documents not called with user_id: {fake_list.calls}"
    )
    assert fake_preview.calls == [
        {"doc_id": "doc-1", "user_id": "user-1", "conversation_id": "conv-1"}
    ], f"get_document_preview not called with full scope: {fake_preview.calls}"


def test_get_document_preview_threads_scope_to_full_content():
    agent = _build_minimal_agent()

    async def fake_full(doc_id, *, user_id=None, conversation_id=None):
        fake_full.calls.append(
            {
                "doc_id": doc_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
            }
        )
        return "the whole document content"

    fake_full.calls = []
    agent.get_document_full_content = fake_full

    result = asyncio.run(
        agent.get_document_preview("doc-1", user_id="user-1", conversation_id="conv-1")
    )

    assert result == "the whole document content"
    assert fake_full.calls == [
        {"doc_id": "doc-1", "user_id": "user-1", "conversation_id": "conv-1"}
    ]


def test_get_document_images_uses_scoped_repository_when_scope_present(tmp_path):
    agent = _build_minimal_agent()

    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    fake_image = SimpleNamespace(
        id=uuid4(),
        image_path=str(image_path),
        image_caption="cap",
        page_number=1,
        mime_type="image/png",
    )

    image_repo = MagicMock()
    image_repo.get_by_document_for_scope.return_value = [fake_image]
    # Catch accidental fallthrough to unscoped path.
    image_repo.get_by_document_id.side_effect = AssertionError(
        "Unscoped lookup must not be used when scope is provided"
    )

    document_id = uuid4()
    with patch("app.ai.agents.rag_agent.DocumentImageRepository") as repo_cls:
        repo_cls.return_value = image_repo
        result = asyncio.run(
            agent.get_document_images(
                str(document_id),
                user_id="user-1",
                conversation_id="conv-1",
            )
        )

    image_repo.get_by_document_for_scope.assert_called_once()
    call = image_repo.get_by_document_for_scope.call_args
    assert call.kwargs.get("user_id") == "user-1"
    assert call.kwargs.get("conversation_id") == "conv-1"
    assert len(result) == 1
    assert result[0]["caption"] == "cap"


def test_search_documents_action_scan_all_passes_server_scope():
    from unittest.mock import AsyncMock

    import app.ai.rag_tool_actions as actions_module

    rag_agent = MagicMock()
    rag_agent.scan_all_documents = AsyncMock(return_value="DOCUMENT SCAN: ...")

    asyncio.run(
        actions_module.execute_search_documents_action(
            rag_agent=rag_agent,
            conversation_id="conv-1",
            tool_args={"action": "scan_all"},
            context={},
            max_agentic_images=6,
            user_id="user-1",
        )
    )

    rag_agent.scan_all_documents.assert_awaited_once_with("conv-1", user_id="user-1")


def test_search_documents_action_read_document_resolves_filename_reference():
    from unittest.mock import AsyncMock

    import app.ai.rag_tool_actions as actions_module

    document_id = str(uuid4())
    rag_agent = MagicMock()
    rag_agent.list_conversation_documents = AsyncMock(
        return_value=[
            {
                "document_id": document_id,
                "filename": "02_Huntington_Medication_Tip_Sheet.pdf",
                "chunk_count": 1,
            }
        ]
    )

    async def fake_full_content(doc_ref, *, user_id=None, conversation_id=None):
        if doc_ref == document_id:
            return "resolved document text"
        return None

    rag_agent.get_document_full_content = AsyncMock(side_effect=fake_full_content)

    result, _, evidence = asyncio.run(
        actions_module.execute_search_documents_action(
            rag_agent=rag_agent,
            conversation_id="conv-1",
            tool_args={
                "action": "read_document",
                "document_id": "[02_Huntington_Medication_Tip_Sheet.pdf]",
            },
            context={},
            max_agentic_images=6,
            user_id="user-1",
        )
    )

    assert result == f"DOCUMENT CONTENT ({document_id}):\n\nresolved document text"
    assert evidence["document"]["document_id"] == document_id
    rag_agent.list_conversation_documents.assert_awaited_once_with(
        "conv-1",
        user_id="user-1",
    )
    rag_agent.get_document_full_content.assert_awaited_once_with(
        document_id,
        user_id="user-1",
        conversation_id="conv-1",
    )


def test_search_documents_action_view_images_passes_server_scope():
    from unittest.mock import AsyncMock

    import app.ai.rag_tool_actions as actions_module

    rag_agent = MagicMock()
    rag_agent.get_document_images = AsyncMock(return_value=[])

    asyncio.run(
        actions_module.execute_search_documents_action(
            rag_agent=rag_agent,
            conversation_id="conv-1",
            tool_args={"action": "view_images", "document_id": "doc-1"},
            context={},
            max_agentic_images=6,
            user_id="user-1",
        )
    )

    rag_agent.get_document_images.assert_awaited_once_with(
        "doc-1",
        user_id="user-1",
        conversation_id="conv-1",
    )


# ---------------------------------------------------------------------------
# Phase 3 / Task 3.1: tool refresh on every process_message invocation.
# ---------------------------------------------------------------------------


def test_rag_process_message_refreshes_tools_every_invocation():
    """Every RAG turn must refresh tool catalog (mirrors other agents)."""
    from unittest.mock import AsyncMock

    from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole

    agent = object.__new__(RAGAgent)
    # mcp_manager already populated — historical bug short-circuited refresh here.
    agent.mcp_manager = object()
    agent._init_tools = AsyncMock()
    agent._process_message_agentic = AsyncMock(
        return_value=AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="done"),
            metadata={},
        )
    )

    message = AgentMessage(
        role=MessageRole.USER,
        content="read the uploaded document",
        metadata={},
    )

    asyncio.run(agent.process_message(message, "conv-1"))

    agent._init_tools.assert_awaited_once()


def test_rag_reranker_initialization_is_serialized(monkeypatch):
    """CrossEncoder construction is not safe to run concurrently during cold start."""
    active_creations = 0
    max_active_creations = 0
    counter_lock = threading.Lock()
    start_barrier = threading.Barrier(6)

    class FakeCrossEncoder:
        def __init__(self, model_name, *, local_files_only=False, **_kwargs):
            nonlocal active_creations, max_active_creations
            self.model_name = model_name
            self.local_files_only = local_files_only
            with counter_lock:
                active_creations += 1
                max_active_creations = max(max_active_creations, active_creations)
            time.sleep(0.05)
            with counter_lock:
                active_creations -= 1

    monkeypatch.setattr(rag_agent_module, "CrossEncoder", FakeCrossEncoder)

    def initialize_reranker():
        agent = object.__new__(RAGAgent)
        agent.settings = SimpleNamespace(reranker_model="fake-cross-encoder")
        start_barrier.wait(timeout=5)
        agent._init_reranker()
        return agent.reranker

    with ThreadPoolExecutor(max_workers=6) as executor:
        rerankers = list(executor.map(lambda _idx: initialize_reranker(), range(6)))

    assert len(rerankers) == 6
    assert max_active_creations == 1


def test_rag_reranker_uses_canonical_rag_setting(monkeypatch):
    """RAG_RERANKER_MODEL is the documented setting and must win over the legacy alias."""
    constructed: list[str] = []

    class FakeCrossEncoder:
        def __init__(self, model_name, *, local_files_only=False, **_kwargs):
            constructed.append(model_name)
            self.model_name = model_name
            self.local_files_only = local_files_only

    monkeypatch.setattr(rag_agent_module, "CrossEncoder", FakeCrossEncoder)

    agent = object.__new__(RAGAgent)
    agent.settings = SimpleNamespace(
        rag_reranker_model="canonical-cross-encoder",
        reranker_model="legacy-cross-encoder",
    )

    agent._init_reranker()

    assert constructed == ["canonical-cross-encoder"]
    assert agent.reranker.model_name == "canonical-cross-encoder"


def test_rag_reranker_loads_from_local_cache_first(monkeypatch):
    """Cold start must not block on a huggingface.co HEAD request.

    The model is revalidated against the hub on every construction unless
    local_files_only is set, so a slow/unreachable hub times out even when the
    model is already cached. The reranker must load offline-first.
    """
    calls: list[bool] = []

    class FakeCrossEncoder:
        def __init__(self, model_name, *, local_files_only=False, **_kwargs):
            calls.append(local_files_only)
            self.model_name = model_name
            self.local_files_only = local_files_only

    monkeypatch.setattr(rag_agent_module, "CrossEncoder", FakeCrossEncoder)

    agent = object.__new__(RAGAgent)
    agent.settings = SimpleNamespace(reranker_model="fake-cross-encoder")

    agent._init_reranker()

    assert calls == [True]
    assert agent.reranker.local_files_only is True


def test_rag_reranker_downloads_when_not_cached(monkeypatch):
    """When the model is absent from the local cache, fall back to a download."""
    calls: list[bool] = []

    class FakeCrossEncoder:
        def __init__(self, model_name, *, local_files_only=False, **_kwargs):
            calls.append(local_files_only)
            if local_files_only:
                raise OSError("not in local cache")
            self.model_name = model_name
            self.local_files_only = local_files_only

    monkeypatch.setattr(rag_agent_module, "CrossEncoder", FakeCrossEncoder)

    agent = object.__new__(RAGAgent)
    agent.settings = SimpleNamespace(reranker_model="fake-cross-encoder")

    agent._init_reranker()

    assert calls == [True, False]
    assert agent.reranker.local_files_only is False


# ---------------------------------------------------------------------------
# Phase 3 / Task 3.3 + 3.4: shared runtime behavior + single agentic
# invocation method.
# ---------------------------------------------------------------------------


def _build_agentic_invocation_agent():
    """Construct a RAGAgent without invoking BaseAgent.__init__, populated
    with the minimum attributes _invoke_agentic_rag_model needs."""
    agent = object.__new__(RAGAgent)
    agent.settings = MagicMock()
    agent.agentic_max_iterations = 5
    agent.agentic_preview_chars = 500
    agent._last_thinking_summary = None
    agent.runtime_model_resolver = None
    agent.tools = []
    agent.mcp_manager = None
    agent._tools_generation_seen = -1
    return agent


def test_rag_agent_exposes_invoke_agentic_rag_model():
    """The single agentic invocation method must exist with disable_tools support."""
    assert hasattr(RAGAgent, "_invoke_agentic_rag_model")
    sig = inspect.signature(RAGAgent._invoke_agentic_rag_model)
    params = sig.parameters
    assert "disable_tools" in params, "_invoke_agentic_rag_model must accept disable_tools"
    assert "tools" in params, "_invoke_agentic_rag_model must accept tools"


def test_invoke_agentic_rag_model_disable_tools_skips_tool_binding():
    """When disable_tools=True, tools must NOT be bound to the model."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.core.runtime_modeling import ResolvedRuntimeModelConfig

    agent = _build_agentic_invocation_agent()

    fake_response = SimpleNamespace(content="final answer", tool_calls=None)
    bind_calls = []

    async def fake_invoke(llm, msgs):
        return fake_response

    def fake_create_model(rc, *, user_id=None, enable_reasoning_summary=False):
        return (SimpleNamespace(__name__="fakellm"), False)

    def fake_bind(llm, tools, tool_choice="auto"):
        bind_calls.append({"tools": tools, "tool_choice": tool_choice})
        return llm

    agent._ainvoke_with_retries = fake_invoke
    agent._create_langchain_model_from_runtime = fake_create_model

    runtime_config = ResolvedRuntimeModelConfig(
        agent_key="rag",
        provider="gemini",
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=None,
        key_source="settings",
        source="agent_default",
        capabilities={"supports_vision": False},
        fallback_config=None,
        warnings=[],
        provider_fallback=None,
        is_custom_model=False,
    )

    with patch("app.ai.agents.rag_agent.ModelFactory.bind_tools_to_model", side_effect=fake_bind):
        response = asyncio.run(
            agent._invoke_agentic_rag_model(
                conversation_id="conv-1",
                messages=[
                    SystemMessage(content="sys"),
                    HumanMessage(content="hi"),
                ],
                tools=[MagicMock(name="search_documents")],
                disable_tools=True,
                user_id="user-1",
                runtime_config=runtime_config,
            )
        )

    assert bind_calls == [], (
        f"bind_tools_to_model must not be called when disable_tools=True; got {bind_calls}"
    )
    assert response.message.content == "final answer"
    assert response.message.tool_calls is None


def test_invoke_agentic_rag_model_binds_tools_when_enabled():
    """When disable_tools=False, tools are bound through ModelFactory."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.core.runtime_modeling import ResolvedRuntimeModelConfig

    agent = _build_agentic_invocation_agent()

    fake_response = SimpleNamespace(
        content="",
        tool_calls=[{"name": "search_documents", "args": {}, "id": "t1"}],
    )

    bind_calls = []

    async def fake_invoke(llm, msgs):
        return fake_response

    def fake_create_model(rc, *, user_id=None, enable_reasoning_summary=False):
        return (SimpleNamespace(__name__="fakellm"), False)

    def fake_bind(llm, tools, tool_choice="auto"):
        bind_calls.append({"tools": tools, "tool_choice": tool_choice})
        return llm

    agent._ainvoke_with_retries = fake_invoke
    agent._create_langchain_model_from_runtime = fake_create_model

    runtime_config = ResolvedRuntimeModelConfig(
        agent_key="rag",
        provider="gemini",
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=None,
        key_source="settings",
        source="agent_default",
        capabilities={"supports_vision": False},
        fallback_config=None,
        warnings=[],
        provider_fallback=None,
        is_custom_model=False,
    )

    fake_tool = MagicMock(name="search_documents")
    with patch("app.ai.agents.rag_agent.ModelFactory.bind_tools_to_model", side_effect=fake_bind):
        response = asyncio.run(
            agent._invoke_agentic_rag_model(
                conversation_id="conv-1",
                messages=[SystemMessage(content="sys"), HumanMessage(content="hi")],
                tools=[fake_tool],
                disable_tools=False,
                user_id="user-1",
                runtime_config=runtime_config,
            )
        )

    assert len(bind_calls) == 1, f"Expected one bind call, got {bind_calls}"
    assert bind_calls[0]["tools"] == [fake_tool]
    assert (
        response.message.tool_calls and response.message.tool_calls[0]["name"] == "search_documents"
    )


def test_invoke_agentic_rag_model_populates_runtime_metadata():
    """Runtime metadata must include provider/model from the shared method."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.core.runtime_modeling import ResolvedRuntimeModelConfig

    agent = _build_agentic_invocation_agent()
    fake_response = SimpleNamespace(content="ok", tool_calls=None)

    async def fake_invoke(llm, msgs):
        return fake_response

    def fake_create_model(rc, *, user_id=None, enable_reasoning_summary=False):
        return (SimpleNamespace(__name__="fakellm"), False)

    agent._ainvoke_with_retries = fake_invoke
    agent._create_langchain_model_from_runtime = fake_create_model

    runtime_config = ResolvedRuntimeModelConfig(
        agent_key="rag",
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=None,
        key_source="settings",
        source="agent_default",
        capabilities={"supports_vision": True},
        fallback_config=None,
        warnings=[],
        provider_fallback=None,
        is_custom_model=False,
    )

    response = asyncio.run(
        agent._invoke_agentic_rag_model(
            conversation_id="conv-1",
            messages=[SystemMessage(content="sys"), HumanMessage(content="hi")],
            tools=[],
            disable_tools=True,
            user_id="user-1",
            runtime_config=runtime_config,
        )
    )

    assert response.metadata["provider"] == "openai"
    assert response.metadata["model"] == "gpt-4o-mini"
    assert response.metadata["agentic_mode"] is True


def test_invoke_agentic_rag_model_merges_context_window_usage_from_total_tokens():
    """Agentic RAG bypasses BaseAgent.invoke_model_with_history, so its shared
    invocation helper must explicitly attach dynamic context usage."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.core.runtime_modeling import ResolvedRuntimeModelConfig

    agent = _build_agentic_invocation_agent()
    fake_response = SimpleNamespace(
        content="ok",
        tool_calls=None,
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 140,
            "output_token_details": {"reasoning": 15},
        },
    )

    async def fake_invoke(llm, msgs):
        return fake_response

    def fake_create_model(rc, *, user_id=None, enable_reasoning_summary=False):
        return (SimpleNamespace(__name__="fakellm"), False)

    agent._ainvoke_with_retries = fake_invoke
    agent._create_langchain_model_from_runtime = fake_create_model

    runtime_config = ResolvedRuntimeModelConfig(
        agent_key="rag",
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=None,
        key_source="settings",
        source="agent_default",
        capabilities={"supports_vision": True},
        fallback_config=None,
        warnings=[],
        provider_fallback=None,
        is_custom_model=False,
        context_window={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "context_window_tokens": 128000,
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
            "source": "registry",
            "known": True,
        },
    )

    response = asyncio.run(
        agent._invoke_agentic_rag_model(
            conversation_id="conv-1",
            messages=[SystemMessage(content="sys"), HumanMessage(content="hi")],
            tools=[],
            disable_tools=True,
            user_id="user-1",
            runtime_config=runtime_config,
        )
    )

    breakdown = response.metadata["token_breakdown"]
    assert breakdown["actual"]["input_tokens"] == 100
    assert breakdown["actual"]["output_tokens"] == 25
    assert breakdown["actual"]["total_tokens"] == 140
    assert breakdown["actual"]["reasoning_tokens"] == 15

    context_window = response.metadata["context_window"]
    assert context_window["used_tokens"] == 140
    assert context_window["used_token_source"] == "actual_total"
    assert context_window["usage_ratio"] == 140 / 128000
    assert context_window["display_state"] == "ok"


def test_rag_system_prompt_includes_dynamic_delegation_roster():
    """RAG advertises only its graph-injected live handoff roster."""
    from app.ai.hand_off_tool import create_hand_off_tool
    from app.ai.schemas import AgentMessage, MessageRole
    from app.core.runtime_modeling import ResolvedRuntimeModelConfig

    agent = _build_agentic_invocation_agent()
    agent.tools = []

    captured_messages = {}

    async def fake_invoke_model(
        *, conversation_id, messages, tools, disable_tools, user_id, runtime_config, **kwargs
    ):
        captured_messages["messages"] = messages
        from app.ai.schemas import AgentResponse, AgentType

        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="ok"),
            metadata={},
        )

    runtime_config = ResolvedRuntimeModelConfig(
        agent_key="rag",
        provider="gemini",
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=None,
        key_source="settings",
        source="agent_default",
        capabilities={"supports_vision": False},
        fallback_config=None,
        warnings=[],
        provider_fallback=None,
        is_custom_model=False,
    )

    agent._invoke_agentic_rag_model = fake_invoke_model
    agent._resolve_runtime_model_config = lambda *a, **kw: runtime_config
    agent._create_fallback_runtime_config = lambda *a, **kw: None
    agent._build_skills_suffix = lambda **kw: ""
    agent._get_tools_for_binding = lambda **kw: []

    msg = AgentMessage(
        role=MessageRole.USER,
        content="What is in the document?",
        metadata={"original_query": "What is in the document?"},
    )

    asyncio.run(
        agent._process_message_agentic(
            msg,
            "conv-1",
            internal_tools=[create_hand_off_tool(["search_agent"])],
            handoff_target_descriptions={"search_agent": "Current web research."},
        )
    )

    system_msg = captured_messages["messages"][0]
    rendered = system_msg.content if hasattr(system_msg, "content") else str(system_msg)
    assert "hand_off" in rendered
    assert "search_agent: Current web research." in rendered


def test_rag_system_prompt_has_compact_complex_query_policy_without_hardcoded_phrases():
    """The model-facing RAG policy should guide hard queries without bloating every turn."""
    from app.ai.prompts import AGENTIC_RAG_SYSTEM_PROMPT

    section_start = AGENTIC_RAG_SYSTEM_PROMPT.index("Complex questions:")
    section_end = AGENTIC_RAG_SYSTEM_PROMPT.index(
        "When providing your final answer:",
        section_start,
    )
    section = AGENTIC_RAG_SYSTEM_PROMPT[section_start:section_end]
    section_lower = section.lower()

    assert len(section) <= 520

    required_terms = [
        "decompose",
        "comparisons",
        "counts",
        "exclusions",
        "conditions",
        "evidence sufficiency",
    ]
    missing = [term for term in required_terms if term not in section_lower]
    assert not missing, f"RAG prompt missing compact complex-query guidance: {missing}"

    forbidden_literals = [
        '"not"',
        '"no"',
        '"cannot"',
        '"unsupported"',
        '"without"',
        '"fails"',
        '"less than"',
        '"greater than"',
        '"<"',
        '">"',
    ]
    found_literals = [term for term in forbidden_literals if term in section_lower]
    assert not found_literals, f"RAG prompt contains hardcoded query phrases: {found_literals}"
    assert "Hard query strategy" not in AGENTIC_RAG_SYSTEM_PROMPT
