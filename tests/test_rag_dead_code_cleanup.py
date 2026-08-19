from pathlib import Path

RAG_AGENT_SOURCE = Path("app/ai/agents/rag_agent.py")
DOCUMENT_PARSE_SERVICE_SOURCE = Path("app/services/document_parse_service.py")
DOCUMENT_PROCESSING_SERVICE_SOURCE = Path("app/services/document_processing_service.py")


def test_rag_agent_does_not_use_langchain_create_agent() -> None:
    source = RAG_AGENT_SOURCE.read_text(encoding="utf-8")

    assert "from langchain.agents import create_agent" not in source
    assert "create_agent(" not in source


def test_rag_agent_has_no_legacy_generation_helpers() -> None:
    source = RAG_AGENT_SOURCE.read_text(encoding="utf-8")

    legacy_helpers = [
        "def _create_agent_executor(",
        "def _generate_with_tools_stream(",
        "def _generate_with_tools(",
        "def _generate_with_vision(",
        "def _generate(",
        "def _verify_citations(",
        "def _augment_prompt_with_tool_context(",
    ]

    for helper in legacy_helpers:
        assert helper not in source


# ---------------------------------------------------------------------------
# Task 15: proven-dead adapters and delegators removed from the RAG pipeline.
#
# Each name below was proven unreachable by a repository-wide caller search
# (docs/rag-cleanup-inventory.md carries the evidence per candidate) before
# removal. These are source/AST-level absence checks, not behavior tests —
# the corresponding behavior is proven through the replacement boundary in:
#   * tests/test_rag_agent.py (RAGAgent._search reranking + chunk-window auth)
#   * tests/test_rag_document_scope_repository.py (scoped chunk-window reads)
#   * tests/test_document_normalizer.py + tests/test_document_processing_service.py
#     (MinerU table content reaching indexed chunks via DocumentNormalizer +
#     DocumentChunkBuilder instead of the retired character-count chunker)
# ---------------------------------------------------------------------------


def test_rag_agent_has_no_legacy_search_dict_adapters() -> None:
    """``_rerank_results`` and ``get_document_full_content`` had zero non-test
    callers: the live search path (``_search``, called from
    ``execute_search_documents_action``) reranks typed ``RetrievalCandidate``
    objects directly, and ``get_document_chunk_window`` replaced the unbounded
    full-content read for every production caller."""
    source = RAG_AGENT_SOURCE.read_text(encoding="utf-8")

    assert "def _rerank_results(" not in source
    assert "def get_document_full_content(" not in source

    # The replacement boundary must still be present and doing the work.
    assert "def _search(" in source
    assert "def get_document_chunk_window(" in source


def test_document_parse_service_has_no_legacy_character_chunker() -> None:
    """The character-count chunker (RecursiveCharacterTextSplitter-backed)
    and its page-metadata variant had zero production callers once
    DocumentNormalizer + DocumentChunkBuilder became the only chunking path."""
    source = DOCUMENT_PARSE_SERVICE_SOURCE.read_text(encoding="utf-8")

    retired = [
        "RecursiveCharacterTextSplitter",
        "langchain_text_splitters",
        "def _legacy_char_chunk_size(",
        "def _legacy_char_overlap(",
        "def _create_chunks(",
        "def _create_chunks_with_page_metadata(",
    ]
    for token in retired:
        assert token not in source, f"retired token still present: {token}"

    # The replacement primitives remain the only chunking/normalizing path.
    assert "DocumentNormalizer" in source
    assert "DocumentChunkBuilder" in source


def test_all_supported_parse_formats_reach_document_normalizer() -> None:
    """Every extension DocumentParseService.parse_document dispatches on must
    reach DocumentNormalizer — directly, or through a helper that does — so
    no format has a private normalization path left over from the character
    chunker."""
    import inspect

    from app.services.document_parse_service import DocumentParseService

    parse_source = inspect.getsource(DocumentParseService.parse_document)
    assert "self.document_normalizer.normalize_text(" in parse_source
    assert "self._process_excel_workbook(" in parse_source
    assert "self._process_with_mineru(" in parse_source

    excel_source = inspect.getsource(DocumentParseService._process_excel_workbook)
    assert "self.document_normalizer.normalize_excel(" in excel_source
    assert "self.document_normalizer.normalize_text(" in excel_source

    mineru_source = inspect.getsource(DocumentParseService._process_with_mineru)
    assert "self.document_normalizer.normalize_mineru(" in mineru_source
    assert "self.document_normalizer.normalize_markdown(" in mineru_source


def test_document_processing_service_has_no_orphaned_parse_delegators() -> None:
    """These delegator methods forwarded to DocumentParseService but had zero
    callers anywhere in the repository, production or test: process_document
    only ever calls ``_process_excel_workbook`` and ``_process_with_mineru``,
    both of which are retained because process_document (the persist-before-
    vector-writes reference implementation, Task 15 out of scope for removal)
    depends on them."""
    source = DOCUMENT_PROCESSING_SERVICE_SOURCE.read_text(encoding="utf-8")

    retired = [
        "def _legacy_char_chunk_size(",
        "def _legacy_char_overlap(",
        "def _create_chunks(",
        "def _extract_excel_rows(",
        "def _excel_rows_to_markdown(",
        "def _stringify_excel_cell(",
        "def _escape_markdown_table_cell(",
        "def _parse_content_list_json(",
        "def _create_chunks_with_page_metadata(",
        "def _normalize_filename_token(",
        "def _collapse_filename_token(",
        "def _build_filename_aliases(",
        "def _resolve_mineru_output_dir(",
        "def _resolve_markdown_file(",
        "def _attach_prepared_images_to_chunks(",
        "def _display_page_number(",
    ]
    for token in retired:
        assert token not in source, f"retired delegator still present: {token}"

    # process_document and the two delegators it actually calls are retained
    # deliberately (see docs/rag-cleanup-inventory.md) — this is not an
    # oversight, so pin their continued presence.
    assert "async def process_document(" in source
    assert "def _process_excel_workbook(" in source
    assert "async def _process_with_mineru(" in source
    assert "def _get_parse_service(" in source


def test_rag_agent_search_is_wired_to_the_typed_retriever_and_gate() -> None:
    """Light wiring check for two Step 2 contracts whose full behavior is
    already proven in dedicated suites (not duplicated here):
      * all searches use RAGRetriever — see RAGAgent._search plus
        tests/test_rag_retrieval.py and tests/test_rag_multi_user_isolation.py
      * every accepted answer passes GroundedAnswerGate — see
        tests/test_rag_grounding.py and the enforced/disabled-gate cases in
        tests/test_rag_tool_loop_finalization.py
    """
    import inspect

    from app.ai.agents.rag_agent import RAGAgent

    init_source = inspect.getsource(RAGAgent.__init__)
    assert "self.retriever = retriever or RAGRetriever(" in init_source
    assert "self.grounded_answer_gate = grounded_answer_gate or GroundedAnswerGate(" in init_source

    search_source = inspect.getsource(RAGAgent._search)
    assert "retriever.search(" in search_source
