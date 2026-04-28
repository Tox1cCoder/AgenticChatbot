"""Phase 9 + Phase 11 guards: model & config selection for the RAG path.

  * Container wires the embedding service from ``rag_embedding_*`` settings.
  * Reranker defaults to the cross-encoder model the plan specifies.
  * RAG agent model standardizes on ``gemini-3.1-pro-preview``.
  * Retired settings are absent from the Settings schema.
  * Phase 11: active embedding provider is Gemini and ``embedding_dimension``
    is no longer a field on Settings.
"""

from __future__ import annotations

import inspect


def test_rag_agent_model_defaults_to_gemini_3_1_pro_preview():
    from app.core.config import Settings

    assert Settings.model_fields["rag_agent_model"].default == "gemini-3.1-pro-preview"


def test_rag_embedding_settings_are_present():
    from app.core.config import Settings

    fields = Settings.model_fields
    assert fields["rag_embedding_provider"].default == "gemini"
    assert fields["rag_embedding_model"].default == "gemini-embedding-2"
    assert fields["rag_embedding_dimension"].default == 768
    assert fields["rag_embedding_query_task"].default == "search result"
    assert fields["rag_multimodal_image_embeddings_enabled"].default is False
    assert fields["rag_reranker_model"].default == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    for name in ("rag_chunk_target_tokens", "rag_chunk_overlap_tokens", "rag_chunk_max_tokens"):
        assert name in fields, f"{name} must be declared on Settings"


def test_qdrant_collection_default_matches_phase11_namespace():
    from app.core.config import Settings

    assert (
        Settings.model_fields["qdrant_collection_name"].default
        == "documents_gemini_embedding_2_768"
    )


def test_retired_settings_are_removed():
    from app.core.config import Settings

    retired = {
        "agentic_rag_enabled",
        "rag_max_context_tokens",
        "rag_chunks_in_prompt",
        "max_chunk_chars_in_prompt",
        "document_chunk_size",
        "document_chunk_overlap",
        "preserve_cross_page_context",
        # Phase 11: legacy embedding_dimension is gone — only rag_embedding_dimension remains.
        "embedding_dimension",
    }
    leaked = retired & set(Settings.model_fields)
    assert not leaked, f"Retired RAG settings still declared: {sorted(leaked)}"


def test_container_does_not_instantiate_sentencetransformer_for_active_path():
    """When provider is ``gemini``, the active RAG embedding adapter must
    not be a local SentenceTransformer."""
    import app.core.container as container_module

    source = inspect.getsource(container_module)
    # Hardcoded Gemma model name must be absent.
    assert '"google/embeddinggemma-300m"' not in source, (
        "Embedding model name must come from settings, not a hardcoded literal"
    )
    # The active embedding adapter must come from the new service module.
    assert "rag_embedding_service" in source, (
        "Container must wire the rag_embedding_service for the active embedding path"
    )


def test_provider_service_uses_settings_rag_agent_model():
    import app.services.provider_service as mod

    source = inspect.getsource(mod)
    # No retired `gemini-3-pro-preview` references.
    assert "gemini-3-pro-preview" not in source, (
        "Retired model ID gemini-3-pro-preview must be removed from provider_service"
    )


def test_no_settings_embedding_dimension_reads_in_app_or_tests():
    """No code reads the legacy ``settings.embedding_dimension`` after Phase 11."""
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    pattern = re.compile(r"settings\.embedding_dimension")
    leaks: list[str] = []
    for folder in ("app", "tests"):
        for path in (repo_root / folder).rglob("*.py"):
            if path.name == Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8")
            if pattern.search(text):
                leaks.append(str(path.relative_to(repo_root)))
    assert not leaks, (
        f"Legacy settings.embedding_dimension still read from: {leaks}"
    )
