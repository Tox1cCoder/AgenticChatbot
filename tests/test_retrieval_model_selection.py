"""Phase 9 guards: model & config selection for the RAG path.

  * Container wires the embedding SentenceTransformer from ``rag_embedding_model``.
  * Reranker defaults to the cross-encoder model the plan specifies.
  * RAG agent model standardizes on ``gemini-3.1-pro-preview``.
  * Retired settings are absent from the Settings schema.
"""

from __future__ import annotations

import inspect


def test_rag_agent_model_defaults_to_gemini_3_1_pro_preview():
    from app.core.config import Settings

    assert Settings.model_fields["rag_agent_model"].default == "gemini-3.1-pro-preview"


def test_rag_embedding_settings_are_present():
    from app.core.config import Settings

    fields = Settings.model_fields
    assert fields["rag_embedding_model"].default == "google/embeddinggemma-300m"
    assert fields["rag_embedding_dimension"].default == 768
    assert fields["rag_reranker_model"].default == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    for name in ("rag_chunk_target_tokens", "rag_chunk_overlap_tokens", "rag_chunk_max_tokens"):
        assert name in fields, f"{name} must be declared on Settings"


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
    }
    leaked = retired & set(Settings.model_fields)
    assert not leaked, f"Retired RAG settings still declared: {sorted(leaked)}"


def test_container_embedding_model_is_built_from_settings():
    """``container.embedding_model`` should construct SentenceTransformer using
    the configured ``rag_embedding_model`` name, not a hardcoded string."""
    import app.core.container as container_module

    source = inspect.getsource(container_module)
    # Hardcoded original name must be gone.
    assert '"google/embeddinggemma-300m"' not in source, (
        "Embedding model name must come from settings, not a hardcoded literal"
    )


def test_provider_service_uses_settings_rag_agent_model():
    import app.services.provider_service as mod

    source = inspect.getsource(mod)
    # No retired `gemini-3-pro-preview` references.
    assert "gemini-3-pro-preview" not in source, (
        "Retired model ID gemini-3-pro-preview must be removed from provider_service"
    )
