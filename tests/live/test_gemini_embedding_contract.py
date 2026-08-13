"""Opt-in contract coverage against the configured Gemini embedding provider."""

from __future__ import annotations

import os

import pytest

from app.services.rag_embedding_service import GeminiRAGEmbeddingService


@pytest.mark.live_provider
def test_gemini_batched_text_embeddings_are_distinct_and_configured_size():
    """Gemini accepts two Content inputs and returns one vector for each."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY is not configured")

    service = GeminiRAGEmbeddingService(api_key=api_key, dimension=768)
    vectors = service.embed_documents(
        ["A lighthouse flashes at sunset.", "A database transaction commits atomically."],
        titles=["coast.txt", "systems.txt"],
    )

    assert len(vectors) == 2
    assert all(len(vector) == service.dimension for vector in vectors)
    assert vectors[0] != vectors[1]
