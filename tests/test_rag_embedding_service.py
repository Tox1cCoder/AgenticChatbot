"""Phase 11 guards: GeminiRAGEmbeddingService.

The active RAG embedding path is the Gemini Embeddings API
(``gemini-embedding-2``) accessed through ``GeminiRAGEmbeddingService``.

Tests pin:
  * Document inputs are formatted ``title: ... | text: ...`` and the model
    is called with ``output_dimensionality`` equal to the configured dim.
  * Query inputs are prefixed with ``task: ... | query: ...``.
  * The service returns ``list[list[float]]`` for documents and
    ``list[float]`` for a single query.
  * Mismatched response counts raise a clear ``RuntimeError``.
  * ``embed_image`` only runs when ``rag_multimodal_image_embeddings_enabled``
    is true (the gate lives at the caller, but the service must accept
    bytes + mime_type without error).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _make_response(*vectors: list[float]):
    embeddings = [SimpleNamespace(values=list(v)) for v in vectors]
    return SimpleNamespace(embeddings=embeddings)


def _build_service(monkeypatch, *, dimension: int = 768, query_task: str = "search result"):
    from app.services import rag_embedding_service as mod

    fake_client = MagicMock()
    fake_client.models = MagicMock()
    monkeypatch.setattr(mod.genai, "Client", lambda **_kwargs: fake_client)

    service = mod.GeminiRAGEmbeddingService(
        api_key="test-key",
        model_name="gemini-embedding-2",
        dimension=dimension,
        query_task=query_task,
    )
    return service, fake_client


def test_embed_documents_uses_doc_format_and_output_dimensionality(monkeypatch):
    service, client = _build_service(monkeypatch)
    client.models.embed_content.side_effect = [
        _make_response([0.1] * 768),
    ]

    vectors = service.embed_documents(["body"], titles=["file.pdf"])

    assert vectors == [[0.1] * 768]
    call = client.models.embed_content.call_args
    kwargs = call.kwargs
    assert kwargs["model"] == "gemini-embedding-2"
    assert kwargs["contents"] == "title: file.pdf | text: body"
    config = kwargs["config"]
    assert getattr(config, "output_dimensionality", None) == 768


def test_embed_documents_returns_list_of_list_of_floats(monkeypatch):
    service, client = _build_service(monkeypatch)
    client.models.embed_content.side_effect = [
        _make_response([0.1, 0.2, 0.3]),
        _make_response([0.4, 0.5, 0.6]),
    ]

    vectors = service.embed_documents(["a", "b"], titles=["x.pdf", None])

    assert isinstance(vectors, list)
    assert all(isinstance(v, list) for v in vectors)
    assert all(isinstance(x, float) for v in vectors for x in v)


def test_embed_documents_titles_must_match_texts_length(monkeypatch):
    service, _ = _build_service(monkeypatch)
    with pytest.raises(ValueError):
        service.embed_documents(["a", "b"], titles=["only one"])


def test_embed_query_prefixes_with_task_and_returns_single_vector(monkeypatch):
    service, client = _build_service(monkeypatch)
    client.models.embed_content.side_effect = [_make_response([0.7] * 768)]

    vector = service.embed_query("what changed?")

    assert isinstance(vector, list)
    assert len(vector) == 768
    assert all(isinstance(v, float) for v in vector)
    call = client.models.embed_content.call_args
    assert call.kwargs["contents"] == "task: search result | query: what changed?"


def test_embed_query_uses_configured_query_task(monkeypatch):
    service, client = _build_service(monkeypatch, query_task="question answering")
    client.models.embed_content.side_effect = [_make_response([0.0] * 768)]

    service.embed_query("explain")

    call = client.models.embed_content.call_args
    assert call.kwargs["contents"] == "task: question answering | query: explain"


def test_response_count_mismatch_raises_runtime_error(monkeypatch):
    service, client = _build_service(monkeypatch)
    # Return zero embeddings for a single document — count mismatch.
    client.models.embed_content.side_effect = [SimpleNamespace(embeddings=[])]

    with pytest.raises(RuntimeError, match="Expected one embedding"):
        service.embed_documents(["body"], titles=["t.pdf"])


def test_embed_image_accepts_bytes_and_mime_type(monkeypatch):
    """The image path is gated by a config flag at the caller — the service
    method must still accept (bytes, mime_type) when invoked."""
    service, client = _build_service(monkeypatch)
    client.models.embed_content.side_effect = [_make_response([0.0] * 768)]

    vector = service.embed_image(b"\x89PNG...", mime_type="image/png")

    assert isinstance(vector, list)
    assert len(vector) == 768
    call = client.models.embed_content.call_args
    # contents should be a Part (or anything truthy) — we don't assert on the
    # exact Part shape because google-genai's Part API surface evolves.
    assert call.kwargs["contents"] is not None
    config = call.kwargs["config"]
    assert getattr(config, "output_dimensionality", None) == 768


def test_service_exposes_provider_model_dimension(monkeypatch):
    service, _ = _build_service(monkeypatch)
    assert service.provider == "gemini"
    assert service.model_name == "gemini-embedding-2"
    assert service.dimension == 768
