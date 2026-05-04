"""RAG embedding adapter.

The active production path is :class:`GeminiRAGEmbeddingService`, which
wraps the Gemini Embeddings API (``gemini-embedding-2``) and produces
deterministic per-input vectors at the configured ``output_dimensionality``.

Key invariants:

* Document inputs are formatted as ``title: {title} | text: {text}``.
* Query inputs are prefixed with ``task: {query_task} | query: {query}``.
* The service returns ``list[list[float]]`` for documents and a flat
  ``list[float]`` for a single query.
* Response shape mismatches raise a clear :class:`RuntimeError`. We never
  silently fall back to a stale or partial vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from google import genai
from google.genai import types


class RAGEmbeddingService(Protocol):
    provider: str
    model_name: str
    dimension: int

    def embed_documents(
        self,
        texts: list[str],
        *,
        titles: list[str | None] | None = None,
    ) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


@dataclass
class SentenceTransformerRAGEmbeddingService:
    """Offline-only fallback adapter wrapping a local SentenceTransformer model.

    Selected when ``rag_embedding_provider='sentence_transformers'``. Used for
    development without GEMINI_API_KEY. Production runs with the Gemini
    adapter; this class exists so the container fallback branch stays
    importable.
    """

    model: Any
    model_name: str
    dimension: int
    provider: str = field(default="sentence_transformers", init=False)

    def embed_documents(
        self,
        texts: list[str],
        *,
        titles: list[str | None] | None = None,
    ) -> list[list[float]]:
        _ = titles
        if not texts:
            return []
        vectors = self.model.encode(texts)
        return [self._to_float_list(vector) for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(query)
        return self._to_float_list(vector)

    @staticmethod
    def _to_float_list(vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(value) for value in list(vector)]


@dataclass
class GeminiRAGEmbeddingService:
    api_key: str
    model_name: str = "gemini-embedding-2"
    dimension: int = 768
    query_task: str = "search result"
    provider: str = field(default="gemini", init=False)
    client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.client = genai.Client(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_documents(
        self,
        texts: list[str],
        *,
        titles: list[str | None] | None = None,
    ) -> list[list[float]]:
        titles = list(titles) if titles is not None else [None] * len(texts)
        if len(titles) != len(texts):
            raise ValueError("titles must match texts length")

        vectors: list[list[float]] = []
        for text, title in zip(texts, titles, strict=True):
            payload = self._format_document(text, title)
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=payload,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimension,
                ),
            )
            vectors.append(self._single_embedding(response))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=f"task: {self.query_task} | query: {query}",
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimension,
            ),
        )
        return self._single_embedding(response)

    def embed_image(self, image_bytes: bytes, *, mime_type: str) -> list[float]:
        try:
            part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        except AttributeError:
            # Older google-genai exposes a different Part constructor; fall
            # back to passing bytes inline so the service still produces a
            # vector. Keeping this defensive avoids a hard dependency on a
            # specific Part API surface.
            part = {"inline_data": {"mime_type": mime_type, "data": image_bytes}}
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=part,
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimension,
            ),
        )
        return self._single_embedding(response)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _format_document(text: str, title: str | None) -> str:
        clean_title = title.strip() if title and title.strip() else "none"
        return f"title: {clean_title} | text: {text}"

    @staticmethod
    def _single_embedding(response: Any) -> list[float]:
        embeddings = list(getattr(response, "embeddings", []) or [])
        if len(embeddings) != 1:
            raise RuntimeError(f"Expected one embedding, got {len(embeddings)}")
        values = getattr(embeddings[0], "values", None)
        if values is None:
            raise RuntimeError("Embedding response missing values")
        return [float(value) for value in values]
