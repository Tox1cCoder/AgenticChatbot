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

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.services.gemini_retry import is_rate_limit_error, parse_retry_delay

logger = logging.getLogger(__name__)


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
    # Gemini Embeddings API accepts up to 100 contents per embed_content call.
    # Default 32 is conservative; raise via config if throughput matters.
    embedding_batch_size: int = 32
    # Number of batches submitted concurrently via ThreadPoolExecutor.
    embedding_max_concurrency: int = 4
    provider: str = field(default="gemini", init=False)
    client: Any = field(default=None, init=False, repr=False)

    # Hard ceiling imposed by the Gemini Embeddings API for gemini-embedding-2.
    _API_MAX_BATCH: ClassVar[int] = 100
    # Maximum retry attempts per batch on 429 / RESOURCE_EXHAUSTED.
    _MAX_RETRY_ATTEMPTS: ClassVar[int] = 5
    # Minimum sleep between retries even when the server hint is very small.
    _MIN_RETRY_DELAY: ClassVar[float] = 0.5
    # Base back-off (seconds) when no retryDelay hint is present.
    _BASE_RETRY_DELAY: ClassVar[float] = 5.0

    def __post_init__(self) -> None:
        # Application owns retries (see gemini_retry helpers); attempts=1
        # disables the Gen AI SDK's own retry of the original request.
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
        )
        # Clamp to the documented API maximum so misconfigured values fail safe.
        self.embedding_batch_size = min(self.embedding_batch_size, self._API_MAX_BATCH)

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

        if not texts:
            return []

        pairs = list(zip(texts, titles, strict=True))
        batch_size = self.embedding_batch_size
        batches = [pairs[start : start + batch_size] for start in range(0, len(pairs), batch_size)]

        # Submit all batches concurrently; collect results in submission order
        # to preserve input ordering.
        with ThreadPoolExecutor(max_workers=self.embedding_max_concurrency) as executor:
            futures = [
                executor.submit(
                    self._embed_batch,
                    [text for text, _ in batch],
                    [title for _, title in batch],
                )
                for batch in batches
            ]
            results = [f.result() for f in futures]

        vectors: list[list[float]] = []
        for batch_vectors in results:
            vectors.extend(batch_vectors)
        return vectors

    def _embed_batch(
        self, batch_texts: list[str], batch_titles: list[str | None]
    ) -> list[list[float]]:
        """Call the Gemini embed_content API for one batch, retrying on 429s.

        Returns a list of float vectors in the same order as *batch_texts*.
        """
        contents = [
            self._format_document(text, title)
            for text, title in zip(batch_texts, batch_titles, strict=True)
        ]
        for attempt in range(1, self._MAX_RETRY_ATTEMPTS + 1):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimension,
                    ),
                )
                embeddings = list(getattr(response, "embeddings", []) or [])
                if len(embeddings) != len(contents):
                    raise RuntimeError(
                        f"Embedding count mismatch: sent {len(contents)} contents, "
                        f"got {len(embeddings)} embeddings back"
                    )
                vectors: list[list[float]] = []
                for emb in embeddings:
                    values = getattr(emb, "values", None)
                    if values is None:
                        raise RuntimeError("Embedding response missing values")
                    vectors.append([float(v) for v in values])
                return vectors
            except genai_errors.ClientError as exc:
                if not is_rate_limit_error(exc) or attempt == self._MAX_RETRY_ATTEMPTS:
                    raise
                delay_hint = parse_retry_delay(exc)
                if delay_hint is not None:
                    delay = max(delay_hint, self._MIN_RETRY_DELAY)
                else:
                    delay = self._BASE_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini rate limit on embedding batch "
                    f"(attempt {attempt}/{self._MAX_RETRY_ATTEMPTS}). "
                    f"Waiting {delay:.2f}s before retry."
                )
                time.sleep(delay)
        # Unreachable — the loop raises on the final attempt.
        raise RuntimeError("_embed_batch exhausted retries without raising")  # pragma: no cover

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
