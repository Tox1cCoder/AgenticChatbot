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

import contextlib
import logging
import random
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.services.gemini_retry import is_rate_limit_error, parse_retry_delay
from app.usage import begin_usage_operation, bind_usage_context, current_usage_context
from app.usage.types import NormalizedUsage, UsageContext, UsageOperation

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
        usage_context: UsageContext | None = None,
    ) -> list[list[float]]: ...

    def embed_query(
        self, query: str, *, usage_context: UsageContext | None = None
    ) -> list[float]: ...


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
    # Task 12: version tag for the document-embedding cache key. Bump this if
    # the input formatting given to the model ever changes, so stale cache
    # entries keyed on the old formatting are never reused.
    document_format_version: str = field(default="doc-fmt-v1", init=False)

    def embed_documents(
        self,
        texts: list[str],
        *,
        titles: list[str | None] | None = None,
        usage_context: UsageContext | None = None,
    ) -> list[list[float]]:
        # Local sentence-transformer inference is not a provider call, so it
        # never produces a usage event; ``usage_context`` is accepted only to
        # satisfy the shared protocol.
        _ = titles, usage_context
        if not texts:
            return []
        vectors = self.model.encode(texts)
        return [self._to_float_list(vector) for vector in vectors]

    def embed_query(self, query: str, *, usage_context: UsageContext | None = None) -> list[float]:
        _ = usage_context
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
    # Optional model-usage recorder (Task 10). When present, every
    # ``embed_content`` provider attempt is recorded as an ``embedding``
    # operation. Left ``None`` in offline/dev construction (no recording).
    recorder: Any = field(default=None, repr=False)
    provider: str = field(default="gemini", init=False)
    client: Any = field(default=None, init=False, repr=False)
    # Task 12: version tag for the document-embedding cache key. Bump this if
    # ``_format_document``'s template ever changes, so a cache entry keyed on
    # the old formatting is never mistaken for a hit under the new one.
    document_format_version: str = field(default="doc-fmt-v1", init=False)

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
        usage_context: UsageContext | None = None,
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
        # to preserve input ordering. ``usage_context`` is passed explicitly to
        # each worker because ContextVars do NOT propagate into ThreadPoolExecutor
        # threads, so the bound request context would otherwise be lost.
        with ThreadPoolExecutor(max_workers=self.embedding_max_concurrency) as executor:
            futures = [
                executor.submit(
                    self._embed_batch,
                    [text for text, _ in batch],
                    [title for _, title in batch],
                    usage_context=usage_context,
                )
                for batch in batches
            ]
            results = [f.result() for f in futures]

        vectors: list[list[float]] = []
        for batch_vectors in results:
            vectors.extend(batch_vectors)
        return vectors

    def _embed_batch(
        self,
        batch_texts: list[str],
        batch_titles: list[str | None],
        *,
        usage_context: UsageContext | None = None,
    ) -> list[list[float]]:
        """Call the Gemini embed_content API for one batch, retrying on 429s.

        Returns a list of float vectors in the same order as *batch_texts*.
        Each provider attempt (including retries) is recorded as one
        ``embedding`` usage event under a single operation.
        """
        contents = [
            self._text_content(self._format_document(text, title))
            for text, title in zip(batch_texts, batch_titles, strict=True)
        ]
        # Gemini's config supports one title per request. Preserve it for a
        # singleton document; batched documents retain their individual title
        # in their Content text so no title is incorrectly applied to a peer.
        config_title = batch_titles[0] if len(batch_titles) == 1 else None
        with self._usage_scope(usage_context) as operation:
            return self._embed_with_retries(
                contents=contents,
                expected_count=len(batch_texts),
                task_type="RETRIEVAL_DOCUMENT",
                title=config_title,
                operation=operation,
            )

    def _embed_with_retries(
        self,
        *,
        contents: Any,
        expected_count: int,
        task_type: str,
        title: str | None,
        operation: UsageOperation | None,
    ) -> list[list[float]]:
        for attempt in range(1, self._MAX_RETRY_ATTEMPTS + 1):
            try:
                response = self._run_embed_content(
                    contents=contents,
                    task_type=task_type,
                    title=title,
                    operation=operation,
                )
                vectors = self._response_vectors(response)
                self._validate_vectors(
                    vectors, expected_count=expected_count, dimension=self.dimension
                )
                return vectors
            except genai_errors.ClientError as exc:
                if not is_rate_limit_error(exc) or attempt == self._MAX_RETRY_ATTEMPTS:
                    raise
                delay_hint = parse_retry_delay(exc)
                exponential_delay = self._BASE_RETRY_DELAY * (2 ** (attempt - 1))
                delay = max(
                    delay_hint or 0.0,
                    self._MIN_RETRY_DELAY,
                    exponential_delay + random.uniform(0.0, exponential_delay),
                )
                logger.warning(
                    "Gemini rate limit on embedding request "
                    f"(attempt {attempt}/{self._MAX_RETRY_ATTEMPTS}). "
                    f"Waiting {delay:.2f}s before retry."
                )
                time.sleep(delay)
        # Unreachable — the loop raises on the final attempt.
        raise RuntimeError(
            "Embedding request exhausted retries without raising"
        )  # pragma: no cover

    def embed_query(self, query: str, *, usage_context: UsageContext | None = None) -> list[float]:
        with self._usage_scope(usage_context) as operation:
            vectors = self._embed_with_retries(
                contents=self._text_content(f"task: {self.query_task} | query: {query}"),
                expected_count=1,
                task_type="RETRIEVAL_QUERY",
                title=None,
                operation=operation,
            )
        return vectors[0]

    def embed_image(
        self, image_bytes: bytes, *, mime_type: str, usage_context: UsageContext | None = None
    ) -> list[float]:
        try:
            part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        except AttributeError:
            # Older google-genai exposes a different Part constructor; fall
            # back to passing bytes inline so the service still produces a
            # vector. Keeping this defensive avoids a hard dependency on a
            # specific Part API surface.
            part = {"inline_data": {"mime_type": mime_type, "data": image_bytes}}
        with self._usage_scope(usage_context) as operation:
            vectors = self._embed_with_retries(
                contents=types.Content(role="user", parts=[part]),
                expected_count=1,
                task_type="RETRIEVAL_DOCUMENT",
                title=None,
                operation=operation,
            )
        return vectors[0]

    # ------------------------------------------------------------------
    # Usage recording (Task 10)
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _usage_scope(self, usage_context: UsageContext | None) -> Iterator[UsageOperation | None]:
        """Bind an ``embedding`` operation for the wrapped provider call(s).

        Yields ``None`` (and binds nothing) when no recorder is configured, so
        the offline/dev path is byte-for-byte unchanged. Otherwise binds the
        supplied context (or the current bound context, for the query path that
        inherits the authenticated chat context) with ``operation="embedding"``
        and starts one operation shared across a batch's retries.
        """
        if self.recorder is None:
            yield None
            return
        base = usage_context if usage_context is not None else current_usage_context()
        with bind_usage_context(base.child(operation="embedding")), begin_usage_operation() as op:
            yield op

    def _run_embed_content(
        self,
        *,
        contents: Any,
        task_type: str,
        title: str | None,
        operation: UsageOperation | None,
    ) -> Any:
        """Invoke ``embed_content`` once, recording the attempt when enabled."""

        def _call() -> Any:
            return self.client.models.embed_content(
                model=self.model_name,
                contents=contents,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimension,
                    task_type=task_type,
                    title=title,
                ),
            )

        if self.recorder is None or operation is None:
            return _call()
        return self.recorder.record_one_sync_attempt(
            call=_call,
            provider="gemini",
            model=self.model_name,
            operation=operation,
            estimate=lambda _response: self._estimate_input_usage(contents),
        )

    def _estimate_input_usage(self, contents: Any) -> NormalizedUsage:
        """Locally estimate embedding input tokens when the SDK omits usage."""
        from app.ai.token_counter import TokenCounter

        counter = TokenCounter()
        items = contents if isinstance(contents, list) else [contents]
        texts = [
            part.text
            for item in items
            for part in getattr(item, "parts", []) or []
            if getattr(part, "text", None)
        ]
        if not texts:
            return NormalizedUsage(source="unavailable")
        total = sum(
            counter.count_text(provider="gemini", model=self.model_name, text=text).tokens
            for text in texts
        )
        return NormalizedUsage(input_tokens=total, source="locally_estimated")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _format_document(text: str, title: str | None) -> str:
        clean_title = title.strip() if title and title.strip() else "none"
        return f"title: {clean_title} | text: {text}"

    @staticmethod
    def _text_content(text: str) -> types.Content:
        return types.Content(role="user", parts=[types.Part(text=text)])

    @staticmethod
    def _response_vectors(response: Any) -> list[list[float]]:
        embeddings = list(getattr(response, "embeddings", []) or [])
        vectors: list[list[float]] = []
        for embedding in embeddings:
            values = getattr(embedding, "values", None)
            if values is None:
                raise RuntimeError("Embedding response missing values")
            vectors.append([float(value) for value in values])
        return vectors

    @staticmethod
    def _validate_vectors(
        vectors: list[list[float]], *, expected_count: int, dimension: int
    ) -> None:
        if len(vectors) != expected_count:
            raise RuntimeError(
                f"Embedding count mismatch: expected {expected_count}, got {len(vectors)}"
            )
        bad_indices = [index for index, vector in enumerate(vectors) if len(vector) != dimension]
        if bad_indices:
            raise RuntimeError(
                f"Embedding dimension mismatch at indices {bad_indices}: expected {dimension}"
            )
