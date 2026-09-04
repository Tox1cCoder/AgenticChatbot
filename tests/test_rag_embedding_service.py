"""Phase 11 guards: GeminiRAGEmbeddingService.

The active RAG embedding path is the Gemini Embeddings API
(``gemini-embedding-2``) accessed through ``GeminiRAGEmbeddingService``.

Tests pin:
  * Document inputs are formatted ``title: ... | text: ...`` and the model
    is called with ``output_dimensionality`` equal to the configured dim.
  * ``embed_content`` receives a *list* of ``Content`` objects (one per text
    in the batch), not a single string.
  * Two texts in the same batch → one ``embed_content`` call with two
    ``contents`` items.
  * With ``embedding_batch_size=N``, K texts are split into ceil(K/N) calls.
  * Input ordering is preserved across batches even under concurrency.
  * Response-count mismatches raise a clear ``RuntimeError``.
  * 429 / RESOURCE_EXHAUSTED errors are retried with the suggested delay.
  * Concurrency is capped at ``embedding_max_concurrency`` workers.
  * Query inputs are prefixed with ``task: ... | query: ...``.
  * Requests carry ``output_dimensionality`` only: gemini-embedding-2 rejects
    ``task_type``, and the title travels in the document text.
  * The service returns ``list[list[float]]`` for documents and
    ``list[float]`` for a single query.
  * ``embed_image`` only runs when ``rag_multimodal_image_embeddings_enabled``
    is true (the gate lives at the caller, but the service must accept
    bytes + mime_type without error).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai import types


def _make_response(*vectors: list[float]):
    embeddings = [SimpleNamespace(values=list(v)) for v in vectors]
    return SimpleNamespace(embeddings=embeddings)


def _build_service(
    monkeypatch,
    *,
    dimension: int = 2,
    query_task: str = "search result",
    embedding_batch_size: int = 32,
    embedding_max_concurrency: int = 4,
):
    from app.services import rag_embedding_service as mod

    fake_client = MagicMock()
    fake_client.models = MagicMock()
    monkeypatch.setattr(mod.genai, "Client", lambda **_kwargs: fake_client)

    service = mod.GeminiRAGEmbeddingService(
        api_key="test-key",
        model_name="gemini-embedding-2",
        dimension=dimension,
        query_task=query_task,
        embedding_batch_size=embedding_batch_size,
        embedding_max_concurrency=embedding_max_concurrency,
    )
    return service, fake_client


# ------------------------------------------------------------------
# Existing tests — updated for batched list API
# ------------------------------------------------------------------


def test_embed_documents_uses_doc_format_and_output_dimensionality(monkeypatch):
    service, client = _build_service(monkeypatch, dimension=3072)
    # Single text → one embed_content call with a one-element contents list.
    client.models.embed_content.side_effect = [
        _make_response([0.1] * 3072),
    ]

    vectors = service.embed_documents(["body"], titles=["file.pdf"])

    assert vectors == [[0.1] * 3072]
    call = client.models.embed_content.call_args
    kwargs = call.kwargs
    assert kwargs["model"] == "gemini-embedding-2"
    # Each text is its own Gemini Content, not a bare string.
    assert kwargs["contents"] == [
        types.Content(role="user", parts=[types.Part(text="title: file.pdf | text: body")])
    ]
    config = kwargs["config"]
    assert getattr(config, "output_dimensionality", None) == 3072


def test_each_batched_text_is_a_separate_content(monkeypatch):
    """A broken batch payload must not collapse two documents into one input."""
    service, client = _build_service(monkeypatch, dimension=2)
    client.models.embed_content.return_value = _make_response([1.0, 0.0], [0.0, 1.0])

    assert service.embed_documents(["alpha", "beta"]) == [[1.0, 0.0], [0.0, 1.0]]

    contents = client.models.embed_content.call_args.kwargs["contents"]
    assert len(contents) == 2
    assert all(isinstance(item, types.Content) for item in contents)
    assert [item.parts[0].text for item in contents] == [
        "title: none | text: alpha",
        "title: none | text: beta",
    ]


def test_wrong_vector_dimension_fails_closed(monkeypatch):
    """A provider vector with the wrong length must never reach the index."""
    service, client = _build_service(monkeypatch, dimension=3)
    client.models.embed_content.return_value = _make_response([1.0, 2.0])

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        service.embed_documents(["alpha"])


def test_embed_documents_returns_list_of_list_of_floats(monkeypatch):
    service, client = _build_service(monkeypatch, dimension=3)
    # Two texts in the same batch → one call, two embeddings in the response.
    client.models.embed_content.side_effect = [
        _make_response([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]),
    ]

    vectors = service.embed_documents(["a", "b"], titles=["x.pdf", None])

    assert isinstance(vectors, list)
    assert len(vectors) == 2
    assert all(isinstance(v, list) for v in vectors)
    assert all(isinstance(x, float) for v in vectors for x in v)


def test_embed_documents_titles_must_match_texts_length(monkeypatch):
    service, _ = _build_service(monkeypatch)
    with pytest.raises(ValueError):
        service.embed_documents(["a", "b"], titles=["only one"])


def test_embed_query_prefixes_with_task_and_returns_single_vector(monkeypatch):
    service, client = _build_service(monkeypatch, dimension=3072)
    client.models.embed_content.side_effect = [_make_response([0.7] * 3072)]

    vector = service.embed_query("what changed?")

    assert isinstance(vector, list)
    assert len(vector) == 3072
    assert all(isinstance(v, float) for v in vector)
    call = client.models.embed_content.call_args
    assert call.kwargs["contents"] == types.Content(
        role="user", parts=[types.Part(text="task: search result | query: what changed?")]
    )


def test_embed_query_uses_configured_query_task(monkeypatch):
    service, client = _build_service(monkeypatch, dimension=3072, query_task="question answering")
    client.models.embed_content.side_effect = [_make_response([0.0] * 3072)]

    service.embed_query("explain")

    call = client.models.embed_content.call_args
    assert call.kwargs["contents"] == types.Content(
        role="user", parts=[types.Part(text="task: question answering | query: explain")]
    )


def test_embed_query_retry_rate_limit_before_returning_vector(monkeypatch):
    """A transient query rate limit is retried within the configured bound."""
    from google.genai import errors as genai_errors

    service, client = _build_service(monkeypatch, dimension=2)
    monkeypatch.setattr("app.services.rag_embedding_service.time.sleep", lambda _seconds: None)
    rate_limit_err = genai_errors.ClientError(
        429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}
    )
    client.models.embed_content.side_effect = [rate_limit_err, _make_response([0.7, 0.8])]

    assert service.embed_query("what changed?") == [0.7, 0.8]
    assert client.models.embed_content.call_count == 2


def test_response_count_mismatch_raises_runtime_error(monkeypatch):
    service, client = _build_service(monkeypatch)
    # Return zero embeddings for a single-item batch — count mismatch.
    client.models.embed_content.side_effect = [SimpleNamespace(embeddings=[])]

    with pytest.raises(RuntimeError, match="mismatch"):
        service.embed_documents(["body"], titles=["t.pdf"])


def test_embed_image_accepts_bytes_and_mime_type(monkeypatch):
    """The image path is gated by a config flag at the caller — the service
    method must still accept (bytes, mime_type) when invoked."""
    service, client = _build_service(monkeypatch, dimension=3072)
    client.models.embed_content.side_effect = [_make_response([0.0] * 3072)]

    vector = service.embed_image(b"\x89PNG...", mime_type="image/png")

    assert isinstance(vector, list)
    assert len(vector) == 3072
    call = client.models.embed_content.call_args
    # contents should be a Part (or anything truthy) — we don't assert on the
    # exact Part shape because google-genai's Part API surface evolves.
    assert call.kwargs["contents"] is not None
    config = call.kwargs["config"]
    assert getattr(config, "output_dimensionality", None) == 3072


def test_embed_image_retries_rate_limit_before_returning_vector(monkeypatch):
    """Image embeddings receive the same bounded retry treatment as text."""
    from google.genai import errors as genai_errors

    service, client = _build_service(monkeypatch, dimension=2)
    monkeypatch.setattr("app.services.rag_embedding_service.time.sleep", lambda _seconds: None)
    client.models.embed_content.side_effect = [
        genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}),
        _make_response([0.1, 0.2]),
    ]

    assert service.embed_image(b"image", mime_type="image/png") == [0.1, 0.2]
    assert client.models.embed_content.call_count == 2


def test_invalid_response_shape_is_not_retried(monkeypatch):
    """A malformed provider response fails closed instead of spending retries."""
    service, client = _build_service(monkeypatch, dimension=2)
    client.models.embed_content.return_value = _make_response([0.1])

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        service.embed_query("broken")

    assert client.models.embed_content.call_count == 1


def test_service_exposes_provider_model_dimension(monkeypatch):
    service, _ = _build_service(monkeypatch, dimension=3072)
    assert service.provider == "gemini"
    assert service.model_name == "gemini-embedding-2"
    assert service.dimension == 3072


# ------------------------------------------------------------------
# New tests — T005-a through T005-e
# ------------------------------------------------------------------


def test_embed_documents_splits_into_batches_of_batch_size(monkeypatch):
    """T005-a: 3 texts with batch_size=2 → 2 embed_content calls."""
    service, client = _build_service(monkeypatch, embedding_batch_size=2)

    # Batch 1: ["a", "b"] → 2 vectors; Batch 2: ["c"] → 1 vector.
    client.models.embed_content.side_effect = [
        _make_response([1.0, 0.0], [2.0, 0.0]),
        _make_response([3.0, 0.0]),
    ]

    vectors = service.embed_documents(["a", "b", "c"], titles=["t1", "t2", "t3"])

    assert client.models.embed_content.call_count == 2
    first_call = client.models.embed_content.call_args_list[0].kwargs
    second_call = client.models.embed_content.call_args_list[1].kwargs
    assert len(first_call["contents"]) == 2
    assert len(second_call["contents"]) == 1
    assert vectors == [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]


def test_embed_documents_preserves_input_order_across_batches(monkeypatch):
    """T005-b: 4 texts with batch_size=2, concurrency=2 — order is preserved."""
    service, client = _build_service(
        monkeypatch, dimension=1, embedding_batch_size=2, embedding_max_concurrency=2
    )

    # Each batch returns distinctly recognisable vectors so we can tell
    # which batch supplied which result position.
    batch_a = _make_response([1.0], [2.0])  # texts 0, 1
    batch_b = _make_response([3.0], [4.0])  # texts 2, 3

    # Use a lock-protected list to supply responses in call order rather than
    # relying on side_effect ordering (which is call-count based, safe here).
    call_lock = threading.Lock()
    responses = [batch_a, batch_b]
    call_index = [0]

    def fake_embed_content(**kwargs):
        with call_lock:
            idx = call_index[0]
            call_index[0] += 1
        # Slight delay to let both batches start before either finishes.
        time.sleep(0.005)
        return responses[idx]

    client.models.embed_content.side_effect = fake_embed_content

    vectors = service.embed_documents(["a", "b", "c", "d"], titles=["t1", "t2", "t3", "t4"])

    assert len(vectors) == 4
    assert vectors[0] == [1.0]
    assert vectors[1] == [2.0]
    assert vectors[2] == [3.0]
    assert vectors[3] == [4.0]


def test_embed_documents_raises_on_response_count_mismatch(monkeypatch):
    """T005-c: stub returns fewer embeddings than batch_size → RuntimeError."""
    service, client = _build_service(monkeypatch, embedding_batch_size=3)

    # Batch has 3 contents but response only has 1 embedding.
    client.models.embed_content.side_effect = [
        _make_response([0.1, 0.2]),  # only 1 embedding for a 3-text batch
    ]

    with pytest.raises(RuntimeError, match="mismatch"):
        service.embed_documents(["x", "y", "z"], titles=["t1", "t2", "t3"])


def test_embed_documents_429_retry_waits_for_delay_hint(monkeypatch):
    """T005-d: first call raises 429 with retryDelay='2s'; sleep >= 2s; second
    call succeeds."""
    from google.genai import errors as genai_errors

    service, client = _build_service(
        monkeypatch, embedding_batch_size=1, embedding_max_concurrency=1
    )

    response_json = {
        "error": {
            "code": 429,
            "message": "Resource exhausted",
            "status": "RESOURCE_EXHAUSTED",
            "details": [{"retryDelay": "2s"}],
        }
    }
    rate_limit_err = genai_errors.ClientError(429, response_json)

    sleep_calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("app.services.rag_embedding_service.time.sleep", fake_sleep)

    client.models.embed_content.side_effect = [
        rate_limit_err,
        _make_response([0.5, 0.6]),
    ]

    vectors = service.embed_documents(["hello"], titles=["doc.pdf"])

    assert vectors == [[0.5, 0.6]]
    assert sleep_calls, "time.sleep must be called on retry"
    assert sleep_calls[0] >= 2.0, (
        f"sleep must honour the retryDelay hint (≥2s); got {sleep_calls[0]}"
    )


def test_embed_documents_concurrency_cap_respected(monkeypatch):
    """T005-e: ThreadPoolExecutor is created with max_workers=embedding_max_concurrency."""
    from app.services import rag_embedding_service as mod

    service, client = _build_service(
        monkeypatch, dimension=1, embedding_batch_size=1, embedding_max_concurrency=2
    )

    # 6 texts → 6 single-item batches.
    client.models.embed_content.side_effect = [_make_response([float(i)]) for i in range(6)]

    executor_init_kwargs: list[dict] = []
    _original_executor = ThreadPoolExecutor

    class _SpyExecutor(_original_executor):
        def __init__(self, **kwargs):
            executor_init_kwargs.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(mod, "ThreadPoolExecutor", _SpyExecutor)

    service.embed_documents(
        ["a", "b", "c", "d", "e", "f"],
        titles=["t1", "t2", "t3", "t4", "t5", "t6"],
    )

    assert executor_init_kwargs, "ThreadPoolExecutor must be instantiated"
    assert executor_init_kwargs[0].get("max_workers") == 2, (
        f"max_workers must equal embedding_max_concurrency=2; got {executor_init_kwargs[0]}"
    )


# ------------------------------------------------------------------
# Task 10: per-batch usage recording
# ------------------------------------------------------------------

from uuid import uuid4  # noqa: E402

from prometheus_client import CollectorRegistry  # noqa: E402

from app.observability.model_usage import ModelUsageMetrics  # noqa: E402
from app.repositories.model_usage import RecordEventCommand, RecordResult  # noqa: E402
from app.usage.context import bind_usage_context  # noqa: E402
from app.usage.recorder import ModelUsageRecorder  # noqa: E402
from app.usage.types import UsageContext  # noqa: E402


class _FakeUsageRepo:
    def __init__(self) -> None:
        self.commands: list[RecordEventCommand] = []

    def record_event(self, command: RecordEventCommand) -> RecordResult:
        self.commands.append(command)
        return RecordResult(inserted=True, event_id=uuid4())


def _build_recording_service(monkeypatch, **kwargs):
    service, client = _build_service(monkeypatch, **kwargs)
    repo = _FakeUsageRepo()
    service.recorder = ModelUsageRecorder(
        repository=repo,
        enqueue_failed_write=lambda payload: None,
        metrics=ModelUsageMetrics(registry=CollectorRegistry()),
    )
    return service, client, repo


def test_embed_documents_records_one_event_per_batch(monkeypatch):
    service, client, repo = _build_recording_service(monkeypatch, embedding_batch_size=1)
    client.models.embed_content.side_effect = [
        _make_response([0.1, 0.2]),
        _make_response([0.3, 0.4]),
    ]
    user_id = uuid4()

    service.embed_documents(
        ["a", "b"],
        titles=["t1", "t2"],
        usage_context=UsageContext(user_id=user_id, operation="document_index"),
    )

    assert len(repo.commands) == 2
    for command in repo.commands:
        assert command.status == "success"
        assert command.provider == "gemini"
        assert command.model == "gemini-embedding-2"
        assert command.context.operation == "embedding"
        assert command.context.user_id == user_id


def test_embed_batch_locally_estimates_input_when_no_usage(monkeypatch):
    service, client, repo = _build_recording_service(monkeypatch, embedding_batch_size=32)
    client.models.embed_content.side_effect = [_make_response([0.1, 0.2])]

    service.embed_documents(
        ["some document body"],
        titles=["doc.pdf"],
        usage_context=UsageContext(user_id=uuid4(), operation="document_index"),
    )

    command = repo.commands[0]
    assert command.usage.source == "locally_estimated"
    assert command.usage.input_tokens is not None and command.usage.input_tokens > 0


def test_image_embedding_without_provider_usage_is_unavailable(monkeypatch):
    service, client, repo = _build_recording_service(monkeypatch)
    client.models.embed_content.return_value = _make_response([0.1, 0.2])

    service.embed_image(
        b"image-bytes",
        mime_type="image/png",
        usage_context=UsageContext(user_id=uuid4(), operation="document_index"),
    )

    usage = repo.commands[0].usage
    assert usage.source == "unavailable"
    assert usage.input_tokens is None


def test_embed_batch_records_one_event_per_retry_attempt(monkeypatch):
    from google.genai import errors as genai_errors

    service, client, repo = _build_recording_service(
        monkeypatch, embedding_batch_size=1, embedding_max_concurrency=1
    )
    monkeypatch.setattr("app.services.rag_embedding_service.time.sleep", lambda _s: None)
    rate_limit_err = genai_errors.ClientError(
        429,
        {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [{"retryDelay": "1s"}]}},
    )
    client.models.embed_content.side_effect = [rate_limit_err, _make_response([0.5, 0.6])]

    service.embed_documents(
        ["hello"],
        titles=["doc.pdf"],
        usage_context=UsageContext(user_id=uuid4(), operation="document_index"),
    )

    assert len(repo.commands) == 2
    assert [c.attempt for c in repo.commands] == [1, 2]
    assert repo.commands[0].status == "error"
    assert repo.commands[1].status == "success"


def test_embed_query_records_with_bound_chat_context(monkeypatch):
    service, client, repo = _build_recording_service(monkeypatch)
    client.models.embed_content.side_effect = [_make_response([0.7, 0.8])]
    user_id, conversation_id = uuid4(), uuid4()

    with bind_usage_context(
        UsageContext(user_id=user_id, conversation_id=conversation_id, operation="workflow")
    ):
        service.embed_query("what changed?")

    command = repo.commands[0]
    assert command.context.operation == "embedding"
    assert command.context.user_id == user_id
    assert command.context.conversation_id == conversation_id


def test_embed_documents_without_recorder_records_nothing(monkeypatch):
    service, client = _build_service(monkeypatch, embedding_batch_size=1)
    client.models.embed_content.side_effect = [_make_response([0.1, 0.2])]

    # No recorder configured -> no recording, behavior unchanged.
    vectors = service.embed_documents(["a"], titles=["t1"])
    assert vectors == [[0.1, 0.2]]


# ------------------------------------------------------------------
# gemini-embedding-2 request contract
# ------------------------------------------------------------------


def _sent_configs(client):
    return [call.kwargs["config"] for call in client.models.embed_content.call_args_list]


def test_document_request_omits_task_type_unsupported_by_embedding_2(monkeypatch):
    """``task_type`` is rejected by gemini-embedding-2.

    The provider documents that the field cannot be used with this model and
    that the task instruction belongs in the prompt text instead, which
    ``_format_document`` already provides.
    """
    service, client = _build_service(monkeypatch, dimension=2)
    client.models.embed_content.side_effect = [_make_response([0.1, 0.2])]

    service.embed_documents(["body"], titles=["file.pdf"])

    config = _sent_configs(client)[0]
    assert getattr(config, "task_type", None) is None


def test_document_request_omits_title_field_unsupported_by_embedding_2(monkeypatch):
    """The title belongs in the formatted text, never in the request config."""
    service, client = _build_service(monkeypatch, dimension=2)
    client.models.embed_content.side_effect = [_make_response([0.1, 0.2])]

    service.embed_documents(["body"], titles=["file.pdf"])

    config = _sent_configs(client)[0]
    assert getattr(config, "title", None) is None
    assert client.models.embed_content.call_args.kwargs["contents"] == [
        types.Content(role="user", parts=[types.Part(text="title: file.pdf | text: body")])
    ]


def test_query_request_omits_task_type_unsupported_by_embedding_2(monkeypatch):
    """The query task hint is carried by the ``task: ... | query: ...`` prefix."""
    service, client = _build_service(monkeypatch, dimension=2)
    client.models.embed_content.side_effect = [_make_response([0.3, 0.4])]

    service.embed_query("what changed?")

    config = _sent_configs(client)[0]
    assert getattr(config, "task_type", None) is None
    assert client.models.embed_content.call_args.kwargs["contents"] == types.Content(
        role="user", parts=[types.Part(text="task: search result | query: what changed?")]
    )


def test_image_request_omits_task_type_unsupported_by_embedding_2(monkeypatch):
    service, client = _build_service(monkeypatch, dimension=2)
    client.models.embed_content.side_effect = [_make_response([0.5, 0.6])]

    service.embed_image(b"\x89PNG...", mime_type="image/png")

    config = _sent_configs(client)[0]
    assert getattr(config, "task_type", None) is None


def test_output_dimensionality_is_still_requested(monkeypatch):
    """Removing the unsupported fields must not drop the supported one."""
    service, client = _build_service(monkeypatch, dimension=2)
    client.models.embed_content.side_effect = [_make_response([0.1, 0.2])]

    service.embed_documents(["body"], titles=[None])

    assert getattr(_sent_configs(client)[0], "output_dimensionality", None) == 2
