from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import replace
from uuid import uuid4

import pytest

from app.services.rag_retrieval import RetrievalCandidate


def _candidates(count: int) -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(
            document_id=uuid4(),
            chunk_id=uuid4(),
            image_id=None,
            modality="text",
            content=f"passage {index}",
            filename=f"document-{index}.pdf",
            page_start=index + 1,
            page_end=index + 1,
            section_path=("Section", str(index)),
            dense_rank=index + 1,
            dense_score=1.0 - index / 100,
            lexical_rank=index + 2,
            lexical_score=0.5 - index / 100,
            fused_score=1.0 / (index + 1),
            chunk_index=index,
            metadata={"origin": index},
        )
        for index in range(count)
    ]


class _Model:
    def __init__(self, scores=None, error: Exception | None = None) -> None:
        self.scores = scores
        self.error = error
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs):
        self.calls.append(pairs)
        if self.error is not None:
            raise self.error
        return self.scores


class _Metrics:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def degraded(self, component: str, failure_code: str) -> None:
        self.events.append((component, failure_code))


@pytest.mark.asyncio
async def test_rank_runs_prediction_off_event_loop(monkeypatch):
    from app.services.rag_reranker import RAGReranker

    called = False
    original_to_thread = asyncio.to_thread

    async def fake_to_thread(function, *args, **kwargs):
        nonlocal called
        called = True
        return await original_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    model = _Model([0.1, 0.2])
    reranker = RAGReranker(model_loader=lambda _name: model, output_limit=2)

    await reranker.rank("query", _candidates(2))

    assert called is True


@pytest.mark.asyncio
async def test_rank_caps_input_and_output_and_preserves_candidate_provenance():
    from app.services.rag_reranker import RAGReranker

    rows = _candidates(5)
    model = _Model([0.4, 0.9, 0.9])
    reranker = RAGReranker(
        model_loader=lambda _name: model,
        candidate_pool=3,
        output_limit=2,
    )

    ranked = await reranker.rank("quarterly revenue", rows)

    assert model.calls == [
        [
            ["quarterly revenue", "passage 0"],
            ["quarterly revenue", "passage 1"],
            ["quarterly revenue", "passage 2"],
        ]
    ]
    assert ranked == [
        replace(rows[1], rerank_score=0.9),
        replace(rows[2], rerank_score=0.9),
    ]
    assert ranked[0].metadata is rows[1].metadata


@pytest.mark.asyncio
async def test_rank_keeps_model_scores_raw_and_marks_them_uncalibrated():
    from app.services.rag_reranker import RAGReranker

    rows = _candidates(2)
    reranker = RAGReranker(
        model_loader=lambda _name: _Model([-3.5, 12.25]),
        output_limit=2,
    )

    ranked = await reranker.rank("query", rows)

    assert [row.rerank_score for row in ranked] == [12.25, -3.5]
    assert reranker.last_trace["score_semantics"] == "uncalibrated_model_score"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scores", "expected_reason"),
    [
        ([0.5], "score_count_mismatch"),
        ([0.5, math.nan], "non_finite_score"),
        ([0.5, math.inf], "non_finite_score"),
    ],
)
async def test_malformed_scores_fail_open_without_mutating_fused_order(scores, expected_reason):
    from app.services.rag_reranker import RAGReranker

    rows = _candidates(2)
    metrics = _Metrics()
    reranker = RAGReranker(
        model_loader=lambda _name: _Model(scores),
        output_limit=2,
        metrics=metrics,
    )

    ranked = await reranker.rank("query", rows)

    assert ranked == rows
    assert all(actual is original for actual, original in zip(ranked, rows[:2], strict=True))
    assert reranker.last_fallback_reason == expected_reason
    assert metrics.events == [("reranker", expected_reason)]


@pytest.mark.asyncio
async def test_provider_exception_fails_open_with_observable_reason():
    from app.services.rag_reranker import RAGReranker

    rows = _candidates(3)
    metrics = _Metrics()
    reranker = RAGReranker(
        model_loader=lambda _name: _Model(error=RuntimeError("provider unavailable")),
        output_limit=2,
        metrics=metrics,
    )

    ranked = await reranker.rank("query", rows)

    assert ranked == rows[:2]
    assert all(actual is original for actual, original in zip(ranked, rows[:2], strict=True))
    assert reranker.last_fallback_reason == "provider_exception"
    assert metrics.events == [("reranker", "provider_exception")]


@pytest.mark.asyncio
async def test_model_load_failure_fails_open_with_bounded_reason():
    from app.services.rag_reranker import RAGReranker

    rows = _candidates(2)
    metrics = _Metrics()

    def failing_loader(_name):
        raise OSError("credentials and path must not leak into metric labels")

    reranker = RAGReranker(
        model_loader=failing_loader,
        output_limit=2,
        metrics=metrics,
    )

    ranked = await reranker.rank("query", rows)

    assert ranked == rows
    assert reranker.last_fallback_reason == "model_load_failure"
    assert metrics.events == [("reranker", "model_load_failure")]


@pytest.mark.asyncio
async def test_timeout_fails_open_and_reports_timeout():
    from app.services.rag_reranker import RAGReranker

    rows = _candidates(3)
    metrics = _Metrics()

    class SlowModel:
        def predict(self, _pairs):
            time.sleep(0.05)
            return [0.0, 1.0, 2.0]

    reranker = RAGReranker(
        model_loader=lambda _name: SlowModel(),
        output_limit=3,
        timeout_seconds=0.005,
        metrics=metrics,
    )

    ranked = await reranker.rank("query", rows)

    assert ranked == rows
    assert all(actual is original for actual, original in zip(ranked, rows, strict=True))
    assert reranker.last_fallback_reason == "timeout"
    assert metrics.events == [("reranker", "timeout")]


@pytest.mark.asyncio
async def test_timed_out_worker_keeps_concurrency_permit_until_thread_finishes():
    from app.services.rag_reranker import RAGReranker

    started = threading.Event()
    release = threading.Event()
    calls = 0

    class BlockingModel:
        def predict(self, _pairs):
            nonlocal calls
            calls += 1
            started.set()
            release.wait(timeout=1)
            return [1.0]

    reranker = RAGReranker(
        model_loader=lambda _name: BlockingModel(),
        max_concurrency=1,
        timeout_seconds=0.01,
        output_limit=1,
    )
    first = asyncio.create_task(reranker.rank("first", _candidates(1)))
    await asyncio.to_thread(started.wait, 0.5)
    second = asyncio.create_task(reranker.rank("second", _candidates(1)))

    try:
        await asyncio.gather(first, second)
        assert calls == 1
    finally:
        release.set()
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_missing_candidate_identity_fails_open_before_provider_call():
    from app.services.rag_reranker import RAGReranker

    rows = _candidates(2)
    rows[1] = replace(rows[1], chunk_id=None, image_id=None)
    model = _Model([0.0, 1.0])
    reranker = RAGReranker(model_loader=lambda _name: model, output_limit=2)

    ranked = await reranker.rank("query", rows)

    assert ranked == rows
    assert model.calls == []
    assert reranker.last_fallback_reason == "missing_candidate_id"


@pytest.mark.asyncio
async def test_disabled_reranker_never_constructs_or_calls_provider():
    from app.services.rag_reranker import RAGReranker

    rows = _candidates(4)
    loader_calls = 0

    def loader(_name):
        nonlocal loader_calls
        loader_calls += 1
        return _Model([0.0] * 4)

    reranker = RAGReranker(
        enabled=False,
        model_loader=loader,
        candidate_pool=3,
        output_limit=2,
    )

    ranked = await reranker.rank("query", rows)

    assert ranked == rows[:2]
    assert loader_calls == 0
    assert reranker.last_fallback_reason is None


@pytest.mark.asyncio
async def test_concurrency_is_bounded_by_configured_limit():
    from app.services.rag_reranker import RAGReranker

    active = 0
    maximum_active = 0
    lock = threading.Lock()

    class BlockingModel:
        def predict(self, _pairs):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return [1.0]

    reranker = RAGReranker(
        model_loader=lambda _name: BlockingModel(),
        output_limit=1,
        timeout_seconds=1,
        max_concurrency=2,
    )

    await asyncio.gather(*(reranker.rank(f"query-{index}", _candidates(1)) for index in range(6)))

    assert maximum_active == 2


@pytest.mark.asyncio
async def test_concurrent_first_calls_construct_one_shared_model():
    from app.services.rag_reranker import RAGReranker

    loader_calls = 0

    def loader(_name):
        nonlocal loader_calls
        loader_calls += 1
        time.sleep(0.02)
        return _Model([1.0])

    reranker = RAGReranker(
        model_loader=loader,
        output_limit=1,
        max_concurrency=2,
    )

    await asyncio.gather(
        reranker.rank("first", _candidates(1)),
        reranker.rank("second", _candidates(1)),
    )

    assert loader_calls == 1


def test_cross_encoder_loader_uses_cache_before_network(monkeypatch):
    import app.services.rag_reranker as reranker_module

    calls: list[bool] = []

    class FakeCrossEncoder:
        def __init__(self, _model_name, *, local_files_only=False):
            calls.append(local_files_only)
            if local_files_only:
                raise OSError("cache miss")

    monkeypatch.setattr(reranker_module, "CrossEncoder", FakeCrossEncoder)

    reranker_module.load_cross_encoder("fake-model")

    assert calls == [True, False]


def test_cross_encoder_loader_does_not_use_network_on_cache_hit(monkeypatch):
    import app.services.rag_reranker as reranker_module

    calls: list[bool] = []

    class FakeCrossEncoder:
        def __init__(self, _model_name, *, local_files_only=False):
            calls.append(local_files_only)

    monkeypatch.setattr(reranker_module, "CrossEncoder", FakeCrossEncoder)

    reranker_module.load_cross_encoder("fake-model")

    assert calls == [True]


def test_reranker_settings_define_bounded_canonical_defaults():
    from app.core.config import Settings

    fields = Settings.model_fields

    assert fields["rag_rerank_candidate_pool"].default == 40
    assert fields["rag_evidence_candidate_limit"].default == 10
    assert fields["rag_reranker_timeout_seconds"].default == 5.0
    assert fields["rag_reranker_max_concurrency"].default == 2
    assert fields["reranker_model"].default == fields["rag_reranker_model"].default


def test_container_reranker_provider_is_lazy_and_uses_canonical_settings():
    from app.core.container import Container

    container = Container()
    reranker = container.rag_reranker()

    assert reranker.model_name == container.rag_reranker.kwargs["model_name"]
    assert reranker.model is None
