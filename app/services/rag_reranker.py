"""Bounded, fail-open reranking for authorized retrieval candidates."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Protocol

from sentence_transformers import CrossEncoder

from app.services.rag_retrieval import RetrievalCandidate

logger = logging.getLogger(__name__)

RERANK_SCORE_SEMANTICS = "uncalibrated_model_score"


class RerankerMetrics(Protocol):
    def degraded(self, component: str, failure_code: str) -> None: ...


class _RerankerFailure(ValueError):
    def __init__(self, failure_code: str) -> None:
        super().__init__(failure_code)
        self.failure_code = failure_code


class _ModelLoadFailure(RuntimeError):
    pass


def load_cross_encoder(model_name: str) -> CrossEncoder:
    """Load from the local Hugging Face cache before attempting a download."""
    try:
        return CrossEncoder(model_name, local_files_only=True)
    except OSError:
        logger.info(
            "Reranker '%s' is not in the local cache; downloading it once",
            model_name,
        )
        return CrossEncoder(model_name)


def apply_rerank_scores(
    candidates: Sequence[RetrievalCandidate], scores: Sequence[Any]
) -> list[RetrievalCandidate]:
    """Attach finite model-native scores and order ties by fused position.

    Scores are intentionally not normalized. Cross-encoder outputs are
    model-specific, uncalibrated ordering values and must not be interpreted as
    probabilities or compared to a global answerability threshold.
    """
    if len(scores) != len(candidates):
        raise _RerankerFailure("score_count_mismatch")

    ranked: list[tuple[int, RetrievalCandidate]] = []
    for original_position, (candidate, raw_score) in enumerate(
        zip(candidates, scores, strict=True)
    ):
        try:
            score = float(raw_score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _RerankerFailure("invalid_score") from exc
        if not math.isfinite(score):
            raise _RerankerFailure("non_finite_score")
        ranked.append((original_position, replace(candidate, rerank_score=score)))

    ranked.sort(
        key=lambda item: (
            -float(item[1].rerank_score),
            item[0],
        )
    )
    return [candidate for _, candidate in ranked]


class RAGReranker:
    """Run lazy cross-encoder inference without blocking the event loop.

    The service caps input/output sizes, keeps timed-out worker calls inside the
    configured concurrency budget until their threads finish, and always
    returns the original fused order when reranking cannot be trusted.
    """

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enabled: bool = True,
        candidate_pool: int = 40,
        output_limit: int = 10,
        timeout_seconds: float = 5.0,
        max_concurrency: int = 2,
        model_loader: Callable[[str], Any] = load_cross_encoder,
        metrics: RerankerMetrics | None = None,
    ) -> None:
        self.model_name = str(model_name)
        self.enabled = bool(enabled)
        self.candidate_pool = max(1, int(candidate_pool))
        self.output_limit = max(1, int(output_limit))
        self.timeout_seconds = max(0.001, float(timeout_seconds))
        self.max_concurrency = max(1, int(max_concurrency))
        self._model_loader = model_loader
        self.metrics = metrics
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(self.max_concurrency)

    @property
    def model(self) -> Any | None:
        """Expose an already-loaded model without triggering construction."""
        return self._model

    async def rank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        pool = list(candidates[: self.candidate_pool])
        fallback = pool[: self.output_limit]
        if not self.enabled or not pool:
            return fallback
        if any(candidate.chunk_id is None and candidate.image_id is None for candidate in pool):
            return self._fail_open(fallback, "missing_candidate_id")

        started_at = time.monotonic()
        acquired = False
        worker: asyncio.Task[Any] | None = None
        try:
            await self._acquire_permit(started_at)
            acquired = True
            remaining = self.timeout_seconds - (time.monotonic() - started_at)
            if remaining <= 0:
                raise TimeoutError
            worker = asyncio.create_task(asyncio.to_thread(self._predict, query, pool))
            scores = await asyncio.wait_for(asyncio.shield(worker), timeout=remaining)
            try:
                score_values = list(scores)
            except TypeError as exc:
                raise _RerankerFailure("score_count_mismatch") from exc
            ranked = apply_rerank_scores(pool, score_values)
            return ranked[: self.output_limit]
        except TimeoutError:
            return self._fail_open(fallback, "timeout")
        except _RerankerFailure as exc:
            return self._fail_open(fallback, exc.failure_code)
        except _ModelLoadFailure:
            return self._fail_open(fallback, "model_load_failure")
        except Exception:
            logger.exception("Reranker provider failed; using fused retrieval order")
            return self._fail_open(fallback, "provider_exception")
        finally:
            if acquired:
                if worker is not None and not worker.done():
                    worker.add_done_callback(self._release_after_worker)
                else:
                    self._semaphore.release()

    def _predict(self, query: str, candidates: Sequence[RetrievalCandidate]) -> Sequence[Any]:
        model = self._get_model()
        pairs = [[query, candidate.content] for candidate in candidates]
        return model.predict(pairs)

    async def _acquire_permit(self, started_at: float) -> None:
        """Acquire the process-wide permit without binding state to an event loop."""
        while not self._semaphore.acquire(blocking=False):
            remaining = self.timeout_seconds - (time.monotonic() - started_at)
            if remaining <= 0:
                raise TimeoutError
            await asyncio.sleep(min(0.005, remaining))

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                try:
                    self._model = self._model_loader(self.model_name)
                except Exception as exc:
                    raise _ModelLoadFailure from exc
        return self._model

    def _release_after_worker(self, worker: asyncio.Task[Any]) -> None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            worker.exception()
        self._semaphore.release()

    def _fail_open(
        self,
        fused_order: list[RetrievalCandidate],
        failure_code: str,
    ) -> list[RetrievalCandidate]:
        if self.metrics is not None:
            try:
                self.metrics.degraded("reranker", failure_code)
            except Exception:
                logger.exception("Failed to record reranker degraded metric")
        return fused_order
