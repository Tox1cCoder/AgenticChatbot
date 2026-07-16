"""Celery orchestration for durable conversation compaction jobs."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Any, NamedTuple
from uuid import UUID

from app.ai.conversation_compactor import (
    CompactionCredentialResolver,
    ConversationCompactor,
)
from app.ai.conversation_memory import ConversationMemory
from app.ai.model_context import resolve_model_context_window
from app.ai.model_factory import ModelFactory
from app.ai.token_counter import TokenCounter
from app.core.config import settings
from app.observability.conversation_compaction import conversation_compaction_metrics
from app.repositories.conversation_compaction import ConversationCompactionRepository
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

_PERMANENT_ERRORS = {
    "credential_unavailable",
    "invalid_json",
    "invalid_memory",
    "empty_memory",
    "memory_over_budget",
    "ownership_invalid",
    "invariant_invalid",
    "compaction_input_over_budget",
}
_SKIPPED_ERRORS = {"not_triggered", "no_complete_turn"}


class WorkerOutcome(NamedTuple):
    status: str
    code: str


def compute_retry_delay(
    attempt_count: int,
    *,
    base_seconds: int,
    max_seconds: int,
    jitter=random.uniform,
) -> float:
    """Return capped exponential delay with bounded positive jitter."""
    exponent = max(0, int(attempt_count))
    raw = min(float(max_seconds), float(base_seconds) * (2**exponent))
    jittered = raw + float(jitter(0, raw * 0.25))
    return min(float(max_seconds), jittered)


class ConversationCompactionWorker:
    """Execute one lease-scoped compaction without holding database locks."""

    def __init__(
        self,
        *,
        repository: ConversationCompactionRepository,
        compactor: ConversationCompactor,
        settings: Any,
        jitter=random.uniform,
    ) -> None:
        self.repository = repository
        self.compactor = compactor
        self.settings = settings
        self.jitter = jitter
        self.metrics = conversation_compaction_metrics

    async def execute(self, conversation_id: UUID, *, force: bool = False) -> WorkerOutcome:
        started = time.perf_counter()
        claim = self.repository.claim_job(
            conversation_id,
            lease_seconds=self.settings.conversation_summary_lease_seconds,
        )
        if claim is None:
            self.metrics.record_job_outcome("skipped", error_code="not_claimed")
            return WorkerOutcome("noop", "not_claimed")

        compaction_input = self.repository.load_compaction_input(claim)
        if compaction_input is None:
            self.repository.fail_claim(
                claim,
                error_code="ownership_invalid",
                permanent=True,
                max_attempts=self.settings.conversation_summary_max_attempts,
            )
            self.metrics.record_job_outcome("dead", error_code="ownership_invalid")
            return WorkerOutcome("dead", "ownership_invalid")

        previous_memory = self._previous_memory(compaction_input.summary_payload)
        mode = "emergency" if force else "background"
        try:
            result = await asyncio.wait_for(
                self.compactor.compact(
                    compaction_input.messages,
                    previous_memory=previous_memory,
                    user_id=compaction_input.owner_id,
                    force=force,
                ),
                timeout=self.settings.conversation_summary_timeout_seconds,
            )
        except TimeoutError:
            self.metrics.record_compaction(
                mode=mode,
                outcome="timeout",
                provider=self.compactor.provider,
                model=self.compactor.model,
                content_class="text",
                input_tokens=0,
                output_tokens=0,
                duration_seconds=time.perf_counter() - started,
            )
            return self._retry(claim, "provider_timeout")

        self.metrics.record_compaction(
            mode=mode,
            outcome="success" if result.success else "failure",
            provider=self.compactor.provider,
            model=self.compactor.model,
            content_class="text",
            input_tokens=result.input_token_count,
            output_tokens=result.summary_token_count,
            duration_seconds=time.perf_counter() - started,
            cost_amount=result.reported_cost_amount,
            cost_currency=result.reported_cost_currency,
        )
        if result.reported_input_tokens is not None:
            self.metrics.record_token_calibration(
                provider=self.compactor.provider,
                model=self.compactor.model,
                content_class="text",
                estimated_tokens=result.input_token_count,
                actual_tokens=result.reported_input_tokens,
            )

        if not result.success:
            error_code = result.error_code or "generation_failed"
            if error_code in _SKIPPED_ERRORS:
                status = self.repository.complete_claim(claim)
                self.metrics.record_job_outcome("skipped", error_code=error_code)
                return WorkerOutcome("skipped", status or error_code)
            if error_code in _PERMANENT_ERRORS:
                self.repository.fail_claim(
                    claim,
                    error_code=error_code,
                    permanent=True,
                    max_attempts=self.settings.conversation_summary_max_attempts,
                )
                self.metrics.record_job_outcome("dead", error_code=error_code)
                return WorkerOutcome("dead", error_code)
            return self._retry(claim, error_code)

        persisted = self.repository.persist_memory_cas(
            claim,
            base_summary_version=compaction_input.summary_version,
            base_cursor=compaction_input.last_summarized_sequence,
            summary_payload=result.memory.model_dump(),
            summary_schema_version=1,
            last_summarized_sequence=result.last_summarized_sequence,
            source_message_count=len(result.selection.compactable_prefix),
            source_token_count=result.trigger.token_count,
            summary_token_count=result.summary_token_count,
            provider=self.compactor.provider,
            model=self.compactor.model,
            tokenizer=result.trigger.token_strategy,
            prompt_version=self.compactor.prompt_version,
        )
        if not persisted:
            self._retry(claim, "cas_conflict", record_outcome=False)
            self.metrics.record_job_outcome("conflict", error_code="cas_conflict")
            return WorkerOutcome("conflict", "cas_conflict")
        status = self.repository.complete_claim(claim)
        self.metrics.record_job_outcome("success", error_code=None)
        return WorkerOutcome("completed", status or "lease_lost")

    def _retry(
        self,
        claim,
        error_code: str,
        *,
        record_outcome: bool = True,
    ) -> WorkerOutcome:
        delay = compute_retry_delay(
            claim.attempt_count,
            base_seconds=self.settings.conversation_summary_retry_base_seconds,
            max_seconds=self.settings.conversation_summary_retry_max_seconds,
            jitter=self.jitter,
        )
        retry_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
        self.repository.fail_claim(
            claim,
            error_code=error_code,
            permanent=False,
            retry_at=retry_at,
            max_attempts=self.settings.conversation_summary_max_attempts,
        )
        if claim.attempt_count + 1 >= self.settings.conversation_summary_max_attempts:
            if record_outcome:
                self.metrics.record_job_outcome("dead", error_code=error_code)
            return WorkerOutcome("dead", error_code)
        if record_outcome:
            self.metrics.record_job_outcome("retry", error_code=error_code)
        return WorkerOutcome("retry", error_code)

    @staticmethod
    def _previous_memory(payload: dict[str, Any] | None) -> ConversationMemory | None:
        if not payload:
            return None
        try:
            return ConversationMemory.model_validate(payload)
        except Exception:
            return None


def dispatch_due_jobs(
    repository: ConversationCompactionRepository,
    publisher,
    *,
    debounce_seconds: int,
    limit: int,
) -> int:
    """Reserve due rows before publishing content-free task notifications."""
    conversation_ids = repository.reconcile_due_jobs(
        limit=limit,
        dispatch_debounce_seconds=debounce_seconds,
    )
    published = 0
    for conversation_id in conversation_ids:
        try:
            publisher(conversation_id)
        except Exception:
            logger.warning("Summary task publication failed code=broker_publish_failed")
        else:
            published += 1
    return published


def request_historical_backfill(
    repository: ConversationCompactionRepository,
    publisher,
    *,
    batch_size: int,
) -> int:
    """Request and publish one bounded idempotent historical batch."""
    published = 0
    for conversation_id, target in repository.list_backfill_candidates(limit=batch_size):
        if not repository.request_backfill(conversation_id, target):
            continue
        try:
            publisher(str(conversation_id))
        except Exception:
            logger.warning("Summary backfill publication failed code=broker_publish_failed")
        else:
            published += 1
    return published


async def _langchain_generate(
    *,
    prompt: str,
    provider: str,
    model: str,
    api_key: str | None,
    **_kwargs,
) -> Any:
    if not api_key:
        raise ValueError("credential_unavailable")
    client = ModelFactory.create_model(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=0,
        timeout=settings.conversation_summary_timeout_seconds,
    )
    return await client.ainvoke(prompt)


def _build_worker() -> ConversationCompactionWorker:
    from app.core.container import get_container

    container = get_container()
    repository = container.conversation_compaction_repository()
    provider_service = container.provider_service()
    resolver = CompactionCredentialResolver(
        provider=settings.conversation_summary_provider,
        server_credentials={"gemini": settings.gemini_api_key},
        user_credential_resolver=provider_service.resolve_provider_credentials,
        allow_user_credentials=True,
    )
    context_window = resolve_model_context_window(
        settings.conversation_summary_provider,
        settings.conversation_summary_model,
    )
    max_compaction_input = None
    if context_window.max_input_tokens is not None:
        max_compaction_input = (
            context_window.max_input_tokens
            - settings.conversation_summary_default_reserved_output_tokens
            - settings.conversation_summary_safety_margin_tokens
        )
        if max_compaction_input <= 0:
            raise ValueError("conversation_summary_available_input_must_be_positive")
    compactor = ConversationCompactor(
        token_counter=TokenCounter(),
        generator=_langchain_generate,
        provider=settings.conversation_summary_provider,
        model=settings.conversation_summary_model,
        trigger_messages=settings.conversation_summary_trigger_messages,
        trigger_tokens=settings.conversation_summary_trigger_tokens,
        keep_recent_turns=settings.conversation_summary_keep_recent_turns,
        max_summary_tokens=settings.conversation_summary_max_tokens,
        max_input_tokens=max_compaction_input,
        credential_resolver=resolver,
    )
    return ConversationCompactionWorker(
        repository=repository,
        compactor=compactor,
        settings=settings,
    )


@celery_app.task(
    name="app.workers.conversation_compaction.compact_conversation_task",
    ignore_result=True,
)
def compact_conversation_task(conversation_id: str) -> dict[str, str]:
    """Process one content-free conversation notification."""
    if not settings.conversation_summary_enabled:
        return WorkerOutcome("disabled", "compaction_disabled")._asdict()
    try:
        parsed_id = UUID(str(conversation_id))
    except (TypeError, ValueError):
        return WorkerOutcome("dead", "invalid_conversation_id")._asdict()
    outcome = asyncio.run(_build_worker().execute(parsed_id))
    return outcome._asdict()


async def run_conversation_compaction_now(
    conversation_id: UUID,
    *,
    force: bool = False,
) -> WorkerOutcome:
    if not settings.conversation_summary_enabled:
        return WorkerOutcome("disabled", "compaction_disabled")
    return await _build_worker().execute(conversation_id, force=force)


@celery_app.task(
    name="app.workers.conversation_compaction.compact_backfill_conversation_task",
    ignore_result=True,
    rate_limit="10/m",
)
def compact_backfill_conversation_task(conversation_id: str) -> dict[str, str]:
    """Process one rate-limited historical backfill notification."""
    return compact_conversation_task(conversation_id)


def publish_conversation_compaction(conversation_id: UUID) -> None:
    """Publish a content-free hint after message/job commit."""
    if settings.conversation_summary_enabled:
        compact_conversation_task.delay(str(conversation_id))


@celery_app.task(
    name="app.workers.conversation_compaction.reconcile_conversation_summaries_task",
    ignore_result=True,
)
def reconcile_conversation_summaries_task(limit: int = 100) -> int:
    if not settings.conversation_summary_enabled:
        return 0
    worker = _build_worker()
    return dispatch_due_jobs(
        worker.repository,
        lambda conversation_id: compact_conversation_task.delay(str(conversation_id)),
        debounce_seconds=settings.conversation_summary_reconcile_seconds,
        limit=limit,
    )


@celery_app.task(
    name="app.workers.conversation_compaction.backfill_conversation_summaries_task",
    ignore_result=True,
)
def backfill_conversation_summaries_task(batch_size: int = 100) -> int:
    if not settings.conversation_summary_enabled:
        return 0
    worker = _build_worker()
    return request_historical_backfill(
        worker.repository,
        compact_backfill_conversation_task.delay,
        batch_size=batch_size,
    )
