"""One-attempt model-usage recorder with non-blocking, fault-tolerant persistence.

The recorder wraps a single provider call. It:

* executes the supplied ``call`` **exactly once** and never adds a provider
  retry loop -- the application owns retry boundaries (SDK-internal retries are
  disabled at every client-construction site);
* allocates the attempt number from the shared :class:`UsageOperation` so
  concurrent retries stay monotonic and operation-scoped;
* classifies the outcome (``success`` / ``timeout`` / ``cancelled`` / ``error``)
  and records it to the Task 3 :class:`ModelUsageRepository`, which owns its own
  SQLAlchemy session; the async path runs that blocking write via
  :func:`asyncio.to_thread` so it never blocks the event loop;
* on any persistence failure, enqueues a **content-free** Celery failed-write
  retry task carrying only the normalized :class:`RecordEventCommand` -- no
  prompts, responses, tool args, or raw payloads -- and if the broker enqueue
  itself fails, logs at bounded cardinality and swallows.

Regardless of persistence outcome the provider response is returned and
provider exceptions re-raise.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from app.observability.model_usage import ModelUsageMetrics, model_usage_metrics
from app.repositories.model_usage import RecordEventCommand
from app.usage.context import current_usage_context
from app.usage.normalizers import normalize_provider_usage
from app.usage.types import (
    NormalizedUsage,
    UsageContext,
    UsageOperation,
    UsageStatus,
)

logger = logging.getLogger("app.usage.model_usage_recorder")

_TIMEOUT_TYPES: tuple[type[BaseException], ...] = (asyncio.TimeoutError, TimeoutError)

EstimateCallback = Callable[[Any], NormalizedUsage]
FailedWriteEnqueue = Callable[[dict[str, Any]], None]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def classify_status(exc: BaseException) -> UsageStatus:
    """Map a provider-call exception to a bounded :data:`UsageStatus`.

    Cancellation is handled by the caller (it re-raises without classification
    here). Timeouts are recognised both by concrete type and by a
    ``timeout``-in-class-name heuristic so provider-specific timeout classes are
    covered without importing every provider SDK.
    """
    if isinstance(exc, _TIMEOUT_TYPES):
        return "timeout"
    if "timeout" in type(exc).__name__.lower():
        return "timeout"
    return "error"


def classify_error(exc: BaseException) -> str:
    """Return a bounded error code: the exception class name, never its message.

    The message can contain prompt fragments or PII, so it is never persisted;
    the class name is bounded by the set of exception types the app can raise.
    """
    return type(exc).__name__


def _default_enqueue_failed_write(payload: dict[str, Any]) -> None:
    """Enqueue the content-free failed-write retry task (lazy import)."""
    from app.workers.model_usage import retry_model_usage_write_task

    retry_model_usage_write_task.delay(payload)


class ModelUsageRecorder:
    """Record one provider attempt reliably without owning the provider's retry."""

    def __init__(
        self,
        *,
        repository: Any,
        enqueue_failed_write: FailedWriteEnqueue | None = None,
        metrics: ModelUsageMetrics | None = None,
        context_provider: Callable[[], UsageContext] = current_usage_context,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._repository = repository
        self._enqueue_failed_write = enqueue_failed_write or _default_enqueue_failed_write
        self._metrics = metrics or model_usage_metrics
        self._context_provider = context_provider
        self._clock = clock

    async def record_one_async_attempt(
        self,
        *,
        call: Callable[[], Awaitable[Any]],
        provider: str,
        model: str,
        operation: UsageOperation,
        estimate: EstimateCallback | None = None,
    ) -> Any:
        """Execute ``call`` once, record its outcome off the event loop, return it."""
        context = self._context_provider()
        attempt = operation.allocate_attempt()
        started_at = self._clock()
        started_monotonic = time.monotonic()
        try:
            response = await call()
        except asyncio.CancelledError:
            await asyncio.to_thread(
                self._persist,
                self._build_command(
                    operation=operation,
                    attempt=attempt,
                    context=context,
                    provider=provider,
                    model=model,
                    status="cancelled",
                    usage=NormalizedUsage(source="unavailable"),
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error_code=None,
                ),
            )
            raise
        except Exception as exc:
            await asyncio.to_thread(
                self._persist,
                self._build_command(
                    operation=operation,
                    attempt=attempt,
                    context=context,
                    provider=provider,
                    model=model,
                    status=classify_status(exc),
                    usage=NormalizedUsage(source="unavailable"),
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error_code=classify_error(exc),
                ),
            )
            raise
        await asyncio.to_thread(
            self._persist,
            self._build_command(
                operation=operation,
                attempt=attempt,
                context=context,
                provider=provider,
                model=model,
                status="success",
                usage=self._resolve_usage(provider, response, estimate),
                started_at=started_at,
                started_monotonic=started_monotonic,
                error_code=None,
            ),
        )
        return response

    def record_one_sync_attempt(
        self,
        *,
        call: Callable[[], Any],
        provider: str,
        model: str,
        operation: UsageOperation,
        estimate: EstimateCallback | None = None,
    ) -> Any:
        """Synchronous twin of :meth:`record_one_async_attempt` for worker threads.

        Already running on a worker thread, this persists by calling
        ``record_event`` directly (no :func:`asyncio.to_thread`). Semantics are
        identical: one call, no provider retry, same classification and fallback.
        """
        context = self._context_provider()
        attempt = operation.allocate_attempt()
        started_at = self._clock()
        started_monotonic = time.monotonic()
        try:
            response = call()
        except asyncio.CancelledError:
            self._persist(
                self._build_command(
                    operation=operation,
                    attempt=attempt,
                    context=context,
                    provider=provider,
                    model=model,
                    status="cancelled",
                    usage=NormalizedUsage(source="unavailable"),
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error_code=None,
                )
            )
            raise
        except Exception as exc:
            self._persist(
                self._build_command(
                    operation=operation,
                    attempt=attempt,
                    context=context,
                    provider=provider,
                    model=model,
                    status=classify_status(exc),
                    usage=NormalizedUsage(source="unavailable"),
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error_code=classify_error(exc),
                )
            )
            raise
        self._persist(
            self._build_command(
                operation=operation,
                attempt=attempt,
                context=context,
                provider=provider,
                model=model,
                status="success",
                usage=self._resolve_usage(provider, response, estimate),
                started_at=started_at,
                started_monotonic=started_monotonic,
                error_code=None,
            )
        )
        return response

    @staticmethod
    def _resolve_usage(
        provider: str, response: Any, estimate: EstimateCallback | None
    ) -> NormalizedUsage:
        usage = normalize_provider_usage(provider=provider, payload=response)
        if usage.source == "unavailable" and estimate is not None:
            return estimate(response)
        return usage

    def _build_command(
        self,
        *,
        operation: UsageOperation,
        attempt: int,
        context: UsageContext,
        provider: str,
        model: str,
        status: UsageStatus,
        usage: NormalizedUsage,
        started_at: datetime,
        started_monotonic: float,
        error_code: str | None,
        provider_request_id: str | None = None,
    ) -> RecordEventCommand:
        completed_at = self._clock()
        latency_ms = max(0, round((time.monotonic() - started_monotonic) * 1000))
        return RecordEventCommand(
            operation_id=operation.operation_id,
            attempt=attempt,
            context=context,
            usage=usage,
            provider=provider,
            model=model,
            status=status,
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=completed_at,
            provider_request_id=provider_request_id,
            error_code=error_code,
        )

    def begin_streaming_attempt(
        self,
        *,
        provider: str,
        model: str,
        operation: UsageOperation,
    ) -> StreamingAttemptHandle:
        """Open a handle for one streaming provider attempt.

        Unlike the one-call wrappers, a streaming provider yields usage only
        after the stream is exhausted. The handle reserves the operation-scoped
        attempt number now (so retries stay monotonic) and its start instant,
        but persists **exactly one** event only when :meth:`finalize` is called
        with the terminal status -- never at stream start.
        """
        context = self._context_provider()
        attempt = operation.allocate_attempt()
        return StreamingAttemptHandle(
            recorder=self,
            provider=provider,
            model=model,
            operation=operation,
            context=context,
            attempt=attempt,
            started_at=self._clock(),
            started_monotonic=time.monotonic(),
        )

    def _persist(self, command: RecordEventCommand) -> None:
        """Write ``command`` to the ledger; never propagate a persistence failure."""
        try:
            self._repository.record_event(command)
        except Exception as exc:
            self._handle_persist_failure(command, exc)
            return
        self._metrics.record_attempt(
            provider=command.provider,
            operation=command.context.operation,
            status=command.status,
            source=command.usage.source,
        )
        self._metrics.record_persistence("stored")

    def _handle_persist_failure(self, command: RecordEventCommand, exc: Exception) -> None:
        failure_class = type(exc).__name__
        self._metrics.record_attempt(
            provider=command.provider,
            operation=command.context.operation,
            status=command.status,
            source=command.usage.source,
        )
        try:
            self._enqueue_failed_write(serialize_record_command(command))
        except Exception as broker_exc:
            self._metrics.record_persistence("dropped", failure_class=type(broker_exc).__name__)
            logger.error(
                "model_usage ledger write dropped: persist_failure=%s enqueue_failure=%s",
                failure_class,
                type(broker_exc).__name__,
            )
            return
        self._metrics.record_persistence("retry_enqueued", failure_class=failure_class)


class StreamingAttemptHandle:
    """One streaming provider attempt, finalized exactly once.

    A streaming provider (image generation) yields its token accounting only
    after every image has been delivered, so the outcome cannot be recorded up
    front like the one-call wrappers do. The caller feeds terminal usage in via
    :meth:`set_usage`, counts delivered outputs via :meth:`note_generated_image`,
    and calls :meth:`finalize` once with the terminal status. Repeat
    ``finalize`` calls are no-ops, so a stream that raises after partial output
    still records a single event. Persistence runs off the event loop via
    :func:`asyncio.to_thread`, exactly like the async wrapper.
    """

    def __init__(
        self,
        *,
        recorder: ModelUsageRecorder,
        provider: str,
        model: str,
        operation: UsageOperation,
        context: UsageContext,
        attempt: int,
        started_at: datetime,
        started_monotonic: float,
    ) -> None:
        self._recorder = recorder
        self._provider = provider
        self._model = model
        self._operation = operation
        self._context = context
        self._attempt = attempt
        self._started_at = started_at
        self._started_monotonic = started_monotonic
        self._usage = NormalizedUsage(source="unavailable")
        self._provider_request_id: str | None = None
        self._generated_images = 0
        self._finalized = False

    def set_usage(self, usage: NormalizedUsage, *, provider_request_id: str | None = None) -> None:
        """Record the provider's terminal usage for this attempt."""
        self._usage = usage
        if provider_request_id is not None:
            self._provider_request_id = provider_request_id

    def note_generated_image(self) -> None:
        """Count one delivered image; the ledger's ``generated_images`` field."""
        self._generated_images += 1

    async def finalize(self, status: UsageStatus, *, error_code: str | None = None) -> None:
        """Persist this attempt's single event with ``status`` (idempotent)."""
        if self._finalized:
            return
        self._finalized = True
        usage = self._usage
        if self._generated_images:
            usage = replace(usage, generated_images=self._generated_images)
        await asyncio.to_thread(
            self._recorder._persist,
            self._recorder._build_command(
                operation=self._operation,
                attempt=self._attempt,
                context=self._context,
                provider=self._provider,
                model=self._model,
                status=status,
                usage=usage,
                started_at=self._started_at,
                started_monotonic=self._started_monotonic,
                error_code=error_code,
                provider_request_id=self._provider_request_id,
            ),
        )


# --- content-free (de)serialization for the failed-write retry payload -----


def _uuid_to_str(value: UUID | None) -> str | None:
    return str(value) if value is not None else None


def _str_to_uuid(value: str | None) -> UUID | None:
    return UUID(value) if value is not None else None


def serialize_record_command(command: RecordEventCommand) -> dict[str, Any]:
    """Serialize a :class:`RecordEventCommand` to JSON-safe primitives.

    The command carries no prompts, responses, tool arguments, or raw provider
    payloads -- only attempt identity, resolved dimensions, normalized token
    counts, timing, and a bounded error code -- so the resulting payload is
    safe to place on the broker.
    """
    context = command.context
    usage = command.usage
    return {
        "operation_id": str(command.operation_id),
        "attempt": command.attempt,
        "provider": command.provider,
        "model": command.model,
        "status": command.status,
        "latency_ms": command.latency_ms,
        "started_at": command.started_at.isoformat(),
        "completed_at": command.completed_at.isoformat(),
        "provider_request_id": command.provider_request_id,
        "error_code": command.error_code,
        "context": {
            "user_id": _uuid_to_str(context.user_id),
            "conversation_id": _uuid_to_str(context.conversation_id),
            "request_message_id": _uuid_to_str(context.request_message_id),
            "document_id": _uuid_to_str(context.document_id),
            "correlation_id": context.correlation_id,
            "langsmith_run_id": _uuid_to_str(context.langsmith_run_id),
            "operation": context.operation,
            "agent_id": context.agent_id,
        },
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "input_text_tokens": usage.input_text_tokens,
            "input_image_tokens": usage.input_image_tokens,
            "output_text_tokens": usage.output_text_tokens,
            "output_image_tokens": usage.output_image_tokens,
            "generated_images": usage.generated_images,
            "source": usage.source,
        },
    }


def deserialize_record_command(payload: dict[str, Any]) -> RecordEventCommand:
    """Rebuild the exact :class:`RecordEventCommand` from a serialized payload.

    The rebuilt command yields the same ``event_key``, so replaying it through
    ``record_event`` is idempotent (``ON CONFLICT DO NOTHING`` on ``event_key``).
    """
    context_data = payload["context"]
    usage_data = payload["usage"]
    return RecordEventCommand(
        operation_id=UUID(payload["operation_id"]),
        attempt=int(payload["attempt"]),
        context=UsageContext(
            user_id=_str_to_uuid(context_data["user_id"]),
            conversation_id=_str_to_uuid(context_data["conversation_id"]),
            request_message_id=_str_to_uuid(context_data["request_message_id"]),
            document_id=_str_to_uuid(context_data["document_id"]),
            correlation_id=context_data["correlation_id"],
            langsmith_run_id=_str_to_uuid(context_data["langsmith_run_id"]),
            operation=context_data["operation"],
            agent_id=context_data["agent_id"],
        ),
        usage=NormalizedUsage(
            input_tokens=usage_data["input_tokens"],
            output_tokens=usage_data["output_tokens"],
            total_tokens=usage_data["total_tokens"],
            reasoning_tokens=usage_data["reasoning_tokens"],
            cached_input_tokens=usage_data["cached_input_tokens"],
            input_text_tokens=usage_data["input_text_tokens"],
            input_image_tokens=usage_data["input_image_tokens"],
            output_text_tokens=usage_data["output_text_tokens"],
            output_image_tokens=usage_data["output_image_tokens"],
            generated_images=usage_data["generated_images"],
            source=usage_data["source"],
        ),
        provider=payload["provider"],
        model=payload["model"],
        status=payload["status"],
        latency_ms=int(payload["latency_ms"]),
        started_at=datetime.fromisoformat(payload["started_at"]),
        completed_at=datetime.fromisoformat(payload["completed_at"]),
        provider_request_id=payload["provider_request_id"],
        error_code=payload["error_code"],
    )
