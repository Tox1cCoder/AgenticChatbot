"""Canonical v3 → Vercel AI SDK UI Message Stream adapter.

Consumes canonical :class:`V3StreamEvent` events and emits the AI SDK
UI Message Stream SSE chunks assistant-ui expects. The response header
``x-vercel-ai-ui-message-stream: v1`` is set by the endpoint that wraps this.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections.abc import AsyncGenerator, Callable
from typing import Any

from app.core.config import settings
from app.core.exceptions import CustomHTTPException
from app.core.rich_response import select_transient_upsert_items
from app.models.generation import GenerationStatus
from app.observability.streaming import StreamingMetrics, streaming_metrics

from .ai_sdk_projection import (
    VisualMarkerStreamFilter,
    attach_image_parts_to_message,
    clean_tool_output,
    coerce_json_object,
    find_message_metadata,
    is_v1_rich_items_message,
    project_ai_sdk_message_event,
    project_ai_sdk_message_for_capability,
    visible_image_file_parts,
)
from .events import (
    IMAGE_PREVIEW_STATUS_SKIPPED,
    SUBAGENT_PHASE_BY_EVENT,
    V3StreamEvent,
    apply_inline_preview_wire_budget,
    make_event,
    resolve_image_preview_delivery,
)

logger = logging.getLogger(__name__)

_AI_SDK_HEARTBEAT_INTERVAL_SECONDS = 15.0

#: Sent when the service stream ends without a terminal outcome: an upstream
#: EOF, or a Stop (which ends the stream without one). The client recovers the
#: saved text and the generation status; the adapter never invents either.
STREAM_INCOMPLETE_ERROR_CODE = "stream_incomplete"
_STREAM_INCOMPLETE_ERROR_TEXT = "The response ended before it was complete."
STREAM_EVENT_TOO_LARGE_ERROR_CODE = "stream_event_too_large"

#: Published generation statuses that end an epoch at a Stop. ``stop_requested``
#: is deliberately absent: it is pending, and the client polls for the outcome.
_STOP_STATUSES = frozenset(
    {GenerationStatus.STOPPED.value, GenerationStatus.COMPLETED_PARTIAL.value}
)

#: Canonical lifecycle event -> the ``phase`` on the shared ``data-generation``
#: part. `run_start` projects as ``start`` rather than earning a second
#: canonical event that would carry the identical payload.
_GENERATION_LIFECYCLE_PHASES = {
    "run_start": "start",
    "generation_status": "status",
    "continuation_available": "continuation_available",
}

#: Internal fields of the pause event that never cross to a client: the
#: validated text already arrived as deltas and lives in the persisted message,
#: while the budget and the checkpoint thread are bookkeeping and a resume
#: handle respectively.
_PRIVATE_PAUSE_FIELDS = frozenset(
    {"validated_content", "budget", "thread_id", "type", "active_agent_id"}
)


_HEARTBEAT = object()

#: Frames that put answer text in front of the reader. ``_sse`` writes compact
#: JSON with ``type`` first, so a prefix identifies them without parsing.
_ANSWER_FRAME_PREFIXES = (
    'data: {"type":"text-delta"',
    'data: {"type":"data-answer-preview","data":{"operation":"replace"',
    'data: {"type":"data-answer-preview","data":{"operation":"append"',
)


async def _drain_one_at_a_time(
    queue: asyncio.Queue[Any],
    producer: asyncio.Task[None],
    interval: float,
) -> AsyncGenerator[Any, None]:
    """Yield queued items in order, or ``_HEARTBEAT`` after ``interval`` of silence.

    Ends once the producer has finished and everything it queued is delivered.
    Completion is read from the task itself, never from a sentinel the producer
    would have to enqueue -- into a full queue, from ``finally``, possibly while
    being cancelled.
    """
    getter: asyncio.Future[Any] | None = None
    try:
        while True:
            if getter is None:
                if not queue.empty():
                    yield queue.get_nowait()
                    continue
                if producer.done():
                    return
                getter = asyncio.ensure_future(queue.get())
            waiting = {getter} if producer.done() else {getter, producer}
            await asyncio.wait(waiting, timeout=interval, return_when=asyncio.FIRST_COMPLETED)
            if getter.done():
                item = getter.result()
                getter = None
                yield item
            elif producer.done():
                if queue.empty():
                    return
            else:
                yield _HEARTBEAT
    finally:
        if getter is not None:
            getter.cancel()
            # Let it finish unwinding, so no task outlives the stream.
            with contextlib.suppress(asyncio.CancelledError):
                await getter


def _frame_exceeds(frame: str, limit: int) -> bool:
    """Whether ``frame`` is over ``limit`` UTF-8 bytes; encodes only when it could be."""
    return len(frame) * 4 > limit and len(frame.encode("utf-8")) > limit


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _ai_sdk_error_frames(
    error_text: str,
    *,
    status_code: int | None = None,
    error_code: str | None = None,
) -> list[str]:
    """Frame an error for the AI SDK: optional metadata, then the strict error.

    The SDK's ``error`` chunk is a strict object carrying only ``errorText``, so
    the HTTP status and domain code travel in a preceding transient
    ``data-stream-error`` part. Metadata goes first because some consumers stop
    processing at ``error``.
    """
    frames: list[str] = []
    metadata: dict[str, Any] = {}
    if status_code is not None:
        metadata["statusCode"] = status_code
    if error_code is not None:
        metadata["errorCode"] = error_code
    if metadata:
        frames.append(_sse({"type": "data-stream-error", "data": metadata, "transient": True}))
    frames.append(_sse({"type": "error", "errorText": error_text}))
    return frames


def _ai_sdk_error_frames_from_exception(exc: Exception) -> list[str]:
    """Project a server exception to AI SDK error frames.

    Only a ``CustomHTTPException`` detail is written for the client; any other
    exception's text (provider URLs, SQL, paths) is logged and replaced by its
    type name.
    """
    if isinstance(exc, CustomHTTPException):
        return _ai_sdk_error_frames(
            str(exc.detail),
            status_code=exc.status_code,
            error_code=exc.error_code,
        )
    logger.error("AI SDK stream failed", exc_info=exc)
    return _ai_sdk_error_frames(f"Response generation failed ({type(exc).__name__}).")


def _ai_sdk_error_frames_from_data(data: dict[str, Any]) -> list[str]:
    """Project known V3 error metadata to AI SDK error frames."""
    return _ai_sdk_error_frames(
        str(data.get("error") or data.get("message") or ""),
        status_code=data.get("status_code"),
        error_code=data.get("error_code"),
    )


_VISUAL_MARKER_RE = re.compile(r"<!--rich:visual:[^\n]*?-->")


def _overlay_text(state: AISDKV6StreamState, text: str) -> str:
    """Preview overlay text without visual markers the client cannot render.

    An overlay is replaced wholesale by the terminal answer, so complete
    markers are simply dropped rather than held across chunks.
    """
    if state.answer_visuals_v1:
        return text
    return _VISUAL_MARKER_RE.sub("", text)


class AISDKV6StreamState:
    """Streaming state carried across an AI SDK UI message stream response."""

    def __init__(
        self,
        message_id: str,
        text_id: str,
        reasoning_id: str,
        *,
        inline_rich_response_v1: bool = False,
        stream_web_answer_v1: bool = False,
        answer_visuals_v1: bool = False,
        source_figures_v1: bool = False,
        operation: str = "unknown",
    ) -> None:
        self.message_id = message_id
        # chat | resume | continue: a metrics label, never sent to the client.
        self.operation = operation
        self.text_id = text_id
        self.reasoning_id = reasoning_id
        self.text_started = False
        self.text_ended = False
        self.reasoning_started = False
        self.reasoning_ended = False
        # complete | interrupt | paused | stopped | error; None while unresolved. A
        # presentation fact only: durable generation status is the service's.
        self.outcome: str | None = None
        self.closed = False
        self.any_text_delta = False
        # Legacy preview overlay (client without streamWebAnswerV1).
        self.preview_active = False
        self.stream_web_answer_v1 = bool(stream_web_answer_v1)
        self.web_answer_streamed = False
        # The draft as the SDK holds it, to tell whether reconciliation changed it.
        self.streamed_draft = ""
        self.web_repair_mode = False
        # Repair overlay shown over the streamed draft (opted-in clients).
        self.repair_preview_active = False
        self.web_reconciled = False
        # Set once provisional text has been retired; later previews are stale.
        self.answer_settled = False
        self.tool_seq = 0
        self.pending_tool_call_ids: list[str] = []
        self.inline_rich_response_v1 = bool(inline_rich_response_v1)
        # Effective: already requires the base contract and its rollout gate.
        self.answer_visuals_v1 = bool(answer_visuals_v1 and inline_rich_response_v1)
        # A figure needs no stream filter: without the flag a rich client still
        # renders its marker, as a legacy image, once the terminal registry lands.
        self.source_figures_v1 = bool(source_figures_v1 and inline_rich_response_v1)
        # A client without visuals never sees their markers; when one is
        # withheld, the terminal answer replaces the draft with the fallback.
        self.visual_marker_filter = None if self.answer_visuals_v1 else VisualMarkerStreamFilter()

    @property
    def visual_markers_withheld(self) -> bool:
        return bool(self.visual_marker_filter and self.visual_marker_filter.withheld)

    def answer_text(self, text: str) -> str:
        """The part of streamed answer text this client may see now."""
        if self.visual_marker_filter is None:
            return text
        return self.visual_marker_filter.feed(text)


class AISDKV6StreamAdapter:
    def __init__(
        self,
        event_source_factory: Callable[[], AsyncGenerator[V3StreamEvent, None]],
        state: AISDKV6StreamState,
        *,
        heartbeat_interval_seconds: float | None = None,
        metrics: StreamingMetrics | None = None,
    ) -> None:
        self._event_source_factory = event_source_factory
        self._state = state
        self._metrics = metrics or streaming_metrics
        self._started_at = 0.0
        self._first_answer_recorded = False
        self._heartbeat_interval = (
            heartbeat_interval_seconds
            if heartbeat_interval_seconds is not None
            else _AI_SDK_HEARTBEAT_INTERVAL_SECONDS
        )

    # -- public API --------------------------------------------------------

    async def iter_sse(self) -> AsyncGenerator[str, None]:
        state = self._state
        self._started_at = time.monotonic()
        try:
            # aclosing: closing this generator must close the source before it
            # returns, not leave the inner generator to the GC finalizer.
            async with contextlib.aclosing(self._iter_frames()) as frames:
                async for frame in frames:
                    yield frame
        finally:
            # Recorded once per response. Not closed means the client went away
            # (or we were cancelled) before the stream finished.
            outcome = (state.outcome or "unknown") if state.closed else "disconnect"
            self._metrics.record_end(operation=state.operation, outcome=outcome)

    async def _iter_frames(self) -> AsyncGenerator[str, None]:
        state = self._state
        try:
            yield _sse({"type": "start", "messageId": state.message_id})
            yield _sse({"type": "start-step"})
            yield _sse({"type": "text-start", "id": state.text_id})
            state.text_started = True

            # aclosing: an outer aclose() or cancellation closes every nested
            # generator -- and through them the source -- before it returns.
            async with contextlib.aclosing(self._events_with_heartbeats()) as events:
                async for event in events:
                    received_at = time.monotonic()
                    async with contextlib.aclosing(self._map_event(event)) as frames:
                        async for chunk in frames:
                            frame = self._within_ceiling(chunk)
                            self._observe_first_answer(frame, received_at)
                            yield frame
                    if event.type in {"complete", "interrupt", "error"}:
                        break

            for chunk in self._terminate():
                yield self._within_ceiling(chunk)
        except Exception as exc:
            if state.closed:
                logger.error("AI SDK stream failed after it was closed", exc_info=exc)
                return
            state.outcome = "error"
            for frame in self._settle_answer(None):
                yield frame
            for frame in _ai_sdk_error_frames_from_exception(exc):
                yield frame
            for frame in self._close(finish_reason="error"):
                yield frame
        # CancelledError and GeneratorExit propagate: a disconnected client gets
        # no further frames, and a socket close is never a lifecycle outcome.

    def _observe_first_answer(self, frame: str, received_at: float) -> None:
        """Time the first frame that shows the reader answer text."""
        if self._first_answer_recorded or not frame.startswith(_ANSWER_FRAME_PREFIXES):
            return
        self._first_answer_recorded = True
        now = time.monotonic()
        operation = self._state.operation
        self._metrics.record_first_answer(now - self._started_at, operation=operation)
        self._metrics.record_delta_to_wire(now - received_at, operation=operation)

    @staticmethod
    def _within_ceiling(frame: str) -> str:
        """Refuse a frame over the wire ceiling rather than write or truncate it."""
        if _frame_exceeds(frame, settings.stream_max_event_bytes):
            raise CustomHTTPException(
                413,
                "A streamed event exceeded the maximum size.",
                STREAM_EVENT_TOO_LARGE_ERROR_CODE,
            )
        return frame

    # -- provisional answer text ------------------------------------------

    def _preview_clear_frames(self) -> list[str]:
        state = self._state
        if not (state.preview_active or state.repair_preview_active):
            return []
        state.preview_active = False
        state.repair_preview_active = False
        return [
            _sse({"type": "data-answer-preview", "data": {"operation": "clear"}, "transient": True})
        ]

    def _settle_answer(self, content: str | None) -> list[str]:
        """Retire provisional answer text: the one path every terminal takes.

        Clears any preview overlay, then -- once per stream, and only when a
        draft went out as ``text-delta`` -- replaces the draft with ``content``,
        the persisted answer. ``None`` means nothing authoritative was supplied:
        the draft is discarded (``""``) and the client recovers the saved text
        from history or generation status. Idempotent.
        """
        state = self._state
        state.answer_settled = True
        frames = self._preview_clear_frames()
        held = state.visual_marker_filter.flush() if state.visual_marker_filter else ""
        if held and not state.text_ended:
            # Text that only looked like the start of a visual marker.
            state.any_text_delta = True
            frames.append(_sse({"type": "text-delta", "id": state.text_id, "delta": held}))
        needs_reconcile = state.web_answer_streamed or state.visual_markers_withheld
        if needs_reconcile and not state.web_reconciled:
            state.web_reconciled = True
            self._metrics.record_reconcile(changed=(content or "") != state.streamed_draft)
            frames.append(
                _sse(
                    {
                        "type": "data-answer-reconcile",
                        "data": {"messageId": state.message_id, "content": content or ""},
                        "transient": True,
                    }
                )
            )
        return frames

    def _authoritative_content(self, message: dict[str, Any]) -> str:
        """A persisted message's text exactly as history will project it."""
        projected = project_ai_sdk_message_for_capability(
            message,
            inline_rich_response_v1=self._state.inline_rich_response_v1,
            answer_visuals_v1=self._state.answer_visuals_v1,
            source_figures_v1=self._state.source_figures_v1,
        )
        return str(projected.get("content") or "")

    async def _saved_partial(self, message: dict[str, Any]) -> AsyncGenerator[str, None]:
        """A paused turn's persisted partial replaces whatever was provisional."""
        state = self._state
        content = self._authoritative_content(message)
        for frame in self._settle_answer(content):
            yield frame
        if not state.any_text_delta and content.strip():
            # Web previews are not text: a client without the draft capability
            # would otherwise see the validated partial only after a refetch.
            state.any_text_delta = True
            yield _sse({"type": "text-delta", "id": state.text_id, "delta": content})

    # -- terminal chunks ---------------------------------------------------

    def _terminate(self) -> list[str]:
        """Close a stream whose event source ended, by outcome."""
        state = self._state
        if state.closed:
            return []
        if state.outcome is None:
            # No terminal event arrived: truncation must not look like success.
            return [
                *self._settle_answer(None),
                *_ai_sdk_error_frames(
                    _STREAM_INCOMPLETE_ERROR_TEXT,
                    error_code=STREAM_INCOMPLETE_ERROR_CODE,
                ),
                *self._close(finish_reason="error"),
            ]
        finish_reason = "error" if state.outcome == "error" else None
        return [*self._settle_answer(None), *self._close(finish_reason=finish_reason)]

    def _end_blocks(self) -> list[str]:
        """End the open text and reasoning blocks, each exactly once."""
        state = self._state
        frames: list[str] = []
        if state.text_started and not state.text_ended:
            state.text_ended = True
            frames.append(_sse({"type": "text-end", "id": state.text_id}))
        if state.reasoning_started and not state.reasoning_ended:
            state.reasoning_ended = True
            frames.append(_sse({"type": "reasoning-end", "id": state.reasoning_id}))
        return frames

    def _close(self, *, finish_reason: str | None) -> list[str]:
        """The one closing sequence: blocks, finish-step, finish, [DONE]."""
        state = self._state
        if state.closed:
            return []
        state.closed = True
        finish: dict[str, Any] = {"type": "finish"}
        if finish_reason is not None:
            finish["finishReason"] = finish_reason
        return [
            *self._end_blocks(),
            _sse({"type": "finish-step"}),
            _sse(finish),
            "data: [DONE]\n\n",
        ]

    # -- heartbeat wrapper -------------------------------------------------

    async def _events_with_heartbeats(
        self,
    ) -> AsyncGenerator[V3StreamEvent, None]:
        """Pump the source in its own task, holding at most one event.

        The source is created, iterated, and closed in the producer task, so a
        ContextVar it binds is reset in the task that set it. A slow reader
        makes the producer wait in ``put`` instead of buffering the answer.
        """
        queue: asyncio.Queue[V3StreamEvent] = asyncio.Queue(maxsize=1)

        async def produce_events() -> None:
            source = self._event_source_factory()
            try:
                async for event in source:
                    await queue.put(event)
            finally:
                aclose = getattr(source, "aclose", None)
                if callable(aclose):
                    with contextlib.suppress(Exception):
                        await aclose()

        producer = asyncio.create_task(produce_events())
        try:
            drained = _drain_one_at_a_time(queue, producer, self._heartbeat_interval)
            async with contextlib.aclosing(drained) as items:
                async for item in items:
                    yield make_event("heartbeat", sequence=0) if item is _HEARTBEAT else item
            # A source error surfaces once, after every event it queued. A
            # cancelled producer is a Stop (the registry cancels the task the
            # service runs in), not this consumer's cancellation: the stream
            # ends as unresolved and _terminate reports it.
            if not producer.cancelled():
                producer.result()
        finally:
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await producer

    # -- canonical event mapping ------------------------------------------

    async def _map_event(self, event: V3StreamEvent) -> AsyncGenerator[str, None]:
        state = self._state
        etype = event.type
        data = event.data or {}

        if etype == "answer_preview" and state.answer_settled:
            return  # a preview after the authoritative answer is stale

        if state.stream_web_answer_v1 and etype == "answer_preview":
            operation = data.get("operation")
            if operation == "clear":
                if state.web_answer_streamed:
                    state.web_repair_mode = True
                    state.repair_preview_active = False
                    yield _sse(
                        {
                            "type": "data-answer-preview",
                            "data": {"operation": "clear"},
                            "transient": True,
                        }
                    )
                return
            chunk = data.get("text")
            if operation not in {"replace", "append"} or not isinstance(chunk, str) or not chunk:
                return
            if operation == "replace" and state.web_answer_streamed:
                state.web_repair_mode = True
            if state.web_repair_mode:
                state.repair_preview_active = True
                yield _sse(
                    {
                        "type": "data-answer-preview",
                        "data": {"operation": operation, "text": _overlay_text(state, chunk)},
                        "transient": True,
                    }
                )
            else:
                state.web_answer_streamed = True
                visible = state.answer_text(chunk)
                if visible:
                    state.any_text_delta = True
                    state.streamed_draft += visible
                    yield _sse({"type": "text-delta", "id": state.text_id, "delta": visible})
            return

        if state.stream_web_answer_v1 and state.web_answer_streamed and etype == "message_delta":
            # The graph replays the fully grounded answer after the model run.
            # Completion carries it as a replacement, so this is not appended.
            return

        if state.preview_active and etype == "message_delta":
            # Authoritative text has started; the provisional overlay goes.
            for frame in self._preview_clear_frames():
                yield frame

        if etype == "message_end":
            message = data.get("message")
            if isinstance(message, dict):
                # Only a budget pause publishes its persisted partial this way.
                state.outcome = "paused"
                async for chunk in self._saved_partial(message):
                    yield chunk
            return

        if etype == "answer_preview":
            operation = data.get("operation")
            if operation == "clear":
                if state.preview_active:
                    state.preview_active = False
                    yield _sse(
                        {
                            "type": "data-answer-preview",
                            "data": {"operation": "clear"},
                            "transient": True,
                        }
                    )
                return
            if operation not in {"replace", "append"}:
                return
            text = data.get("text")
            if not isinstance(text, str) or not text:
                return
            state.preview_active = True
            yield _sse(
                {
                    "type": "data-answer-preview",
                    "data": {"operation": operation, "text": _overlay_text(state, text)},
                    "transient": True,
                }
            )
            return

        if etype == "message_delta":
            delta = state.answer_text(data.get("text") or "")
            if delta:
                state.any_text_delta = True
                yield _sse({"type": "text-delta", "id": state.text_id, "delta": delta})
            return

        if etype == "reasoning_delta":
            delta = data.get("text") or ""
            if not delta:
                return
            if not state.reasoning_started:
                state.reasoning_started = True
                yield _sse({"type": "reasoning-start", "id": state.reasoning_id})
            yield _sse({"type": "reasoning-delta", "id": state.reasoning_id, "delta": delta})
            return

        if etype == "tool_call_available":
            async for chunk in self._tool_input(event):
                yield chunk
            return

        if etype == "tool_execution_end":
            async for chunk in self._tool_output(event):
                yield chunk
            return

        if etype == "rich_items":
            async for chunk in self._rich_items(data):
                yield chunk
            return

        if etype == "sources":
            for source in data.get("sources") or []:
                if not isinstance(source, dict) or not source.get("url"):
                    continue
                yield _sse(
                    {
                        "type": "source-url",
                        "sourceId": str(source.get("source_id") or source["url"]),
                        "url": str(source["url"]),
                        "title": str(source.get("title") or source["url"]),
                    }
                )
            return

        if etype == "image_preview":
            async for chunk in self._image_preview(data):
                yield chunk
            return

        if etype == "agent_selected":
            yield _sse(
                {
                    "type": "data-agent-selected",
                    "data": {"agent": event.agent or data.get("agent")},
                    "transient": True,
                }
            )
            return

        if etype in _GENERATION_LIFECYCLE_PHASES:
            if etype == "continuation_available":
                state.outcome = "paused"
            elif etype == "generation_status" and data.get("status") in _STOP_STATUSES:
                # The service's own word that the epoch ended at a Stop.
                state.outcome = "stopped"
            # One custom part for all three lifecycle events, discriminated by
            # `phase`, so a client subscribes once. Not `transient`: a client
            # that reconnects still needs the identity and the version, which
            # are what make a Stop or Continue addressable at all.
            yield _sse(
                {
                    "type": "data-generation",
                    "id": str(data.get("generation_id") or ""),
                    "data": {
                        "phase": _GENERATION_LIFECYCLE_PHASES[etype],
                        **{
                            key: value
                            for key, value in data.items()
                            if key not in _PRIVATE_PAUSE_FIELDS
                        },
                    },
                }
            )
            return

        if etype == "user_message_created":
            message = data.get("message")
            if isinstance(message, dict):
                message = project_ai_sdk_message_event(message, include_content=True)
            yield _sse(
                {
                    "type": "data-user-message",
                    "data": {"message": message},
                    "transient": True,
                }
            )
            return

        if etype == "state_snapshot":
            legacy_type = data.get("legacy_type")
            if legacy_type == "continuation_start":
                yield _sse(
                    {
                        "type": "data-continuation",
                        "data": {
                            "round": data.get("round"),
                            "max_rounds": data.get("max_rounds"),
                            "reason": data.get("reason"),
                        },
                        "transient": True,
                    }
                )
            elif legacy_type == "node_complete":
                yield _sse(
                    {
                        "type": "data-node-complete",
                        "data": {"node": data.get("node") or event.node},
                        "transient": True,
                    }
                )
            return

        if etype == "heartbeat":
            # An SSE comment: the SDK's strict chunk schema has no heartbeat.
            yield ": keepalive\n\n"
            return

        if etype == "interrupt":
            async for chunk in self._interrupt(data):
                yield chunk
            return

        if etype == "error":
            async for chunk in self._error(data):
                yield chunk
            return

        if etype == "complete":
            async for chunk in self._complete(data):
                yield chunk
            return

        if etype in SUBAGENT_PHASE_BY_EVENT:
            async for chunk in self._subagent(event):
                yield chunk
            return

        # message_start/end, reasoning_start/end, tool_call_delta,
        # title_updated carry no assistant-ui projection without a capability flag.

    # -- tool handlers -----------------------------------------------------

    def _resolve_tool_call_id(self, event: V3StreamEvent, *, is_end: bool) -> str:
        state = self._state
        if event.tool_call_id:
            return str(event.tool_call_id)
        if is_end and state.pending_tool_call_ids:
            return state.pending_tool_call_ids.pop(0)
        state.tool_seq += 1
        return f"tool_{state.tool_seq}"

    async def _tool_input(self, event: V3StreamEvent) -> AsyncGenerator[str, None]:
        state = self._state
        tool_call_id = self._resolve_tool_call_id(event, is_end=False)
        tool_name = event.tool_name or "unknown"
        if tool_call_id not in state.pending_tool_call_ids:
            state.pending_tool_call_ids.append(tool_call_id)
        yield _sse({"type": "tool-input-start", "toolCallId": tool_call_id, "toolName": tool_name})
        yield _sse(
            {
                "type": "tool-input-available",
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "input": coerce_json_object(clean_tool_output((event.data or {}).get("args"))),
            }
        )

    async def _tool_output(self, event: V3StreamEvent) -> AsyncGenerator[str, None]:
        state = self._state
        tool_call_id = self._resolve_tool_call_id(event, is_end=True)
        if tool_call_id in state.pending_tool_call_ids:
            with contextlib.suppress(ValueError):
                state.pending_tool_call_ids.remove(tool_call_id)
        data = event.data or {}
        render = data.get("render")
        if isinstance(render, dict):
            # Keyed by tool call: the strict tool-output chunk has no room for it.
            yield _sse(
                {
                    "type": "data-tool-render",
                    "data": {"toolCallId": tool_call_id, "render": render},
                    "transient": True,
                }
            )
        yield _sse(
            {
                "type": "tool-output-available",
                "toolCallId": tool_call_id,
                "output": coerce_json_object(clean_tool_output(data.get("output"))),
            }
        )

    async def _rich_items(self, data: dict[str, Any]) -> AsyncGenerator[str, None]:
        if not self._state.inline_rich_response_v1:
            return
        safe_items = select_transient_upsert_items(data.get("items") or [])
        if not safe_items:
            return
        yield _sse(
            {
                "type": "data-rich-items",
                "data": {"operation": data.get("operation") or "upsert", "items": safe_items},
                "transient": True,
            }
        )

    async def _image_preview(self, data: dict[str, Any]) -> AsyncGenerator[str, None]:
        """Project an early-delivery image preview as a transient data part.

        The stable part ``id`` (one per image index) lets AI SDK clients
        replace a partial/preview with the next partial/final in place. Reads
        both schema-v1 (top-level ``data_b64``) and schema-v2 (``delivery``)
        payloads. Inline deliveries carry an assembled ``data:`` URL; a final
        reference carries the protected relative ``/chat-images/...`` URL
        verbatim so a credentialed client can fetch it. The authoritative image
        still arrives as a ``file`` part on ``complete`` for clients that ignore
        preview events.
        """
        item_id = data.get("item_id")
        if not item_id:
            return

        # Second-defense inline budget at serialization time (FR-IMG-007).
        budget = getattr(settings, "image_stream_preview_max_b64_chars", 0)
        data = apply_inline_preview_wire_budget(data, budget=budget)

        projection = resolve_image_preview_delivery(data)

        if projection["status"] == IMAGE_PREVIEW_STATUS_SKIPPED:
            # Structured status — never a silent drop.
            yield _sse(
                {
                    "type": "data-image-preview",
                    "id": str(item_id),
                    "data": {
                        "imageIndex": projection["image_index"],
                        "status": IMAGE_PREVIEW_STATUS_SKIPPED,
                        "seq": projection["seq"],
                        "reason": projection["reason"],
                    },
                    "transient": True,
                }
            )
            return

        url = projection["url"]
        if not url:
            return
        yield _sse(
            {
                "type": "data-image-preview",
                "id": str(item_id),
                "data": {
                    "imageIndex": projection["image_index"],
                    "status": projection["status"],
                    "mediaType": projection["media_type"],
                    "url": url,
                    "seq": projection["seq"],
                },
                "transient": True,
            }
        )

    async def _subagent(self, event: V3StreamEvent) -> AsyncGenerator[str, None]:
        data = event.data or {}
        payload_data: dict[str, Any] = {
            "phase": SUBAGENT_PHASE_BY_EVENT.get(event.type, "update"),
            "subagent": event.subagent.model_dump(mode="json") if event.subagent else None,
        }
        if event.tool_call_id:
            payload_data["toolCallId"] = event.tool_call_id
        if event.tool_name:
            payload_data["toolName"] = event.tool_name
        for src_key, out_key in (
            ("task", "task"),
            ("output", "output"),
            ("summary", "summary"),
            ("thinking", "thinking"),
            ("status", "status"),
            ("error", "error"),
            ("render", "render"),
            ("text", "text"),
            ("channel", "channel"),
            ("elapsed_ms", "elapsedMs"),
            ("requested_model", "requestedModel"),
            ("resolved_model", "resolvedModel"),
        ):
            value = data.get(src_key)
            if value is not None:
                payload_data[out_key] = value
        yield _sse({"type": "data-subagent", "data": payload_data, "transient": True})

    # -- terminal-bearing handlers ----------------------------------------

    async def _interrupt(self, data: dict[str, Any]) -> AsyncGenerator[str, None]:
        state = self._state
        message = data.get("message")
        saved: str | None = None
        if isinstance(message, dict):
            saved = self._authoritative_content(message)
        elif isinstance(message, str):
            saved = message.strip()
        for frame in self._settle_answer(saved):
            yield frame
        if isinstance(message, str) and message.strip():
            yield _sse({"type": "text-delta", "id": state.text_id, "delta": message.strip()})
        projected_message = message
        if isinstance(message, dict):
            projected_message = project_ai_sdk_message_event(message, include_content=True)
        for frame in self._end_blocks():
            yield frame
        yield _sse(
            {
                "type": "data-interrupt",
                "data": {
                    "threadId": data.get("thread_id"),
                    "next": data.get("next"),
                    "interrupt": data.get("interrupt"),
                    "message": projected_message,
                },
            }
        )
        state.outcome = "interrupt"
        for frame in self._close(finish_reason=None):
            yield frame

    async def _error(self, data: dict[str, Any]) -> AsyncGenerator[str, None]:
        state = self._state
        for frame in self._settle_answer(None):
            yield frame
        message = data.get("message")
        if isinstance(message, dict):
            message = project_ai_sdk_message_event(message, include_content=True)
        if message:
            yield _sse(
                {"type": "data-error-message", "data": {"message": message}, "transient": True}
            )
        for frame in _ai_sdk_error_frames_from_data(data):
            yield frame
        state.outcome = "error"
        for frame in self._close(finish_reason="error"):
            yield frame

    async def _complete(self, data: dict[str, Any]) -> AsyncGenerator[str, None]:
        state = self._state
        state.outcome = "complete"
        original_message = data.get("message") or {}
        message = original_message
        is_v1 = False
        file_parts: list[dict[str, str]] = []
        if isinstance(original_message, dict):
            # v1-ness is decided on the original metadata so hidden image
            # candidates cannot fall back onto legacy `images` after the
            # capability projection strips the rich keys.
            is_v1 = is_v1_rich_items_message(find_message_metadata(original_message))
            file_parts = visible_image_file_parts(
                original_message,
                is_v1=is_v1,
                inline_rich_response_v1=state.inline_rich_response_v1,
            )
            message = project_ai_sdk_message_for_capability(
                original_message,
                inline_rich_response_v1=state.inline_rich_response_v1,
                answer_visuals_v1=state.answer_visuals_v1,
                source_figures_v1=state.source_figures_v1,
            )
            message = attach_image_parts_to_message(
                message,
                image_parts=file_parts,
                is_v1=is_v1,
            )

        # ``message`` is already capability-projected, as history projects it.
        saved = str(message.get("content") or "") if isinstance(message, dict) else None
        for frame in self._settle_answer(saved):
            yield frame

        if not state.any_text_delta:
            content = message.get("content") or "" if isinstance(message, dict) else ""
            if isinstance(content, str) and content.strip():
                state.any_text_delta = True
                yield _sse({"type": "text-delta", "id": state.text_id, "delta": content})

        if isinstance(message, dict):
            for file_part in file_parts:
                yield _sse(
                    {
                        "type": "file",
                        "url": file_part["url"],
                        "mediaType": file_part["mediaType"],
                    }
                )

        if isinstance(message, dict) and message:
            message_meta = project_ai_sdk_message_event(message)
            # A message carrying only an id has no side-channel metadata worth a
            # dedicated data part (the streamed text already conveyed the body).
            if message_meta and any(key != "id" for key in message_meta):
                yield _sse(
                    {
                        "type": "data-assistant-message",
                        "data": {"message": message_meta},
                    }
                )
