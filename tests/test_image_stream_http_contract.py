"""Phase-0 RED characterization: full-path image streaming contract.

These tests drive a *deterministic* fake image generation through the REAL
canonical FastAPI stacks:

* the internal Streamlit SSE route  ``POST /messages/stream``
  (``app.api.messages.create_message_stream``), and
* the Vercel AI SDK UI Message Stream route ``POST /api/chat/{conversation_id}``
  (``app.api.ai_sdk.chat_ui_message_stream``).

Both routes call the REAL ``MessageService.create_message_stream`` /
``resume_message_creation_stream`` (mounted via ``Container.message_service``
override), the REAL internal-SSE / AI-SDK wire adapters, and the REAL
``ImagePreviewPublisher`` emission policy. No wire-adapter helper is called
directly; every assertion is made against bytes produced by an HTTP route.

Injection-seam note (documented for the controller): the production plan's
per-run *media delivery service* injected at the graph boundary (T002) does
not exist yet, so a fake ``ImageGenerationProvider`` cannot be threaded through
the real multi-agent graph deterministically (that needs a live router LLM +
checkpointer). The faithful full-HTTP-path seam available today is the
``ai_service`` event source consumed by the real ``MessageService``; the fake
source runs the REAL ``ImagePreviewPublisher`` so the emitter-drop defect
(``app/ai/image_generation/emitter.py:70-78``) is exercised by real code, and
emits a terminal ``complete`` carrying a protected ``/chat-images/{id}``
reference so the projection defect
(``app/services/event_streaming/ai_sdk_projection.py:104-154``) is exercised by
real code.

Expected terminal state: RED. Each assertion documents a currently-broken
contract that Phase-1 (T002-T005) will repair.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from dependency_injector import providers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ai.image_generation import (
    ImagePreviewPublisher,
    MediaDeliveryService,
    use_image_preview_emitter,
)
from app.ai.image_generation.base import ImageGenerationProvider
from app.ai.image_generation.models import (
    ImageFinal,
    ImageGenerationRequest,
    ImagePartial,
    ImageStreamEvent,
    NarrativeDelta,
)
from app.api.ai_sdk import router as ai_sdk_router
from app.api.messages import router as messages_router
from app.core.auth import get_current_user_id
from app.core.config import settings
from app.core.container import Container, setup_auto_injection
from app.models.enums import MessageRole
from app.schemas.message import MessageRead
from app.schemas.workflow import (
    WorkflowExecutionRequest,
    WorkflowPlanningContext,
    WorkflowResponse,
    WorkflowResponseMessage,
)
from app.services.event_streaming.events import make_event
from app.services.event_streaming.subagents import SubagentEventSink
from app.services.message_service import MessageService
from app.utils.exception_handler import register_exception_handlers

# ---------------------------------------------------------------------------
# Deterministic fake image payloads
# ---------------------------------------------------------------------------

# One 128 KiB partial preview (well under the 4,000,000-char preview cap).
_PARTIAL_BYTES = b"\x00" * (128 * 1024)
PARTIAL_B64 = base64.b64encode(_PARTIAL_BYTES).decode("ascii")

# One FINAL image whose base64 exceeds the 4,000,000-char preview cap so the
# current ImagePreviewPublisher drops it with only a logger.info (no event).
_FINAL_BYTES = b"\x00" * 3_000_003  # -> 4,000,004 base64 chars
FINAL_B64 = base64.b64encode(_FINAL_BYTES).decode("ascii")

NARRATIVE = "Here is the generated image you asked for."

_PREVIEW_CAP = settings.image_stream_preview_max_b64_chars


class _FakeImageProvider:
    """Deterministic ``ImageGenerationProvider``: one 128 KiB partial, one
    FINAL image whose base64 exceeds the 4,000,000-char preview cap, then a
    narrative delta. No network/provider I/O.
    """

    def __init__(self, *, partial_b64: str = PARTIAL_B64, final_b64: str = FINAL_B64):
        self._partial_b64 = partial_b64
        self._final_b64 = final_b64

    async def stream_generate(
        self, request: ImageGenerationRequest
    ) -> AsyncIterator[ImageStreamEvent]:
        yield ImagePartial(index=0, data_b64=self._partial_b64, mime="image/png", seq=1)
        yield ImageFinal(index=0, data_b64=self._final_b64, mime="image/png")
        yield NarrativeDelta(text=NARRATIVE)


def _sizes_banner() -> str:
    return (
        f"[sizes] partial_b64={len(PARTIAL_B64)} chars "
        f"({len(_PARTIAL_BYTES)} bytes); "
        f"final_b64={len(FINAL_B64)} chars ({len(_FINAL_BYTES)} bytes); "
        f"preview_cap={_PREVIEW_CAP} chars"
    )


# ---------------------------------------------------------------------------
# Real-MessageService harness (stubbed collaborators, real streaming logic)
# ---------------------------------------------------------------------------


def _message_row(*, conversation_id: UUID, sender: int, content: str, metadata: dict):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id=uuid4(),
        conversation_id=conversation_id,
        sender=sender,
        content=content,
        message_metadata=metadata,
        feedback=None,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _assistant_message_read(conversation_id: UUID, image_url: str) -> MessageRead:
    return MessageRead.model_validate(
        _message_row(
            conversation_id=conversation_id,
            sender=MessageRole.assistant.value,
            content=NARRATIVE,
            metadata={"images": [{"url": image_url, "mime": "image/png"}]},
        )
    )


class _HarnessImageStorage:
    """Deterministic ChatImageStorageService stand-in for the media delivery
    seam. Returns a fixed protected reference (the test's ``image_url``) so the
    early reference event and the terminal file part point at the same URL. No
    byte cap is enforced — final persistence is bounded by the storage cap, not
    the transient SSE preview cap.
    """

    def __init__(self, url: str):
        self._url = url
        self._image_id = url.rsplit("/", 1)[-1]

    def store(self, *, conversation_id, user_id, mime, data_b64, name):
        return {
            "image_id": self._image_id,
            "url": self._url,
            "mime": mime,
            "name": name or "generated-image",
            "content_hash": "harness-hash",
        }


async def _main_image_event_source(_request):
    """Fake ai_service.execute_request_stream mirroring the graph main path.

    Mirrors ``graph.execute_request_stream`` + the real
    ``ImageGeneratorAgent._consume_image_stream``: bind a real preview emitter to
    a real ``SubagentEventSink`` and drive the REAL per-run
    ``MediaDeliveryService`` (T002). Partials go through the REAL
    ``ImagePreviewPublisher`` policy; the FINAL is persisted and delivered early
    by protected reference the moment it arrives (T003). Then narrative text +
    a terminal complete carrying the same protected image reference.
    """
    image_url = _main_image_event_source.image_url
    sink = SubagentEventSink()

    def _emit(payload: dict) -> None:
        sink.emit_event(make_event("image_preview", sequence=0, data=payload))

    provider = _FakeImageProvider()
    request = ImageGenerationRequest(prompt="draw a cat", model="gemini-3-pro-image-preview")
    with use_image_preview_emitter(_emit):
        publisher = ImagePreviewPublisher(
            enabled=settings.enable_image_streaming,
            max_b64_chars=settings.image_stream_preview_max_b64_chars,
        )
        media = MediaDeliveryService(
            storage=_HarnessImageStorage(image_url),
            conversation_id=uuid4(),
            user_id=uuid4(),
            preview_publisher=publisher,
        )
        async for item in provider.stream_generate(request):
            if isinstance(item, ImagePartial):
                media.publish_partial(
                    image_index=item.index,
                    mime=item.mime,
                    data_b64=item.data_b64,
                    seq=item.seq,
                )
            elif isinstance(item, ImageFinal):
                media.persist_final(
                    image_index=item.index,
                    mime=item.mime,
                    data_b64=item.data_b64,
                )

    for event in await sink.drain():
        yield event

    yield make_event("message_delta", sequence=0, data={"text": NARRATIVE})
    yield make_event(
        "complete",
        sequence=0,
        data={
            "response": WorkflowResponse(
                message=WorkflowResponseMessage(content=NARRATIVE),
                metadata={"images": [{"url": image_url, "mime": "image/png"}]},
            )
        },
    )


async def _resume_image_event_source(**_kwargs):
    """Fake ai_service.resume_interrupted_execution_stream mirroring the graph
    resume path, which installs NO preview sink (``graph.py:2488-2490``).

    Because no emitter is bound (exactly as the current resume path leaves it),
    the REAL ``ImagePreviewPublisher`` returns False even for a *small* final
    image, so no early preview event is produced. This reproduces the
    resume-parity defect (FR-IMG-008).
    """
    image_url = _resume_image_event_source.image_url
    small_b64 = base64.b64encode(b"\x00" * 4096).decode("ascii")

    # NO use_image_preview_emitter(...) here, mirroring the resume path.
    publisher = ImagePreviewPublisher(
        enabled=settings.enable_image_streaming,
        max_b64_chars=settings.image_stream_preview_max_b64_chars,
    )
    emitted = publisher.publish(
        image_index=0, status="final", mime="image/png", data_b64=small_b64, seq=1
    )
    _resume_image_event_source.publisher_emitted = emitted  # False on current source

    yield make_event("message_delta", sequence=0, data={"text": NARRATIVE})
    yield make_event(
        "complete",
        sequence=0,
        data={
            "response": WorkflowResponse(
                message=WorkflowResponseMessage(content=NARRATIVE),
                metadata={"images": [{"url": image_url, "mime": "image/png"}]},
            )
        },
    )


def _build_message_service(*, conversation_id: UUID, user_id: UUID, image_url: str, resume: bool):
    """A REAL ``MessageService`` with stubbed persistence/validation only."""
    service = MessageService.__new__(MessageService)
    service.repository = SimpleNamespace(
        create=lambda entity: _message_row(
            conversation_id=conversation_id,
            sender=MessageRole.user.value,
            content=entity["content"],
            metadata={},
        )
    )
    service.conversation_validation_utils = SimpleNamespace(
        validate_conversation_access=lambda *_a: None,
        conversation_repository=SimpleNamespace(
            get_by_id=lambda _cid: SimpleNamespace(title="Existing chat")
        ),
    )
    workflow_request = WorkflowExecutionRequest(
        message="draw a cat",
        conversation_id=str(conversation_id),
        user_id=str(user_id),
        planning=WorkflowPlanningContext(),
    )
    service._build_user_message_workflow_request = AsyncMock(
        return_value=(user_id, None, workflow_request)
    )

    if resume:
        _resume_image_event_source.image_url = image_url
        service.hitl_interrupt_repository = None
        service._validate_and_claim_interrupt_resume = lambda **_k: None
        service._get_conversation_context = lambda *_a: (user_id, None)
        service._revalidate_resume_custom_agent = lambda *_a: None
        service._audit_interrupt_resume_decisions = lambda **_k: None
        service._resolve_custom_agents_state = lambda *_a: {}
        service._clear_redis_interrupt = lambda *_a: None
        service._sync_response_plan_state = lambda **_k: False
        service.ai_service = SimpleNamespace(
            invalidate_history_cache=lambda *_a: None,
            resume_interrupted_execution_stream=_resume_image_event_source,
        )
    else:
        _main_image_event_source.image_url = image_url
        service.ai_service = SimpleNamespace(
            invalidate_history_cache=lambda *_a: None,
            execute_request_stream=_main_image_event_source,
        )

    service._persist_completed_workflow_response = AsyncMock(
        return_value=_assistant_message_read(conversation_id, image_url)
    )
    service._compact_checkpoint_after_persist = AsyncMock()
    return service


def _build_app(message_service, user_id: UUID) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(messages_router)
    app.include_router(ai_sdk_router)
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    return app


# ---------------------------------------------------------------------------
# SSE parsing helpers (parse only; never call adapter helpers directly)
# ---------------------------------------------------------------------------


def _parse_sse(body: str):
    """Return the ordered list of ``data:`` payloads from an SSE body.

    ``[DONE]`` is kept as the literal string; everything else is JSON-decoded.
    """
    payloads = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[len("data: ") :]
        payloads.append(raw if raw.strip() == "[DONE]" else json.loads(raw))
    return payloads


def _types(payloads) -> list[str]:
    out: list[str] = []
    for p in payloads:
        if p == "[DONE]":
            out.append("[DONE]")
        elif isinstance(p, dict):
            out.append(str(p.get("type")))
    return out


def _first_index_or_raise(payloads, predicate, *, label: str, order: list[str]) -> int:
    """Return the index of the first payload matching ``predicate``.

    Raises an informative ``AssertionError`` (not a bare ``ValueError``) when
    no matching event exists at all, so an explicit FR-IMG-002 ordering
    comparison (``index(final ref) < index(narrative completion)``) fails for
    the defect reason — the event is simply never emitted — rather than for an
    unrelated ``IndexError``/``TypeError``.
    """
    for i, p in enumerate(payloads):
        if predicate(p):
            return i
    raise AssertionError(f"no {label} found in stream; event order={order}")


@pytest.fixture(autouse=True)
def _restore_wiring():
    setup_auto_injection(Container)
    yield
    setup_auto_injection(Container)


# ---------------------------------------------------------------------------
# Internal Streamlit SSE route
# ---------------------------------------------------------------------------


def test_internal_sse_delivers_oversized_final_image_early_by_reference():
    """RED: the oversized FINAL image must still surface an early
    reference-delivery ``image_preview`` (status=final) strictly before the
    terminal ``complete`` (FR-IMG-002: "an authenticated image reference is
    emitted before narrative completion"), asserted as an explicit
    ``index(final ref) < index(complete)`` comparison, not mere presence in
    the payload list. Current source drops the final preview entirely in
    ImagePreviewPublisher, so only the 128 KiB partial survives and the index
    lookup for the final reference fails outright.
    """
    conversation_id = uuid4()
    user_id = uuid4()
    image_url = f"/chat-images/{uuid4()}"
    service = _build_message_service(
        conversation_id=conversation_id, user_id=user_id, image_url=image_url, resume=False
    )

    with Container.message_service.override(providers.Object(service)):
        client = TestClient(_build_app(service, user_id))
        resp = client.post(
            "/messages/stream",
            json={
                "conversation_id": str(conversation_id),
                "content": "draw a cat",
                "role": 1,
            },
        )

    assert resp.status_code == 200
    payloads = _parse_sse(resp.text)
    previews = [
        p for p in payloads if isinstance(p, dict) and p.get("type") == "image_preview"
    ]
    statuses = [p.get("status") for p in previews]
    order = _types(payloads)

    partial_seen = any(s == "partial" for s in statuses)

    assert partial_seen, (
        "expected the 128 KiB partial preview to be delivered.\n"
        f"{_sizes_banner()}\nevent order: {order}"
    )

    # FR-IMG-002 ("an authenticated image reference is emitted before
    # narrative completion"), asserted as an explicit index comparison rather
    # than mere presence. Characterized defect: emitter.py:70-78 drops the
    # oversized final entirely, so this lookup itself fails/raises -- there is
    # no early final-status reference event to find an index for at all.
    final_ref_index = _first_index_or_raise(
        payloads,
        lambda p: isinstance(p, dict)
        and p.get("type") == "image_preview"
        and p.get("status") == "final",
        label=(
            "DEFECT (emitter.py:70-78): an early reference-delivery "
            "image_preview (status=final); the oversized FINAL image only "
            "arrives with the terminal `complete`, violating "
            f"FR-IMG-002/FR-IMG-003. {_sizes_banner()} "
            f"image_preview statuses seen: {statuses}"
        ),
        order=order,
    )
    complete_index = order.index("complete")
    assert final_ref_index < complete_index, (
        "FR-IMG-002 violated: final-status image_preview at index "
        f"{final_ref_index} must precede the terminal complete at index "
        f"{complete_index}; event order: {order}"
    )


def test_internal_sse_terminates_with_single_complete():
    """Ordering guard: the internal stream ends with exactly one terminal
    ``complete`` and the partial preview precedes it."""
    conversation_id = uuid4()
    user_id = uuid4()
    image_url = f"/chat-images/{uuid4()}"
    service = _build_message_service(
        conversation_id=conversation_id, user_id=user_id, image_url=image_url, resume=False
    )

    with Container.message_service.override(providers.Object(service)):
        client = TestClient(_build_app(service, user_id))
        resp = client.post(
            "/messages/stream",
            json={
                "conversation_id": str(conversation_id),
                "content": "draw a cat",
                "role": 1,
            },
        )

    payloads = _parse_sse(resp.text)
    order = _types(payloads)
    completes = [p for p in payloads if isinstance(p, dict) and p.get("type") == "complete"]
    assert len(completes) == 1, f"expected exactly one terminal complete; order={order}"
    assert order[-1] == "complete", f"stream must end on complete; order={order}"


# ---------------------------------------------------------------------------
# AI SDK route
# ---------------------------------------------------------------------------


def _drive_ai_sdk(conversation_id: UUID, user_id: UUID, service) -> str:
    with Container.message_service.override(providers.Object(service)):
        client = TestClient(_build_app(service, user_id))
        resp = client.post(
            f"/api/chat/{conversation_id}",
            json={"messages": [{"role": "user", "content": "draw a cat"}]},
        )
    assert resp.status_code == 200, resp.text
    return resp.text


def test_ai_sdk_terminal_file_part_preserves_protected_reference_url():
    """RED: the terminal AI SDK ``file`` part must preserve the protected
    ``/chat-images/{id}`` URL verbatim. Current source runs it through loose
    base64 decoding and emits a corrupt ``data:`` URL instead.
    """
    conversation_id = uuid4()
    user_id = uuid4()
    image_url = f"/chat-images/{uuid4()}"
    service = _build_message_service(
        conversation_id=conversation_id, user_id=user_id, image_url=image_url, resume=False
    )
    body = _drive_ai_sdk(conversation_id, user_id, service)
    payloads = _parse_sse(body)
    file_parts = [p for p in payloads if isinstance(p, dict) and p.get("type") == "file"]

    assert file_parts, (
        "expected a terminal `file` part carrying the generated image.\n"
        f"event order: {_types(payloads)}"
    )
    urls = [p.get("url") for p in file_parts]
    assert image_url in urls, (
        "DEFECT (ai_sdk_projection.py:104-154): the protected relative image URL "
        "was NOT preserved as a `file` url; it was reinterpreted as loose base64 "
        "and mangled into a corrupt data: URL, so a credentialed fetch is "
        "impossible (FR-IMG-006).\n"
        f"expected url: {image_url}\nactual file urls: {urls}"
    )


def test_ai_sdk_emits_exactly_one_done_and_ends_with_it():
    """Wire guard: the AI SDK stream carries exactly one ``[DONE]`` and ends
    with it (contract lock; already correct on current source)."""
    conversation_id = uuid4()
    user_id = uuid4()
    image_url = f"/chat-images/{uuid4()}"
    service = _build_message_service(
        conversation_id=conversation_id, user_id=user_id, image_url=image_url, resume=False
    )
    body = _drive_ai_sdk(conversation_id, user_id, service)
    payloads = _parse_sse(body)
    order = _types(payloads)
    done_count = order.count("[DONE]")
    assert done_count == 1, f"expected exactly one [DONE]; order={order}"
    assert order[-1] == "[DONE]", f"stream must end on [DONE]; order={order}"


def test_ai_sdk_delivers_oversized_final_image_early():
    """RED: an oversized FINAL image must produce an early
    ``data-image-preview`` with status=final (or an early reference file part)
    strictly before narrative completion (FR-IMG-002: "an authenticated image
    reference is emitted before narrative completion"), asserted as an
    explicit ``index(final ref) < index(text-end)`` comparison, not mere
    presence in the payload list. Current source drops the oversized final
    entirely in the publisher, so only the partial ``data-image-preview``
    appears and the index lookup for the final reference fails outright.
    """
    conversation_id = uuid4()
    user_id = uuid4()
    image_url = f"/chat-images/{uuid4()}"
    service = _build_message_service(
        conversation_id=conversation_id, user_id=user_id, image_url=image_url, resume=False
    )
    body = _drive_ai_sdk(conversation_id, user_id, service)
    payloads = _parse_sse(body)
    order = _types(payloads)
    previews = [
        p
        for p in payloads
        if isinstance(p, dict) and p.get("type") == "data-image-preview"
    ]
    statuses = [(p.get("data") or {}).get("status") for p in previews]

    assert any(s == "partial" for s in statuses), (
        "expected the 128 KiB partial data-image-preview.\n"
        f"{_sizes_banner()}\nstatuses={statuses}\norder={order}"
    )

    # FR-IMG-002 ordering, asserted explicitly rather than by presence.
    # Characterized defect: emitter.py:70-78 drops the oversized final
    # entirely, so this lookup itself fails/raises -- there is no early
    # final-status data-image-preview to find an index for at all.
    final_ref_index = _first_index_or_raise(
        payloads,
        lambda p: isinstance(p, dict)
        and p.get("type") == "data-image-preview"
        and (p.get("data") or {}).get("status") == "final",
        label=(
            "DEFECT (emitter.py:70-78): an early final-status "
            "data-image-preview; the oversized FINAL image is dropped and "
            f"never shown before completion (FR-IMG-002/FR-IMG-003). "
            f"{_sizes_banner()} data-image-preview statuses seen: {statuses}"
        ),
        order=order,
    )
    narrative_complete_index = order.index("text-end")
    assert final_ref_index < narrative_complete_index, (
        "FR-IMG-002 violated: final-status data-image-preview at index "
        f"{final_ref_index} must precede narrative completion (text-end) at "
        f"index {narrative_complete_index}; event order: {order}"
    )


# ---------------------------------------------------------------------------
# Resume (post-HITL) parity
# ---------------------------------------------------------------------------


def test_resume_stream_emits_early_image_preview_after_hitl():
    """RED: an image generated after a HITL resume must surface the same early
    ``image_preview`` as a fresh run. Current source installs no preview sink
    on the resume path (``graph.py:2488-2490``), so even a small image emits no
    early preview.
    """
    conversation_id = uuid4()
    user_id = uuid4()
    image_url = f"/chat-images/{uuid4()}"
    service = _build_message_service(
        conversation_id=conversation_id, user_id=user_id, image_url=image_url, resume=True
    )

    with Container.message_service.override(providers.Object(service)):
        client = TestClient(_build_app(service, user_id))
        resp = client.post(
            "/messages/resume-interrupt",
            json={
                "thread_id": str(conversation_id),
                "conversation_id": str(conversation_id),
                "interrupt_id": "int-1",
                "decisions": [],
            },
        )

    assert resp.status_code == 200, resp.text
    payloads = _parse_sse(resp.text)
    order = _types(payloads)
    previews = [p for p in payloads if isinstance(p, dict) and p.get("type") == "image_preview"]
    assert previews, (
        "DEFECT (graph.py:2488-2490 resume path installs no preview sink): an "
        "image generated after HITL resume produced NO early `image_preview` "
        "event (FR-IMG-008). publisher.publish returned "
        f"{getattr(_resume_image_event_source, 'publisher_emitted', 'n/a')} because no "
        f"emitter was bound on resume.\nevent order: {order}"
    )


# ---------------------------------------------------------------------------
# Deterministic fake provider shape + opt-in real-provider smoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_image_provider_emits_specified_deterministic_shape():
    """The deterministic fixture provider conforms to
    ``ImageGenerationProvider`` and emits exactly: one 128 KiB partial, one
    FINAL image above the 4,000,000-char preview cap, then narrative text."""
    provider = _FakeImageProvider()
    assert isinstance(provider, ImageGenerationProvider)

    items = [
        item
        async for item in provider.stream_generate(
            ImageGenerationRequest(prompt="draw a cat", model="gemini-3-pro-image-preview")
        )
    ]
    kinds = [type(item).__name__ for item in items]
    assert kinds == ["ImagePartial", "ImageFinal", "NarrativeDelta"], kinds

    partial, final, narrative = items
    assert len(partial.data_b64) == len(PARTIAL_B64) < _PREVIEW_CAP, _sizes_banner()
    assert len(final.data_b64) == len(FINAL_B64) > _PREVIEW_CAP, _sizes_banner()
    assert narrative.text == NARRATIVE


_RUN_PROVIDER_SMOKE = os.getenv("RUN_IMAGE_PROVIDER_SMOKE") == "1"


@pytest.mark.skipif(
    not _RUN_PROVIDER_SMOKE,
    reason="opt-in real-provider smoke; set RUN_IMAGE_PROVIDER_SMOKE=1 (needs credentials)",
)
@pytest.mark.asyncio
async def test_real_image_provider_smoke():
    """Opt-in real-provider smoke (never runs in CI; the deterministic tests are
    the gate). When enabled, resolve a REAL image provider and stream one
    generation, asserting a non-empty final image. Skips if no provider or
    credentials are available; records only byte sizes, never image bytes.
    """
    from app.ai.image_generation.registry import resolve_image_provider

    model = os.getenv("IMAGE_PROVIDER_SMOKE_MODEL", "gpt-image-1")
    provider = resolve_image_provider(
        model, gemini_client=None, openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    if provider is None:
        pytest.skip(f"no image provider available for model {model!r}")

    finals: list[ImageFinal] = []
    async for item in provider.stream_generate(
        ImageGenerationRequest(prompt="a red circle on white", model=model, max_images=1)
    ):
        if isinstance(item, ImageFinal):
            finals.append(item)

    assert finals, "real provider produced no ImageFinal"
    assert finals[0].data_b64, "real provider returned an empty final image"
    # Record size only; never surface raw image bytes in artifacts.
    print(f"[real-provider smoke] model={model} final_b64_len={len(finals[0].data_b64)}")
