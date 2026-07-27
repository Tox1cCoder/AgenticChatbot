# Image Streaming (Early Delivery) — Design

Date: 2026-07-17
Status: approved direction (user requested implementation; Gemini + OpenAI compatible, Claude-ready)

## Problem

Generated images ship only in the terminal `complete` event. The image itself is
ready long before the user sees it: after generation the agent makes a second
LLM call for the user-facing narrative, then finalization builds the rich-items
registry. Providers cannot stream partial pixels (Gemini emits whole images;
OpenAI `gpt-image-1` emits a few progressive previews), so the goal is **early
delivery**: push each image (and provider partial previews) to the client the
moment it exists.

## Non-goals

- Changing image persistence or the rich-items finalization contract
  (`select_transient_upsert_items` still excludes image registry items;
  previews use a dedicated event type and are never persisted).
- Per-user runtime keys for the image model (generation model stays global,
  behavior parity with today).
- A Claude image provider (Anthropic has no image generation API; the registry
  documents the extension point).

## Architecture

### 1. Provider abstraction — `app/ai/image_generation/`

- `models.py` — `ImageGenerationRequest` (prompt, source_images, max_images,
  aspect_ratio, model) and stream events: `ImagePartial(index, data_b64, mime,
  seq)`, `ImageFinal(index, data_b64, mime)`, `NarrativeDelta(text)`.
- `base.py` — `ImageGenerationProvider` protocol:
  `stream_generate(request) -> AsyncIterator[event]`.
- `gemini.py` — wraps `google-genai` async streaming
  (`client.aio.models.generate_content_stream`); yields `ImageFinal` per
  `inline_data` part, `NarrativeDelta` per text part. Replaces today's
  **blocking** `models.generate_content` call (also fixes event-loop blocking).
- `openai_provider.py` — wraps `AsyncOpenAI().images.generate/edit`
  (`stream=True, partial_images=2`); `image_generation.partial_image` events →
  `ImagePartial`, `image_generation.completed` → `ImageFinal`. Key comes from
  the standard `OPENAI_API_KEY` env var (SDK default) or explicit arg. Aspect
  ratio maps to nearest supported size. One image per request.
- `registry.py` — `resolve_image_provider(model, *, gemini_client)`:
  `gpt-image*`/`dall-e*` → OpenAI; anything else → Gemini (parity). Future
  Claude support = new provider module + one registry line.

### 2. Event escape hatch (mid-node → stream)

Reuse the existing live-merge infrastructure (`SubagentEventSink` +
`stream_with_subagent_events`, already merged into
`execute_request_stream`):

- `SubagentEventSink.emit_event(event)` (new, sync `put_nowait`) accepts a
  prebuilt `V3StreamEvent`.
- `_image_generator_node` resolves the sink from
  `state.context.subagent_event_sink_token` and installs a per-request emitter
  via a `ContextVar` (`app/ai/image_generation/emitter.py`), so no agent
  signature changes and concurrent requests stay isolated. Non-stream runs
  resolve no sink → no previews, no errors.

  **Superseded (T005 — resume parity, FR-IMG-008):** resumed runs now install
  the SAME request-scoped media sink as a fresh run. The token persisted in the
  checkpoint outlives its original sink (its weakref dies with that stream and
  resolves to None), so `resume_with_decisions_stream` creates a live
  `SubagentEventSink`, rebinds it under the persisted token
  (`rebind_subagent_event_sink`) — or injects a fresh token via the resume
  state update when none was persisted — and merges it with
  `stream_with_subagent_events`. An image generated after a HITL resume now
  emits the same early preview / final-by-reference as a new run.

### 3. Canonical event + wire projection

New `StreamEventType`: `image_preview`. Payload:
`{item_id, image_index, status: "partial"|"final", mime, data_b64, seq}`.

- `GraphPublicStreamProjector.map_event` → public dict
  `{"type": "image_preview", ...}` (not affected by `suppress_tokens`).
- `AIService._map_workflow_stream` → `make_event("image_preview", ...)`.
- `internal_sse.legacy_event_from_v3` → same-shape legacy dict for Streamlit.
- `ai_sdk_v6._map_event` → `{"type": "data-image-preview", "id": item_id,
  "data": {imageIndex, status, mediaType, url: <data URL>, seq},
  "transient": true}` — stable part id lets AI SDK clients reconcile partial →
  final in place. The terminal `file` parts at `complete` remain the
  authoritative images (unchanged).

### 4. Emission policy (`ImagePreviewPublisher`)

- Kill switch: `settings.enable_image_streaming` (default true).
- Size cap: `settings.image_stream_preview_max_b64_chars` (default 4,000,000
  ≈ 3 MB binary) applies to INLINE (transient) previews only.
- Emit-once per index for `final`; partials replace by `(index, seq)`.
- Publisher failures never break generation (log + continue).

**Superseded (T003/T005 — oversized delivery + lossless final):** an oversized
image is never silently dropped.
- A FINAL image is ALWAYS delivered EARLY and by protected reference
  (`MediaDeliveryService.persist_final` → schema-v2 `image_preview` with
  `delivery.kind = reference`), so its early delivery does not depend on the
  inline SSE size cap. The bytes are bounded by the storage byte cap, not the
  transient preview char cap.
- An oversized INLINE partial is downgraded to a structured `preview_skipped`
  status (carries no base64) rather than being silently dropped (FR-IMG-007).
- Backpressure on the shared `SubagentEventSink` treats FINAL references as
  LOSSLESS: only in-progress partials coalesce, a newer partial for an item
  evicts the stale one before any unrelated frame, and a final reference is
  never discarded.
- Cross-run idempotency: `MediaDeliveryService.persist_final` is idempotent
  per instance, and `ChatImageStorageService.store()` is idempotent at the
  ownership-row level on `(user_id, sha256)`. A resumed run (fresh
  `MediaDeliveryService`) that re-persists identical content reuses the existing
  `chat_images` row instead of inserting a duplicate.

### 5. Demo (Streamlit)

Dedicated preview placeholder in the stream loop: `image_preview` events render
`st.image` (latest payload per index) with a "Generating image…" caption for
partials; cleared on `complete` (finalize renders the authoritative message).
RichStreamState/rich segments machinery untouched.

## Testing

- Provider units with fake SDK streams (Gemini chunk shapes, OpenAI event
  objects): ordering, b64 handling, narrative capture, aspect-ratio mapping.
- Publisher policy: flag off, size cap, emit-once, no-emitter contexts.
- Pipeline: sink `emit_event` → projector → ai_service → both adapters
  (payload shape, transient flags, non-suppression with `suppress_tokens`).
- Existing rich-response contract tests must stay green (no changes to them).

## Risks

- google-genai/openai streaming API surface drift → defensive parsing,
  provider tests use fakes, integration failures degrade to "no previews"
  (never to failed generation).
- Oversized SSE frames → the inline size cap now applies to transient previews
  only; a FINAL image is delivered by protected reference (no multi-megabyte
  base64 frame), and oversized inline partials degrade to a `preview_skipped`
  status. Previews remain single-shot (no re-send per upsert like rich items).
