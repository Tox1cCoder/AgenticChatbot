# AI_SDK_FE_CONTRACT.md — Contract Updates

## 2026-07-27 (plan T004) — protected media references + consumer renderers

**Status: APPLIED 2026-07-27 — in the contract.** Kept as the audit record;
do not re-apply.

Since schema-v2 (`build_image_preview_reference_data` /
`resolve_image_preview_delivery` in `app/services/event_streaming/events.py`),
a `final` image is delivered by **protected reference** — the AI SDK
`data-image-preview.data.url` and the terminal `file.url` carry a relative
`/chat-images/{id}` served by the local origin's authenticated media route
(new sidecar route `client_backend/api/chat_images.py`, verified by
`tests/client_backend/test_image_stream_proxy.py`). The prior contract only
documented the inline `data:` form and gave no consumer guidance for
credentialed media, so a browser dropping the token onto a bare `<img src>`
would 401/404 the image.

Applied to `plans/AI_SDK_FE_CONTRACT.md`:

1. **`file` (image) part** (Stream Events) — documents the two `url` forms
   (protected relative reference vs absolute/`data:`) and requires an
   authenticated fetch + Blob URL for the relative form.
2. **`data-image-preview`** (Stream Events) — `data.url` may be a protected
   relative reference (finals always are); added `preview_skipped` handling
   and the authenticated-fetch requirement.
3. **Assistant Images + new "Protected Media Rendering" subsection** — one
   shared authenticated `resolveImageSrc` helper (used by `file` parts,
   previews, and history `rich_items`); the `useChat({ onData })` preview
   handler that fetches reference URLs, builds a Blob URL, replaces by
   `id`/`seq`, and **revokes stale Blob URLs on replacement / finish /
   unmount**; the custom terminal `file` renderer for protected relative URLs;
   and the `401/404/413/5xx` semantics (other-user reads return `404`, no
   existence oracle, no upstream-path leak).
4. **Renderer Algorithm** — steps 8–9 reference the shared handler and the
   terminal `file` renderer with Blob-URL revocation.

The React app is not in this repo; this is a documentation-only specification
of the required frontend behavior.

---

# AI_SDK_FE_CONTRACT.md — Contract Updates (2026-07-17)

**Status: APPLIED 2026-07-17 — all 5 items are now in the contract.**
Kept as the audit record; do not re-apply. Line numbers below reference the
pre-fix revision of the file.

Audit of `plans/AI_SDK_FE_CONTRACT.md` against the current backend after the
image-streaming feature. Three required fixes, two recommended additions.

## 1. Required — misplaced paragraph (lines 280–281)

The sentence below belongs to the rich-items/marker discussion but now sits
inside the "Image preview" section (it was orphaned when that section was
inserted):

> A marker streamed in `text-delta` whose item has not arrived yet renders as
> a pending placeholder until `finish`.

**Fix:** move it to the end of the `data-rich-items` block — as its own
paragraph directly after "...`provenance` is `{}` when there is no origin
metadata." (line 258) and before the "Image preview" paragraph (line 260).
Markers are a rich-items concept; image previews have no markers.

## 2. Required — resume streams never emit `data-image-preview`

Verified in `app/ai/graph.py`: `resume_with_decisions_stream` does not
register the live event sink that `execute_request_stream` uses, so image
previews cannot surface on `POST /ai/resume-interrupt`. Same limitation the
contract already documents for live subagent progress.

**Fix:** append to the "Image preview" section (after line 279):

> Emitted only on `POST /api/chat/{conversationId}` streams. Resume streams
> (`POST /ai/resume-interrupt`) do not emit image previews — the completed
> image still arrives via `file` parts and `metadata.rich_items` on the final
> `data-assistant-message`.

## 3. Required — capability flag independence

Verified in `ai_sdk_v6.py`: `_image_preview` has no
`inline_rich_response_v1` check, unlike `_rich_items`. The contract implies
per-capability behavior only for `data-rich-items` (line 168: "capable
requests only") but says nothing for previews, leaving it ambiguous.

**Fix:** add to the "Image preview" section:

> Unlike `data-rich-items`, `data-image-preview` does **not** require
> `inlineRichResponseV1` — it is emitted for any streaming chat request while
> `ENABLE_IMAGE_STREAMING` is on. Clients that do not handle it fall under
> the standard unknown-event rule (ignore it).

## 4. Recommended — renderer guidance

The "Renderer Algorithm" section (line 948) covers rich markers only.
Add one step (or a short note) for previews:

> In `useChat({ onData })`, keep the latest `data-image-preview` payload per
> part `id` and render it under the streaming text. On
> `data-assistant-message` (or `finish`), discard all previews — the final
> message's `file` parts / `rich_items` are authoritative.

## 5. Recommended — ordering note

Image-generator turns suppress token streaming, so `data-image-preview`
events typically arrive **before** the turn's first `text-delta` (the answer
text lands near completion). One sentence in the "Image preview" section
prevents FE assumptions that text always precedes media:

> Previews may arrive before any `text-delta` of the turn.

## Already correct — no action

- `data-image-preview` catalog row (line 169) and payload example
  (lines 262–268) match the wire: stable `id` per image index,
  `data.{imageIndex, status, mediaType, url, seq}`, `transient: true`.
- `status` semantics (`partial` = provider progress previews, `final` = at
  most once per index) match `ImagePreviewPublisher`.
- Rich-items exclusion table (lines 246–251) stays true: image *registry*
  items still never stream transiently; previews are a separate channel.
- Server knobs (`ENABLE_IMAGE_STREAMING`, `IMAGE_STREAM_PREVIEW_MAX_B64_CHARS`)
  are named correctly.
