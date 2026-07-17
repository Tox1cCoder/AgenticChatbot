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
