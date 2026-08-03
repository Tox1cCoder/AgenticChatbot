# Production Rich-Item Image Contract Design

**Status:** Approved architecture; implementation pending

**Date:** 2026-08-03

## Summary

Images remain first-class members of the rich-item system. `image` and
`image_group` use the same typed, versioned registry as `live_widget`,
`canvas_artifact`, and `tool_render`. A finalized assistant message's Markdown
markers and `metadata.rich_items` are the sole durable authority for rich-capable
clients.

The Streamlit and AI SDK adapters will project that authority differently only
where client capability requires it:

- Streamlit and AI SDK clients declaring `inlineRichResponseV1: true` render
  selected images through rich-item markers and do not render duplicate image
  file parts.
- AI SDK clients without the rich-response capability receive selected images
  as compatibility `file` parts, with markers and rich metadata removed.

Protected `/web-images/{id}` and `/chat-images/{id}` references are the only
rendering sources for persisted remote/generated images. External provenance
URLs remain attribution and audit data; they are never media fallbacks.

## Context and verified findings

The frontend report correctly identified a protected relative
`/web-images/{id}` reference and a separate external provenance URL. Its route
diagnosis no longer matches the current source: the local sidecar registers both
`/web-images/{image_id}` and `/api/web-images/{image_id}`, proxies the request to
the canonical server with upstream authentication, and has behavioral tests for
both paths. A deployed sidecar whose OpenAPI omits those paths is stale and must
be rebuilt/restarted; the frontend must not compensate by loading the external
provenance URL.

The report also revealed an ambiguous client contract. Current rich-capable AI
SDK responses can contain the same selected image twice: once as a rich item for
placement and once as an AI SDK `file` part for media. Documentation asks the
frontend to deduplicate the two representations, but the server can eliminate
that ambiguity at the capability projection boundary.

The in-repo Streamlit renderer already resolves `image` and `image_group`
markers through the rich-item registry and fetches protected references with
authentication. However, it currently removes an `image_group` cell when the
protected fetch fails. The production contract requires stable group ordering
and an in-place neutral failure cell.

## Goals

1. Keep `image` and `image_group` as typed `RichItem` variants alongside all
   other rich artifact types.
2. Establish one durable authority for rich-capable rendering: final Markdown
   markers plus final `metadata.rich_items`.
3. Make Streamlit streaming, Streamlit history, AI SDK streaming, and AI SDK
   history converge on the same final selected images and placements.
4. Eliminate duplicate image rendering by construction for rich-capable AI SDK
   clients.
5. Preserve backward compatibility for AI SDK clients that do not advertise
   rich-response v1.
6. Keep protected media authenticated, user-scoped, bounded, MIME-validated,
   and independent from answer completion.
7. Publish one concise, sendable frontend contract and retire contradictory
   update documents.

## Non-goals

- Removing images from the rich-item union.
- Making third-party publisher image URLs public rendering fallbacks.
- Changing image discovery, ranking, query anchoring, or selection policy.
- Migrating legacy persisted messages that have no `rich_items_version`.
- Adding a React frontend to this repository. The external frontend team will
  implement the published contract in its own repository.
- Removing generated-image preview events. Previews remain transient transport
  state and are not durable rich items.

## Canonical domain model

`RichItem` remains a discriminated union whose image variants are:

- `type: "image"` with `display_policy: "inline_only"` and one image payload;
- `type: "image_group"` with `display_policy: "inline_only"` and an ordered
  collection of image cells.

They share the registry, marker syntax, validation, finalization, provenance,
and versioning rules used by widgets, canvas artifacts, and tool renders.

For every finalized v1 message:

1. `content` contains zero or more standalone `<!--rich:<id>-->` markers.
2. `metadata.rich_items_version` equals `1`.
3. `metadata.rich_items` contains only public, finalized records.
4. A selected image renders only at its matching marker.
5. An unreferenced `inline_only` image is not appended as a gallery.
6. Unselected candidates and internal `_rich_item_candidates` never reach a
   client.
7. The final registry replaces transient upserts; it is not merged with them.

The rich item owns semantic identity, placement, alt text, structured caption,
source attribution, dimensions, and protected media reference. A transport
`file` part has no independent selection or placement authority.

## Transport projections

### Streamlit streaming and history

Streamlit advertises rich-response v1 and uses the canonical internal SSE path.
While streaming, safe non-image `rich_items` upserts may progressively resolve
markers. Image candidates remain private until final selection. At terminal
completion, Streamlit replaces its transient registry with the final message's
`metadata.rich_items` and renders all markers from that registry.

History uses the same `build_rich_response_view` marker resolver as terminal
stream rendering. It must not invoke the legacy image gallery for a message with
`rich_items_version == 1`.

Generated-image `image_preview` events remain temporary. On completion, the
final rich-item registry is authoritative and the preview state is cleared or
reconciled so the final visual renders once.

### Rich-capable AI SDK streaming and history

A request is rich-capable only when it declares
`inlineRichResponseV1: true` (or the documented snake-case alias) and the server
rollout setting permits the capability.

For a rich-capable v1 assistant message:

- marker-bearing text and final `metadata.rich_items` stay on the wire;
- safe non-image transient records may arrive through `data-rich-items`;
- finalized `image` and `image_group` records arrive in the terminal assistant
  message metadata;
- the server does not emit selected-image `file` stream events;
- the server does not attach selected-image `file` parts to the terminal or
  history message;
- the frontend renders images only through the marker-resolved rich-item
  component.

This suppression is a projection rule, not a domain-model change. The persisted
message still contains image rich items.

Generated-image previews may arrive before final text and remain independent of
the rich capability. They are discarded when the terminal message becomes
authoritative.

### Non-rich AI SDK compatibility

For a client that does not advertise rich-response v1:

- standalone marker lines are stripped;
- `rich_items`, `rich_items_version`, and rich-reference warnings are removed
  from client-facing metadata;
- only finalized selected images are flattened into AI SDK `file` events/parts;
- hidden or unselected candidates never fall back through legacy metadata;
- existing legacy messages without `rich_items_version` retain their current
  legacy image projection.

This preserves visible images for older clients without requiring them to
understand rich-item markers.

### Projection matrix

| Path | Markers | Final rich registry | Image file parts | Rendering authority |
|---|---:|---:|---:|---|
| Streamlit v1 stream/history | yes | yes | no | rich items |
| AI SDK v1 stream/history | yes | yes | no | rich items |
| AI SDK non-rich projection of v1 | no | no | yes | selected file parts |
| AI SDK legacy message | no | no | legacy-compatible | legacy projection |

## Protected media contract

### Routes

The supported protected references are:

- `/web-images/{id}` and `/api/web-images/{id}` for selected remote web
  visuals;
- `/chat-images/{id}` and `/api/chat-images/{id}` for generated or stored chat
  images.

Both canonical-server and local-sidecar reads require authentication and enforce
user ownership. The local sidecar forwards the authenticated upstream read,
streams rather than eagerly buffering the upstream response, preserves safe
media/cache headers, bounds declared and undeclared response sizes, closes the
upstream stream on cancellation, and does not expose upstream error bodies.

The release check must assert that all four paths are present in the sidecar
OpenAPI schema. Missing `/web-images` paths identify an outdated sidecar build,
not a signal to use a publisher URL directly.

### Rendering rules

Protected relative references cannot be assigned directly to a browser
`<img src>` because an image element cannot attach the application Bearer token.

- React/AI SDK frontends fetch the reference with the authenticated chat
  transport, validate success and an `image/*` content type, create an object
  URL, and revoke it on replacement or unmount.
- Streamlit fetches through its authenticated local-backend client, validates
  status and MIME, and supplies trusted bytes/data to its renderer.
- Tokens never appear in query parameters, provenance, logs, DOM attributes, or
  public/shared cache keys.
- `payload.source_url` may be displayed as a source link only.
- `provenance.original_image_url` is diagnostic provenance only and must not be
  rendered or fetched by the client.

The response's restrictive CSP is intentional defense in depth. The design does
not expand `img-src` to arbitrary remote origins.

## Rich image rendering behavior

### Single image

The renderer reserves space using valid dimensions when available, loads the
protected reference independently of the text answer, and renders at natural
size up to the answer-column limit without upscaling small images. `alt_text`
is used as the image's accessible alternative, not duplicated as visible text.
A trusted structured `caption` and source attribution may appear once in the
figure footer.

If loading or decoding fails, the single visual is omitted for that attempt.
The assistant's text, other rich items, and message completion remain intact.
The client must not retry against external provenance.

### Image group

Group cell order is stable. Each cell loads and fails independently. A failed
cell remains in place as a compact neutral `Visual unavailable` state so other
cells do not move and the group does not collapse. A successful cell remains
visible even if another cell fails. A persisted one-cell group is rendered as a
single-image layout.

Provider descriptions are not trusted captions. They may inform alt text but
must not become visible footer text automatically.

### Other rich items

Widgets, canvas artifacts, and tool renders continue using their existing
marker-resolved renderers. The image changes must not create a parallel registry
or special placement pipeline.

## Failure and security semantics

- `401`: authentication/session handling may refresh or reauthenticate according
  to the existing client policy; it must not expose a broken external fallback.
- `404`: treat as unavailable without revealing whether another user owns the
  reference.
- `413`: media exceeded the configured sidecar bound; fail the visual only.
- `502`/`504`: upstream visual retrieval failed or timed out; fail the visual
  only.
- invalid or missing `image/*` MIME: reject the visual.
- decode failure: reject that visual/cell.
- client disconnect: abort the fetch and release object/upstream resources.

No media failure may replace, truncate, roll back, or mark the assistant text
answer as failed.

## Implementation boundaries

### Backend projection

The shared AI SDK projection helpers remain the single source for deciding which
selected images are visible. They will receive the negotiated rich capability
when attaching/emitting image file parts. For a rich-capable v1 message they
return no image file parts; for a non-rich projection they derive parts only
from finalized image rich items.

Both streaming completion and history serialization must call the same helper
with the same capability semantics. No adapter may independently inspect legacy
`metadata.images` for a v1 message.

### Streamlit renderer

The shared protected-image resolver will validate MIME and retain failure state
needed by the group renderer. Group rendering will preserve one output cell per
persisted input cell rather than filtering failed cells before HTML generation.
Streaming completion and history continue to use the same rich-response view.

### Frontend documentation

Implementation will create one concise sendable contract at
`plans/AI_SDK_FE_RICH_ITEM_CONTRACT.md`. It will contain the capability request,
wire shapes, marker algorithm, protected fetch helper requirements, component
lifecycle, error behavior, and acceptance checklist.

`plans/AI_SDK_FE_CONTRACT_UPDATES.md` will become a short superseded notice that
points to the new contract. Relevant contradictory image sections in
`plans/AI_SDK_FE_CONTRACT.md` and `docs/frontend/rich-image-rendering.md` will
also point to the new contract instead of preserving a second normative source.

## Testing strategy

### Shared contract tests

- `image` and `image_group` remain accepted public rich-item variants.
- finalization persists only selected, marker-referenced images.
- externalization converts every persisted remote image/cell to a protected
  `/web-images/{id}` reference.
- source/provenance URLs never become protected-media fallbacks.

### AI SDK tests

- Rich-capable streaming emits marker-bearing text and terminal rich metadata
  with no selected-image `file` event.
- Rich-capable history contains image rich items with no image `file` part.
- Non-rich streaming/history strip markers and rich metadata while emitting
  exactly the finalized selected images as file parts.
- `image_group` compatibility projection flattens cells in stable order and
  deduplicates identical URLs.
- Streaming and history apply identical capability rules.
- Preview events remain transient and are reconciled at terminal completion.

### Streamlit tests

- Streaming and history resolve the same marker to the same image item/group.
- Final registry replacement removes stale transient records.
- Protected `/web-images` and `/chat-images` references use authenticated
  loading.
- Single-image failure leaves text intact.
- Group failure preserves the failed cell in place and successful siblings.
- A one-cell persisted group uses the single-image layout.
- V1 messages never invoke the legacy image gallery.

### Media-route tests

- Sidecar OpenAPI contains `/web-images`, `/api/web-images`, `/chat-images`, and
  `/api/chat-images` reads.
- Owner reads stream bytes with safe MIME/cache/hardening headers.
- unauthenticated/invalid tokens never contact upstream.
- other-user reads remain non-enumerable `404` responses.
- declared and chunked oversized responses are bounded.
- cancellation closes the upstream stream.

### Regression suite

Run focused rich-response, AI SDK projection/streaming/history, Streamlit rich
rendering, article image flow, and sidecar media proxy suites, followed by the
repository's normal lint/type/test gates for modified modules.

## Rollout and compatibility

The change is gated by the existing per-request rich-response capability; no
database migration or new feature flag is required. Deploy the canonical server
and local sidecar together. A sidecar build that lacks `/web-images` is rejected
by the release smoke test.

External AI SDK clients can migrate independently:

1. continue without the capability and receive compatibility file parts; or
2. implement the new contract and advertise `inlineRichResponseV1: true`.

Once rich capability is advertised, the client must render marker-resolved rich
items; it must not expect terminal image file parts.

## Observability

Existing rich-image eligibility, presentation, final-selection, anchoring, and
protected-fetch metrics remain authoritative. Add no new high-cardinality URL or
image-id labels. Release monitoring should compare:

- finalized selected-image counts against rendered marker counts;
- protected media status classes;
- rich-capable versus compatibility projection counts if already available;
- image-group per-cell load failures on the client without logging source URLs
  or tokens.

## Acceptance criteria

1. Images remain `RichItem` variants beside widgets, canvas artifacts, and tool
   renders.
2. Streamlit and rich-capable AI SDK clients render final images only from
   markers plus `metadata.rich_items`.
3. Rich-capable AI SDK streaming and history contain no duplicate selected-image
   file events/parts.
4. Non-rich AI SDK clients still receive exactly the finalized selected images.
5. Streaming and history converge on identical final selected images and order.
6. All four protected media paths exist on the sidecar and require
   authentication.
7. No client loads `source_url` or `original_image_url` as a media fallback.
8. A media failure never fails the text answer.
9. Streamlit preserves failed image-group cells in place and keeps successful
   siblings visible.
10. One concise FE contract is normative; older update/image documents point to
    it and contain no conflicting instructions.

