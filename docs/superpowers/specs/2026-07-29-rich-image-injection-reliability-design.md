# Rich Image Injection Reliability Design

Date: 2026-07-29
Status: approved direction

## Goal

Make inline web images feel natural and remain visually stable when third-party
media is slow, blocked, expired, or unavailable. Preserve the existing rich-item
registry, marker placement, capability negotiation, and selected-only
persistence contract.

The design must not add dataset-dependent evaluation, online vision reranking,
or an additional caption-generation model call.

## Current Problems

1. Caption ownership is duplicated. The response prompt tells the agent to
   write Markdown below an image marker, while the renderer creates another
   visible caption from `title` or `alt_text`. The renderer caption is centered;
   the separate Markdown caption is not.
2. General Tavily searches request images by default, even when a visual is not
   useful. The current normalizer also ignores images associated with individual
   Tavily results and loses their page-level provenance.
3. Brave returns better display metadata for dedicated image discovery, but the
   integration displays the original hotlinked URL while leaving Brave's proxied
   thumbnail in provenance.
4. External URLs pass directly to clients. A blocked or hotlink-protected image
   can therefore fail only in the browser, after the backend has persisted it.
5. `demo.py` hides only the failed `<img>`, leaving its figure and caption
   visible. Other frontends receive the same unstable remote URL through rich
   metadata and AI SDK file parts.

## Non-goals

- A labeled image-quality dataset or offline scoring pipeline.
- Runtime vision-model evaluation or semantic image reranking.
- Post-completion image updates. Selected images remain terminal rich items in
  the v1 protocol.
- Image generation changes.
- A licensing classifier. Provider terms and source attribution still apply.
- A general-purpose open proxy for arbitrary user-controlled URLs.

## Decisions

### 1. Retrieval routing

Brave Image Search is the primary provider for focused visual discovery.
Tavily remains the primary web-research provider and a complementary image
source when an image should be tied to a cited result.

`tavily_search` gains a per-call `include_images` option. During rollout, an
omitted value preserves the configured default. Once visual routing is verified,
nonvisual calls should pass `false`; source-bound visual calls pass `true`.

When images are enabled, Tavily normalization must preserve both:

- top-level query-related images, marked as lacking page-level provenance when
  no source page is available; and
- `results[].images`, carrying the parent result URL, title, domain, and result
  rank/score.

Focused visual queries call Brave directly. If Brave is unavailable, Tavily
images may be used as a fallback. The system does not call both providers in a
serial retry chain unless the first provider actually fails or returns no usable
candidates.

### 2. Lightweight candidate selection

Selection uses existing provider ordering plus deterministic checks only:

- strict SafeSearch;
- valid HTTPS image URL;
- exact URL deduplication;
- source-page provenance when available;
- rejection of known tiny images using provider dimensions; and
- a bounded candidate and selected-image count.

The existing text/description overlap may assist placement, but it must not be
presented as image verification. Provider descriptions are metadata, and Tavily
descriptions are not treated as publisher-authored captions.

There is no vision call, learned ranker, evaluation threshold, or dataset.

### 3. Single caption owner

The agent owns selection, marker placement, and surrounding prose. It does not
write a Markdown caption after the marker.

The renderer owns the one visible figure footer. For web images that footer is
source attribution, linked to `source_url`, rather than a second generated
description. `alt_text` remains accessibility text and is not automatically
shown as a caption. An explicit structured caption remains optional for image
types that have a trustworthy caption, such as document figures or generated
images.

Both prompt instructions that request a post-marker caption are removed.

### 4. Stable display delivery

Clients should not depend on an original third-party image URL as the primary
inline source.

The backend exposes an authenticated, same-origin, opaque media route for every
selected remote rich image. For Brave candidates, that route fetches Brave's
proxied thumbnail as its preferred upstream instead of hotlinking the original.
For Tavily candidates, it fetches the selected remote image. The route is
limited to server-selected provider results and is not an arbitrary URL fetch
endpoint.

The route applies:

- HTTPS-only upstream URLs;
- DNS/IP checks that reject private, loopback, link-local, and reserved targets;
- the same checks after every redirect;
- bounded redirects, connection/read timeouts, response bytes, and dimensions;
- raster image MIME validation; and
- optional bounded caching only when provider terms permit it.

Fetching occurs when the client loads the image, not while the assistant answer
is being finalized. A failed fetch returns an ordinary media error response so
the client enters its failure state. The original publisher page remains
available separately as `source_url`.

The opaque route follows the existing protected chat-image access pattern.
`demo.py` resolves it server-side with the user's Bearer token. Browser clients
perform a credentialed fetch and render the returned bytes through an object
URL; they must not place a protected relative URL directly in `<img src>`, where
the Bearer token cannot be attached. A future short-lived signed delivery URL
may be added, but is not required by this design.

The public image record keeps `payload.url` as the preferred display URL for
backward compatibility and adds optional `width`, `height`, and `caption`
metadata. Provider, original image URL, thumbnail URL, and discovery details
remain provenance fields. Existing clients can continue using `payload.url`.

### 5. Frontend failure contract

Every frontend uses one rich-image component with `loading`, `loaded`, and
`failed` states.

- Loading reserves the known aspect ratio and shows a quiet skeleton.
- Loaded renders the image, optional trustworthy caption, and linked source
  attribution.
- Failed replaces the entire figure with a compact neutral `Visual unavailable`
  block. It may include `Open source` and `Retry`; it never leaves a caption
  without an image or a large empty gap.

`demo.py` must replace its current `onerror` behavior, which hides only the
`<img>`. A missing protected image and a failed external image use the same
neutral fallback component.

AI SDK frontends must treat v1 rich-item images as the authoritative placement
records and deduplicate matching `file` parts rather than rendering an extra
gallery. A protected media URL in either record is fetched with credentials and
converted to an object URL before rendering.

Client-side failure handling remains mandatory even when the backend media route
works, because a user's firewall or browser policy can differ from the backend's
network.

## Latency Policy

This design adds no model-based image evaluation and no dataset-based runtime
work.

Image search remains the only request-path provider cost. When an answer needs
both web research and visual discovery, the workflow should issue Tavily and
Brave calls concurrently where the tool runtime permits it. It must avoid the
serial pattern of Tavily, another model decision, Brave, then another model.

The same-origin media fetch is client-driven and therefore does not block text
time-to-first-token or assistant completion. Slow media displays a skeleton and
then either loads or becomes the compact failure state. Optional images never
delay completion solely to prove availability.

Existing tool `duration_ms` telemetry measures provider-call latency. Add
separate bounded counters/timings for provider result counts, selected images,
media-route fetch outcomes, upstream timeout/MIME/size rejection, and sampled
client load failures. Do not log full signed URLs or image bytes.

## Error Handling

- Brave missing/timeout: use source-bound Tavily candidates when already
  available; otherwise complete text-only.
- Tavily images absent: use Brave when visual intent exists; otherwise complete
  text-only.
- No candidate passes deterministic checks: omit the image and marker.
- Same-origin media route cannot fetch upstream: return an error status; the
  frontend shows its compact fallback.
- Image expires after persistence: the same frontend fallback applies on history
  rendering.
- Source page is unavailable but display media loads: show the image without an
  active source link rather than inventing attribution.

Image failure must never fail the assistant response.

## Testing

### Backend

- Tavily `include_images` parameter forwarding and default compatibility.
- Tavily top-level and per-result normalization, provenance, and deduplication.
- Brave thumbnail chosen as the preferred display source.
- Candidate rejection for invalid schemes, duplicates, and known-small images.
- Media route authentication, ownership, redirect/IP protections, byte/MIME
  limits, timeouts, and success/error status behavior using fake transports.
- Rich-item serialization remains backward compatible and selected-only.
- No image failure can fail message persistence or terminal streaming.

### `demo.py`

- Loading markup reserves aspect ratio.
- `onerror` replaces the complete figure with the fallback.
- A failed image never leaves a visible caption by itself.
- Source link and Retry are escaped and shown only when available.
- Protected relative and remote display URLs use the same component states.

### External frontend contract

- Marker order and rich-item placement remain authoritative.
- Matching AI SDK file parts are deduplicated.
- Loading, success, and failure snapshots are covered at narrow and wide chat
  widths.
- Browser/network-block simulation produces the compact fallback without layout
  shift.

No labeled relevance dataset or vision-quality test suite is required.

## Rollout

1. Remove duplicate caption instructions and update the Streamlit figure failure
   state.
2. Prefer Brave thumbnails and normalize Tavily per-result image provenance.
3. Add the constrained same-origin media route and optional dimensions/caption
   fields.
4. Publish the frontend rich-image state contract and update AI SDK clients.
5. Add operational telemetry, then switch nonvisual Tavily calls to
   `include_images=false` once explicit Brave routing is confirmed.

The existing `INLINE_RICH_RESPONSE_ENABLED` kill switch remains the emergency
rollback for marker-bearing rich responses.
