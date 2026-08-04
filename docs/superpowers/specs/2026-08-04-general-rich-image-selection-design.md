# General Rich Image Selection Design

**Date:** 2026-08-04
**Status:** Approved for implementation planning
**Scope:** Web-search images used as inline rich items in assistant answers

## Problem

The chatbot can retrieve relevant images and still display a small, incidental,
or unrelated asset. The failure occurs before rendering: retrieval results from
different providers are accumulated in arrival order, and the current per-answer
cap is applied to that order before provider intent and image quality are
considered.

The public LangSmith trace supplied for this investigation demonstrates the
problem class:

- Tavily returned 114 image URLs, including a 145x94 thumbnail and many 45-pixel
  logo assets.
- Candidate construction retained the first eight Tavily images. Tavily did not
  provide dimensions, so the existing minimum-dimension gate could not reject
  them reliably.
- Brave image search returned six relevant T1/Faker photographs, collapsed into
  one image group.
- The combined candidate list contained the eight Tavily candidates followed by
  the Brave image group.
- The prompt and placement cap exposed only the first two image candidates, so
  the dedicated image-search group was not available to the model or automatic
  placement.
- The legacy `metadata.images` collection contained images outside the typed
  candidate set, creating a separate path through which unselected assets could
  reach a client.

The trace's final marker references a KED Global roster photograph, while the
provided screenshot shows a different response containing a small Korean flag.
They are not the same rendered message. They nevertheless expose the same
architectural defect: incidental web-page images can occupy the selection window
before a deliberate image-search result is considered.

## Goals

1. Select relevant, usable images across general subjects without subject-specific
   rules.
2. Prefer deliberate image-search results over incidental images scraped from web
   results.
3. Apply the display cap after eligibility, deduplication, and ranking.
4. Use one canonical selected set across model inventory, automatic placement,
   persistence, streaming, history, and rendering.
5. Fail to a complete text-only answer when no candidate is trustworthy.
6. Add no meaningful latency to the normal response path.
7. Persist only selected images, not per-candidate classification decisions.

## Non-goals

- Pixel-level image classification or vision-model reranking.
- Downloading candidate bytes or issuing HTTP `HEAD` requests during selection.
- A second search attempt on the normal path.
- Perfect image recall for every possible subject.
- Subject-specific rules for T1, Korea, Wikipedia, flags, logos, people, or any
  other entity.
- Replacing Brave or Tavily.
- Changing generated-image or user-upload attachment behavior.

## Design principles

### Precision before recall

The system does not promise to display an image for every visual subject. It
promises not to display one without adequate evidence that it is relevant and
usable. A missing optional image is preferable to an unrelated or tiny asset.

### Eligibility is not relevance

URL safety and payload validity decide whether a candidate can be considered.
Provider intent, result order, provenance, quality evidence, and descriptive
evidence decide how a surviving candidate is ranked. These stages remain
separate so an image does not become "relevant" merely because it passed schema
validation.

### One selection boundary

All consumers receive the same selected set. No downstream component performs an
independent fallback to the unselected discovery pool.

### Content-free decisions

Selection is transient. Rejection or ranking labels are not stored on messages,
web-image rows, rich items, or client payloads. Aggregate telemetry may count
bounded outcome categories without candidate identifiers, URLs, queries, titles,
descriptions, or user content.

## Architecture

The system introduces one deterministic image-candidate selector between
candidate discovery and every presentation consumer:

```text
Tavily and Brave execute concurrently
                |
                v
Per-tool candidate normalization and bounded collection
                |
                v
Canonical selector
  - validate
  - filter image-group cells
  - deduplicate
  - rank
  - apply final display cap
                |
                v
Transient selected image set
  - model-facing rich-item inventory
  - automatic placement
  - final rich-item registry
  - protected URL registration
  - streaming/history projections
  - Streamlit and AI SDK renderers
```

Per-tool collection caps remain as resource bounds. They are not presentation
decisions. The final per-answer cap is applied only by the canonical selector
after candidates from every completed tool call are available.

## Selector contract

The selector accepts the turn-scoped typed candidate sequence and the configured
maximum image-item count. It returns only public-shape image or image-group
records that may be presented. Non-image rich items continue through their
existing paths and are not ranked against images.

Selection is a pure, synchronous, in-process operation:

- no network access;
- no model invocation;
- no database access;
- no mutation of the original tool artifacts;
- deterministic output for identical ordered input and settings.

The transient workflow state retains the selected records for the remainder of
the turn. `build_rich_response_guidance`, image anchoring, and final metadata
construction consume this same sequence rather than slicing the raw discovery
sequence independently.

## Eligibility policy

A candidate or image-group cell is rejected when it has any of these general
defects:

- no usable URL or supported inline data;
- a remote URL with a disallowed or insecure scheme;
- malformed payload shape or unsupported media type;
- an exact duplicate display locator or duplicate original-image locator;
- known positive raster dimensions below the configured display minimum;
- known positive dimensions with an extreme configured aspect ratio;
- an explicit non-content asset path such as a favicon, sprite, spacer, avatar
  placeholder, or tracking pixel;
- a URL transformation that unambiguously requests a known-below-minimum raster
  size.

The policy does not classify the words `logo`, `flag`, `map`, `portrait`,
`diagram`, `chart`, or `icon` as junk. Those are legitimate visual subjects. A
small raster rendition can be rejected for display quality while a usable version
of the same subject remains eligible.

Unknown dimensions are not treated as proof of either quality or failure. Such a
candidate can remain eligible when it has strong source-bound and descriptive
evidence, but it ranks behind a comparable known-usable result.

### Image groups

Every image-group cell passes the same eligibility and deduplication policy.

- Two or more surviving cells remain an image group.
- One surviving cell becomes a normal image item with the group identity and
  provenance projected into the single-image shape.
- Zero surviving cells remove the candidate.

This normalization occurs before the answer-level image-item cap, so a partially
bad group cannot smuggle unusable cells through a valid group wrapper.

## Ranking policy

The selector uses a deterministic lexicographic ordering rather than an opaque
weighted score. Earlier keys have higher priority:

1. **Retrieval intent**
   - deliberate image-search item;
   - source-bound web-search image;
   - weak or query-level image.
2. **Provider result order** within the same retrieval-intent tier.
3. **Quality evidence**, preferring known usable dimensions to unknown dimensions.
4. **Descriptive evidence**, using bounded normalized overlap among the image
   query, title, description, alt text, and source title as a tie-breaker rather
   than an absolute semantic classifier.
5. **Stable discovery order** as the final tie-breaker.

Query-level Tavily images have weak provenance and are not selected on the normal
path. Source-bound Tavily images serve as fallback supply when no eligible
dedicated image-search item fills the available slot. A weak candidate never
displaces an eligible dedicated image-search result.

The configured per-answer image-item cap is applied after this ordering. An image
group counts as one item, regardless of its cell count.

## End-to-end data flow

### Discovery and normalization

Tavily and Brave continue to execute in the same parallel tool block. Existing
tool adapters preserve provider rank, query, source URL, source title, original
image URL, dimensions when supplied, and descriptions. Candidate IDs remain
stable and tool-call scoped.

### Selection before presentation

After tool artifacts are lifted into turn context, the selector reads the entire
bounded image pool. It returns the canonical selected sequence. Only that
sequence is supplied to the model-facing rich-item inventory.

This ordering fixes the traced failure: the Brave T1/Faker group is evaluated in
the deliberate-image-search tier before Tavily candidates, even though Tavily's
tool result appeared first.

### Placement

Automatic image anchoring reads the selected sequence, not raw candidates. A
valid model-authored marker for a selected item remains authoritative. Selected
but unreferenced deliberate image-search items may use the existing query anchor
and first-prose fallback rules. Source-bound web-search images retain the stricter
no-fallback placement rule.

A marker for an unselected or unknown image ID is removed before persistence. It
does not revive the candidate and does not create a public unavailable-image
placeholder.

### Persistence and public projection

Only referenced or automatically placed members of the selected sequence enter
`metadata.rich_items`. Selection reasons and rejected candidates do not persist.

For a rich-response-capable turn, web-search images are projected exclusively
through selected `rich_items`. The legacy `metadata.images` path must not carry
raw Tavily or Brave discovery results. Generated images and user attachments keep
their existing independent contracts.

The selected rich registry remains the source for:

- protected web-image reference registration;
- AI SDK terminal streaming parts;
- history reconstruction;
- Streamlit rendering;
- client-backend projections.

No renderer may fall back to the raw discovery pool when a selected item fails.

## Failure handling

- **One provider fails:** select eligible candidates from the surviving provider.
- **Both providers fail or nothing qualifies:** preserve the full text answer and
  emit no web-search image.
- **Selector failure:** fail closed, discard turn-scoped web-image candidates,
  and preserve the text answer.
- **Selected marker becomes invalid:** remove only that marker and image.
- **Protected reference registration fails for one group cell:** remove the cell,
  retain or collapse the group according to the surviving-cell rules.
- **All registrations fail:** remove the image item and its marker.
- **Client image load fails:** remove the failed figure or cell without showing a
  hidden candidate or legacy gallery.

There is no automatic second search in the initial release, including for an
explicit image request. The model's one deliberate image-search call remains the
only image-search round. If it returns no usable image, the answer is text-only.
A bounded retry can be designed later from production evidence without changing
the selector contract.

## Latency

Normal-path latency remains dominated by the existing provider calls and model
generation. Tavily and Brave continue to run concurrently. Selection adds only
bounded local validation, hashing/deduplication, token normalization, and sorting
over a small candidate pool.

The design explicitly excludes:

- image downloads;
- HTTP metadata probes;
- vision or language-model reranking;
- sequential provider calls;
- normal-path retries.

Selector duration is measured with content-free latency telemetry. A local
microbenchmark establishes a baseline, while production p50 and p95 measurements
verify that the selector remains lost in ordinary network/model noise. No strict
wall-clock assertion is added to unit tests because shared CI timing is unstable.

## Observability and privacy

Existing rich-image metrics may be extended or rewired to reflect the canonical
stages:

- candidates discovered by provider;
- candidates eligible after transient filtering;
- image items selected after ranking and cap;
- selected images successfully registered and protected-delivery fetch outcomes;
- selector duration.

Outcome dimensions remain bounded enumerations. Metrics and application logs do
not contain URLs, queries, titles, descriptions, candidate IDs, conversation IDs,
user IDs, or source text. Per-candidate outcome tags are not written to message
metadata, database records, public APIs, or frontend state.

## Security

Existing HTTPS, public-network, MIME, and protected-reference policies remain in
force. The selector performs string and metadata inspection only and does not
contact candidate hosts. Protected web-image registration and render-time fetch
remain the only approved remote-image delivery boundary.

Removing raw provider results from legacy public image metadata reduces the risk
of clients contacting unselected third-party hosts directly.

## Testing strategy

### Selector unit tests

- Eight Tavily candidates followed by one good Brave group select the Brave group.
- Known tiny rasters are rejected.
- A URL with an unambiguous below-minimum resize transformation is rejected even
  when explicit dimensions are absent.
- Unknown-dimension candidates do not outrank comparable known-good candidates.
- High-resolution logos and flags remain eligible, proving subject words are not
  globally blocked.
- Portraits, maps, diagrams, transparent images, and acceptable unusual aspect
  ratios remain eligible.
- Favicons, sprites, spacers, tracking pixels, malformed payloads, and insecure
  URLs are rejected.
- Exact display URLs and original-image variants deduplicate stably.
- Query-level images do not enter the normal selected set.
- Provider result order and final discovery-order tie-breaking are deterministic.
- Group filtering preserves two cells, collapses one, and removes zero.
- The final cap applies after ranking, and a group counts as one item.
- Selection performs no network, database, or model calls.

### Pipeline integration tests

- The same selected IDs feed prompt inventory and automatic placement.
- A model marker cannot select an ID outside the canonical set.
- Final metadata persists selected-and-placed images only.
- A synthetic 114-entry legacy list does not leak unselected web-search images.
- Generated images and user attachments are unaffected.
- Partial and total protected-registration failures follow the group rules.
- AI SDK streaming, history, client-backend, and Streamlit project the same final
  rich registry.
- A provider or selector failure preserves the text response.

### Trace regression fixture

A compact fixture derived from the supplied trace contains:

- source-bound Tavily roster images;
- a known tiny Tavily thumbnail;
- logo-like Tavily noise with unknown dimensions;
- a late-arriving Brave T1/Faker image group with known usable dimensions.

The expected result selects the Brave group before applying the answer cap and
exposes no raw legacy gallery.

### Performance verification

A local benchmark measures selector throughput over candidate pools larger than
the configured production bounds. Production telemetry monitors selector p50 and
p95 duration. These measurements validate the latency claim without adding a
flaky CI timing gate.

## Rollout

The selector replaces independent first-N slicing at the model inventory and
placement boundaries. No database migration is required because no new persisted
field is introduced. Existing selected rich-item and protected-reference schemas
remain valid.

Deployment should compare the following stage counts before and after rollout:

- discovered candidates;
- eligible candidates;
- selected image items;
- registered images;
- answers with no selected web image;
- selector duration.

A rise in text-only answers is acceptable when accompanied by a reduction in
tiny, irrelevant, or failed images. If production evidence later shows excessive
false negatives, ranking policy or an explicit-request retry can be revised
without restoring multiple selection paths.

## Acceptance criteria

1. A deliberate image-search result cannot be excluded merely because Tavily's
   result completed or was lifted first.
2. The display cap is applied after cross-provider eligibility and ranking.
3. The trace-derived T1/Faker fixture selects the relevant Brave group.
4. Known 45-pixel and 145x94 raster assets cannot be selected for normal inline
   display.
5. Legitimate high-quality logos, flags, portraits, maps, and diagrams remain
   selectable.
6. Unknown dimensions alone neither prove quality nor cause unconditional
   rejection.
7. Prompt inventory, placement, persistence, streaming, history, and renderers use
   one canonical selected sequence.
8. Unselected web-search images never appear in `rich_items`, legacy public image
   metadata, or client fallback galleries.
9. No per-candidate classification tag or rejected candidate is persisted.
10. Empty or failed selection produces a complete text-only answer.
11. Normal selection performs no network, model, or database operation.
12. Existing generated-image and user-attachment behavior remains unchanged.
