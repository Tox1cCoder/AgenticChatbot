# Rich Image Selection Hardening Design

Date: 2026-07-29
Status: approved direction, pending written-spec review

This revision replaces the earlier precision-only version of this document. That
version made a model-authored marker the only path to a displayed image. Review
against the code showed the mechanism is unreliable in this repository — the
existence of `_repair_unprefixed_markers` in `app/core/rich_placement.py` is
direct evidence that models mangle the marker syntax, and its docstring records
the consequence: "the image is discarded." Removing the compensating placement
path while adding nothing to make marker emission reliable would have driven
inline images close to zero. This revision keeps every precision gain of that
version and restores recall through a display-intent mechanism modelled on how
ChatGPT and Gemini actually inject images.

## Goal

Make inline web images relevant, suitable, and naturally placed, at a rate a
user would recognize as normal assistant behavior, while preserving the existing
rich-item protocol, protected image delivery, and graceful text-only fallback.

This is a focused follow-up to
`2026-07-29-rich-image-injection-reliability-design.md`, which established safe
delivery and resilient rendering. This document fixes retrieval intent,
candidate quality, image grouping, placement, and production observability.

## The mechanism being adopted

ChatGPT and Gemini do not rely on the answering model remembering a formatting
token. Two properties of their behavior are what this design copies:

1. **Relevance comes from a ranked image provider queried with a purpose-written
   image query**, not from a language model judging scraped candidates by title.
2. **Running the image search is itself the intent to display.** The renderer
   places what the search returned; the model refines placement rather than
   authorizing it.

Both are available here already. `brave_image_search` is a bound, pinned tool
(`app/ai/deferred_tool_binding.py`) and returns its `query` in the normalized
payload (`app/ai/mcp_servers/brave_image_search_server.py`), which gives
placement a trustworthy anchor signal.

## Constraints and Non-goals

- Do not build or require an offline image-quality dataset or evaluation
  pipeline.
- Do not add an online vision-model reranker, image-captioning call, embedding
  service, or separate intent-classification model call.
- Do not fetch candidate image bytes during answer generation or persistence.
- Do not add a second model pass for placement.
- Do not change image-generation behavior.
- Do not redesign the public rich-item v1 marker contract. Adding one new item
  type to the existing discriminated union is in scope; changing the marker
  grammar is not.
- Do not add product, place, shopping, or other domain-specific card schemas.
- Keep images optional: failure to retrieve, select, register, or render an
  image must never fail or delay a valid text answer.

## Current Failure Modes

Each was verified against the code.

### Images are retrieved without visual intent

`tavily_search_include_images` defaults to `True` (`app/core/config.py:326`), so
ordinary text research manufactures image candidates unless the model explicitly
passes `include_images=false`.

### Candidate acceptance is recorded as relevance

The candidate builder checks URL scheme, exact duplicates, known dimensions, and
count limits, then records every accepted candidate as `selected`
(`app/ai/tool_execution.py:241`). Those checks establish transport eligibility,
not relevance, and the metric name hides the difference.

### Strong Tavily signals are dropped in the handoff

The adapter emits `source_title`, `source_domain`, `result_rank`, and
`result_score` (`app/ai/mcp_servers/tavily_server.py:194`), but the candidate
builder's copy list omits `source_title` (`app/ai/tool_execution.py:213-220`),
and rank and score are never used for ordering.

### Placement anchors on the weakest available text

`_image_placement_entries` matches an image using its title, alt text, and
provider description. For Tavily, the description is frequently absent; for
Brave it falls back to `title or source or hostname`
(`app/ai/mcp_servers/brave_image_search_server.py:123`), which is often just a
hostname. Anchoring on that text is what places a photo of fruit next to a
paragraph about Apple Inc.

### Marker emission is unreliable

Models drop the `rich:` prefix often enough that a repair function exists
(`app/core/rich_placement.py:185-201`). A design in which the marker is the only
path to display inherits that failure rate.

### Dimension gates never fire for Tavily

`_normalize_search_image` returns `{url, description, provider}` only
(`app/ai/mcp_servers/tavily_server.py:226-237`). Tavily supplies no width or
height, so the existing 320x180 minimums — and any other dimension-based gate —
apply solely to Brave candidates.

### Tavily automatic depth is inert

The adapter always sends an explicit `search_depth` alongside `auto_parameters`
(`app/ai/mcp_servers/tavily_server.py:129-137`). Tavily cannot choose a depth
when an explicit depth is present, so enabling
`TAVILY_SEARCH_AUTO_PARAMETERS` has no effect.

## Architecture

Five stages. The display decision moves from "did the model remember the syntax"
to "did the model ask for images."

| Stage | Owner | Rule |
|---|---|---|
| Visual intent | answer agent | calls `brave_image_search` for visual discovery, `tavily_search(include_images=true)` for source-bound visuals, concurrently with text research |
| Query formulation | answer agent | writes a disambiguated image query, not the verbatim chat question |
| Eligibility | deterministic code | existing gates plus aspect-ratio and junk-URL gates |
| Grouping | deterministic code | 1 eligible candidate becomes an `image` item; 2 or more from one image search become one `image_group` item |
| Anchoring | model marker, else query anchor | a marker's position wins; absent a marker, anchor on the image query text |

No new model call. The model invocation that already writes the answer still
owns final selection when it chooses to exercise it.

## Detailed Decisions

### 1. Explicit visual retrieval intent

Change the Tavily deployment default to `include_images=false`.

Routes defined by tool descriptions and shared media guidance:

- Ordinary web research calls Tavily without images.
- Research needing an image tied to a cited source calls Tavily with
  `include_images=true`.
- Focused visual discovery calls `brave_image_search`.
- An answer needing both broad research and visuals issues the Tavily and Brave
  calls in the same parallel tool block. The guidance explicitly forbids the
  serial pattern of searching, answering partially, then searching again for
  images.
- Provider failure or zero eligible candidates produces a text-only answer. No
  unconditional serial provider retry chain.

The agent's existing tool-selection turn is the intent decision. No separate
heuristic or classifier.

### 2. Image query formulation

The image query, not the image's metadata, is the relevance and placement
signal, so its quality governs the whole feature. Tool guidance requires:

- a concrete subject with any disambiguator the conversation implies — company
  versus fruit, language versus island, city versus person;
- a form qualifier when the useful visual has a form: `photo`, `diagram`,
  `map`, `chart`, `screenshot`;
- no verbatim reuse of the user's chat sentence, and no question words; and
- one subject per call. Two subjects means two calls.

The query is preserved on the candidate for anchoring and accessibility text.

### 3. Tavily search-depth contract

Add a per-call `auto_parameters: bool | None` to `tavily_search`, keeping
`search_depth` optional. Resolution is deterministic:

- An explicit `search_depth` wins: send that depth and `auto_parameters=false`.
- No explicit depth and effective `auto_parameters=true`: send
  `auto_parameters=true` and omit `search_depth`.
- No explicit depth and automatic parameters disabled: send the configured
  default depth, initially `basic`.
- `max_results`, raw-content behavior, and image behavior stay explicit because
  they control response size and rich-item creation.

Reject an empty query and a query longer than 400 characters with a structured,
non-retryable argument error rather than truncating silently.

Automatic parameters remain disabled by default for predictable cost.

**Deliberately out of scope.** The `topic`, `time_range`, `start_date`,
`end_date`, `include_domains`, `exclude_domains`, `country`, `exact_match`, and
`chunks_per_source` controls are moved to a separate Tavily-controls spec. None
of them serves image relevance, and including them roughly doubles the surface
area of this change.

### 4. Candidate eligibility and ordering

Retain the existing hard gates: HTTPS display URL, exact display-URL
deduplication, known minimum dimensions, supported raster MIME at delivery time,
and a bounded candidate count.

Add two cheap gates. Both are metadata-only arithmetic and string matching
inside the candidate loop that already runs; neither fetches or decodes an image,
so neither adds request-path latency.

- **Aspect ratio.** When both dimensions are known, reject
  `ratio > rich_image_max_aspect_ratio` (default 5.0) or
  `ratio < rich_image_min_aspect_ratio` (default 0.2). The bounds are settings so
  tuning needs no code change, mirroring `rich_image_min_width_px`. They are
  deliberately loose: a tight photo-shaped band such as 0.4-3.0 rejects tall
  infographics, full-page screenshots, wide timelines, and panoramas, which are
  often the most useful visual in an explanatory answer. The loose band catches
  only what the existing minimums miss, principally wide hero strips such as
  2000x200.
- **Junk URL patterns.** Reject display URLs whose path matches an unambiguous
  non-content marker: `favicon`, `sprite`, `spacer`, `1x1`, `pixel.gif`,
  `avatar`. The list is a bounded module constant, not a setting, so no
  unvalidated pattern can be injected through configuration. `logo` is
  deliberately absent: "what does the new X logo look like" is a legitimate
  visual query.

**State the coverage asymmetry.** Tavily supplies no dimensions, so for Tavily
candidates precision rests only on HTTPS, deduplication, junk-URL patterns,
provider rank and score ordering, and the anchor threshold in Decision 6. This is
an additional reason Brave is the primary visual path, and it must be documented
rather than implied away.

Preserve this normalized metadata through the handoff, including the currently
dropped `source_title`: provider; source URL, title, and domain; provider result
rank and score; source-bound versus query-level; original image URL and preferred
display thumbnail; provider description and dimensions when available; and the
originating image query.

Order Tavily candidates deterministically: source-bound before query-level, then
higher parent result score, then lower parent result rank, then original provider
order as a stable tie-breaker. Sorting never rejects an otherwise eligible
candidate. Because the adapter currently defaults a missing score to `0`, the
adapter must emit `None` for an absent score so the "missing sorts last" rule is
real rather than vacuous.

Do not invent an uncalibrated relevance threshold. Result score is an ordering
signal, not an acceptance boundary.

### 5. One rich item per image search

A `brave_image_search` result produces **one** rich item, not one per image.

- Exactly 1 eligible candidate: emit the existing `image` item.
- 2 or more eligible candidates: emit one `image_group` item whose cells are the
  first `rich_image_group_max_items` (default 3) eligible candidates in provider
  order. Surplus candidates are discarded, not held in reserve.

This is the Gemini-style row, and it fixes three problems at once: the model
copies one id instead of four, prompt noise drops, and grouped display becomes
possible without a per-image placement decision.

Tavily images remain individual `image` items. Grouping applies only to
deliberate image search.

New type in the existing discriminated union:

```
type: "image_group"
id: "imagegroup:tool:<tool_call_id>"          # valid under ^[A-Za-z0-9_\-.:]+$, <=128 chars
source: "image_search"
display_policy: "inline_only"
alt_text: <derived from the image query>
payload:
  items:                                       # 2..rich_image_group_max_items
    - url, mime_type, source_url?, description?, width?, height?
provenance:
  tool_call_id, tool, provider, query
```

The inventory presents at most `rich_auto_place_max_images` image-typed entries,
where a group counts as one. `rich_auto_place_max_images` default drops from 3 to
2 and governs the per-answer image-item cap on both the new and legacy paths.

### 6. Query-anchored placement

An image is displayed when the model writes its marker, or when the model ran an
image search and did not write one. Placement reuses the existing block
segmentation and overlap scoring in `app/core/rich_placement.py`, with the image
query replacing the description as the match text.

Algorithm, run once at persistence time:

1. If a valid marker for the item exists, respect it. Its position is
   authoritative. Repair of a known bare marker remains supported.
2. Otherwise tokenize the item's image query with the existing tokenizer, whose
   stopword list already discards `image`, `photo`, `picture`, and `view`.
3. Score each non-code prose block by the fraction of query tokens it contains.
4. Anchor after the highest-scoring block whose score is at least
   `rich_image_anchor_min_score` (default 0.34, roughly one third of query
   tokens). At most one item per block.

Fallback behavior differs by origin, and this is the recall mechanism:

| Origin | No marker written |
|---|---|
| `brave_image_search` | anchor at the best qualifying block; if no block qualifies, anchor after the first prose block holding at least 12 tokens as counted by the existing tokenizer after stopword removal |
| `tavily_search`, source-bound | anchor only if a block clears the threshold; no fallback |
| `tavily_search`, query-level | never auto-anchored; model marker only |

The Brave fallback is deliberate. A user who asks what something looks like, for
whom a successful ranked image search ran, must not receive an image-free answer
because the model omitted a formatting token. Tavily source-bound images get no
fallback because their query described text research, not the image.

`RICH_QUERY_ANCHORED_IMAGES_ENABLED` enables this path. While it is disabled, the
legacy description-anchored path runs unchanged, so the flag is a straight
rollback control. The legacy path is removed once the new path is verified.
Widget auto-placement is untouched by either flag.

The shared prompt still tells the model to place zero, one, or at most two image
items per answer, and that surrounding prose must remain useful without the
image.

### 7. Accessibility and caption ownership

`alt_text` derives from the image query when no genuine provider description
exists, replacing the current `"Image from tool result"` fallback. Alt text is
accessibility text, not a visible footer, so deriving it from the query the model
wrote is both accurate and safe. For an `image_group`, the item-level `alt_text`
describes the set from the query; each cell prefers its own provider description
and falls back to the query plus its source domain, so no cell announces itself
only as "Image".

`source_title` still never becomes the public `title` or `caption`: an article
title is not a trustworthy description of an image and must not render as a
figure footer. The renderer keeps sole ownership of the one visible footer, which
remains source attribution.

### 8. Delivery and partial failure

Every cell of an `image_group` gets its own authenticated `/web-images/{id}`
reference at persistence time, with no answer-path byte fetch. Externalization
currently handles `type == "image"` only and must iterate group cells.

`selected_image_file_parts_from_rich_items` filters on `type != "image"`
(`app/services/event_streaming/ai_sdk_projection.py:221`), so without change an
`image_group` yields no AI SDK file parts and images vanish for non-rich clients.
It must flatten group cells into file parts. Projections must skip unknown rich
types so a group degrades safely on older clients rather than rendering raw.

Failure is per cell: a failed cell shows the neutral state in place; all cells
failing collapses the group to one neutral `Visual unavailable` block. A failed
cell never leaves a caption without an image.

### 9. Metrics with truthful stage names

Replace the overloaded selection counter with bounded stage-specific outcomes:

- discovery result count by provider;
- candidate outcome: `eligible` or a fixed rejection reason, including the new
  `aspect_ratio` and `junk_url` reasons;
- presentation count: items included in the model inventory;
- **anchor outcome: `marker`, `query_anchored`, `fallback_anchored`, or
  `unplaced`**;
- final selection count: items surviving finalization;
- registration outcome for the protected reference; and
- existing render-time fetch outcome and duration.

The anchor outcome is the counter the previous revision lacked. `unplaced`
rising is the signal that images are not reaching users, and it is the rollback
trigger for Decision 6.

No query, caption, title, URL, tenant ID, conversation ID, or candidate ID may be
a Prometheus label. The existing unused `omitted` selection label folds into the
new stage counters. Keep existing metric names for one compatibility release,
emit new counters in parallel, migrate dashboards, then remove the old counter in
a separate change. Metrics failures remain non-fatal.

## Data Flow

1. The answer agent decides whether a visual materially improves the response.
2. It issues `brave_image_search` with a purpose-written query, and/or
   `tavily_search` with or without images, concurrently.
3. Provider adapters normalize image and source metadata, preserving the query.
4. Candidate construction applies safety and shape gates, orders candidates,
   groups Brave results into one item, and adds bounded items to turn-scoped
   context.
5. The bounded inventory exposes at most two image-typed entries to the answer
   model.
6. The model writes zero or more markers near supporting prose.
7. Finalization respects markers, then query-anchors any unreferenced
   image-search item per Decision 6.
8. Persistence keeps only placed items and converts each selected remote image,
   including every group cell, into an authenticated `/web-images/{id}`
   reference without fetching bytes.
9. The frontend fetches protected media per cell and renders its
   loading/loaded/failed component.

## Error Handling

- Invalid Tavily parameter combinations normalize deterministically; an explicit
  depth disables automatic depth.
- Invalid query length returns a structured non-retryable argument error.
- Provider timeout or failure leaves the answer text-only.
- No eligible candidates leaves the inventory empty and does not trigger a second
  provider automatically.
- A group whose cells all fail eligibility is never emitted.
- Invalid or invented markers use the existing warning contract.
- Reference-registration failure removes only the affected cell; losing every
  cell removes the item and its marker.
- Render-time timeout, MIME, size, dimension, DNS, redirect, or decode failure
  changes only that cell's state.
- No image failure may fail or delay message persistence or terminal streaming.

## Testing

Ordinary deterministic regression coverage with stubbed providers. No dataset,
evaluation harness, or scoring threshold. No test requires a model call.

### Retrieval and parameters

- Default Tavily call sends `basic`, disables automatic parameters, and does not
  request images.
- Automatic mode omits `search_depth` and sends `auto_parameters=true`.
- Explicit depth overrides automatic mode.
- `include_images` true/false/omitted behavior is deterministic.
- Empty and over-400-character queries return structured errors.

### Eligibility

- Aspect ratio at and beyond both bounds; a 2000x200 strip is rejected; a tall
  infographic at ratio 0.3 and a panorama at ratio 4.5 are accepted.
- Each junk URL pattern rejects; a URL containing `logo` is accepted.
- A Tavily candidate with no dimensions passes dimension and aspect gates
  unchanged.
- Existing HTTPS, duplicate, minimum-dimension, and candidate-count gates remain
  covered.
- `source_title` survives into candidate provenance and the inventory.
- Source-bound images precede query-level; score then rank order the rest; an
  absent score sorts last.

### Grouping and placement

- One eligible Brave candidate yields an `image`; two or more yield one
  `image_group` capped at three cells.
- A model marker is respected at its exact position and never double-placed.
- A Brave item with no marker and a matching block anchors after that block.
- A Brave item with no marker and no matching block anchors after the first prose
  block of at least 12 tokens.
- A Tavily source-bound item with no matching block is not placed.
- A Tavily query-level item is never auto-anchored.
- Ambiguous entities such as Apple, Java, Gemini, and Paris anchor by query text
  or not at all, never by scraped description.
- At most two image items per answer; the inventory lists at most two image-typed
  entries with a group counting as one.
- Widget auto-placement is unchanged when the image path is disabled.
- The legacy description-anchored path remains available only behind its flag.

### Integration

- Every group cell externalizes to a protected reference with no answer-path
  fetch.
- `selected_image_file_parts_from_rich_items` flattens group cells into file
  parts.
- A non-capable projection strips rich items and never renders a group verbatim.
- Registration and media failures preserve the complete text answer.
- Stream and persisted message projections contain the same placed item set.
- One failing cell leaves sibling cells rendered; all failing collapses to one
  neutral block.
- Metrics labels stay bounded and content-free, including the new anchor
  outcomes.

## Rollout

1. Fix the Tavily `auto_parameters` resolver and add its tests. No production
   default changes.
2. Flip `tavily_search_include_images` to `false` and update tool and prompt
   guidance, including query formulation. **This removes the misplacement cause
   first**, before any placement behavior changes, so the effect of the cause-fix
   alone is observable.
3. Preserve `source_title`, add ordering, add the aspect-ratio and junk-URL
   gates, and rename metrics to truthful stages.
4. Add `image_group`, grouping, query-anchored placement, and per-cell protected
   delivery behind `RICH_QUERY_ANCHORED_IMAGES_ENABLED`, shipped disabled.
5. Ship the `demo.py` group renderer and the AI SDK flattening fix.
6. Enable query-anchored placement. Watch `unplaced`, `fallback_anchored`,
   provider latency, registration, and fetch failures. Remove the legacy
   description-anchored path in a separate change.

Rollback controls: the global `INLINE_RICH_RESPONSE_ENABLED` kill switch,
`RICH_QUERY_ANCHORED_IMAGES_ENABLED`, and `rich_auto_place_enabled`. If
`rich_auto_place_enabled` is false, nothing auto-anchors regardless of the
query-anchoring flag.

## Acceptance Criteria

- Ordinary Tavily research creates no image candidates by default.
- Tavily automatic depth actually omits explicit depth when enabled.
- Source-bound Tavily images retain parent source title, URL, domain, rank, and
  score through selection.
- **A deliberate `brave_image_search` with at least one eligible candidate
  results in a displayed image, with or without a model marker.** Verified
  deterministically with a stubbed provider.
- A model marker always wins on selection and position when present.
- Ambiguous-entity cases are anchored by the model's image query or left
  unplaced; never by scraped provider description.
- Tavily query-level images never appear without an explicit model marker.
- Aspect-ratio and junk-URL gates reject their targets while accepting
  infographics, panoramas, screenshots, and legitimate logo queries.
- Widgets retain existing placement reliability.
- Selected images and every group cell retain protected delivery, source
  attribution, selected-only persistence, and per-cell graceful failure.
- Metrics distinguish eligibility, presentation, anchoring, final selection,
  registration, and delivery without content-bearing labels.
- No offline dataset, evaluation pipeline, vision call, embedding service,
  additional intent classifier, or second model pass is required.

## Deferred

- Tavily `topic`, `time_range`, date range, domain lists, `country`,
  `exact_match`, and `chunks_per_source` controls: separate spec.
- Signed short-lived media URLs in place of credentialed fetch.
- Any vision-based or learned relevance scoring.
- Post-completion image updates. Placed images remain terminal v1 rich items.
