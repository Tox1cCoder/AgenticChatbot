# Rich Image Selection Hardening Design

Date: 2026-07-29
Status: approved direction, pending written-spec review

## Goal

Make inline web images relevant, suitable, and naturally placed while preserving
the existing rich-item protocol, protected image delivery, and graceful
text-only fallback.

This design is a focused follow-up to
`2026-07-29-rich-image-injection-reliability-design.md`. That work established
safe delivery and resilient rendering. This follow-up fixes retrieval intent,
candidate quality, model selection, image placement, Tavily search controls,
and production observability.

## Constraints and Non-goals

- Do not build or require an offline image-quality dataset or evaluation
  pipeline.
- Do not add an online vision-model reranker, image-captioning call, embedding
  service, or separate intent-classification model call.
- Do not fetch candidate image bytes during answer generation or persistence.
- Do not change image-generation behavior.
- Do not redesign the public rich-item v1 marker or registry contract.
- Do not add product, place, shopping, or other domain-specific card schemas.
- Keep images optional: failure to retrieve, select, register, or render an
  image must never fail or delay a valid text answer.

## Current Failure Modes

### Images are retrieved without visual intent

`tavily_search` currently inherits a deployment default that enables images.
Ordinary text research therefore creates image candidates unless the model
explicitly passes `include_images=false`.

### Candidate acceptance is mistaken for relevance

The current candidate builder checks URL scheme, exact duplicates, known
dimensions, and count limits. Those checks establish transport eligibility, not
semantic relevance. Accepted candidates are nevertheless recorded as
`selected` in metrics.

### Strong Tavily signals are lost or unused

Tavily result-bound images carry the parent result title, URL, domain, rank, and
score. The current candidate handoff drops the parent title and does not use
rank, score, or query-level provenance when ordering candidates for the model.

### Post-processing overrides the model's omission

If the model chooses not to write an image marker, deterministic auto-placement
still considers every candidate. Its low keyword-overlap threshold can place a
wrong image for ambiguous entities such as Apple, Java, Gemini, or Paris. A
model omission must be treated as a valid decision, not as missing formatting.

### Tavily automatic depth is ineffective

The adapter always sends an explicit `search_depth`. Tavily automatic parameters
cannot choose a different depth when an explicit depth is present, so enabling
the current `TAVILY_SEARCH_AUTO_PARAMETERS` setting does not provide dynamic
depth.

## Architecture

The pipeline is precision-first and has five distinct stages:

1. **Retrieval intent:** the existing answer agent chooses whether it needs
   ordinary research, source-bound visuals, or focused image discovery.
2. **Provider retrieval:** Tavily or Brave executes with explicit visual intent.
3. **Eligibility and ordering:** deterministic code filters unsafe/unusable
   candidates, preserves provider signals, and presents a small ordered set.
4. **Model selection and placement:** the answer model explicitly selects an
   image by writing its existing rich marker near the supporting prose.
5. **Delivery:** the existing selected-only persistence, protected media route,
   and frontend state machine render the chosen image or degrade text-only.

No new model call is introduced. The model invocation that already writes the
answer owns the final image-selection decision.

## Detailed Decisions

### 1. Explicit visual retrieval intent

Change the Tavily deployment default to `include_images=false`.

The tool descriptions and shared media guidance define these routes:

- Ordinary web research calls Tavily without images.
- Research that needs an image tied to a cited source calls Tavily with
  `include_images=true`.
- Focused visual discovery, such as “what does X look like,” calls Brave Image
  Search.
- A request needing both broad research and focused visual discovery may call
  Tavily and Brave concurrently through the existing multi-tool execution path.
- Provider failure or zero eligible candidates produces a text-only answer. The
  adapters do not start an unconditional serial provider retry chain.

The agent's existing tool-selection turn is the intent decision. There is no
separate heuristic or classifier service.

### 2. Tavily search-depth and filtering contract

Add a per-call `auto_parameters: bool | None` option to `tavily_search` while
keeping `search_depth` optional.

Parameter resolution is deterministic:

- An explicit `search_depth` wins. The adapter sends that depth and sends
  `auto_parameters=false`.
- With no explicit depth and effective `auto_parameters=true`, the adapter sends
  `auto_parameters=true` and omits `search_depth`, allowing Tavily to choose.
- With no explicit depth and automatic parameters disabled, the adapter sends
  the configured default depth, initially `basic`.
- `max_results`, raw-content behavior, and image behavior remain explicit
  because they control response size and rich-item creation.

Expose bounded optional Tavily controls supported by the installed SDK:

- `topic`: `general`, `news`, or `finance`;
- `time_range`: `day`, `week`, `month`, or `year`;
- `start_date` and `end_date` in `YYYY-MM-DD` form;
- `include_domains` and `exclude_domains`, each capped at 20 normalized entries;
- `country` for general-topic localization;
- `exact_match`; and
- `chunks_per_source` from 1 to 3, sent only for explicit advanced depth.

Reject an empty query and a query longer than 400 characters with a structured
argument error rather than silently truncating it. Reject malformed dates and a
date range whose start is after its end. Tool guidance tells the model to split
complex research into focused subqueries.

Automatic parameters remain disabled by default for predictable cost. Callers
choose the automatic mode when query intent is genuinely ambiguous or detailed.

### 3. Candidate eligibility and ordering

Retain the existing hard gates:

- HTTPS display URL;
- exact display-URL deduplication;
- known minimum dimensions;
- supported raster MIME at delivery time; and
- bounded candidate count.

Preserve the following normalized metadata through the candidate handoff:

- provider;
- source URL, title, and domain;
- provider result rank and score;
- whether the image is source-bound or query-level;
- original image URL and preferred display thumbnail; and
- provider description and dimensions when available.

Order Tavily candidates deterministically:

1. source-bound images before query-level images;
2. higher parent result score before lower score;
3. lower parent result rank before higher rank; and
4. original provider order as the stable tie-breaker.

A missing or non-numeric result score sorts after a numeric score. A missing or
invalid result rank sorts after a valid rank. Sorting never rejects an otherwise
eligible candidate.

Do not invent an uncalibrated relevance threshold without evaluation data.
Result score is an ordering signal, not a hard acceptance boundary. Query-level
images remain eligible but are lower priority because they lack publisher-page
provenance.

Present at most four image candidates from a tool result to the model by default.
The prompt inventory includes the descriptive text and an allowlisted,
length-bounded `provenance.source_title`, but never raw URLs, signed values,
image bytes, or unrestricted provenance. The source title is not copied into the
public image `title` or `caption`, because an article title is not a trustworthy
description of the image and must not become a visible figure footer.

### 4. Model-owned selection and placement

An image is selected only when the answer model writes its existing
`<!--rich:<id>-->` marker. The marker's position remains the authoritative
placement anchor.

Add a separate `RICH_AUTO_PLACE_IMAGES_ENABLED` setting with a production
default of `false`. The existing deterministic auto-placement remains available
for widgets and other rich items whose creation implies display. The legacy
image auto-placement path can remain temporarily behind the new setting for
rollback, but it is not the production default.

Consequences:

- Model omission means “do not display this image.”
- An unreferenced image candidate is discarded by the existing selected-only
  finalizer.
- Explicitly selected images retain current persistence and rendering behavior.
- The server does not guess that a missing image marker is a formatting error.
- Marker repair for a known model-authored bare marker remains supported.

The shared prompt tells the model to select zero, one, or at most two images for
an ordinary answer. Multiple images are appropriate only when the user asks for
a comparison, sequence, examples, or gallery. Surrounding prose must remain
useful without the image.

### 5. Metrics with truthful stage names

Add bounded stage-specific outcomes to replace the overloaded selection meaning:

- discovery result count by provider;
- candidate outcome: `eligible` or a fixed rejection reason;
- presentation count: candidates included in the model inventory;
- final selection count: referenced image IDs that survive finalization;
- registration outcome for the protected reference; and
- existing render-time fetch outcome and duration.

No query, caption, title, URL, tenant ID, conversation ID, or candidate ID may be
a Prometheus label. Metrics failures remain non-fatal.

Keep the existing metric names for one compatibility release while emitting the
new counters in parallel. Mark the old selection counter deprecated in
operations documentation, migrate dashboards and alerts, then remove it in a
separate change. This feature must not silently break production monitoring.

During rollout, compare eligible, presented, and final-selection counts. This is
operational observation, not an offline evaluation dataset or quality-scoring
pipeline.

## Data Flow

1. The answer agent decides whether a visual materially improves the response.
2. It calls Tavily without images for ordinary research, Tavily with images for
   source-bound visual research, or Brave for focused image discovery.
3. The provider adapter normalizes image and source metadata.
4. Candidate construction applies safety/shape gates, orders candidates, and
   adds at most the configured cap to turn-scoped context.
5. The bounded inventory exposes the best candidates to the existing answer
   model.
6. The model writes zero or more valid markers near supporting prose.
7. Finalization persists only referenced images.
8. Persistence converts each selected remote image into an authenticated
   `/web-images/{id}` reference without fetching its bytes.
9. The frontend fetches the protected media and renders its existing
   loading/loaded/failed component.

## Error Handling

- Invalid Tavily parameter combinations are normalized deterministically; an
  explicit depth disables automatic depth.
- Invalid dates, topics, ranges, domain-list shapes, or query lengths return a
  structured non-retryable argument error.
- Provider timeout or failure leaves the answer text-only.
- No eligible candidates leaves the inventory empty and does not trigger a
  second provider automatically.
- Invalid or invented markers are handled by the existing warning contract.
- Reference-registration failure removes only the affected image and marker.
- Render-time timeout, MIME, size, dimension, DNS, redirect, or decode failure
  changes only the figure to the existing neutral failure state.

## Testing Without an Offline Dataset

Testing remains ordinary deterministic regression coverage; no dataset,
evaluation harness, precision benchmark, or scoring threshold is introduced.

### Tavily parameter tests

- Default call sends `basic`, disables automatic parameters, and does not request
  images.
- Automatic mode omits `search_depth` and sends `auto_parameters=true`.
- Explicit depth overrides automatic mode without hidden advanced-search cost.
- Image true/false/omitted behavior is deterministic.
- Topic, date, domain, country, exact-match, and chunk bounds are enforced.
- Empty and over-400-character queries return structured errors.

### Candidate tests

- Tavily parent source title survives into candidate provenance and inventory.
- Source-bound images precede query-level images.
- Result score and rank order otherwise equivalent Tavily candidates.
- Brave thumbnails remain the preferred display upstream.
- Existing HTTPS, duplicate, size, dimension, and candidate-count gates remain
  covered.

### Selection and placement tests

- A candidate omitted by the model is not auto-injected.
- Ambiguous entity examples such as Apple, Java, Gemini, and Paris remain
  text-only without an explicit marker.
- An explicit valid marker persists exactly one image at that position.
- Widget auto-placement remains unchanged when image auto-placement is disabled.
- Legacy image auto-placement remains available only when its separate rollback
  setting is enabled.

### Integration tests

- Selected images still externalize to protected references without an
  answer-path fetch.
- Registration and media failures preserve the complete text answer.
- Stream and persisted message projections contain the same selected image set.
- Existing frontend deduplication and whole-figure failure behavior remain
  unchanged.

## Rollout

1. Add the Tavily parameter resolver and tests without changing production
   defaults.
2. Preserve and order Tavily source metadata; rename metrics to truthful stages.
3. Add the separate image auto-placement flag and ship it disabled in production.
4. Change the Tavily image default to false and update prompt/tool guidance.
5. Monitor provider latency, candidate counts, final selections, registration,
   and fetch failures. Keep the existing global rich-response kill switch and
   the temporary legacy image auto-placement flag as rollback controls.

## Acceptance Criteria

- Ordinary Tavily research does not create image candidates by default.
- Tavily automatic depth actually omits explicit depth when enabled.
- Source-bound Tavily images retain parent source title, URL, domain, rank, and
  score through model selection.
- The server never injects an image that the answer model did not explicitly
  reference under production defaults.
- Ambiguous-entity regression cases remain image-free unless the model selects a
  candidate explicitly.
- Widgets retain their existing placement reliability.
- Selected web images retain protected delivery, source attribution,
  selected-only persistence, and graceful failure behavior.
- Metrics distinguish eligibility, model presentation, final selection,
  registration, and delivery without content-bearing labels.
- No offline dataset, evaluation pipeline, vision call, embedding service, or
  additional intent-classifier call is required.
