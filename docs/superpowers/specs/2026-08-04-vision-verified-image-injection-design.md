# Vision-Verified Image Injection Design

**Date:** 2026-08-04

**Status:** Approved for implementation planning

## Problem

The current remote-image path cannot establish semantic relevance. It ranks
Tavily page assets from text metadata such as the search query, source title,
description, dimensions, and result rank. Those fields describe the page, not
necessarily the pixels. In the supplied T1 trace, an author portrait labeled
`Moi` inherited a T1 article title and was therefore admitted beside the
answer even though the picture did not depict T1.

The same trace exposes a second failure. Tavily returned a large scraped-image
array before its useful research results. Tool-result offloading retained only
an image-heavy preview, so the model did not receive enough factual content and
issued additional Tavily searches. The trace contains three sequential Tavily
queries for one broad T1 request.

The previous deterministic selector improved structural safety, provider
ordering, and deduplication, but its metadata-only relevance premise is not
sufficient. The remote-image selection path will be replaced rather than
further tuned.

## Goals

- Never show a remote image unless its visible content is confidently relevant
  to the user's subject and materially supports the answer.
- Prefer a text-only answer over an uncertain or unrelated image.
- Make Tavily a text-and-source retrieval path only.
- Use dedicated image search as the only source of remote web images.
- Perform at most one remote image search and one batched visual-verification
  call per answer.
- Keep search and image discovery parallel where their dependency graph allows.
- Prevent redundant, near-duplicate Tavily network calls within a turn.
- Keep visual decisions transient. Do not persist classifications, confidence,
  labels, or rejected candidate details.
- Preserve existing structural validation, safe URL handling, deduplication,
  rendering contracts, and direct image sources.

## Non-Goals

- Training or hosting a custom image classifier.
- Building a durable image-label database or classification cache.
- Guaranteeing that every answer contains an image.
- Using source-page images as a fallback when dedicated image search is empty.
- Retrying failed image search or visual verification in the same answer.
- Applying remote-web verification to user uploads, RAG document images,
  generated images, or explicit image-producing tools.

## Architecture

### Model-facing research operation

Chat and search agents use one model-facing `web_research` operation instead of
independently coordinating Tavily and Brave. Its relevant input is:

```python
class WebResearchInput:
    query: str
    image_query: str | None = None
    max_results: int | None = None
    search_depth: str | None = None
```

`query` is the factual research query. `image_query` is a short, concrete visual
subject or `None` when an image would not improve the answer. The tool
description must explicitly say that an uncertain image decision uses `None`.

The server owns orchestration:

1. Tavily always runs with `include_images=False`.
2. When `image_query` is non-empty and the turn has not consumed its image
   budget, Brave Image Search starts alongside Tavily.
3. Tavily text and sources are returned to the model normally.
4. Brave results are converted to private transient candidates and do not enter
   the model inventory yet.
5. The visual verifier processes those candidates once. Only approved items are
   added to the rich-item inventory.

The underlying Tavily and Brave implementations remain independently testable,
but answer agents must not depend on issuing the two calls in a particular
prompted sequence. Their normal model-facing route is `web_research`.

### Turn-local research budget

The orchestration state tracks completed factual queries and the single image
search without persisting either outside the turn.

- A normalized exact or near-duplicate factual query reuses the existing
  Tavily result instead of making another network call.
- Query normalization uses NFKC case-folding and alphanumeric tokens, retaining
  short identifiers such as `T1`, `F1`, and `3M`.
- Two queries are near-duplicates when the intersection of their token sets is
  at least 75 percent of the smaller token set.
- At most two distinct Tavily network requests are allowed per user turn.
  Additional calls return the accumulated research result with a structured
  budget-reused indication rather than invoking Tavily again.
- At most one Brave request is allowed per user turn. Later `image_query` values
  reuse the existing approved image result and never launch another search.

These controls suppress repeated broad searches while retaining one distinct
follow-up query for genuinely incomplete research.

## Candidate Acquisition and Safety

Brave returns no more than six candidates to the verifier. Candidate order is
the provider's order after canonical URL/original-image deduplication.

Before visual verification, every remote candidate must pass structural and
transport validation:

- HTTPS URL with safe parsing.
- Public-network destination; redirects are revalidated.
- Allowed image MIME determined from response bytes, not only a URL suffix.
- Maximum response size and decoded dimensions from configuration.
- Existing minimum dimensions and aspect-ratio safety bounds.
- No duplicate display URL or original-image identity.
- Per-thumbnail timeout and a shared batch deadline.

Thumbnail downloads run concurrently. A failed, malformed, blocked, oversized,
or non-image candidate is removed individually. It must not abort the batch.

## Transient Visual Verifier

### Input

One fast, configurable vision-capable model call receives:

- The user's current request.
- The normalized `image_query`.
- The factual Tavily query and any bounded result titles already available when
  verification is dispatched.
- Up to six downloaded thumbnails.
- A stable, unguessable temporary ID for each thumbnail.
- Candidate title and description as untrusted supporting metadata.

Metadata cannot establish relevance by itself. The verifier prompt instructs
the model to decide from visible content, using metadata only to disambiguate
what it can actually see.

### Structured output

The verifier returns a bounded record for every recognized candidate:

```python
class VisualCandidateDecision:
    candidate_id: str
    depicts_requested_subject: bool
    materially_supports_answer: bool
    confidence: float
    content_kind: Literal[
        "photo", "portrait", "logo", "diagram", "map", "chart", "screenshot", "other"
    ]
```

Only IDs from the submitted batch are accepted. Duplicate IDs, missing fields,
out-of-range confidence, extra records, malformed output, or an unrecognized ID
cause the affected candidate to be rejected. A response-level parse failure
rejects the entire remote batch.

### Admission policy

A candidate is admitted only when all conditions hold:

- `depicts_requested_subject` is true.
- `materially_supports_answer` is true.
- `confidence` is at least the configured threshold, initially `0.85`.
- The visual content is not a generic author portrait, advertisement,
  navigation asset, decorative stock image, or unrelated page graphic.
- A portrait, logo, map, chart, diagram, screenshot, or similar specialized kind
  is used only when that kind is relevant to the requested subject. Explicit
  user requests for a kind override the generic preference against it.

The final pool preserves Brave order among approved candidates and contains at
most two image items. Zero approved candidates is a successful text-only
outcome, not an error.

Verifier decisions, confidence values, kinds, and rejection reasons stay in
turn-local memory only. They are excluded from public rich items, response
metadata, message persistence, logs, and metrics.

## Prompt Inventory and Placement

Only verifier-approved remote candidates enter `rich_item_candidates` and the
model-visible marker inventory. The answering model never sees rejected remote
candidate IDs, URLs, titles, or descriptions and therefore cannot place them.

Placement remains optional. The answer must be complete without an image, and
an approved marker is used only near prose it directly supports. At most two
remote image items may be placed in one answer.

User-uploaded images, RAG document images, generated images, and explicit
image-producing tool results retain their current direct-source path. They skip
the remote web relevance verifier but continue through schema, size, MIME, URL,
and rendering validation.

The following old remote-selection behavior is removed:

- Tavily image candidate construction and inventory admission.
- Source-title/query token overlap as semantic evidence.
- Provider-priority ranking as a substitute for visual relevance.
- Query-level or source-bound Tavily image fallbacks.

Canonical payload validation, safe URL rules, original-image deduplication,
inventory caps, marker validation, and persistence sanitization remain.

## Latency and Failure Policy

Tavily and Brave start concurrently when `image_query` is present. Visual
verification begins as soon as the Brave candidates and safe thumbnails are
available. It may use bounded Tavily result titles if that parallel request has
already completed, but it never waits solely for Tavily because the user request
and factual query provide the required subject context.

- Target normal added latency: less than two seconds.
- Initial configurable end-to-end verification deadline: four seconds.
- No verification retry in the same answer.
- No persistent or cross-turn decision cache.
- Brave failure, thumbnail failure, verifier timeout, provider refusal, invalid
  structured output, or no high-confidence match yields a normal text-only
  answer.
- Tavily failure remains a research error and is handled independently from the
  optional image path.

The response must never wait beyond the hard image deadline. The image path may
be abandoned while factual answer generation continues.

## Observability

Content-free aggregate metrics may record:

- Image search attempted/succeeded/failed.
- Number of candidates discovered, fetched, submitted, approved, and placed.
- Thumbnail batch duration and verifier duration.
- Text-only fallback reason category such as timeout, no-match, malformed
  output, or transport rejection.
- Tavily network calls, exact/near-duplicate reuse, and budget reuse.

Metrics must not contain image URLs, queries, titles, descriptions, verifier
labels, confidence values, candidate IDs, or user content.

## Acceptance Tests

### Supplied T1 regression

Recreate the essential `example_run.txt` conditions:

- User asks for information about T1 League of Legends.
- A candidate is the `Moi` author portrait from a T1 article.
- A separate candidate visibly depicts T1.
- The verifier rejects the author portrait and approves only the T1 visual.
- The answer inventory contains no marker for the portrait.

### Research payload and call budget

- Tavily is always invoked with `include_images=False` through `web_research`.
- Tavily tool content contains text and sources but no scraped image array.
- Three normalized near-duplicate factual queries result in one Tavily network
  request and turn-local result reuse.
- A genuinely distinct follow-up may use the second Tavily request.
- A third distinct request reuses accumulated results without network I/O.
- Multiple image requests in one turn result in exactly one Brave call.

### Strict semantic fallback

- An all-uncertain batch produces no remote rich item.
- A low-confidence candidate is rejected even when provider rank is first.
- A relevant candidate is rejected when it does not materially support the
  requested answer.
- Explicit requests for a portrait, logo, map, diagram, chart, screenshot, or
  product photo can admit that kind when confidence is high.
- Abstract questions use `image_query=None` and make no Brave or verifier call.

### Safety and robustness

- Malformed URLs, redirects to private networks, invalid MIME, oversized bytes,
  extreme dimensions, and duplicate originals are rejected per candidate.
- Hallucinated verifier IDs and malformed structured responses fail closed.
- Partial thumbnail failures do not discard other valid candidates.
- Verifier timeout produces a text-only answer inside the hard deadline.
- No verifier output fields appear in persisted or public metadata.

### Performance

- Tavily and Brave concurrency is proven with a controlled integration test.
- Thumbnail downloads are concurrent and obey the batch deadline.
- The verifier is called at most once with no more than six images.
- Metrics contain only aggregate counts, durations, and bounded reason enums.

## Migration and Rollout

The replacement remains behind the existing inline-rich-response capability
gate plus a new vision-verification rollout flag. During rollout:

1. Enable Tavily text-only behavior unconditionally to protect research output.
2. Keep remote image injection disabled unless the vision-verification flag is
   enabled and a verifier model is configured.
3. When the verifier is unavailable, fail closed to text-only rather than
   falling back to the old metadata selector.
4. Remove the old Tavily remote-image selection tests and code only after the
   new trace-shaped regression, safety suite, and persistence checks pass.

Rollback disables remote web image injection. It must not restore unverified
Tavily page images.
