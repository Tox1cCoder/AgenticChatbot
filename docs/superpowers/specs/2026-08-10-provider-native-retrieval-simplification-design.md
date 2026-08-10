# Provider-Native Retrieval Simplification Design

**Date:** 2026-08-10

## Goal

Improve the relevance and reliability of automatic visual enrichment while
removing the latency and complexity of the separate vision-model quality gate.
Align both Brave Image Search and Tavily Search with their current provider
contracts, retain deterministic safety checks, and leave the answer model as
the only LLM that synthesizes a response.

## Context

The current research path runs Tavily for factual retrieval and Brave for image
discovery. Brave candidates are downloaded and submitted to a second LLM for
pixel-level relevance classification before they can become rich items. Manual
testing shows that this gate adds substantial latency without reliably
improving the final images.

The Brave adapter also discards provider-native relevance metadata, drops a
result when its original image URL is missing even if its Brave-proxied
thumbnail is usable, and lets structured provider failures look like successful
empty searches. The Tavily adapter requests an LLM-generated answer even though
the application answer model performs the final synthesis, does not expose the
provider's most useful recency controls, and drops news publication dates.

## Goals

- Use Brave's native rank and confidence as the image relevance signal.
- Select high-confidence images first and use medium-confidence images only
  when no high-confidence candidate survives deterministic checks.
- Prefer Brave-proxied thumbnails for display and protected delivery.
- Remove the image-specific LLM call and its supporting fetch/cache machinery.
- Keep Tavily focused on ranked source retrieval rather than answer generation.
- Expose a small set of Tavily controls that materially improves current, news,
  and finance retrieval.
- Preserve text-only success whenever optional image enrichment fails.
- Preserve authenticated `/web-images/{id}` delivery and its SSRF, redirect,
  MIME, byte-size, decode, dimension, and aspect-ratio protections.

## Non-goals

- Building another custom semantic or lexical reranker.
- Inspecting remote image pixels before selection.
- Using Tavily page images as a fallback image source.
- Adding automatic provider retries within one answer.
- Exposing every Brave or Tavily API parameter to the answer model.
- Changing the rich-item marker, placement, persistence, or client rendering
  contracts beyond removal of verifier-specific byte caching.

## Architecture

`web_research` remains the single model-facing research tool. It starts Tavily
text retrieval and Brave image discovery concurrently when the existing
capability, feature, and turn-budget gates permit images. The two providers have
separate responsibilities:

- Tavily returns ranked factual sources and query-aligned content chunks.
- Brave returns ranked image candidates and provider-native confidence.

The application performs deterministic normalization and bounded selection. It
does not ask a second LLM to summarize Tavily results or judge image pixels.
The answer model receives Tavily sources plus the selected rich-item inventory
and remains responsible for the final prose and marker placement.

## Brave Image Search

### Request contract

The adapter continues using `GET /res/v1/images/search` with the subscription
token header. Each request explicitly sends:

- the focused `q` string;
- the existing bounded result count;
- `safesearch=strict` by default;
- `spellcheck=true`.

`country` and `search_lang` remain optional parameters. The server does not
guess them from the query or user locale. The current small answer-enrichment
count remains preferable to Brave's much larger endpoint maximum because only
one or two figure images, or one bounded gallery, can be presented.

### Normalization

For each result, preserve:

- provider rank;
- `confidence` (`high`, `medium`, or `low`);
- title, source domain, source-page URL, and crawl time;
- Brave-proxied thumbnail URL and thumbnail dimensions;
- original image URL and dimensions when present.

Also preserve the response's original and spell-corrected queries and
`extra.might_be_offensive` safety signal for internal diagnostics.

If `extra.might_be_offensive` is true, select no candidate from that response.
Strict SafeSearch remains enabled even with this response-level safety check.

The Brave-proxied thumbnail is the preferred display URL. A result remains
eligible when the thumbnail exists but `properties.url` does not. The source
page remains attribution metadata, not an image URL. The original image URL is
never exposed in public message metadata.

### Deterministic selection

Candidate construction retains the existing deterministic checks for:

- HTTPS display URLs;
- malformed and known junk URLs;
- duplicates;
- known minimum dimensions;
- known aspect-ratio bounds;
- supported MIME types and valid rich-item shape.

Aspect-ratio checks use original dimensions when Brave supplies them and fall
back to thumbnail dimensions otherwise. Minimum source-dimension checks apply
only to known original dimensions; the dimensions of Brave's resized proxy
thumbnail do not establish that the source asset is intrinsically too small.

Candidates with `low`, missing, or unknown Brave confidence are excluded.
Among surviving candidates:

1. Select from the `high` tier in Brave result order.
2. If and only if the high tier is empty, select from the `medium` tier in
   Brave result order.
3. Apply the existing figure or gallery presentation cap.

No score is invented and no metadata keyword reranker is added. A deliberate
image query that produces no surviving candidate yields a text-only answer.

### Failures

The adapter preserves HTTP status and provider error category internally.
Timeouts, HTTP 429, and HTTP 5xx responses are retryable search failures.
Authentication, subscription, invalid-argument, and other HTTP 4xx responses
are non-retryable. A provider failure is not normalized to an empty successful
result and is not reported as `no_match`.

No retry occurs inside the same answer. Optional image failure never converts a
successful Tavily result into a failed `web_research` call.

## Removing Visual Verification

The image path becomes discovery and deterministic selection rather than
discovery, download, LLM verification, and admission. Remove:

- the visual-verifier model call, prompt, schemas, and usage recording;
- verifier confidence, media-resolution, thinking-level, timeout, candidate,
  and rollout settings that exist only for pixel verification;
- the pre-verification thumbnail download batch;
- the turn-scoped verified-image byte hand-off;
- verifier-specific stage/outcome metrics and tests.

Rename verification-oriented orchestration and descriptions to discovery or
selection terminology. Continue using the existing discovery, candidate,
registration, placement, and final-selection metrics where applicable.

Message persistence registers selected Brave thumbnails as protected web-image
references without contacting the upstream. On render, `WebImageService`
securely fetches and validates the Brave-proxied image. A fetch failure affects
only that visual.

Rename `vision_image_verification_enabled` to
`remote_image_enrichment_enabled`, retaining its current default and gate
position. Its description refers to remote image enrichment rather than vision
verification.

## Tavily Search

### Default request

The default search request uses:

- `search_depth="basic"`;
- `max_results=5`, retaining the existing configurable hard cap;
- `auto_parameters=false`;
- `include_answer=false`;
- `include_raw_content=false`;
- `include_images=false`;
- `include_usage=true`.

The adapter passes `timeout=10` directly to `TavilyClient.search`. This is the
Tavily SDK/API timeout mechanism. Do not add an `asyncio` timeout wrapper or a
new application timeout setting.

`basic` remains the default because Tavily recommends it for general-purpose
search and it costs one credit. Explicit `advanced`, `fast`, or `ultra-fast`
requests remain supported. Automatic parameter selection stays disabled by
default because it may choose the two-credit advanced depth.

### Model-facing controls

Add only these high-value optional controls to `web_research` and forward them
to Tavily:

- `topic`: `general`, `news`, or `finance`;
- `time_range`: `day`, `week`, `month`, or `year`.

The tool description directs the answer model to use `news` for current events
and a time range only when recency is part of the request. It does not add
domain, country, exact-match, raw-content, or answer-generation controls to the
combined tool.

### Normalization

Preserve Tavily order and expose compact normalized results containing title,
URL, content, score, and `published_date` when supplied. Preserve usage,
request ID, response time, and selected automatic parameters for diagnostics.

Build a deduplication key by lowercasing the URL host, normalizing an empty path
to `/`, removing a trailing slash from non-root paths, and omitting the query
string and fragment. Keep the first-ranked result's original URL for citation.
When duplicates occur, merge only distinct content chunks without changing the
first result's rank.

Do not apply a global relevance-score threshold: Tavily documents score
filtering as use-case dependent, and a fixed cutoff would suppress niche
queries. Do not request raw page content during search. The existing
`tavily_extract` tool remains the follow-up path when snippets are insufficient
or a specific URL must be read.

### Failures

Map Tavily SDK/API errors into the existing structured provider-error contract:

- API timeout and HTTP 429 are retryable;
- invalid query, authentication, forbidden subscription, and usage-limit
  failures are non-retryable;
- HTTP 5xx failures are retryable.

`web_research` must recognize a structured Tavily error instead of treating it
as a successful result payload. No in-turn retry is added.

## Data Flow

1. The answer model calls `web_research` with a factual query and optional
   image query, image intent, Tavily topic, and Tavily time range.
2. Tavily search and Brave image discovery begin concurrently.
3. Tavily returns ranked source chunks without a provider-generated answer.
4. Brave results are normalized, deterministically filtered, tiered by native
   confidence, and capped.
5. Selected Brave items enter `selected_image_sink`, renamed from
   `verified_image_sink`; no pixel verification occurs.
6. The answer model receives Tavily results and available rich-item IDs, writes
   the final answer, and places only provided markers.
7. Persistence converts selected remote image URLs to protected references.
8. Rendering securely fetches and validates the proxied image on demand.

## Observability

Keep provider request duration, result count, candidate rejection, registration,
placement, and final-selection metrics. Replace verifier-only outcome names with
bounded discovery outcomes:

- `selected`;
- `no_match`;
- `skipped`;
- `unavailable`;
- `search_failure`.

Log one warning for operational provider failures and none for `no_match` or
`skipped`. Never include API keys, full upstream image URLs, user content, or
unbounded provider error bodies in metric labels or logs.

## Testing

Use red-green tests for each behavior change.

Brave adapter tests prove:

- `spellcheck=true` and strict SafeSearch are sent;
- confidence, rank, query correction, safety metadata, and thumbnail dimensions
  survive normalization;
- a proxied thumbnail without an original image URL remains eligible;
- HTTP 429/5xx/timeout and non-retryable 4xx errors stay distinguishable.

Selection tests prove:

- high-confidence results win in provider order;
- medium-confidence results are used only when the high tier is empty;
- low, missing-confidence, unsafe, malformed, duplicate, undersized, and
  invalid-aspect candidates are excluded;
- figure and gallery caps remain intact;
- no verifier model or thumbnail prefetch is invoked.

Tavily tests prove:

- default requests use basic depth, five results, no generated answer, no raw
  content, no images, usage reporting, and the SDK/API `timeout=10` argument;
- no local timeout wrapper or timeout setting exists;
- topic and time range validate and forward correctly;
- publication dates and diagnostics survive normalization;
- duplicate URLs merge deterministically without disturbing rank;
- provider failures map to the correct retryability.

Integration tests prove:

- Tavily and Brave still start concurrently;
- selected provider images reach the model's rich-item inventory without an
  LLM verification call;
- Tavily success plus Brave failure returns a successful text result;
- protected registration and render-time security checks remain active;
- explicit image opt-out and closed image gates avoid Brave calls.

## Documentation and Configuration

Update `.env.example`, `README.md`, active visual-enrichment documentation, and
tool descriptions to remove visual-verifier configuration and terminology.
Document Brave as the focused image provider and Tavily as the ranked factual
source provider. Document the fixed Tavily SDK/API timeout argument without
introducing a corresponding environment variable.

## Acceptance Criteria

- No image-specific LLM call occurs in `web_research`.
- No Tavily-generated answer is requested by default.
- Brave high-confidence candidates are preferred; medium is a fallback only.
- A valid Brave thumbnail is not rejected solely because the original image URL
  is absent.
- Tavily supports topic and relative time filters and preserves news dates.
- Provider errors are not confused with successful empty results.
- Tavily timeout handling uses `TavilyClient.search(timeout=10)` and no local
  timeout wrapper or new timeout setting.
- Tavily and Brave remain concurrent, bounded, and independently optional.
- Protected image delivery and render-time security validation remain intact.
- Focused tests, the complete test suite, linting, and diff checks pass.

## References

- Brave Image Search API:
  https://api-dashboard.search.brave.com/api-reference/images/image_search
- Brave Image Search guide:
  https://api-dashboard.search.brave.com/app/documentation/image-search/responses
- Brave Image Search agent skill:
  https://github.com/brave/brave-search-skills/blob/main/skills/images-search/SKILL.md
- Tavily Search API:
  https://docs.tavily.com/documentation/api-reference/endpoint/search
- Tavily Search best practices:
  https://docs.tavily.com/documentation/best-practices/best-practices-search
- Tavily Python SDK reference:
  https://docs.tavily.com/sdk/python/reference
