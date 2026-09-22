# Web Image Retrieval Quality Design

## Problem

Web images now reach the answer model and render correctly, but the retrieval
session can still choose an inferior representation or prevent a refined
search from contributing any image at all.

The production T1 turn exposed four independent defects:

1. The image provider request ignored `include_domains` and locale even though
   the model supplied both.
2. The source registry and the model's four-to-six-image inspection window
   shared fixed, append-only capacity. Once the first search filled them, the
   second image query omitted all ten newly retrieved candidates.
3. The answer model saw a 1200x675 team photograph but chose a 180x180 roster
   graphic because it valued exact roster recency over the user's literal
   request for a photograph.
4. Every search response repeated the complete accumulated source registry,
   and the synthetic evidence message repeated the source snippets again.

Live provider checks establish that this is primarily an integration problem,
not proof that Brave cannot retrieve useful images:

- The broad T1 query returned a real 1200x675 team photograph at rank four.
- The later five-player query was worse and contained mostly small Reddit
  thumbnails, so blindly replacing the first result set would reduce quality.
- A T1-domain-scoped query returned an official 1920x1080 League of Legends
  image at rank one.
- A Grace Ashcroft hair-close-up query returned a relevant 1024x576 result at
  rank one with high confidence.
- A Windows 11 Bluetooth-settings query returned relevant screenshots in its
  leading results.

## Goals

- Make existing domain and locale inputs affect Brave image discovery.
- Preserve provider-native relevance and quality signals that are currently
  discarded.
- Allow a later image query to improve the candidates shown to the model.
- Keep source IDs stable for citations while making the active image window
  replaceable.
- Require selected images to match the literal visual representation named by
  the user or `image_query`.
- Remove cumulative tool-result duplication without weakening grounding.
- Keep changes inside the existing web-research provider, session, projection,
  and evidence-context boundaries.

## Non-goals

- Do not add `alt_text` as a retrieval or ranking mechanism.
- Do not add a new `visual_kind` enum; `image_query` already carries terms such
  as photo, screenshot, close-up, diagram, and chart.
- Do not create a generic ranking service, semantic keyword taxonomy, or new
  provider abstraction.
- Do not replace Brave or Tavily with OpenAI-hosted search.
- Do not automatically publish provider results without model inspection.
- Do not promise a matching image when no provider returns one.

## Architecture

### Provider fidelity

`BraveImageSearchProvider` will forward locale as Brave `country` and
`search_lang` arguments. When `include_domains` is present, it will first run
the ordinary query and retain only results whose source domain is allowed. If
that does not fill the requested count, it will make **one** supplemental probe
against the first allowed domain with Brave's `site:` operator and merge the
allowed results. It will not fall back to disallowed domains because the
existing request contract defines `include_domains` as a restriction.

Every `include_domains` entry is normalized to a bare host before it is
compared or probed with, because the field carries whatever the model typed --
often a full URL.

Results will be deduplicated by canonical source URL plus image URL. The
implementation will avoid one combined multi-domain query because live
verification showed that a broad `OR site:` expression was dominated by
unrelated results from one allowed domain.

**One probe, not three.** Brave's image endpoint has a 2.5s timeout
(`brave_image_search_timeout_seconds`) and the free tier allows roughly one
request per second, while the whole tool call shares a 30s soft timeout with
the parallel Tavily text search. Three supplemental probes would spend up to
10s sequentially and risk HTTP 429; issuing them concurrently trades the
latency for the rate limit. One probe covers the real case -- the model names
one authoritative site -- at a worst case of two calls and ~5s.

`ProviderImageCandidate` will retain Brave's normalized confidence and source
domain. No custom semantic score or uncalibrated relevance cutoff will be
introduced.

### Three separate stores

`WebResearchSession` will keep three responsibilities distinct:

1. **Citation registry** — immutable `S#` records accumulated for the turn.
   IDs are never reassigned or reused.
2. **Candidate catalog** — bounded metadata from successful image searches.
   It takes at most `web_research_max_candidate_pool` candidates from each
   cohort and retains at most `web_research_max_candidate_catalog` (default 24)
   across the whole turn.

   This is a new setting rather than a product of the existing ones.
   `research_max_image_searches_per_turn` bounds nothing today:
   `ResearchBudget.reserve_image_search`, `may_image_search`, and
   `record_image_search` have no production callers, so deriving catalog
   capacity from it would dress an unenforced number as a budget. Wiring the
   image-search budget is separate work.
3. **Active vision window** — at most four candidates, or six for a gallery,
   whose validated previews are attached to the next model call.

The source registry will reserve enough bounded capacity for the candidate
catalog instead of limiting the entire turn to one image window. Image source
records will be admitted before the operation's text records so text cannot
consume the capacity reserved for that image cohort. This keeps stable citation
IDs without allowing unbounded evidence growth.

### Candidate ordering and replacement

Candidate ordering will be a small deterministic function in the existing
web-research service, not a new class or module. It will use only signals the
provider already supplies:

1. whether declared dimensions are adequate for display (640px longest edge);
2. Brave confidence (`high`, `medium`, `low`, unknown);
3. source-domain match against the request's preferred domains;
4. pixel area when dimensions are known;
5. original Brave rank as the final tie-breaker.

Adequacy leads and provenance sits below it. The provider filters a
domain-restricted cohort strictly, so every candidate it produced is preferred;
if that flag led, any candidate from a restricted search would outrank every
candidate from an unrestricted one, and a 200x150 logo from the named site
would displace a 4000x3000 photo. Provenance breaks ties between comparable
images; it does not buy a small one a window slot.

The adequate-display check is a quality bucket, not a hard rejection. A small
image may remain when it is the only relevant result, but it cannot outrank an
otherwise comparable full-size image merely because it appeared two provider
positions earlier.

This ordering is *quality only*. It cannot distinguish a team photo from a
roster infographic, and a larger graphic will outrank a smaller photo. Matching
the requested visual form is the answer model's judgement, made from the pixels
plus the literal representation contract below.

After each image search, the session merges the new cohort into the catalog,
deduplicates it, recomputes the active window, retains already-prepared images
that remain active, fetches newly active images, and releases pending
references that were evicted. A later low-quality cohort therefore cannot
blindly erase a stronger earlier result, while a better later result can enter
even when the previous window was full.

Candidate IDs remain stable for the lifetime of a prepared image. Newly active
images receive new IDs; an evicted ID is not rebound to different bytes.

### Literal representation contract

The evidence message will name the current `image_query` once before the image
blocks and instruct the answer model to match the requested representation
literally. Examples are behavioral, not a new schema: a roster graphic is not
a team photograph, concept art is not a UI screenshot, and a group scene is
not a close-up. When none of the visible candidates match, the model must
search with a materially different query or answer without an image.

Each candidate label will include trusted normalized facts already known by
the server: candidate ID, source ID, dimensions, and source domain. Retrieved
titles and descriptions remain untrusted evidence rather than instructions.

### Compact evidence projection

The public result for one `web_search` operation will contain only sources
newly admitted by that operation, named explicitly on the bundle rather than
inferred from `query_index` -- a page the model opens belongs to the operation
that opened it even though an earlier search admitted it. Source IDs remain
turn-stable, and earlier tool results remain in model history. Snippets are
bounded by source status, 1,200 characters for a search result and 3,000 for a
deliberately opened page, so the deep-read path is not gutted by a triage
bound.

### Three audiences for the registry

The complete registry, the operation delta, and the published list are three
different questions. The registry stays complete so grounding resolves any
`S#` the model was offered; the delta is what one tool result returns; and
`answer_sources` -- text pages, opened pages, and image pages backing an active
image -- is what a reader is shown. Without the third, a gallery turn would
show the reader roughly two dozen image-host pages it never cited.

The synthetic evidence message will retain the complete source-ID, title, and
URL mapping needed for image attribution, but it will not repeat full source
snippets that are already present in tool messages. Image previews and their
source mappings remain in the synthetic message because that is the only place
the answer model receives the actual pixels.

## Error handling

- If a domain probe fails, already returned allowed-domain candidates remain
  usable and the provider logs the bounded probe failure without erasing them.
- A provider rate limit does not erase candidates already in the catalog.
- An original-image fetch still falls back to the provider proxy.
- If a newly preferred image cannot be fetched or validated, the next ranked
  catalog candidate fills the active slot.
- Evicted pending image references are released immediately; selected
  references retain the existing closeout lifecycle.
- If the answer model has no vision capability, no active image is published.

## Testing

Tests will exercise observable behavior with deterministic provider fixtures:

- Brave calls receive locale and bounded domain-scoped queries.
- Scoped and broad results merge without duplicate images.
- Provider confidence and source domain survive normalization.
- A second search can displace a weaker active image after the first window is
  full.
- A worse second cohort does not displace a stronger official or full-size
  candidate.
- Stable candidate IDs are never rebound to different image bytes.
- Evicted pending references are released.
- One tool result returns only that operation's newly admitted sources.
- Synthetic evidence does not duplicate full source snippets.
- Existing grounding rejects unknown `S#` and `I#` references.

The deterministic fixtures will include the shapes that caused the production
failure: a small roster graphic, a full-size team photograph, a later cohort
of small social thumbnails, a hair close-up, and a Windows settings screenshot.

## Acceptance criteria

- Replaying the recorded T1 request offers an official or full-size team photo
  to the model ahead of the small roster graphic.
- Replaying the refined T1 query does not discard the stronger earlier photo
  and does not omit every new image merely because the first window was full.
- Grace-hair and Windows-settings queries keep their provider-leading visual
  results in the active window.
- The selected rich item still publishes through `/web-images/{id}` with its
  original validated resolution.
- A second search response does not repeat every previous source snippet.
- The published source list is narrower than the registry, while every `S#` the
  model was offered still resolves for citation.
- Focused web-research, grounding, rich-image lifecycle, and model-context tests
  pass without adding a parallel ranking or rendering subsystem.
