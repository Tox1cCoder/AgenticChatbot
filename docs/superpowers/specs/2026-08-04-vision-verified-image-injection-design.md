# Vision-Verified Image Injection Design

**Date:** 2026-08-04

**Status:** Approved for implementation planning

**Revision:** 2026-08-04, after verifying the original draft against the
codebase and replaying the `example_run.txt` trace through the live selection
path. The image-relevance replacement is unchanged in intent. The research
payload, tool-result offload, orchestration home, and candidate-safety sections
are corrected against what the code actually does.

## Problem

### Verified: remote image relevance is not merely unproven, it is inverted

The current remote-image path ranks Tavily page assets from text metadata: the
search query, source title, description, URL-derived dimensions, and result
rank. Those fields describe the page, not the pixels.

Replaying the trace's eight Tavily images through
`build_image_candidates_from_tool_result` and `select_rich_item_candidates`
reproduces the trace inventory exactly — `Moi` and `research`, ids `:2` and
`:5`. Two independent mechanisms produce that outcome:

1. `_description_overlap` tokenizes the *page* title as relevance evidence, so
   an author portrait inherits its article's title. The Leaguepedia asset
   labelled `research` scores highest of all (0.43) purely because its page
   title repeats "league legends esports". The one token that identifies the
   subject, `T1`, is discarded by the >= 3-character token filter and never
   participates in scoring.
2. `_url_dimension_hint` reads the Next.js `&w=3840` resize parameter as an
   intrinsic width and pairs it with the real height `565` parsed from the same
   URL. The resulting fake 6.80 aspect ratio trips the max-5.0 gate and rejects
   the only candidate that actually depicts the team. The portrait's own
   `1200x1600` plus `w=1080` stays inside the bounds and survives.

So the deterministic selector prefers the portrait *and* discards the correct
photograph. Metadata-only relevance is not a premise worth tuning further. The
remote-image selection path is replaced.

### Verified: three separate causes produced the repeated Tavily calls

The trace contains three sequential Tavily searches for one broad request. The
original draft attributed this to the scraped-image array alone. The array is
the volume; it is not the whole cause.

| Cause | Location |
| --- | --- |
| `images` is serialized before `results`, so any head truncation keeps image metadata and drops facts | `app/ai/mcp_servers/tavily_server.py` `_normalize_search_response` |
| The offload preview is a blind character prefix, `output_text[:preview_chars]`, of a payload past the 16000-char threshold, with a 4000-char preview | `app/services/tool_result_blob_service.py` `offload_if_large` |
| The offloaded blob is unreachable by the model. The inline notice says "use blob_id to read the full result", but no `blob_id` appears in the ToolMessage text and no model-facing tool reads blobs — only an authenticated HTTP route at `/tool-results/{blob_id}` | `app/ai/tool_execution.py` `apply_tool_output_offload`, `app/api/tool_result_blobs.py` |
| `include_answer` is never requested from Tavily, so the normalized `answer` field is always null even though the payload reads it | `app/ai/mcp_servers/tavily_server.py` `tavily_search` |
| No per-turn query dedup or research budget exists. `react_agent_max_iterations` is 50 | `app/ai/workflow/tool_loop.py` |

The third row is the decisive one. The model was told that content existed,
given no way to fetch it, and re-searching was its only available recovery.
Capping Tavily calls without repairing the preview would convert a bad answer
into a bad answer with less recourse, so the offload repair is in scope here.

## Goals

- Never show a remote image unless its visible content is confidently relevant
  to the user's subject and materially supports the answer.
- Prefer a text-only answer over an uncertain or unrelated image.
- Make Tavily a text-and-source retrieval path only, with no image array at all.
- Use dedicated image search as the only source of remote web images.
- Perform at most one remote image search and one batched visual-verification
  call per answer.
- Keep search and image discovery parallel where their dependency graph allows.
- Prevent redundant, near-duplicate Tavily network calls within a turn.
- Make a truncated tool result recoverable: preserve facts over image metadata
  in the preview, and give the model a real way to read the rest.
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
- Streaming prose before the image decision. Verification blocks the first
  token by design; see Latency and Failure Policy.
- Raising Tavily search depth or result count. Answer synthesis and the preview
  repair address content sufficiency at lower cost.

## Architecture

### Model-facing research operation

Chat and search agents use one model-facing `web_research` operation instead of
independently coordinating Tavily and Brave. Its relevant input is:

```python
class WebResearchInput:
    query: str
    image_query: str | None = None
    image_intent: Literal["figure", "gallery"] | None = None
    max_results: int | None = None
    search_depth: str | None = None
```

`query` is the factual research query. `image_query` is a short, concrete visual
subject or `None` when an image would not improve the answer. The tool
description must explicitly say that an uncertain image decision uses `None`.

`image_intent` selects the layout, and through it the count. `figure` — the
default whenever `image_query` is set — means up to two individual images placed
beside the prose they support. `gallery` means one grid item holding several
images, and is for requests that ask to see multiple instances or to compare
things: "show me the roster", "what do the team logos look like", "compare the
colours".

The model declares intent only. It never declares a number. How many images a
subject deserves is an unverifiable judgement, and unverifiable model judgement
about images is what this design exists to remove; the server maps intent to a
cap and the verifier decides which candidates survive. For a comparison the
count is derived from the things being compared, not guessed.

`web_research` is an **in-process internal tool**, registered the way
`internal::tool_search` and `internal::dispatch_subagents` already are. It must
not live in an MCP server. The bundled Tavily and Brave servers are stdio
subprocesses shared across conversations; they have no turn context, no request
identity, and no database access, so they cannot own a per-turn budget, a
verifier call, or byte caching.

The orchestrator owns sequencing:

1. Tavily always runs text-only. There is no image path to disable.
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

### Closing the direct-Tavily escape hatches

`tavily::tavily_search` is currently a hard system pin for the search agent, and
`tool_search` can load it for any agent. Two changes make `web_research` the
effective route:

- Replace `tavily::tavily_search` with the `web_research` pin in the search
  agent's required specs, and pin `web_research` for the chat agent alongside
  the existing image-search pin.
- Enforce the turn-local research budget at the tool-execution boundary, keyed
  on the search operation, so a `tavily_search` call loaded through
  `tool_search` is subject to the same dedup and call cap.

`tavily_search` stays discoverable as a text-only tool. It is not deleted;
`tavily_extract`, `tavily_crawl`, and `tavily_map` continue to use it as a peer.

### Tavily payload contract

- The `images` key is removed from the normalized search payload outright, not
  gated on a flag. The `include_images` and `include_image_descriptions`
  parameters are removed from the `tavily_search` signature so no caller and no
  model can reintroduce scraped page images.
- `include_answer` is requested, so the already-present `answer` field carries a
  synthesized answer instead of null.
- `results` is serialized first in the payload, ahead of every optional
  diagnostic key. Field order is part of the contract because truncation is
  order-sensitive.
- `search_depth`, `max_results`, and `include_raw_content` defaults are
  unchanged.

### Turn-local research budget

The orchestration state tracks completed factual queries and the single image
search without persisting either outside the turn.

- A normalized exact or near-duplicate factual query reuses the existing Tavily
  result instead of making another network call.
- Query normalization uses NFKC case-folding and alphanumeric tokens, retaining
  short identifiers such as `T1`, `F1`, and `3M`. No stopword list is used: the
  corpus is multilingual, a per-language table is scope the threshold does not
  need, and the trace outcome is identical either way.
- Two queries are near-duplicates when the intersection of their token sets is
  at least 75 percent of the smaller token set.
- At most two distinct Tavily network requests are allowed per user turn.
  Additional calls return the accumulated research result with a structured
  budget-reused indication rather than invoking Tavily again.
- At most one Brave request is allowed per user turn. Later `image_query` values
  reuse the existing approved image result and never launch another search.

Applied to the trace, the second query is a near-duplicate of the first (8 of 9
shared tokens, 88.9 percent) and is served from the turn cache. The third query is
genuinely distinct (6 of 9, 66.7 percent) and consumes the second budgeted
request. Three tool calls therefore become two network calls plus one reuse —
not one call. The acceptance tests assert those overlap counts explicitly, not
merely which side of the threshold they fall on.

**Corrected 2026-08-05.** An earlier draft stated 7 of 8 (88 percent) and 5 of 8
(63 percent). Those were the counts with stopwords removed, which contradicts the
no-stopword rule stated three paragraphs above — `of` is retained, so both token
sets hold nine members, not eight. The qualitative outcome is unchanged; only the
figures were wrong.

Reserving a search must be atomic. A caller checks the budget, awaits a network
call, then records the result, so two concurrent research calls in one turn would
both pass a bare check before either recorded, and the two-request bound would not
hold. The budget therefore exposes a single reservation step that refuses when a
reuse exists or when claimed-plus-completed searches already fill the budget. A
failed search does not release its slot: a provider error must not buy the model
another attempt at the same broken call.

## Tool-Result Offload Repair

This section is new. It is the fix for the cause that actually drove the retry
loop.

### Structure-aware preview

`ToolResultBlobService.offload_if_large` keeps its threshold and preview budget
but stops emitting a raw character prefix when the payload is a JSON object:

- Scalar identity keys (`provider`, `operation`, `query`, `total_results`) are
  preserved whole.
- `answer` is preserved up to a configured share of the preview budget,
  initially one quarter.
- The remaining budget is divided evenly across `results` entries in order, and
  each entry's `content` is truncated to its share. `title` and `url` are never
  truncated. If the even share falls below a configured per-result floor, later
  entries are dropped whole rather than shrinking every entry into uselessness.
- Any remaining array is omitted and reported by name and length, not sampled.
- A payload that is not a JSON object, or that fails to parse, keeps today's
  character-prefix behavior.

### Reachable full result

- The inline notice carries the actual `blob_id`, the omitted key names, and the
  full length, so the model knows precisely what it is missing.
- A new in-process `read_tool_result` tool returns a bounded slice of a blob.
  It resolves `conversation_id` and `user_id` from the tool-execution context
  and the repository query filters on **both**, extending the existing
  `get_for_user` scoping with the conversation. A blob from another conversation
  or another user is not found, and the tool reports not-found rather than
  distinguishing the two cases.
- Slice size is capped per call by configuration. The tool takes an offset so a
  large result can be walked, and it never returns more than the cap even when
  asked for more.
- The repository is synchronous; the tool executes it without blocking the event
  loop.

## Candidate Acquisition and Safety

Brave returns no more than six candidates to the verifier. Candidate order is
the provider's order after canonical URL/original-image deduplication. Brave
supplies real `width` and `height`, so no URL-derived dimension inference is
needed or permitted.

**Every transport and decode guard this design requires already exists** in
`app/services/web_image_service.py`: HTTPS-only parsing with credential
rejection, DNS resolution with a public-address assertion and an IP-pinned
transport that preserves Host and SNI, per-hop redirect revalidation, a
`Content-Length` and streamed byte cap, decompression-bomb bounds, and a MIME
type decoded from the bytes and cross-checked against the declared header. The
verifier reuses it through a new `fetch_url()` seam that fetches a URL without
first registering a persistent record. No parallel fetcher is written.

Existing minimum-dimension and aspect-ratio bounds continue to apply, evaluated
against decoded dimensions rather than provider or URL claims. Duplicate display
URLs and duplicate original-image identities are rejected as they are today.

Thumbnail downloads run concurrently under a per-thumbnail timeout and a shared
batch deadline. A failed, malformed, blocked, oversized, or non-image candidate
is removed individually and must not abort the batch.

### Verified bytes are cached, not refetched

Today a placed remote image is fetched at render time, after the answer is
persisted, so an image can pass every gate, be placed by the model, and then
fail at render — a broken figure with no fallback. Verification changes that:
the bytes of an approved candidate are already in memory and validated, so they
are stored with the web-image record and served from it. Each approved image is
fetched once, and a rendered figure is one whose bytes were already decoded
successfully. Cache write failure degrades to today's render-time fetch.

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

The model is built through the existing Gemini model factory under its own
configuration key, so the verifier tier is set independently of the answering
agent. Thumbnails are submitted at low media resolution.

Metadata cannot establish relevance by itself. The verifier prompt instructs the
model to decide from visible content, using metadata only to disambiguate what
it can actually see.

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

Only IDs from the submitted batch are accepted. Duplicate IDs, out-of-range
confidence, extra records, or an unrecognized ID cause the affected candidate to
be rejected — these are checked after parsing, so one bad record costs only
itself. A malformed record or any response-level parse failure rejects the
entire remote batch.

**Corrected 2026-08-05.** An earlier draft said malformed output rejected only
the affected candidate. It cannot: the response is validated as one strict
structured-output object, so a single unparseable field sinks the whole batch.
Tolerating per-record failures would mean parsing into a permissive shape and
validating each record by hand, which also weakens the schema that constrains
generation in the first place. Batch rejection is the simpler contract, it fails
safe, and it is what the code does — so the contract says that.

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

The final pool preserves Brave order among approved candidates. Its size follows
the declared intent: `figure` admits at most two individual image items,
`gallery` admits one grid item holding up to six. Zero approved candidates is a
successful text-only outcome in either mode, not an error.

Candidates are discovered, fetched, and verified individually — never as a
pre-grouped grid. Grouping happens only after admission, over the survivors.
Grouping before verification would both hide individual images from the verifier
and cap discovery below the candidate budget.

Raising the gallery ceiling costs nothing at verification time: the verifier
already receives the whole candidate batch in one call, so admitting six rather
than two adds no model call and no thumbnail download. The marginal cost is
render-time fetches for images actually placed.

Verifier decisions, confidence values, kinds, and rejection reasons stay in
turn-local memory only. They are excluded from public rich items, response
metadata, message persistence, logs, and metrics.

## Prompt Inventory and Placement

Only verifier-approved remote candidates enter `rich_item_candidates` and the
model-visible marker inventory. The answering model never sees rejected remote
candidate IDs, URLs, titles, or descriptions and therefore cannot place them.

Placement remains optional. The answer must be complete without an image, and an
approved marker is used only near prose it directly supports. In `figure` mode at
most two remote image items may be placed. In `gallery` mode there is exactly one
marker to place, because the grid is a single item — the answering model never
chooses how many images the grid holds, and must not describe the grid's contents
beyond what the prose already says.

User-uploaded images, RAG document images, generated images, and explicit
image-producing tool results retain their current direct-source path. They skip
the remote web relevance verifier but continue through schema, size, MIME, URL,
and rendering validation.

The media guidance block in the system prompt is rewritten: it describes
`web_research` with an optional `image_query`, and drops the instructions to
call `brave_image_search` directly, to pass `include_images=False`, and to issue
two searches in one tool block.

The following old remote-selection behavior is removed:

- Tavily image candidate construction and inventory admission. The `web_search`
  source disappears from the remote-discovery set; `image_search` and the direct
  sources keep their existing relative ordering.
- Source-title/query token overlap as semantic evidence.
- URL-derived dimension inference, including the resize-parameter heuristic that
  rejected the trace's correct image. Direct sources are unaffected, because
  dimension and aspect gates only ever ran for remote discovery sources.
- Provider-priority ranking of remote web candidates as a substitute for visual
  relevance.
- Query-level or source-bound Tavily image fallbacks.

Canonical payload validation, safe URL rules, original-image deduplication,
inventory caps, marker validation, and persistence sanitization remain.

## Latency and Failure Policy

Verification necessarily precedes answer generation, because the marker
inventory is part of the answering prompt. Streaming prose first and attaching
figures afterwards was considered and rejected: it would move placement from the
model to a server-side heuristic and change the streaming contract. The image
path therefore blocks the first token, under a hard cap.

Tavily and Brave start concurrently when `image_query` is present. Visual
verification begins as soon as the Brave candidates and safe thumbnails are
available. It may use bounded Tavily result titles if that parallel request has
already completed, but it never waits solely for Tavily because the user request
and factual query provide the required subject context.

- Configurable end-to-end image-path deadline, initially four seconds, measured
  from dispatch of the Brave call to the verifier verdict.
- The existing 2.5-second Brave timeout, the thumbnail batch deadline, and the
  verifier timeout are configured so their sum cannot exceed that deadline. The
  Brave timeout is lowered if it cannot fit.
- Expected added latency on a visual question is one to three seconds. The
  four-second cap is the guarantee; a sub-two-second typical case is a target,
  not a contract.
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
- Offload preview mode, omitted key count, and `read_tool_result` call count.

These extend the existing rich-image metrics recorder. Metrics must not contain
image URLs, queries, titles, descriptions, verifier labels, confidence values,
candidate IDs, or user content.

## Acceptance Tests

### Supplied T1 regression

Recreate the essential `example_run.txt` conditions:

- User asks for information about T1 League of Legends.
- A candidate is the `Moi` author portrait from a T1 article.
- A separate candidate visibly depicts T1.
- The verifier rejects the author portrait and approves only the T1 visual.
- The answer inventory contains no marker for the portrait.
- The T1 candidate is admitted on decoded dimensions; a `w=3840`-style resize
  parameter in its URL does not affect eligibility.

### Research payload and call budget

- The `tavily_search` result contains no `images` key under any argument, and
  the parameter that used to request them no longer exists.
- `results` precedes every optional key in the serialized payload.
- `include_answer` is requested and a returned answer reaches the model.
- The trace's three queries produce exactly two Tavily network requests: the
  second is served from turn cache as a near-duplicate, the third is distinct
  and consumes the second budgeted request.
- A fourth distinct request reuses accumulated results without network I/O.
- Multiple image requests in one turn result in exactly one Brave call.
- A `tavily_search` call loaded through `tool_search` is subject to the same
  dedup and cap.

### Offload recovery

- A Tavily payload above the threshold yields a preview containing several
  results with truncated content, and no image metadata.
- The notice contains the real `blob_id`, the omitted key names, and the full
  length.
- `read_tool_result` returns a bounded slice; a request above the cap is
  clamped.
- A blob belonging to another conversation or another user is reported
  not-found, with the two cases indistinguishable to the caller.
- A non-JSON offloaded payload keeps character-prefix truncation.

### Strict semantic fallback

- An all-uncertain batch produces no remote rich item.
- A low-confidence candidate is rejected even when provider rank is first.
- A relevant candidate is rejected when it does not materially support the
  requested answer.
- Explicit requests for a portrait, logo, map, diagram, chart, screenshot, or
  product photo can admit that kind when confidence is high.
- Abstract questions use `image_query=None` and make no Brave or verifier call.

### Layout and count

- `figure` intent admits at most two individual image items.
- `gallery` intent admits exactly one grid item, holding up to six approved
  images, and the answering inventory therefore exposes one marker.
- A gallery request whose candidates are mostly rejected still produces a grid
  from whatever survived; a single survivor is emitted as an individual image
  rather than a one-cell grid.
- Candidates reach the verifier individually even in gallery mode. A test must
  prove the verifier saw every candidate separately and that discovery was not
  capped by any grouping bound.
- An absent `image_intent` alongside a non-empty `image_query` behaves as
  `figure`.

### Safety and robustness

- Malformed URLs, redirects to private networks, invalid MIME, oversized bytes,
  extreme dimensions, and duplicate originals are rejected per candidate,
  through the existing web-image guards rather than a new fetcher.
- Hallucinated verifier IDs and malformed structured responses fail closed.
- Partial thumbnail failures do not discard other valid candidates.
- Verifier timeout produces a text-only answer inside the hard deadline.
- No verifier output fields appear in persisted or public metadata.
- An approved image is served from cached verified bytes; a cache write failure
  falls back to render-time fetch without breaking the figure.

### Performance

- Tavily and Brave concurrency is proven with a controlled integration test.
- Thumbnail downloads are concurrent and obey the batch deadline.
- The verifier is called at most once with no more than six images.
- The configured Brave, thumbnail, and verifier timeouts sum to no more than the
  image-path deadline.
- Metrics contain only aggregate counts, durations, and bounded reason enums.

## Migration and Rollout

The replacement remains behind the existing inline-rich-response capability gate
plus a new vision-verification rollout flag. Remote web images require a
configured Brave key; with no image provider the product is text-only by
construction, which is the intended fail-closed state.

1. Ship the Tavily payload contract and the offload repair first, unflagged.
   They improve research quality on their own and are independent of the image
   path.
2. Keep remote image injection disabled unless the vision-verification flag is
   enabled and a verifier model is configured.
3. When the verifier is unavailable, fail closed to text-only rather than
   falling back to the old metadata selector.
4. Remove the old Tavily remote-image selection tests and code only after the
   new trace-shaped regression, safety suite, and persistence checks pass.

Rollback disables remote web image injection. It must not restore unverified
Tavily page images, the `images` key, or URL-derived dimension inference.
