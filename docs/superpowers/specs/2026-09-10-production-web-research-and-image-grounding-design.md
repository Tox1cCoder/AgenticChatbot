# Production Web Research and Image Grounding Design

**Date:** 2026-09-10 (revised 2026-09-15)
**Status:** Approved design, revised after codebase review
**Scope:** Provider-neutral web research, citations, and image selection

## Purpose

Replace the competing search and image paths with one production web-research pipeline that behaves consistently across answer agents. The pipeline must retrieve current evidence when required, let the answer model inspect image candidates before selecting them, produce only verified citations and media references, and degrade honestly when retrieval fails.

The desired product behavior is comparable to a modern integrated web-search experience: the model can perform a quick lookup or a bounded multi-step investigation, sources are first-class and clickable, and images appear only when they materially support the answer. This design reproduces those observable behaviors without depending on a provider's private ranking or orchestration implementation.

## Goals

- Use one provider-neutral orchestration contract for text search, page opening, and image discovery.
- Support `none`, `quick`, and `agentic` research modes without reintroducing a
  search-specific call ceiling.
- Require research for claims whose accuracy depends on current or externally verifiable information.
- Give the existing post-tool answer-model call a bounded set of actual low-detail image candidates and stable candidate IDs.
- Make the server authoritative for citation validation and rich-image placement.
- Support one image by default, multiple images when the subject or request benefits from them, and an explicit gallery mode.
- Produce stable source records that both Streamlit and the AI SDK path can render.
- Add no classifier, reviewer, or separate vision-model call to the normal runtime path.
- Preserve useful existing validation, budgeting, rich-item, and streaming code while removing superseded paths.
- Provide deterministic automated release gates and optional offline quality evaluation without requiring a human test team.
- Keep the implementation small and traceable: one session object, one evidence
  middleware, and one grounding parser. Do not retain forwarding wrappers or
  duplicate helper layers after callers migrate.

## Non-goals

- Reproducing OpenAI's private search ranking, crawling, or internal orchestration.
- Building a general-purpose crawler or long-running deep-research product.
- Guaranteeing an image for every answer.
- Encoding special cases for example topics or incidents.
- Redesigning the conversation Inputs/Outputs/Sources panel. That is a separate project consuming the source records defined here.
- A repository-wide cleanup unrelated to web research, citations, or rich media.

## Project boundary and sequencing

This is the first of two coordinated projects:

1. This project defines and produces canonical web evidence, public sources, and selected rich media.
2. A later conversation-resource project will persist and present Inputs, Outputs, and Sources across Streamlit and AI SDK clients, including verified files created by device-local MCP tools.

The second project may consume the public source records from this project, but this project must not depend on the panel UI or the general artifact registry.

## Architecture

### Canonical service

Introduce a `WebResearchService` as the only active orchestration boundary for product web tools. Answer agents interact with provider-neutral requests and results. Tavily, Brave, OpenAI web search, or future providers are adapters; their response shapes and configuration do not leak into agents, finalization, persistence, or clients.

The service owns:

- request normalization;
- turn work budgets, cancellation, and transport-liveness policy;
- provider selection, retry, fallback, and circuit breaking;
- URL canonicalization and deduplication;
- page-open policy;
- image transport validation;
- turn-scoped source and image-candidate registries;
- compact model-facing evidence;
- public, JSON-safe provenance and operational outcomes.

Provider adapters remain small and independently testable. The initial configuration may use Tavily for text and Brave for images. A second text adapter can be enabled as an ordered fallback without changing agent or client contracts.

### Research modes

Research modes bound admitted work, not the number of distinct searches:

- `none`: stable knowledge for which retrieval is not required.
- `quick`: up to five admitted public sources and at most two focused page opens.
- `agentic`: up to eight admitted public sources and at most four focused page opens.

The answer model decides how many materially distinct searches the question
requires. Existing exact and near-duplicate rejection remains active, every
result reports the running search count, and the generation-wide soft/hard tool
budgets remain the runaway-loop ceiling. This preserves the current production
contract in `ResearchBudget`; this project must not restore
`research_max_search_calls_per_turn` or change `reserve_search()` back to a
boolean API.

No additional model call classifies the request. Existing routing supplies an explicit `requires_web` decision, and deterministic policy forces at least `quick` for an explicit request to browse or verify, a user-supplied page that must be inspected, or temporal/current-data cues such as latest status, news, schedules, prices, weather, laws, standards, and software versions. Tool intent supplies the normalized objective. The answer model may stop below the ceiling. It may escalate within the ceiling only when current evidence is insufficient.

Invalid requests, rejected duplicate queries, and provider calls that never start do not consume successful-search quota. Attempts and failures are still recorded for observability.

### Core contracts

The exact schema names may follow existing repository conventions, but the boundaries are fixed:

`WebResearchRequest`

- objective;
- normalized query;
- research mode;
- freshness requirement;
- optional allowed domains;
- locale or answer language when known;
- visual intent: `none`, `figure`, `comparison`, or `gallery`;
- turn, conversation, user, and device scope needed by policy.

`SourceRecord`

- stable turn-scoped source ID such as `S1`;
- canonical public URL;
- title and bounded snippet;
- retrieval status such as `search_result`, `opened`, or `snippet_only`;
- publication or modification time when the provider supplies one;
- provider and query provenance kept in bounded metadata;
- safety and validation status.

`ImageCandidate`

- stable candidate ID;
- protected or validated delivery URL;
- public source-page ID and URL;
- MIME type, dimensions, byte size, and content digest when available;
- bounded title and description;
- provider rank and retrieval provenance;
- validation status.

`WebEvidenceBundle`

- normalized request, actual mode, visual intent, and operation sequence;
- deduplicated source records;
- bounded opened-page evidence;
- validated image candidates;
- partial-failure records and timing summary;
- budget consumption.

`WebEvidenceBundle` is the private canonical result used inside the turn. Its
model-visible projection contains source evidence and candidate IDs but no
private origins or bytes. Its public tool-event projection contains source
records and bounded status only; unselected image descriptors are removed
before `tool_execution_end` reaches a client.

Provider-native payloads and untrusted raw HTML never become the public contract.

## Data flow

1. The router and agent establish the turn's research ceiling and freshness requirement.
2. The model calls the provider-neutral web tool with an objective, query, and visual intent. Provider-specific arguments are not exposed.
3. When visual intent exists, text retrieval and image discovery start concurrently. Each branch returns immutable provider records; shared registries are updated only after both branches settle, in deterministic text-then-image order, so completion timing cannot change IDs.
4. Results enter a turn-scoped source registry. URLs are canonicalized, deduplicated, safety-checked, and assigned stable source IDs. An image source page is admitted explicitly or the image is rejected; it never races text results for an ID.
5. Focused page opening runs only when snippets do not provide enough evidence or the user requested analysis of a specific page.
6. Image candidates are downloaded through the server's safe transport, bounded by time and bytes, MIME-sniffed, dimension-checked, deduplicated, and associated with their public source pages.
7. After each tool step, answer-model context is rebuilt from the current evidence bundle. It is not frozen before tool execution.
8. The evidence middleware runs after runtime-model resolution and before request-budget preflight. The post-tool request therefore budgets the exact compact textual evidence and bounded multimodal candidate set it sends. Each `candidate_id=I#` text label immediately precedes its image bytes.
9. The model should cite source IDs with `[[source:S1]]` tokens and requests image placement with `[[image:I1]]` candidate tokens. It does not author final rich markers.
10. One grounding parser admits only IDs and citation URLs present in the current turn registries. It is used incrementally for streaming and once over terminal content. Finalization converts valid source references to clickable links, accepts an already-authored Markdown citation only when its canonical URL is admitted, and performs server-owned rich-item placement.
11. The assistant message persists final linked Markdown, public source records, selected rich items, and a bounded search trace. Unselected images and private retrieval data are discarded from the public message.

## Citation and client transport

The source registry is authoritative. A citation is valid only when its source ID belongs to the current turn and the corresponding record passed public URL validation.

The model should emit `[[source:<source-id>]]` tokens. The streaming filter buffers partial tokens across chunks, resolves valid IDs, and converts them to numbered clickable Markdown links. If the model instead emits an ordinary Markdown citation, it is grounded only when its canonical URL resolves to the current turn's source registry; this equivalent admitted citation may support an explicitly selected image. Unknown, malformed, stale-turn, invented, or unadmitted references are rejected. The same parser is applied during terminal finalization so streamed and reloaded messages agree.

For a required-web turn, answer text is buffered until terminal grounding. Tool,
reasoning, and source events may still stream. When the runtime admitted web
sources but the draft cites none, or when it selects an image without an admitted
source citation, the specialist performs one bounded, tool-free correction while
the same turn-scoped source registry and validated image pixels are still
available. The correction must rewrite the complete answer, use admitted source
tokens, and select only inspected image candidates. Grounding and session
closeout happen after that correction.

The validator never authors or substitutes assistant prose. It validates the
model-authored result and either publishes it unchanged or returns a structured
workflow error. A required-web route with no admitted source is not made
impossible to complete by demanding a citation that cannot exist; the model's
own answer remains responsible for describing unavailable or uncertain evidence.
This keeps stream and history output aligned without presenting a server fallback
as though it came from the model.

The AI SDK projection additionally emits each public URL source as a native `source-url` UI part with `sourceId`, `url`, and optional `title`. Streamlit receives the same records through a canonical `sources_upsert` event. Both paths persist and reload the same source identities; neither parses tool prose to reconstruct sources.

The later conversation-resource project may aggregate these records into a Sources panel without changing this contract.

## Image selection and placement

### Candidate visibility

Metadata-only selection is insufficient. Every candidate offered for model selection must have passed transport validation and must be attached to the post-tool model request as an actual low-detail image alongside its stable ID. Candidate titles and provider confidence may supplement visual input but cannot substitute for it.

The model may select only IDs from the offered set by emitting `[[image:<candidate-id>]]`. Streaming and finalization reject any other ID, translate valid candidate tokens to the corresponding server-owned rich-item markers, and ensure the selected public registry contains those items. The server, not the model, creates final rich-item IDs and records. Selection is derived once from the canonical grounding result; middleware does not keep a second selection parser. Ordered deduplication preserves the model-authored token order.

Image selection is enabled only when the configured answer model and provider adapter accept image inputs. If they do not, the service returns text and source evidence without offering metadata-only image selection. It records a stable `answer_model_not_vision_capable` reason so the UI and operators can distinguish capability limits from provider failure.

### Balanced cardinality policy

- An ordinary request that benefits from a visual may select one image.
- A comparison, multiple named subjects, or materially broader coverage may select two to four images.
- An explicit request for a gallery or many images may select up to six.
- The model may select fewer than the allowed maximum, including zero.
- When no candidate is sufficiently relevant, current enough, or safely deliverable, the answer remains text-only and must not claim an image was shown.

The service should preserve authored order for multiple selected images. It may create either separate inline image items or one image-group item according to the existing rich-response renderer contract, provided source attribution remains per image.

### Cross-language behavior

Selection must not depend on English token overlap between the search query and answer prose. Relevance is established by the multimodal answer model and stable IDs. The active pipeline removes textual-overlap automatic image selection and placement. Once the model emits a valid candidate token, the server deterministically translates it to a deliverable rich item without performing a second relevance decision.

No token means no image. There is no query-anchor, description-overlap, or
provider-rank fallback that can insert an unselected candidate. This guarantees
that every displayed web image was available to the answer model, but it does
not claim that a probabilistic vision model will always make the best semantic
choice.

### Continuation and worker identity

Prepared image references have an explicit `pending`, `selected`, or `released`
lifecycle. A normal completion resolves tokens and releases unselected
references; selected references are marked only after their assistant message
is durable. A validated budget partial follows the same persistence rule before
Continue is advertised. Cancellation, hard failure, persistence failure, and a
human-approval interrupt leave no published image; pending rows are released by
the turn or the expiry sweeper. Image bytes and candidate mappings are
deliberately not checkpointed; after Continue, a model must perform a fresh
canonical search before it can select a new image. The previously persisted
partial retains its own grounded sources and selected image references.

Top-level sessions may use compact `S1`/`I1` tokens. Each Planning worker owns an
isolated registry. The parent remaps worker source records by stable URL order,
but worker web-image markers and rich items are stripped and their references
released before synthesis. Planning cannot publish a worker image until a
future parent-scoped multimodal pass explicitly offers those bytes to the
Planning model.

## Security and privacy

- Permit only configured public HTTP schemes, with HTTPS required in production.
- Resolve and reject loopback, link-local, private, reserved, and metadata-service targets before requests and after every redirect.
- Apply DNS rebinding protections consistent with the existing safe-fetch layer.
- Cap redirects, response bytes, content types, dimensions, decompression, and total image candidates.
- Enforce both per-image and aggregate per-turn downloaded-byte/model-byte budgets, use bounded fetch concurrency, and deduplicate successful candidates by content digest.
- MIME-sniff downloaded media rather than trusting extensions or response headers alone.
- Keep provider credentials, request headers, private image origins, raw HTML, and binary payloads out of public metadata and logs.
- Deliver remote media through existing protected storage/proxy routes when direct URLs are not suitable for durable rendering.
- Scope source and image IDs to the active turn to prevent cross-turn citation or media reuse without explicit history hydration.
- Treat search snippets, opened pages, and image pixels as untrusted evidence;
  model instructions explicitly forbid following commands found inside them.

## Reliability and latency

### Work budgets and progress-aware execution

Research has no fixed end-to-end wall-clock deadline. A guessed elapsed-time cap could interrupt a healthy provider request or page transfer, and this project has no measured production baseline that would justify one.

Work remains bounded structurally:

- `quick` may admit up to five results and open at most two pages;
- `agentic` may admit up to eight unique public sources and open at most four pages;
- distinct search count is model-driven and bounded by the existing generation-wide tool budget;
- ordinary visual research performs one image query and offers at most four validated candidates to the model;
- explicit gallery research may offer at most six validated candidates;
- every fetch retains byte, redirect, content-type, and decompression limits;
- every turn has aggregate downloaded-image-byte and model-image-byte ceilings.

Provider clients retain configurable connection and idle-read liveness controls so a dead socket or a provider that makes no progress cannot hang forever. These controls are not total research deadlines: an operation that continues to make valid progress is not cancelled merely because a fixed number of seconds elapsed. Explicit user stop, client disconnect, application shutdown, and upstream cancellation propagate promptly through all active work.

Latency is observed by mode, provider, operation, and outcome. Initial rollout establishes a production baseline; service objectives may be proposed later from measured distributions rather than embedded in this design as unsupported thresholds.

### Provider behavior

- Use an ordered, configurable provider chain.
- Do not hedge normal calls across providers, avoiding duplicate cost.
- Invoke fallback only after a transport-liveness failure, a retryable provider error, an open circuit, or an objectively unusable result set.
- Permit one bounded retry with jitter for transport failures, `429`, and `5xx` responses.
- Do not retry invalid requests, authentication failures, policy rejection, or unsafe URLs.
- Maintain circuit-breaker state and cooldown per provider configuration or credential pool, so one tenant's throttled credential cannot open a global circuit for unrelated tenants.
- Cache normalized requests briefly, with maximum age constrained by the request's freshness requirement.

### Degraded operation

- Image failure never discards valid text evidence.
- A failed page open leaves its source explicitly marked `snippet_only`.
- Partial valid evidence may support a qualified answer.
- When current information cannot be verified, the response states that limitation instead of silently using stale model knowledge.
- All failure and fallback paths emit stable reason codes and bounded public status; provider internals and sensitive diagnostics remain server-side.

## Observability

The rollout endpoint records low-cardinality operation counts by operation,
mode, outcome, and visual intent. Existing rich-image, model, and routing
telemetry supplies adjacent delivery and model-call signals. This is the
minimum release baseline; do not manufacture high-cardinality labels merely to
fill a dashboard.

After measured production baselines exist, expand telemetry where operationally
useful to cover:

- request count, success, partial success, timeout, retry, fallback, and circuit-open outcomes;
- provider and total research latency;
- results returned, sources admitted, pages opened, and sources cited;
- citation validity and coverage;
- image candidates discovered, validated, shown to the model, selected, delivered, and failed;
- answers that requested visuals but delivered none;
- cache hit rate and budget rejection;
- stream-to-history parity failures.

Logs and traces use request, turn, tool-call, source, and candidate IDs. They must not contain full provider payloads, image bytes, private URLs, credentials, or unbounded page content.

## Automated validation and evaluation

No human reviewer or runtime judge is required.

### Deterministic tests

- Mode selection, structural work budgets, cancellation propagation, transport-liveness behavior, retry eligibility, fallback order, and circuit transitions.
- Provider normalization using sanitized recorded responses; ordinary CI does not call live providers.
- URL canonicalization, deduplication, redirect policy, SSRF rejection, MIME sniffing, byte limits, and dimension limits.
- Dynamic context rebuilding after tool completion, including the class of defect where post-tool images are absent from a statically resolved prompt.
- Request-budget accounting after evidence injection, including fallback to a model with different vision capability.
- Citation tokens split across stream chunks, unknown-ID rejection, valid-link generation, and turn scoping.
- Selection restricted to image candidates actually shown to the model.
- A two-candidate end-to-end specialist test in which selecting `I2` renders the protected bytes and digest for `I2`, never provider-rank `I1`.
- A no-image-token end-to-end test proving that zero images are streamed, persisted, or appended and that every pending reference is released.
- Fail-closed interrupt handling and expiry cleanup for pending references.
- Multi-worker source remapping and web-image isolation before parent synthesis.
- Required-web streaming tests proving unsupported raw answer deltas are never published.
- Figure, comparison, multiple-entity, gallery, zero-valid-image, stalled-image-transport, and cancellation behavior.
- Stream, persistence, history reload, Streamlit, and AI SDK source/image identity parity.

### General quality matrix

Evaluation varies capabilities instead of recognizing named example topics:

- stable and time-sensitive questions;
- single-fact, comparison, exploratory, and multi-source requests;
- same-language and cross-language query/answer pairs;
- clear, ambiguous, and similarly named subjects;
- text-only, optional-image, required-image, comparison, and gallery intents;
- relevant, outdated, low-quality, duplicate, and misleading image candidates;
- provider success, partial response, malformed response, timeout, and total failure.

Synthetic provider recordings keep release checks deterministic. The small
JSON scorer is a supplemental contract smoke check, not an end-to-end pipeline
test; the focused pytest matrix is the release gate for provider normalization,
byte injection, grounding, streaming, persistence, and cleanup. Metamorphic
tests paraphrase prompts and change languages while asserting the same
behavioral invariants. Production code and tests must not contain topic-specific
routing or selection rules derived from reported examples.

An optional scheduled multimodal judge scores general subject match, temporal suitability, and usefulness. It runs offline, adds no user-facing latency, and is a trend/regression signal rather than the sole release authority.

An opt-in live canary may exercise configured providers and write a machine-readable report. Live provider availability is not a blocking dependency for the deterministic test suite.

### Release invariants

- Zero answers claim an image was displayed when no image was delivered.
- Every public citation resolves to an admitted source from that turn.
- Every selected image ID was included in the multimodal candidate set and passed validation.
- No private origin, unsafe URL, credential, or raw binary payload leaks into public metadata.
- Streamed and reloaded messages produce the same visible answer, sources, and selected images.
- Search and image work stays within configured call, source, page-open, byte, redirect, and candidate budgets.

## Migration, consolidation, and cleanup

Cleanup is a required delivery phase.

1. Inventory every active and compatibility-only search, page-open, image-discovery, image-selection, rich-placement, citation, metadata, and client-projection path.
2. Classify each relevant function as reuse unchanged, adapt behind the canonical contract, isolate for historical reads, or delete.
3. Add the canonical service and contracts using existing safety and validation functions where their semantics match. Avoid pass-through wrappers: adapt a caller directly or replace it.
4. Migrate all active answer-agent callers and both streaming projections.
5. Run deterministic parity tests and the general evaluation matrix, then operate a bounded canary cohort.
6. Remove superseded active paths, duplicate provider normalization, obsolete prompt instructions, misleading tool-result notes, compatibility flags that no longer guard a live path, and forwarding helpers left with one trivial caller.
7. Retain only a small, explicit compatibility reader for historical messages. Legacy formats must never remain active write paths.
8. Run dead-code and import checks, update configuration examples and operations documentation, and only then enable the canonical path generally.

Expected candidates for removal or replacement include:

- the out-of-band selected-image sink;
- metadata-only image inventory as the model's selection mechanism;
- competing automatic-placement decisions;
- duplicate legacy `images` projection on active v1 messages;
- provider-native response handling outside adapters;
- tool-result copy that promises unavailable rich items;
- raw provider web tools remaining independently callable when the canonical product tool owns the same behavior.

Expected reuse candidates include:

- existing research-budget primitives;
- URL, MIME, and image safety validation;
- rich-item schemas and renderers;
- protected image storage and delivery routes;
- canonical stream event infrastructure;
- citation chunk buffering and finalization concepts;
- provider tool resolution and focused-result capping where compatible with the new adapter boundary.

Deletion happens in the same delivery branch after active callers migrate and
parity checks pass. The cleanup is limited to feature-owned code and does not
authorize unrelated repository refactoring. Historical compatibility is one
narrow read-only projection; it must not share execution code with new turns.

## Rollout and rollback

- Use one short-lived rollout switch for the canonical path while the canary is active; avoid a permanent matrix of independently interacting feature flags.
- Begin with deterministic tests and provider recordings.
- Run shadow comparison without publishing the new answer when safe and useful.
- Enable a small canary cohort and monitor correctness, failure, cost, and latency metrics.
- Promote only after release invariants and operational thresholds hold.
- During the canary only, roll back by routing new turns to the prior path while preserving messages already written under the new versioned contract.
- Remove the prior active execution path, its rollout switch, and its settings after the canary window and cleanup gate complete. Rollback after cleanup is a code rollback, not a dormant legacy implementation.

## Acceptance criteria

The project is complete when:

- all answer agents use the provider-neutral web-research boundary;
- current-information questions reliably invoke at least quick research under deterministic policy;
- the post-tool answer model receives actual validated image candidates when visual intent applies;
- the server validates citations and image selections against turn-scoped registries;
- ordinary, comparison, and gallery image cardinality follows the balanced policy;
- Streamlit and AI SDK streams expose the same public source identities, with native AI SDK `source-url` parts;
- answers and history reloads contain consistent clickable citations and selected images;
- deterministic release invariants pass without live providers or human review;
- source, page-open, general tool-call, aggregate byte, redirect, concurrency, and candidate limits are enforced, while latency and progress are observable;
- superseded active code paths are removed or isolated as historical readers;
- configuration, rollout, rollback, and operations documentation are current.
