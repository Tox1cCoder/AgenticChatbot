# Web Tool-Loop Safety and User-Controlled Continuation Design

**Date:** 2026-09-04

**Status:** Approved design; implementation not started

**Scope:** Tool-result retrieval, model/tool-call guardrails, user-controlled
Continue/Stop, AI SDK and Streamlit parity, child tracing, and web-query quality.

## Problem statement

Recent production traces show a search specialist repeatedly discovering a web
tool, extracting an oversized page, and paging through the offloaded result with
`read_tool_result`. The trace is expensive and difficult to inspect, and the
model may exhaust its execution limits before giving the user an answer.

Related defects make the failure mode harder to control:

- model and tool limits currently terminate with an error rather than reserving
  an answer-only finalization step;
- Streamlit explicitly asks the backend to stop and reconciles the persisted
  partial message, while an AI SDK client normally only aborts its HTTP stream;
- the in-flight generation registry is process-local and cancellation is checked
  only between workflow events;
- direct Tavily and Brave invocations can appear as root LangSmith runs because
  the active runnable configuration is not propagated;
- the model can issue vague searches or stale-year queries even though runtime
  time context is available;
- Tavily extraction accepts no focused query, allowing a whole page to enter
  tool output, tracing, and blob storage.

These are one product problem: tool execution needs bounded evidence handling,
predictable completion, and a transport-independent generation lifecycle.

## Goals

1. Give every generation a bounded path to a useful answer before a hard limit.
2. Let the user stop active work and continue safe paused work deliberately.
3. Make Continue and Stop semantically identical for the AI SDK and `demo.py`.
4. Preserve exact workflow/checkpoint state without re-routing a continuation.
5. Prevent duplicate or falsely reported side effects across cancellation and
   continuation.
6. Make provider calls children of the conversation trace and bound traced data.
7. Improve web queries with typed intent, authoritative temporal normalization,
   focused extraction, deduplication, and evidence-based stopping.
8. Retain large-result storage without teaching the model to page raw blobs.

## Non-goals

- Reproducing ChatGPT's private search implementation.
- Adding automatic outer-loop continuation.
- Treating transport reconnection as semantic task continuation.
- Guaranteeing cancellation inside a provider that does not support it.
- Refactoring unrelated routing, RAG, planning, or UI code.
- Exposing unrestricted raw provider payloads to the model or public clients.

## Existing constraints

The in-progress production routing refactor remains authoritative:

- each new user turn routes exactly once;
- a resume uses the exact durable checkpoint and does not route again;
- all public answer text is validated and persisted before release;
- there is no automatic top-level continuation loop;
- side-effecting tool retries use durable execution receipts;
- only finalization reaches graph end.

This design extends those rules. A Continue action is not a new user turn and is
not automatic. It resumes a durable control interrupt for the same logical turn,
using a new execution epoch and a new assistant message where output is produced.

## Considered approaches

### A. Submit a visible or hidden user message saying "Continue"

This is easy to build, but it creates a new routed turn, can select another
specialist, repeats searches, loses exact budget state, and makes duplicate side
effects harder to reason about. Rejected.

### B. Finalize the partial answer and create a new checkpoint that skips routing

This gives a clean terminal boundary but invents checkpoint-cloning semantics,
duplicates state, and weakens the routing refactor's exact-resume guarantee.
Rejected.

### C. Pause at a durable control interrupt after bounded finalization

The graph reserves a tool-free synthesis call, validates its partial answer, and
reaches a typed `execution_budget_exhausted` interrupt. The message service
persists the validated partial before presenting it. Continue resumes that
checkpoint. Stop resolves the interrupt to graph finalization. A manual stop
also resumes from the last safe checkpoint when possible. Selected.

## Canonical generation lifecycle

Transport adapters must consume one lifecycle; they must not independently
infer whether work is running, stopped, or continuable.

```text
starting -> running
running -> finalizing_after_limit -> continuable
running -> stop_requested -> stopped
running -> completed
running -> failed
continuable -> continuing -> running
continuable -> stop_requested -> completed_partial
stopped -> continuing -> running          (only when resumable)
```

States with no active work are `continuable`, `stopped`, `completed`,
`completed_partial`, and `failed`. Only `completed`, `completed_partial`, and
`failed` are permanently terminal. `stopped` may carry a continuation when the
last durable checkpoint is safe.

Expected execution-limit exhaustion is not a public error. It produces either
`continuable` with a persisted response or, if synthesis fails, a deterministic
persisted fallback explaining that the work limit was reached and summarizing
the bounded evidence already collected.

### Stable identifiers

- `generation_id`: server-generated UUID allocated before workflow execution;
  identifies the generation across its continuation epochs and is the canonical
  Stop target.
- `logical_turn_id`: the original routed user turn; remains stable through all
  continuations.
- `execution_epoch`: starts at zero and increments for every Continue action.
- `checkpoint_thread_id`: the exact routing-v2 checkpoint ID.
- `continuation_id`: server-generated, opaque, durable, and scoped to one paused
  checkpoint plus expected epoch.
- `user_message_id` and `assistant_message_id`: message identities only; they are
  never reused as generation-control identifiers.

All generation-control commands include an idempotency key. Repeating Stop or
Continue returns the current authoritative state and does not create another
execution epoch.

## Persistence and cancellation control

PostgreSQL is authoritative for generation state and continuation eligibility.
The record contains:

- all stable identifiers above;
- owner user and conversation IDs;
- lifecycle status and monotonically increasing version;
- active agent ID and checkpoint identity;
- current model/tool budget counts and execution epoch;
- partial assistant message ID, when one exists;
- continuation availability and block reason;
- timestamps for started, stop requested, paused, and terminal state;
- a bounded terminal reason enum;
- references to tool-execution receipts, never raw tool output.

The existing Redis infrastructure distributes cancellation to the process that
owns the workflow task. The owning process keeps an in-memory task handle as an
optimization, not as the source of truth. Stop performs these steps:

1. authorize the generation against user and conversation;
2. atomically transition the durable row to `stop_requested`;
3. publish a cancellation message containing only the generation ID and version;
4. have the owner cancel the active model/tool task and observe the durable flag
   at every graph/tool boundary;
5. persist available partial text once;
6. classify in-flight mutation receipts;
7. transition to `stopped` and return the canonical generation snapshot.

The Stop HTTP request has a bounded wait. A timeout returns `stop_requested`, not
`stopped`. The UI continues polling or refetching until a terminal snapshot is
available. The registry is not deleted merely because the HTTP wait expired.

If Redis delivery is delayed or lost, the worker observes the PostgreSQL flag at
bounded safe points. If the process dies, restart reconciliation uses the durable
row and checkpoint rather than claiming the generation completed.

## Model and tool-call guardrails

### Two thresholds

Each execution epoch has a soft threshold and a hard threshold:

- The soft threshold reserves one model call for answer-only synthesis. Once
  reached, no further tools or handoffs are offered.
- The hard threshold protects against middleware, provider, or graph defects. It
  produces a typed internal control failure that is normalized into the same
  persisted continuable fallback, never a generic public execution error.

Limits are server-configured and declared in settings rather than existing only
as call-site fallback literals. Metrics record configured and consumed counts.

Current thread-wide limit accounting cannot back Continue because the resumed
thread remains exhausted. Per-epoch enforcement must use invocation/run limits
or an explicit budget epoch. Cumulative per-turn and account-level quotas remain
separate and cannot be reset by repeated Continue clicks.

### Forced finalization

At the soft threshold, middleware supplies a server-authored instruction that:

- says the evidence-gathering phase has ended;
- removes all tools from the model request;
- asks for the best direct answer supported by gathered evidence;
- requires uncertainty and missing information to be stated;
- forbids requesting or promising another automatic tool round.

The resulting answer enters the ordinary `validate_output` policy, then routes
to a dedicated `continuation_pause` control node instead of directly to graph
finalization. The pause payload carries only the validated public content and
bounded continuation metadata. The message service transactionally persists
that content as a partial assistant message before projecting any public answer
delta or continuation control. This mirrors the existing rule that an interrupt
message is persisted before it is presented.

Resuming the pause with `action="stop"` enters the universal `finalize` node
without creating or persisting the partial message a second time. Resuming with
`action="continue"` re-enters the preserved active specialist under the next
execution epoch. The public client therefore always has a stable answer before
it offers Continue, while `finalize` remains the only route to graph end.

## Continue behavior

Continue is application-level checkpoint resume, not AI SDK transport
`resumeStream()` and not a synthetic user message.

The command:

1. authorizes the opaque `continuation_id`;
2. uses compare-and-set on expected lifecycle version and epoch;
3. verifies the exact checkpoint still exists and belongs to the same user,
   conversation, and logical turn;
4. verifies no unresolved `outcome_unknown` mutation prevents safe replay;
5. reacquires the same-conversation coordinator;
6. increments the execution epoch and establishes fresh per-epoch limits;
7. resumes the exact interrupt with `action="continue"`;
8. preserves the original routing decision and active agent;
9. streams newly validated output as a new assistant message;
10. releases the coordinator on completion, failure, or another pause.

A duplicate Continue call attaches to or returns the already-created epoch. It
must not start a second workflow invocation.

Continue after a manual Stop is offered only when a durable checkpoint can be
resumed without unclassified side effects. If continuation is temporarily
blocked by reconciliation, both clients show that state rather than an enabled
button that is destined to fail.

## Stop behavior

Stop is available while the lifecycle is `starting`, `running`,
`finalizing_after_limit`, `continuing`, or `continuable`. The UI should react
immediately, but client-side stream abortion alone is not a successful Stop.

Both clients dispatch the explicit Stop request and close their local stream.
They reconcile from the Stop response or the generation-status endpoint. If the
stream remains connected, it may also receive a canonical stopped event.

Stopping before the first answer token records the lifecycle without inventing
an empty assistant message. Stopping after partial text persists that text once
with terminal metadata. A late Stop after completion returns the completed
snapshot. It never rewrites a completed message as stopped.

When the generation is already `continuable`, Stop performs no cancellation. It
accepts the persisted partial answer, resumes the control interrupt with
`action="stop"`, and transitions to `completed_partial` through universal graph
finalization.

### Mutation safety

For read-only tools, cancellation can resume from the last durable checkpoint.
For side-effecting tools:

- a completed durable receipt is reused;
- a provider-confirmed cancellation may be retried according to policy;
- an ambiguous in-flight outcome becomes `outcome_unknown`;
- Continue remains blocked until receipt reconciliation establishes whether the
  effect occurred;
- the UI explains the blocked state without exposing provider payloads.

## Public service and API contracts

One generation service owns lifecycle transitions. Separate transport endpoints
only project its canonical event stream.

Suggested endpoints:

- `POST /messages/stream`: existing internal stream; allocates a generation.
- `POST /api/chat/{conversation_id}`: existing AI SDK stream; allocates a
  generation.
- `POST /generations/{generation_id}/stop`: shared idempotent Stop command.
- `GET /generations/{generation_id}`: authoritative lifecycle snapshot.
- `POST /generations/{generation_id}/continue`: internal SSE continuation.
- `POST /ai/generations/{generation_id}/continue`: AI SDK UI Message Stream
  projection of the same continuation operation.

The current `/messages/stop` route remains as a compatibility adapter during a
bounded migration. It resolves the legacy user-message target to a generation
and delegates to the generation service. New clients use `generation_id`.

### Canonical events

- `generation_started`
- `generation_state_changed`
- `generation_stop_requested`
- `generation_stopped`
- `continuation_available`
- `continuation_blocked`
- `generation_completed`
- `generation_failed`

Every event includes generation ID, logical turn ID, execution epoch, lifecycle
version, status, and only the fields relevant to that transition. Event ordering
is monotonic per generation. Adapters ignore duplicate versions.

## Streamlit behavior

`demo.py` remains an internal-SSE consumer but no longer owns a special stop
protocol.

- Capture `generation_id` from the first lifecycle event, before user-message
  creation is required.
- Show Stop only in stoppable states.
- On click, dispatch the shared Stop command and trigger rerun/stream teardown.
- Reconcile the terminal snapshot rather than assuming any successful HTTP
  response means cancellation completed.
- Render Continue and Stop on persisted messages whose generation metadata is
  `continuable`.
- Render Continue on a stopped message only when `continuation.available=true`.
- Disable duplicate actions while their idempotency key is in flight.
- Restore controls from persisted message/generation state after rerun, browser
  refresh, navigation, or backend restart.

The partial preview is display-only. Persisted message content remains the
source of truth after reconciliation.

## AI SDK behavior

The AI SDK adapter consumes the same canonical events.

- Emit a stable `data-generation-state` part at stream start and reconcile it by
  ID as lifecycle versions arrive.
- Copy terminal continuation metadata into persisted assistant-message metadata
  so controls survive refetch and do not depend on transient parts.
- Map a connected `generation_stopped` event to the standard AI SDK `abort` part
  followed by a well-formed termination sequence where the protocol permits.
- The Stop button starts the explicit server Stop request and invokes
  `useChat.stop()` to abort local rendering; the separate command response or a
  refetch provides authoritative terminal state.
- Keep AI SDK `resume: false`. The product Continue button invokes the dedicated
  AI continuation endpoint and consumes its new UI Message Stream.
- Do not map Continue to `resumeStream()`, `regenerate()`, or
  `sendMessage("Continue")`.

The local client backend proxies Stop, status, internal Continue, and AI Continue
without reimplementing lifecycle decisions. It must propagate disconnect and
cancel its upstream reader without manufacturing an extra normal completion.

## Large tool-result handling

The blob store remains because large provider output must not enter model
context. The model-facing contract changes from sequential raw paging to focused
retrieval.

### Default model tool

Replace the paging-oriented instruction with a focused reader accepting:

- `blob_id`;
- required `objective` or `query`;
- optional structured path, section, or source selector;
- bounded maximum excerpts/characters.

It returns ranked excerpts with stable source offsets and a compact coverage
summary. It does not advertise a `next_offset` loop. A model can refine the
objective once, subject to the normal tool budget.

### Diagnostic raw reader

Raw offset paging may remain for administrators and tests, but it is not offered
to ordinary specialists or returned by `tool_search`.

Provider-specific parsers should first retain titles, URLs, relevant passages,
dates, and source identities. Repeated navigation, boilerplate, scripts, and
duplicated page text are removed before storage and tracing.

## Web-search and extraction contract

Ordinary agents discover product-level tools rather than raw provider methods:

- `web_search`
- `web_open`
- `image_search`

Raw Tavily/Brave tools remain available only to explicitly configured custom
agents or diagnostic scopes.

### Structured search intent

`web_search` accepts a typed search plan containing:

- concise query;
- user objective;
- freshness class: `timeless`, `recent`, or `as_of`;
- optional inclusive date range;
- locale/language;
- preferred source domains or source type;
- bounded result count.

The server derives absolute dates from injected runtime time. For terms such as
"today", "current", "latest", or "this year", it fills provider date filters
and repairs an obviously conflicting year before dispatch. It records the
repair in bounded trace metadata. It does not require a separate time-tool call
unless the user asks for actual clock/timezone information.

### Search quality and stopping

- Rewrite vague queries using conversation entities and the user's objective.
- Preserve exact names, quoted phrases, geography, and requested date context.
- Reject empty or generic searches that cannot be tied to the objective.
- Deduplicate normalized queries and canonical URLs per execution epoch.
- Prefer one focused follow-up query over broad repeated searches.
- Stop when the answer has sufficient independent evidence, when new results are
  duplicates, or when the soft execution threshold is reached.

### Focused extraction

`web_open` requires a URL and focused question. Tavily extraction always receives
that query and a bounded chunk count. Whole-page extraction without a question
is rejected in ordinary chat scope. Extracted output contains relevant chunks
and source metadata, not the raw page body.

## Tracing

Every model-visible provider call made for a conversation is a child of the
active conversation/tool span.

- Propagate the active `RunnableConfig`, callbacks, run name, and bounded tags
  through custom middleware and nested `ainvoke` calls.
- Do not create a fresh root span for direct Tavily or Brave execution within a
  conversation.
- Direct administrative provider tests may remain root traces and must use an
  explicit diagnostic tag.
- Offload and sanitize provider output before attaching it to trace outputs.
- Trace blob identifiers, sizes, hashes, provider status, query metadata, and
  bounded previews; never trace unrestricted page text.
- Link generation ID, logical turn ID, execution epoch, agent ID, tool call ID,
  and receipt status as trace metadata, not unbounded metric labels.

## Error handling

Expected control outcomes have typed status rather than generic errors:

- `execution_limit_reached`
- `stop_requested`
- `stopped`
- `continuation_stale`
- `continuation_already_started`
- `continuation_blocked_reconciliation`
- `generation_already_terminal`

Authorization failures do not reveal whether another user's generation exists.
Missing/corrupt checkpoints produce a non-retriable continuation-expired state
while retaining the persisted partial answer. Infrastructure failures keep the
last durable lifecycle state and return a retriable code.

## Observability

Record bounded metrics for:

- soft and hard limit frequency by agent kind and configured budget;
- forced-synthesis success/failure;
- Continue and Stop latency and idempotent duplicate count;
- time from stop request to worker acknowledgement and terminal persistence;
- resumable versus blocked manual stops;
- execution epochs per logical turn and cumulative tool/model calls;
- focused reader calls per blob and repeated-objective rate;
- search query repairs, duplicate suppression, extraction bytes before/after
  sanitization, and evidence sufficiency stops;
- orphan root provider spans and trace output size.

User, conversation, generation, checkpoint, and blob IDs remain out of metric
labels. They may appear in access-controlled logs and traces.

## Testing strategy

### Guardrail tests

- soft model/tool thresholds reserve exactly one tool-free synthesis call;
- tools cannot be called during forced synthesis;
- the partial answer is validated and persisted before continuation is emitted;
- hard-limit failure produces a persisted continuable fallback, not a public
  generic error;
- Continue receives a fresh epoch budget while cumulative quota remains intact;
- repeated Continue clicks start one epoch.

### Lifecycle service tests

- Stop before first token, during model streaming, during read-only tool work,
  during mutation work, after pause, and after completion;
- duplicate and concurrent Stop requests;
- Stop timeout returns `stop_requested` and never deletes live control state;
- owner-process cancellation and cross-process cancellation delivery;
- worker restart reconciliation;
- partial message persisted exactly once;
- safe manual Stop is resumable;
- `outcome_unknown` blocks Continue until reconciliation;
- Continue resumes exact checkpoint, active agent, and routing decision;
- no hidden or duplicate user message is created;
- same-conversation coordinator is released on pause and reacquired on resume.

### Transport parity tests

Run a shared scenario table through both internal SSE and AI SDK projections:

- lifecycle IDs and versions agree;
- Stop reaches the same terminal generation snapshot;
- Continue creates the same execution epoch and assistant content;
- expected limits are not mapped as errors;
- disconnect differs from explicit Stop only in recorded reason, not persistence
  safety;
- normal completion never renders Continue or Stop;
- controls survive reload from persisted state.

Add focused tests for `demo.py`, the AI SDK adapter, both canonical API routes,
the local client-backend proxies, and an end-to-end AI SDK `useChat` fixture.

### Tool and trace tests

- ordinary agents cannot discover the raw paging or unrestricted provider tools;
- focused blob retrieval is bounded and does not invite sequential paging;
- Tavily extraction without a focused question is rejected in ordinary scope;
- current/latest searches use the authoritative current date and provider
  filters, including a regression for a stale model-proposed year;
- duplicate queries and URLs are suppressed;
- nested Tavily/Brave calls share the conversation trace tree;
- trace outputs remain below configured size and contain no raw full page.

## Rollout

1. Add canonical lifecycle persistence and observe-only generation IDs.
2. Propagate IDs through both streams and client-backend proxies.
3. Add the explicit shared Stop path while preserving `/messages/stop`.
4. Replace process-local-only cancellation with durable control plus owner-task
   signalling.
5. Add soft finalization and continuation interrupts behind a server flag.
6. Add Continue/Stop projections to Streamlit and AI SDK together.
7. Introduce product-level web tools and focused extraction.
8. Replace model-facing raw paging with focused blob retrieval.
9. Fix trace parenting and enforce bounded provider outputs.
10. Enable by canary after transport-parity, restart, mutation-receipt, and live
    search-quality gates pass.

Rollback disables new UI controls and soft-limit pausing while retaining durable
generation rows and compatibility Stop behavior. It must not discard active
continuation checkpoints or downgrade an `outcome_unknown` receipt.

## Routing-refactor amendment required

Before implementation, amend the production routing design and plan to state:

- "no auto-continuation" permits explicit user-controlled continuation;
- execution-budget continuation is a typed durable interrupt;
- Continue is the same logical user turn and resumes without routing;
- per-epoch limits reset while cumulative quotas do not;
- validated partial messages may be persisted at a continuation pause;
- the same-conversation coordinator is released while paused and reacquired on
  resume;
- generation-control interrupts are distinct from tool-approval interrupts but
  use the same durable addressing guarantees.

The amendment should be integrated only after the currently active routing-plan
edits settle, to avoid mixing unrelated implementation changes.

## Acceptance criteria

1. A looping agent reaches bounded answer-only synthesis before any public limit
   error and offers Continue from a persisted response.
2. Continue resumes the exact checkpoint and agent without a routing call or new
   user message.
3. Stop is idempotent, cross-process, persistence-safe, and truthful about
   pending or unknown outcomes.
4. AI SDK and Streamlit produce the same durable generation state for every
   tested scenario; only their wire projections differ.
5. AI SDK Stop uses explicit server cancellation plus client abort, and semantic
   Continue does not use AI SDK transport resumption.
6. Large results remain offloaded, while ordinary models receive focused bounded
   excerpts rather than a raw paging loop.
7. Current/latest web requests receive authoritative temporal constraints, and
   unfocused whole-page extraction is unavailable in ordinary chat scope.
8. Nested provider calls appear beneath the conversation trace and trace outputs
   are bounded before raw provider text is attached.
9. Mutation receipts prevent duplicate side effects and block continuation when
   an outcome is genuinely unknown.
10. The routing refactor's one-route-per-user-turn, exact-resume, finalization,
    persistence, concurrency, and checkpoint-retention guarantees still pass.
