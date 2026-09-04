# Routing-v2 rollout

Operational procedure for deploying the routing-v2 workflow.

**Read this first:** the Planning and RAG cutover is **complete**. Tasks 1–9 of
`docs/superpowers/plans/2026-08-26-production-routing-refactor.md` are
implemented and there is no legacy execution path left to fall back to — no
feature flag, no compatibility alias, no old-checkpoint reader. Rollback is a
redeploy of the previous artifact and nothing else.

Two things are still true and change what this deployment can claim:
**routing accuracy has never been measured against a live model** — the release
gate exists and refuses until a reviewer approves the dataset — and **the live
acceptance gate has not been run**. See [Known gaps](#known-gaps).

## What changes at runtime

| Before | After |
|---|---|
| `selected_agent` carried routing, stickiness, preselection, and handoff state | `routing_decision` (immutable), `active_agent_id`, and append-only `agent_history` |
| Router called Gemini directly, defaulted to `chat_agent` on any failure | `RoutingService` resolves the configured provider strictly and returns a typed error |
| Canvas continuity and custom-agent stickiness bypassed routing | Both are router *context*; neither preselects |
| One checkpoint thread per conversation | One per turn: `routing-v2:{conversation_id}:{turn_id}` |
| Agents could reach `END` directly | Only `finalize` reaches `END` |
| Grounding was opt-in behind a flag, default off | Mandatory for every RAG result, no shadow mode |
| Failures returned English strings | Typed `WorkflowError` with `code`, `retriable`, `request_id` |
| Planning fanned out inside a tool call | Fan-out is parent-graph topology; each worker is its own checkpointed task |
| RAG ran its own inline loop | One compiled `RagExecutionGraph` behind a single specialist node |
| A worker's approval-gated tool call was **refused** | It **pauses**; the human decides per worker |
| A mutation replayed after a crash called the provider again | `tool_execution_receipts` makes it run at most once per execution key |

## Topology

The parent graph owns every decision that ends a turn. Two node groups matter
operationally:

| Group | Nodes |
|---|---|
| Subgraph specialists | `chat_agent`, `rag_agent`, `search_agent`, `image_generator_agent`, `canvas_agent`, `custom_agent` |
| Planning (parent-level) | `planning_model`, `planning_dispatch`, `planning_worker`, `planning_collect`, `planning_actions`, `planning_package` |

`TOOL_STAGE_NODES` is **empty**. Neither Planning nor RAG has a parent-level
tool stage any more, and that is the root fix: a loop inside a tool call is one
unit of work to the checkpointer, so a pause anywhere in it re-ran every
sibling that had already finished. `planning_worker` is reached by `Send` from
`planning_dispatch`, so each worker is a framework-managed task with its own
recorded result.

Only `finalize` reaches `END`. A node that answers without `finalize` is a bug,
not a shortcut.

### Limits

Every bound is enforced by the parent before a single worker starts.
Validation is all-or-nothing: an invalid ninth task leaves zero workers
running, and nothing is truncated — an oversized objective is rejected and
reported to the model, because a shortened objective is different work.

| Setting | Default | Enforces |
|---|---:|---|
| `PLANNING_WORKER_MAX_TASKS` | `8` | Total dispatched tasks across all waves in one turn |
| `PLANNING_WORKER_MAX_CONCURRENCY` | `4` | Workers live at once (passed as the invocation's `max_concurrency`) |
| `PLANNING_WORKER_MAX_DISPATCH_WAVES` | `2` | Dispatch calls per turn; a *rejected* proposal consumes no wave |
| `PLANNING_WORKER_OBJECTIVE_MAX_CHARS` | `4000` | One task objective (the schema also caps this at 4000) |
| `PLANNING_PARENT_CONTEXT_MAX_CHARS` | `12000` | The bounded plan scope a worker may see |
| `IMAGE_PREVIEW_MAX_PARTIALS_PER_IMAGE` | `512` | Partial preview frames per image on the graph's custom channel |

Rejection codes a model can receive on its dispatch call, all paired back on
the call's own `tool_call_id`: `dispatch_task_limit`, `duplicate_task_id`,
`objective_too_long`, `recursive_planning`, `unknown_agent`,
`agent_unavailable`, `dispatch_with_handoff`, `dispatch_wave_limit`,
`parent_context_too_long`, `invalid_dispatch_input`, `missing_tool_call_id`.

The preview cap is not optional tidiness: the graph's custom channel does
**not** backpressure its writer. A node emitted 2000 frames in 17 ms while a
deliberately slow consumer still held the first, so a producer of bulky frames
has to bound itself. Only partial frames are droppable; a final delivery
carries a reference, not bytes.

## Deployment order

Migrations first, then the artifact. The receipt table must exist before any
code that writes to it, or every mutating worker call fails at reservation.

```
a7b8c9d0e1f2  (previous head)
b8c9d0e1f2a3  add durable tool execution receipts     <- required by this release
c9d0e1f2a3b4  drop the stray chat_images sha256 index <- head
```

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic check   # must report no new operations
```

`b8c9d0e1f2a3` creates the `tool_execution_receipt_status` enum with
`checkfirst=True` and declares the column's type with `create_type=False`.
Both are required: `op.create_table` auto-creates a column's enum with *no*
`checkfirst`, so without `create_type=False` the `CREATE TYPE` runs twice and
the migration aborts on `DuplicateObject`.

### Drain v1 interrupts before cutting over

**Nothing in the code rejects a v1 thread ID at resume.** Resume passes the
exact `thread_id` stored on the HITL row, and a row written before the per-turn
scheme carries a conversation-named thread that the v2 graph cannot
meaningfully resume. Before deploying, either let pending v1 interrupts expire
or resolve them on the old artifact. Do not rely on a guard that does not
exist.

## Pre-deployment checks

### 1. Startup configuration

`RoutingService.validate_static_configuration()` runs at startup and fails the
process when the configured router provider/model cannot honour a schema. It
validates **static settings only** — user credentials and per-request model
overrides are user-scoped and are validated per request instead.

Confirm before deploying:

```
router_provider                        # must be gemini or openai
router_model                           # non-empty
routing_timeout_seconds                # default 8.0
routing_max_attempts                   # 1 or 2
conversation_turn_lock_timeout_seconds # default 30.0
```

A provider without an installed structured-output adapter fails closed. It does
not fall back.

### 2. Deterministic suites

```powershell
$env:LANGSMITH_TRACING='false'
.\.venv\Scripts\python.exe -m pytest -q -m "not live_provider"
```

### 3. Production scenarios against PostgreSQL

The replay, partial-approval, and crash-gap scenarios need a real checkpointer
and real receipt rows. `MemorySaver` keeps live Python objects, so a "resume"
against it can pass while the production saver fails on anything that does not
round-trip through the serializer.

```powershell
$env:TEST_DATABASE_URL='postgresql+psycopg://<user>:<password>@localhost:5432/chatbot_test'
.\.venv\Scripts\python.exe -m pytest -q tests/integration/test_planning_worker_resume_postgres.py tests/integration/test_tool_execution_receipt_repository_postgres.py
```

`TEST_DATABASE_URL` must be a **dedicated** database.
`tests/integration/conftest.py` fails the whole integration suite outright if it
addresses the same host/port/database as `settings.database_url`: these modules
run `Base.metadata.create_all` (bypassing Alembic) and delete seeded rows.

### 4. Live routing evaluation

**Not available.** The versioned multilingual routing gate (Task 8) is not
built, so there is no golden dataset, no macro-F1, no per-language accuracy,
and no structured-output success rate. Deploy knowing routing quality is
unmeasured, or build that program first.

## Canary

### Scrape it here

```
GET /metrics/routing        # Prometheus text exposition, unauthenticated
```

Every counter below is emitted by production code and reachable at that
endpoint. Both halves of that sentence were false until `2026-09-04`: eight of
the recorder's methods had no caller, and routing was the one observability
surface with no endpoint at all — so the guidance in this section pointed at
numbers that could neither move nor be read. If you are reading an older copy
of this runbook, distrust its canary section specifically.

Counter keys travel as the `name` label on `workflow_routing_counter`, so a
provider or model id containing a dot cannot break a scrape.

### What is emitted

| Signal | Metric | What a spike means |
|---|---|---|
| Router failures | `routing.failed.*` by code | `routing_provider_unavailable` = credentials or adapter; `routing_invalid_output` = the model stopped honouring the schema; `routing_timeout` = provider latency past 8s |
| Router latency | `latencies_ms` p95 (a list on the recorder's `export()`, not a counter) | Routing is on the pre-first-token path; regression here is felt as a slow start |
| Schema retries | `routing.attempts.2` share | Rising = the model needs a second call routinely; the bound is 2, so the next step is failure |
| Schema rejects | `routing.schema.invalid` vs `routing.schema.ok` | The structured-output contract is being violated outright |
| Target races | `routing.target_race` | A custom agent was detached between inventory build and execution |
| Route volume | `routing.completed.{agent}`, `routing.provider.{provider}.{model}`, `routing.inventory.{version}` | Distribution shift after a prompt or inventory change |
| Turn failures | `finalization.failed.*` by code | **Any non-zero value is a turn that produced no answer.** The matching `workflow.error.<code>.{retriable,terminal}` says whether a client retry can help |
| Handoffs | `transition.accepted.<from>.<to>` vs `transition.rejected.*` | `revisited_target` or `over_depth` spikes mean agents are ping-ponging |
| Workers | `worker.<status>.<agent>`, `worker.evidence_total` | A rising `worker.failed.*` share means dispatched work is not completing; `agent_execution_limit` is broken out under `execution.limit.<agent>.<kind>` |
| Grounding | `grounding.accepted` vs `grounding.accepted_with_findings` vs `grounding.clarification` | Grounding is **record-only** — nothing withholds an answer, so `grounding.abstained` does not exist. A rising `accepted_with_findings` share means validation is neutralising citations it cannot resolve; check retrieval |
| Finalization | `finalization.completed`, `finalization.policy.<id>` | Which output contracts actually ran, per turn |

The failure log carries the same information per turn, including the cause:
`Workflow turn failed: code=... retriable=... details={...}`. `details` is
allowlist-sanitised, so it is safe to read and is the only field separating,
say, a missing credential from a model without structured output.

Metric labels are allowlisted enums plus bounded provider/model/inventory
identifiers. Request, conversation, user, custom-agent-instance, message, and
evidence IDs are **never** labels — look for them in access-controlled traces.

### Verify the checkpoint namespace

New turns must write `routing-v2:` threads:

```sql
SELECT DISTINCT left(thread_id, 11) AS ns, count(*)
FROM checkpoints
WHERE thread_id LIKE 'routing-v2:%'
GROUP BY 1;
```

A conversation-id-shaped thread created *after* deployment means a turn reached
the graph without a `turn_id` — investigate rather than ignore.

### Verify receipts are being written

A mutating worker call with no receipt row means the durable path was bypassed,
which is the crash-gap regression returning silently:

```sql
SELECT status, count(*)
FROM tool_execution_receipts
WHERE created_at > now() - interval '1 hour'
GROUP BY 1;
```

Expect `completed` to dominate. `reserved` rows older than a few minutes are
in-flight work that lost its process — see below. A rising `failed` count means
providers are rejecting calls, which is a provider problem, not a receipt one.

## Reconciling `outcome_unknown`

This is the one state that needs a human, and it exists because the honest
answer is "nobody can say".

A LangGraph checkpoint is written *after* a node returns. A mutation that
reached the provider and then lost the process leaves a `reserved` row and no
record of what happened. On the next attempt:

- **the provider deduplicates on a key we supply** — retry under the *same*
  execution key. Safe, automatic, no operator involvement.
- **it does not** — retrying risks a duplicate charge/message/row and reporting
  failure risks denying a real effect. The row becomes `outcome_unknown`, the
  caller receives `MutationOutcomeUnknown` carrying the execution key, and it is
  **never retried again**, including by a later turn.

To find them:

```sql
SELECT execution_key, qualified_tool_id, conversation_id, turn_id, created_at
FROM tool_execution_receipts
WHERE status = 'outcome_unknown'
ORDER BY created_at DESC;
```

`ToolExecutionReceiptRepository.alist_unresolved(user_id=...)` returns the same
set scoped to one owner. Reconciliation is a manual check against the
provider's own records, keyed by `qualified_tool_id` plus the timestamp window.
Resolve it in the provider, then decide whether the row should be closed —
there is deliberately no code path that flips `outcome_unknown` to anything
else, because that decision cannot be made from inside the process that lost
its own outcome.

The execution key is derived from `(thread_id, dispatch_id, task_id,
tool_call_id)` and never supplied by the model. `provider_idempotency` is a
provider capability and is **not** part of the identity, so discovering that a
provider deduplicates does not move a call to a different row.

## Rollback

Redeploy the previous artifact. **Rollback is not a runtime switch**: there is
no feature flag, no compatibility alias, no fallback route, and no dual
execution path in the new code. Nothing can be turned off in place.

- Old v1 checkpoints are ignored by v2 readers rather than migrated, so a
  rollback finds them intact. v2 threads written during the canary are inert to
  the v1 code for the same reason.
- **v1 interrupts do not resume in v2, and v2 interrupts do not resume in v1.**
  A turn paused on a human decision under one artifact is lost when you swap
  artifacts. Drain pending interrupts before rolling in either direction.
- `tool_execution_receipts` is additive. The v1 artifact does not read it, so
  the table can stay; leaving it in place preserves the reconciliation record
  and means a roll-forward does not lose history. Do **not** downgrade
  `b8c9d0e1f2a3` while any `reserved` or `outcome_unknown` row exists — that is
  the only evidence a real effect may have happened.

## Retention and privacy

- **Expired HITL interrupts** — the retention job marks the row expired, then
  deletes the exact `thread_id` stored on it. Resume uses that same stored
  value, so cleanup never reconstructs an ID.
- **Deleted conversations** — cleanup enumerates exact
  `routing-v2:{conversation_id}:{turn_id}` IDs from persisted user-message IDs,
  plus the pre-v2 conversation-named thread. It never deletes by prefix: a
  prefix would reach turns the conversation does not own, including ones still
  paused on a human decision.
- **Active interrupts survive** ordinary expiry cleanup; only rows the
  repository reports as expired are reaped.
- **Idempotent** — reruns produce the same counts, so a retried job is safe.
- **Receipts carry no arguments or results beyond a bounded payload**, and
  `provider_receipt_id` is stripped from anything the model, the stream, or the
  trace can see.

## Concurrency

Turns in the same conversation are serialized by a PostgreSQL advisory lock
held from context snapshot through response persistence. Different
conversations run concurrently.

Acquisition is bounded by `conversation_turn_lock_timeout_seconds`; exhausting
it returns the retriable `conversation_turn_conflict`. Clients should retry
that code, not surface it as a failure.

The in-process backend exists for tests and is **rejected** when
`production=True`, because it coordinates nothing between workers.

## Known gaps

Removal is enforced by `tests/test_routing_legacy_removal.py`, which asserts
both what was deleted and what is still live.

1. **Routing accuracy is unmeasured.** No component has been run against a live
   model. Do not describe routing as validated. Task 8 (the versioned
   multilingual release gate) is not built.
2. **The final acceptance gate has not been run.** Task 9 requires live
   provider calls.
3. **`min_citation_coverage = 0.5`** remains a carried default, never selected
   from evaluation results. Grounding is mandatory, so this threshold decides
   real abstentions.
Closed since the last revision of this document: RAG and Planning no longer run
pre-v2 loops; `_tool_node`/`_approval_node` are deleted; the
`disable_outer_timeout` allowlist is empty, so every interactive tool call is
bounded; and the workflow metrics are wired and scrapeable at
`/metrics/routing`. See `docs/operations/tool-execution-policy.md`.
