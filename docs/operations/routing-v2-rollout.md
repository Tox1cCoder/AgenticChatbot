# Routing-v2 rollout

Operational procedure for deploying the routing-v2 workflow.

**Read this first:** routing-v2 is **not complete**. Tasks 1–11 and part of 13
of `docs/superpowers/plans/2026-08-26-production-routing-refactor.md` are
implemented; the RAG and Planning cutovers are not, and the live routing
evaluation was skipped. The [Known gaps](#known-gaps) section is not a
formality — it changes what this deployment can be claimed to have verified.

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

Expected: only the three known pre-existing failures
(`test_take100_api` ×2, `test_conversation_compaction_legacy_cleanup` ×1),
which fail identically on the pre-refactor commit.

### 3. Live routing evaluation

**Not available.** Task 12 was skipped as redundant, so there is no golden
dataset, no macro-F1, no per-language accuracy, and no structured-output
success rate. Deploy knowing routing quality is unmeasured, or build that
program first.

## Canary

Deploy to canary and watch these before widening. Each has a specific failure
it detects, not just a number to look at:

| Signal | Metric | What a spike means |
|---|---|---|
| Router failures | `routing.failed.*` by code | `routing_provider_unavailable` = credentials or adapter; `routing_invalid_output` = the model stopped honouring the schema; `routing_timeout` = provider latency past 8s |
| Router latency | `routing.latency_ms` p95 | Routing is on the pre-first-token path; regression here is felt as a slow start |
| Schema retries | `routing.attempts.2` share | Rising = the model needs a second call routinely; the bound is 2, so the next step is failure |
| Target races | `routing.target_race` | A custom agent was detached between inventory build and execution |
| Handoff corrections | `transition.rejected.*` | `revisited_target` or `over_depth` spikes mean agents are ping-ponging |
| Grounding | `grounding.abstained` vs `grounding.accepted` | Abstention is now **live**, not shadow. A spike means real answers are being withheld — check retrieval before assuming the gate is wrong |
| Finalization | `finalization.failed.*` | Any non-zero value is a turn that produced no answer |

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

## Rollback

Redeploy the previous artifact. There is no runtime switch and no old branch
retained in the new code; rollback is a deployment operation.

Old v1 checkpoints are ignored by v2 readers rather than migrated, so a
rollback finds them intact. v2 threads written during the canary are inert to
the v1 code for the same reason.

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

Enforced by `tests/test_routing_legacy_removal.py`, which asserts both what was
removed and what is still live.

1. **RAG and Planning still run their pre-v2 loops.** `RagExecutionGraphFactory`
   and `PlanningOrchestrator` are built and tested but are not what production
   executes. Grounding enforcement *did* ship on the live RAG path — that part
   is real.
2. **Routing accuracy is unmeasured.** No component has been run against a live
   model. Do not describe routing as validated.
3. **`_tool_node` / `_approval_node` are retained** — unreachable from the graph,
   but they still hold canvas-edit denial and HITL edit-rewrite behavior the v2
   middleware has not absorbed.
4. **`min_citation_coverage = 0.5`** remains a carried default, never selected
   from evaluation results. Grounding is now mandatory, so this threshold
   decides real abstentions.
