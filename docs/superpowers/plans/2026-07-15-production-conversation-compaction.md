# Production Conversation Compaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace legacy best-effort rolling summaries with durable, sequence-based, provider-aware conversation compaction that remains safe under restarts, duplicate work, transcript edits, and context-window pressure.

**Architecture:** PostgreSQL owns transcript ordering, structured memory, and one coalescing job row per conversation. Celery performs leased background compaction, while the request path uses the same token counter for preflight, bounded emergency compaction, and deterministic complete-turn reduction. Model-generated memory is validated JSON and is hydrated as an untrusted non-system message.

**Tech Stack:** Python 3.13, FastAPI, SQLAlchemy 2, Alembic, PostgreSQL, Celery/Redis, LangChain messages, Pydantic v2, tiktoken, prometheus-client, pytest, Ruff.

---

## Execution Rules

1. Use red-green-refactor for every production behavior. A test must fail for the expected missing behavior before implementation.
2. After every task, run its targeted tests, affected regression tests, Ruff on changed files, and `git diff --check`.
3. Do not begin the next task until the verification output matches the task's expected result. If it does not, use the systematic-debugging workflow, add or refine a reproducer, and iterate.
4. After every task, update its checkboxes and append the actual commands/results to **Progress Log**. Append non-trivial choices to **Decision Log**.
5. Stage and commit only files owned by this plan. Preserve unrelated pre-existing worktree changes.

## File and Responsibility Map

- `app/core/config.py`: only `CONVERSATION_SUMMARY_*` configuration and validation.
- `app/ai/token_counter.py`: provider/model-aware estimated and authoritative counting plus usage extraction.
- `app/ai/conversation_memory.py`: structured payload schema, validation, canonical rendering, and untrusted memory message.
- `app/ai/conversation_compactor.py`: threshold evaluation, complete-turn prefix selection, provider invocation, output validation.
- `app/ai/request_budget.py`: full-request preflight, emergency compaction orchestration, complete-turn trimming, overflow retry reduction.
- `app/models/conversation_summary_job.py`: durable coalescing job and status values.
- `app/models/conversation_memory_summary.py`: one-to-one structured memory row.
- `app/models/conversation.py`, `app/models/message.py`: monotonic sequence state.
- `app/repositories/conversation_compaction.py`: atomic message persistence, job upsert/claim/complete/retry/reconcile, invalidation, CAS.
- `app/repositories/message.py`: sequence-based prompt and compaction windows only.
- `app/workers/conversation_compaction.py`: Celery summary, reconciliation, and backfill tasks.
- `app/observability/conversation_compaction.py`: content-free metrics and health aggregation.
- `app/api/health.py` and `app/main.py`: summary-specific health endpoint.
- `app/alembic/versions/x1y2z3a4b5c6_production_conversation_compaction.py`: forward schema transition.
- `tests/test_conversation_summary_config.py`: configuration boundaries.
- `tests/test_token_counter.py`: English, Thai, JSON, tools, images, unknown models, usage extraction.
- `tests/test_conversation_memory.py`: payload security and non-system hydration.
- `tests/test_conversation_compactor.py`: triggers, turn selection, validation, and retry classification.
- `tests/test_request_budget.py`: full accounting, trimming invariants, emergency and provider-overflow paths.
- `tests/test_conversation_compaction_repository.py`: repository state transitions with fast test doubles.
- `tests/integration/test_conversation_compaction_postgres.py`: PostgreSQL constraints, concurrency, atomicity, leases, CAS, invalidation.
- `tests/test_conversation_compaction_tasks.py`: Celery orchestration and reconciliation.
- `tests/test_conversation_compaction_health.py`: metrics/health content-safety.

### Task 1: Conversation-summary configuration

**Files:**
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Create: `tests/test_conversation_summary_config.py`

- [x] **Step 1: Write failing configuration tests**

Cover defaults, zero disabling one threshold, rejection when both thresholds are zero while enabled, numeric bounds, `soft < hard`, keep-turn/message-threshold compatibility, retry/lease positivity, production provider/model requirements, and production rejection of `preview`.

```python
def test_summary_defaults_are_stable():
    settings = Settings(_env_file=None)
    assert settings.conversation_summary_provider == "gemini"
    assert settings.conversation_summary_model == "gemini-2.5-flash"
    assert settings.conversation_summary_soft_context_ratio == 0.70
    assert settings.conversation_summary_hard_context_ratio == 0.85

def test_production_rejects_preview_model():
    with pytest.raises(ValidationError, match="stable"):
        Settings(
            _env_file=None,
            environment="production",
            conversation_summary_provider="gemini",
            conversation_summary_model="gemini-3-flash-preview",
        )
```

- [x] **Step 2: Verify RED**

Run: `.venv/Scripts/python.exe -m pytest tests/test_conversation_summary_config.py -q`
Expected: collection/assertion failures because `conversation_summary_*` fields do not exist.

- [x] **Step 3: Implement the namespace and validator**

Add every field listed in design section 10 with documented defaults. Use one `@model_validator(mode="after")` that raises explicit messages for each invariant. Add exact `CONVERSATION_SUMMARY_*` `.env.example` entries; retain legacy fields only until their callers are migrated, then remove fields and old example entries together in Task 11.

- [x] **Step 4: Verify GREEN and regressions**

Run:
`pytest tests/test_conversation_summary_config.py tests/test_config_redis.py tests/test_conversation_summarizer.py -q`
Expected: new tests pass; legacy summarizer tests may fail only where they prove removed settings and must be replaced during Task 11, not weakened here.

- [x] **Step 5: Run task gate, update logs, and commit**

Run Ruff on changed files plus `git diff --check`. Record exact results below. Commit only configuration/test/example files with `feat(compaction): add validated summary configuration`.

### Task 2: Unified provider-aware token accounting

**Files:**
- Create: `app/ai/token_counter.py`
- Create: `tests/test_token_counter.py`
- Modify: `requirements.txt` only if the installed tiktoken dependency is not declared.

- [x] **Step 1: Write failing counter tests**

Define the wished-for API:

```python
@dataclass(frozen=True)
class TokenCount:
    tokens: int
    strategy: str
    source: Literal["local", "provider", "reported"]

counter.count_request(
    provider="openai",
    model="gpt-4o",
    messages=messages,
    tools=tools,
    attachments=attachments,
    reserved_output_tokens=1024,
)
```

Test OpenAI model encoding/fallback, conservative Gemini/Anthropic estimates, byte-upper-bound unknown models, Thai not using chars/4, canonical JSON tools, tool calls/results, image metadata/fallback, authoritative callback near boundaries, and extraction of OpenAI/Gemini/Anthropic reported usage.

- [x] **Step 2: Verify RED**

Run: `pytest tests/test_token_counter.py -q`
Expected: import failure for `app.ai.token_counter`.

- [x] **Step 3: Implement minimal strategies and canonical serialization**

`TokenCounter` returns counts with strategy identifiers; it never silently changes provider. Local counters include role/envelope overhead, `json.dumps(..., sort_keys=True, separators=(",", ":"))` tool serialization, image costs, reserved output, and safety margin as separate breakdown values.

- [x] **Step 4: Verify GREEN and boundary cases**

Run: `pytest tests/test_token_counter.py -q`
Expected: all counter tests pass including Thai and unknown-model conservative assertions.

- [x] **Step 5: Run task gate, update logs, and commit**

Run Ruff and `git diff --check`; record results. Commit with `feat(compaction): add unified token counter`.

### Task 3: Sequence, structured-summary, and job schema

**Files:**
- Modify: `app/models/conversation.py`
- Modify: `app/models/message.py`
- Rewrite: `app/models/conversation_memory_summary.py`
- Create: `app/models/conversation_summary_job.py`
- Modify: `app/models/__init__.py`
- Create: `app/alembic/versions/x1y2z3a4b5c6_production_conversation_compaction.py`
- Create: `tests/test_conversation_compaction_schema.py`

- [x] **Step 1: Write failing SQLAlchemy metadata tests**

Assert column types/nullability/defaults, positive/non-negative checks, one summary/job per conversation, unique `(conversation_id, sequence)`, partial prompt index, same-conversation composite FKs, job statuses, and cascades.

- [x] **Step 2: Verify RED**

Run: `pytest tests/test_conversation_compaction_schema.py -q`
Expected: missing columns/model/constraints.

- [x] **Step 3: Implement models and forward migration**

The migration must: add/backfill message sequences using `(created_at, id)` only for migration ordering; set `next_message_sequence = max(sequence)+1`; replace the legacy summary table in place while preserving defensively converted rows as invalid empty payloads; create jobs; add composite uniqueness before composite FKs; validate constraints; and provide a reversible downgrade to the previous schema.

- [x] **Step 4: Verify GREEN and Alembic metadata**

Run: `pytest tests/test_conversation_compaction_schema.py tests/test_database_schema_contract.py -q`
Run: `python -m alembic heads`
Expected: one head `x1y2z3a4b5c6` and schema tests pass.

- [x] **Step 5: Run task gate, update logs, and commit**

Run Ruff, `alembic check`, and `git diff --check`; record results. Commit with `feat(compaction): add sequence summary and job schema`.

### Task 4: Atomic persistence, job state, and invalidation repository

**Files:**
- Create: `app/repositories/conversation_compaction.py`
- Modify: `app/repositories/message.py`
- Modify: `app/core/container.py`
- Create: `tests/test_conversation_compaction_repository.py`
- Create: `tests/integration/test_conversation_compaction_postgres.py`

- [x] **Step 1: Write failing repository tests**

Test atomic sequence allocation/message insert/assistant job upsert; `GREATEST` target coalescing; pending target advancement during a live lease; `SKIP LOCKED` claim; lease token checks; completion to idle/pending; retry/dead transitions; expired-lease reconciliation; ownership-scoped valid-memory read; CAS on summary version/cursor/lease; and edit/delete invalidation plus rebuild request.

- [x] **Step 2: Verify RED**

Run fast repository tests and the PostgreSQL integration file when `TEST_DATABASE_URL` is available. Expected: missing repository/API failures.

- [x] **Step 3: Implement transaction-owning repository**

Expose methods `persist_message`, `invalidate_for_mutation`, `claim_job`, `load_compaction_input`, `persist_memory_cas`, `complete_claim`, `fail_claim`, `reconcile_due_jobs`, and `request_backfill`. No method may keep a transaction open across provider calls.

- [x] **Step 4: Verify GREEN including concurrency**

Run both repository suites. Expected: duplicate/concurrent operations cannot regress sequence, target, or memory; cross-conversation cursors fail at the database.

- [x] **Step 5: Run task gate, update logs, and commit**

Run Ruff, affected repository tests, Alembic check, and `git diff --check`; record and commit with `feat(compaction): add atomic persistence and job repository`.

### Task 5: Structured conversation memory and compactor

**Files:**
- Create: `app/ai/conversation_memory.py`
- Create: `app/ai/conversation_compactor.py`
- Create: `tests/test_conversation_memory.py`
- Create: `tests/test_conversation_compactor.py`

- [ ] **Step 1: Write failing payload and compactor tests**

Test exactly six list keys; bounded strings/items; forbidden unknown/nested/executable/base64/secret content; safe attachment descriptors; prompt injection retained only as quoted data; empty/invalid/over-budget output preserving prior valid memory; exact trigger boundaries; disabled individual thresholds; full-window threshold evaluation; and prefix selection ending on an assistant turn while retaining configured complete turns.

- [ ] **Step 2: Verify RED**

Run both new test files. Expected: module import failures.

- [ ] **Step 3: Implement schema, canonical renderer, selection, and provider adapter boundary**

The compactor accepts injected `TokenCounter` and async generator callback, treats previous payload/transcript as delimited untrusted data, parses JSON only, validates, recounts, and returns a typed result without writing the database. Its provider adapter resolves a credential for the configured provider only: use a permitted user credential when policy allows, otherwise require the explicitly configured server-managed credential; never substitute a key from another provider.

- [ ] **Step 4: Verify GREEN**

Run both test files plus token-counter tests. Expected: all pass without network calls.

- [ ] **Step 5: Run task gate, update logs, and commit**

Run Ruff and `git diff --check`; record and commit with `feat(compaction): add structured memory compactor`.

### Task 6: Celery worker, reconciliation, and historical backfill

**Files:**
- Create: `app/workers/conversation_compaction.py`
- Modify: `app/workers/celery_app.py`
- Modify: `app/workers/start_worker.py`
- Create: `tests/test_conversation_compaction_tasks.py`
- Modify: `tests/test_celery_worker_config.py`
- Modify: `tests/test_start_worker.py`

- [ ] **Step 1: Write failing task tests**

Test task payload contains only `conversation_id`; lease commits before provider call; transient/permanent classification; exponential backoff with bounded jitter; five-attempt default; sanitized error codes; CAS conflict accounting; newer-target preservation; expired lease recovery; reconciler debounce-before-publish; duplicate notifications; and rate-limited idempotent backfill.

- [ ] **Step 2: Verify RED**

Run new task and existing worker-config tests. Expected: missing summary routes/tasks.

- [ ] **Step 3: Implement tasks and queue wiring**

Add `summary` queue routes and Beat schedules for reconciliation/backfill. Worker transactions wrap only claim/load/persist state, never the provider call. Broker publication is best effort and logs sanitized classifications.

- [ ] **Step 4: Verify GREEN**

Run task and worker tests. Expected: all pass using eager/mocked Celery transport and real repository state machines.

- [ ] **Step 5: Run task gate, update logs, and commit**

Run Ruff and `git diff --check`; record and commit with `feat(compaction): add durable summary workers`.

### Task 7: Tenant-scoped prompt hydration outside system instructions

**Files:**
- Modify: `app/ai/history.py`
- Modify: `app/ai/agents/base_agent.py`
- Modify: `app/ai/schemas.py`
- Modify: agent/workflow callers currently passing `history_summary`
- Create: `tests/test_conversation_memory_hydration.py`
- Modify: `tests/test_history_provider.py`

- [ ] **Step 1: Write failing hydration tests**

Assert only owned valid summaries hydrate; canonical JSON is wrapped as untrusted reference data; memory is a dedicated lower-priority message before recent history; no memory bytes appear in any `SystemMessage`; current user turn remains last; cursor prevents overlap; invalid summaries are ignored.

- [ ] **Step 2: Verify RED**

Run hydration/history tests. Expected: current system-prompt injection assertion fails.

- [ ] **Step 3: Replace `history_summary` with typed `conversation_memory`**

Return the memory message separately from transcript messages. Update all agent paths to assemble `System + Memory + Recent History + Current Turn`; remove summary checkpoint state and fallback propagation.

- [ ] **Step 4: Verify GREEN across agents**

Run hydration, history, graph, chat, RAG, planning, search, canvas, and image-agent tests. Expected: no agent injects memory into system instructions.

- [ ] **Step 5: Run task gate, update logs, and commit**

Run Ruff and legacy-symbol scan for graph summary keys; record and commit with `feat(compaction): hydrate untrusted structured memory`.

### Task 8: Full request budgeting and emergency reduction

**Files:**
- Create: `app/ai/request_budget.py`
- Modify: `app/ai/agents/base_agent.py`
- Modify: provider/model request assembly paths
- Create: `tests/test_request_budget.py`
- Modify: `tests/test_context_overflow_retry.py`

- [ ] **Step 1: Write failing budget tests**

Test complete input accounting; below-soft proceed; soft-to-hard durable request; hard synchronous compaction and recount; timeout/failure deterministic oldest-complete-turn removal; no split user/assistant or tool-call/result structures; fixed input overflow safe error; one aggressive provider-overflow retry reducing old turns and tool previews; repeated overflow surfaced.

- [ ] **Step 2: Verify RED**

Run request-budget and overflow tests. Expected: missing preflight/reducer APIs.

- [ ] **Step 3: Implement request budget service and integrate before invocation**

Calculate `available_input = max_input - reserved_output - safety_margin`. Operate on the actual provider request, preserving required system/current/tool schemas. Use bounded `asyncio.timeout` for emergency compaction and the same counter for every recount.

- [ ] **Step 4: Verify GREEN**

Run budget, counter, and agent invocation tests. Expected: every emitted request is within budget or returns the specific safe budget error.

- [ ] **Step 5: Run task gate, update logs, and commit**

Run Ruff and `git diff --check`; record and commit with `feat(compaction): enforce full request budgets`.

### Task 9: Message-service integration and after-commit publication

**Files:**
- Modify: `app/services/message_service.py`
- Modify: `app/repositories/message.py`
- Modify: API/stream/resume persistence paths as required
- Create: `tests/test_conversation_compaction_message_paths.py`
- Modify: `tests/test_message_history_pipeline.py`
- Modify: `tests/test_message_service_event_streaming.py`

- [ ] **Step 1: Write failing terminal-path tests**

Test ordinary, streamed, partial, stopped, resumed, error-terminal assistant persistence all atomically advance the job; hidden placeholders do not become summary content; broker publish happens only after commit; publish failure keeps message/job committed; user message does not request summary.

- [ ] **Step 2: Verify RED**

Run new path tests. Expected: current in-process runner and separate commits violate assertions.

- [ ] **Step 3: Route every persistence path through the transaction-owning repository**

Delete `_summary_refresh_pending`, `_summary_refresh_active`, `_schedule_summary_refresh`, runner, and `refresh_summary_after_turn`. Register/publish notifications only after successful commit and invalidate prompt caches after persistence.

- [ ] **Step 4: Verify GREEN**

Run path, streaming, stop, resume, HITL, and history pipeline tests. Expected: every committed assistant boundary has durable work independent of notification success.

- [ ] **Step 5: Run task gate, update logs, and commit**

Run Ruff and legacy-runner symbol scan; record and commit with `feat(compaction): integrate atomic assistant persistence`.

### Task 10: Metrics, summary health, and operational controls

**Files:**
- Create: `app/observability/__init__.py`
- Create: `app/observability/conversation_compaction.py`
- Create: `app/api/health.py`
- Modify: `app/main.py`
- Create: `tests/test_conversation_compaction_health.py`

- [ ] **Step 1: Write failing observability tests**

Assert required counters/gauges/histograms, provider/model/content-class calibration labels, cost metadata, queue/dead/lease/lag health checks, no transcript/summary/conversation IDs in metrics or response, and aggregate health separation from Celery health.

- [ ] **Step 2: Verify RED**

Run health tests. Expected: missing endpoint/collector.

- [ ] **Step 3: Implement content-free metrics and health service**

Use bounded label cardinality and sanitized error classifications. Health response exposes aggregate counts/ages/lag and status only.

- [ ] **Step 4: Verify GREEN**

Run health and main-app tests. Expected: endpoint reports healthy/degraded/unhealthy deterministically without tenant content.

- [ ] **Step 5: Run task gate, update logs, and commit**

Run Ruff and `git diff --check`; record and commit with `feat(compaction): add summary observability`.

### Task 11: Migrate remaining callers and delete scoped legacy code

**Files:**
- Delete: `app/ai/summarization_middleware.py`
- Delete: `app/ai/conversation_summarizer.py`
- Delete: `app/ai/memory.py`
- Modify: `app/ai/token_instrumentation.py`
- Modify: `app/ai/prompts.py`
- Modify: document chunking/RAG token callers
- Modify/Delete: legacy summary tests
- Create: `tests/test_conversation_compaction_legacy_cleanup.py`

- [ ] **Step 1: Write failing repository-wide cleanup assertions**

Scan outside applied migrations and explicit migration notes for removed modules, `MemoryManager`, refresh-runner symbols, timestamp/UUID summary cursors, graph keys, `history_summary`, legacy env names, duplicate public estimators, `len(text)//4`, and tool-only retry plumbing superseded by request budgeting.

- [ ] **Step 2: Verify RED**

Run cleanup test. Expected: it lists every remaining legacy path.

- [ ] **Step 3: Migrate callers to `TokenCounter` and remove legacy implementation**

Preserve unrelated user-memory functionality. Document chunking receives an explicit tokenizer strategy. Delete obsolete tests rather than rewriting them to preserve removed behavior.

- [ ] **Step 4: Verify GREEN and focused regressions**

Run cleanup, token, prompt, document chunking, RAG, graph, and context-window tests. Expected: no removed symbol outside allowed immutable history/notes.

- [ ] **Step 5: Run task gate, update logs, and commit**

Run Ruff and repository-wide scan; record and commit with `refactor(compaction): remove legacy summarization paths`.

### Task 12: Documentation and deployment configuration

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: worker/deployment documentation present in repository
- Create: `docs/operations/conversation-compaction.md`
- Create: `tests/test_conversation_compaction_docs.py`

- [ ] **Step 1: Write failing documentation contract test**

Assert exact environment-name/default mapping, summary worker/Beat startup, migration/rollback, backfill, health, dashboards/alerts, credential policy, rollout steps, and absence of legacy names outside the migration mapping table.

- [ ] **Step 2: Verify RED**

Run docs test. Expected: missing operations document and legacy README text.

- [ ] **Step 3: Write deployment and operations documentation**

Include explicit production provider/model requirements, `summary` queue, reconciler, backfill, observable thresholds, rollback limitations, and mapping from old names to new names as release guidance only.

- [ ] **Step 4: Verify GREEN**

Run docs and cleanup tests. Expected: exact names/defaults match `Settings`.

- [ ] **Step 5: Run task gate, update logs, and commit**

Run Markdown-relevant checks, Ruff for test, and `git diff --check`; record and commit with `docs: add conversation compaction operations guide`.

### Task 13: PostgreSQL migration matrix and final verification

**Files:**
- Modify only files required by defects proven during this gate, always with a failing regression test first.
- Update: this plan's Progress Log and Decision Log.

- [ ] **Step 1: Run disposable PostgreSQL upgrade and schema inspection**

Run upgrade from previous head to new head, inspect tables/indexes/checks/composite FKs, validate backfill ordering/next sequence, run `alembic current`, `alembic check`, then downgrade to the previous schema and upgrade again.

- [ ] **Step 2: Run PostgreSQL integration and failure suites**

Run concurrency, atomicity, crash windows, duplicate delivery, lease expiry, CAS conflict, edit/delete rebuild, end-to-end hydration, emergency fallback, and backfill tests.

- [ ] **Step 3: Run complete static and test verification**

Run full `pytest`, `ruff check`, `ruff format --check`, configured type checks, `git diff --check`, and repository-wide cleanup scan. Distinguish pre-existing unrelated failures with a clean reproduction; do not mark complete while plan-owned failures remain.

- [ ] **Step 4: Inspect final diff and requirements checklist**

Review every design success criterion against code/tests, confirm no unrelated file was staged, and record exact evidence in Progress Log.

- [ ] **Step 5: Commit final verification fixes and complete branch workflow**

Commit only proven fixes with `test(compaction): complete production verification`, then invoke the finishing-development-branch workflow.

---

## Progress Log

| Time (Asia/Bangkok) | Task | Status | Verification evidence |
|---|---|---|---|
| 2026-07-15 | Plan | Complete | Placeholder scan returned no matches; requirement coverage scan found sequence/jobs, structured memory, token counting/Thai, invalidation, leases/CAS/reconciliation, emergency retry, health/metrics, cleanup, docs, and migration gates; plan-file verification found no whitespace errors. |
| 2026-07-15 | Task 1 | Complete | RED: `pytest tests/test_conversation_summary_config.py -q` produced 26 expected failures for missing fields/invariants. GREEN: targeted plus config/summarizer regressions produced `36 passed`. Ruff formatting reports both changed Python files formatted; the new test has zero lint errors and the diff adds zero overlong lines. Whole-file `config.py` still reports the same 42 pre-existing E501 errors as `HEAD`; `git diff --check` passed. |
| 2026-07-15 | Task 2 | Complete | RED: token-counter tests failed at collection with the expected missing-module error. GREEN: 16 focused tests passed. Final gate: token, context-window, image-history, and model-context suites produced `76 passed`; Ruff lint passed; both files were formatted after one formatter-only iteration; `git diff --check` passed. |
| 2026-07-15 | Task 3 | Complete | RED: schema and migration tests failed for the missing job model/revision. GREEN: model/migration contracts passed. Disposable PostgreSQL verification from supported previous head validated deterministic `1,2,3` sequence backfill, next sequence `4`, named constraints/indexes, legacy payload invalidation/version `7→8`, `alembic current/check`, downgrade restoration, and re-upgrade. Final schema/container gate: `17 passed`; Ruff lint/format and `git diff --check` passed. A from-zero run was blocked in immutable history by the pre-existing duplicate `decision_type` enum migration; no old revision was altered. |
| 2026-07-15 | Task 4, steps 1–2 | In progress | Added fast SQL-contract tests plus PostgreSQL state-machine/invalidation coverage. RED: `.conda\\python.exe -m pytest tests/test_conversation_compaction_repository.py -q` failed during collection with the expected `ModuleNotFoundError` for `app.repositories.conversation_compaction`. |
| 2026-07-15 | Task 4, steps 3–4 | In progress | Fast SQL contracts passed (`4 passed`). A disposable PostgreSQL suite passed (`9 passed`) after proving and fixing nullable eager-join locking and stale mutation-lease races. Coverage includes concurrent `1..12` allocation, target `12`, real `SKIP LOCKED`, live/expired/wrong-token leases, retry/dead/reconcile, ownership, version/cursor/lease CAS, edit/delete rebuild, and cross-conversation composite-FK rejection. Affected history/container regressions passed (`12 passed`). |
| 2026-07-15 | Task 4 | Complete | Final gate: new/affected repository, schema, history, and container suites produced `19 passed`; disposable PostgreSQL produced `9 passed`; Ruff lint passed for all task changes (with pre-existing message-file E501 ignored only for that legacy file); all three new files are Ruff-formatted; `alembic check` reported no new operations; `git diff --check` passed. |

## Decision Log

| Time (Asia/Bangkok) | Decision | Rationale |
|---|---|---|
| 2026-07-15 | Execute inline in the current checkout without a worktree. | Explicit user instruction; all commits and staging will be path-scoped to preserve unrelated dirty changes. |
| 2026-07-15 | Treat each numbered task as the user's “step” and verification gate. | Each task delivers one testable subsystem while its internal red/green actions remain small TDD increments. |
| 2026-07-15 | Keep PostgreSQL authoritative and use Redis/Celery only for dispatch. | Required by the approved design and necessary for restart/broker-failure recovery. |
| 2026-07-15 | Resolve credentials only for the configured compaction provider. | Prevents the legacy class of silently sending one provider's key to another provider and makes server-managed fallback explicit. |
| 2026-07-15 | Add new settings before deleting legacy settings. | Migrating callers and deleting old fields in Task 11 keeps every intermediate gate runnable while still shipping without compatibility aliases. |
| 2026-07-15 | Default reconciliation to 60 seconds, safety margin to 1,024 tokens, and reserved output to 4,096 tokens. | The design names these controls but does not assign numbers; these conservative values match the existing operational scale and are now locked by tests/docs. |
| 2026-07-15 | Split local `estimate_request` from async authoritative `count_request`. | Most calls remain fast and deterministic while near-boundary/background callers can opt into injected provider-native counting without duplicating serialization logic. |
| 2026-07-15 | Use dimension-aware image formulas with a 1,200-token fallback. | OpenAI tile accounting, Anthropic area accounting, and conservative Gemini area accounting improve estimates when metadata exists; the established 1,200 fallback protects metadata-poor images. |
| 2026-07-15 | Name summary/job primary keys and all new foreign keys explicitly. | The legacy table remains temporarily during data conversion; explicit names prevent PostgreSQL schema-wide constraint-name collisions and make inspection deterministic. |
| 2026-07-15 | Convert legacy free-form summaries into invalid empty structured rows with an incremented version. | Free-form text cannot be safely or deterministically promoted into the validated schema; invalidation guarantees it is never reused while preserving conversation/version continuity. |
| 2026-07-15 | Verify migration from `w7x8y9z0a1b2`, not by rewriting old migration history. | The repository's from-zero history has a pre-existing duplicate enum defect; production rollout starts from the applied previous head, and the design explicitly forbids rewriting applied revisions. |
| 2026-07-15 | Do not downgrade the live development database after automatic startup migration advanced it. | Active API processes applied the new head when the revision appeared; destructive rollback would disrupt those sessions, so all downgrade testing remained disposable. |
| 2026-07-15 | Let notification-driven claims bypass reconciliation debounce while generic scans respect `available_at`. | Reconciliation advances `available_at` before publishing to suppress duplicate dispatch; the addressed task must still be able to claim immediately, while opportunistic scanners must honor retry/debounce timing. |
| 2026-07-15 | Revoke an active compaction lease when a covered transcript message is edited or deleted. | Target coalescing normally preserves a live lease, but a mutation makes that worker's captured transcript stale; clearing the lease and forcing `pending` makes its subsequent CAS fail and guarantees a full rebuild. |
| 2026-07-15 | Require an unexpired lease for input loading, memory CAS, completion, and failure transitions. | A token alone does not remain authoritative after its deadline; reconciliation owns recovery once a lease expires. |
| 2026-07-15 | Build disposable repository tests from only their required PostgreSQL tables. | Metadata-wide `create_all` hits a pre-existing malformed `task_plans.task_metadata` JSONB default; isolating the authoritative compaction tables plus mapper-required `feedbacks` tests this subsystem without altering unrelated code. |
