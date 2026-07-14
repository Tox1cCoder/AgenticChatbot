# Production Conversation Compaction Design

**Date:** 2026-07-14

**Status:** Approved for implementation planning

## 1. Purpose

Replace the chatbot's current best-effort rolling-summary implementation with a durable, provider-aware conversation compaction pipeline suitable for a multi-process production deployment.

The replacement must:

- prevent context-window overflow without blocking ordinary replies;
- preserve important conversation state while retaining recent complete turns verbatim;
- survive API restarts, worker restarts, broker failures, duplicate delivery, and concurrent execution;
- use one token-accounting implementation across prompts, history, compaction, and context telemetry;
- never reuse derived memory after a covered source message is edited or deleted;
- keep untrusted, model-generated memory outside system instructions;
- enforce tenant, cursor, and numeric invariants in PostgreSQL; and
- remove the legacy summarization, memory, token-estimation, and overflow-retry code in this refactor's scope after cutover.

Celery and Redis are always present in production. PostgreSQL remains the authoritative source of transcript, summary, cursor, and job state. Redis/Celery transports notifications and execution but is not the only record that work is required.

## 2. Scope

### 2.1 Included

- Message ordering and summary cursors.
- Durable summary-job/outbox state.
- Celery background compaction and reconciliation.
- Request-path emergency compaction and deterministic trimming.
- Structured conversation-memory generation and prompt injection.
- Provider/model-aware token counting and actual-usage extraction.
- Summary invalidation after transcript mutation.
- Summary-specific configuration, metrics, health, backfill, tests, and documentation.
- Removal of directly related legacy and redundant code.

### 2.2 Excluded

- General RAG retrieval redesign.
- Tool authorization redesign beyond preserving application-level authorization and HITL.
- Unrelated cleanup elsewhere in the chatbot.
- Deleting or rewriting applied Alembic migration history.
- Changing the public chat/message API unless a response already exposes internal summary metadata that must be renamed or removed.

## 3. Chosen Architecture

Use a hybrid compaction architecture:

1. Normal compaction runs asynchronously in a dedicated Celery queue.
2. The assistant message and durable summary-job request are committed atomically in PostgreSQL.
3. Celery notification occurs after commit and is best effort; a periodic reconciler republishes pending and retryable database jobs.
4. A request performs bounded synchronous compaction only when the fully assembled model input approaches its hard budget.
5. If synchronous compaction cannot complete within its timeout, deterministic complete-turn trimming keeps the request within budget.
6. Provider context-overflow errors receive one aggressive retry after reducing both history and tool output.

This architecture keeps ordinary response latency independent of the summarizer while providing a hard request-path safety boundary.

## 4. Data Model

### 4.1 Monotonic message sequences

Add `conversations.next_message_sequence BIGINT NOT NULL DEFAULT 1` with a positive-value check.

Add `messages.sequence BIGINT` and backfill existing rows with `row_number()` partitioned by `conversation_id`, ordered by `(created_at, id)`. After validation, make the column non-null and add:

- `UNIQUE (conversation_id, sequence)`; and
- a prompt-history index on `(conversation_id, sequence)` for rows where `deleted_at IS NULL`.

New sequence allocation is atomic:

```sql
UPDATE conversations
SET next_message_sequence = next_message_sequence + 1
WHERE id = :conversation_id
RETURNING next_message_sequence - 1;
```

All prompt windows, summary windows, job targets, and cursor comparisons use `sequence`. Timestamp and UUID tuple ordering is removed from these paths.

### 4.2 `conversation_memory_summaries`

The summary table becomes a strict one-to-one extension of a conversation:

| Column | Type | Rule |
|---|---|---|
| `conversation_id` | UUID | Primary key; FK to `conversations.id` with `ON DELETE CASCADE` |
| `summary_payload` | JSONB | Validated structured memory; empty object when invalidated |
| `summary_schema_version` | SMALLINT | Positive |
| `last_summarized_sequence` | BIGINT nullable | Composite FK with `conversation_id` to `messages(conversation_id, sequence)` |
| `summary_version` | BIGINT | Positive, monotonically increasing |
| `source_message_count` | INTEGER | Non-negative |
| `source_token_count` | INTEGER | Non-negative |
| `summary_token_count` | INTEGER | Non-negative |
| `provider` | VARCHAR | Provider used for compaction |
| `model` | VARCHAR | Model used for compaction |
| `tokenizer` | VARCHAR | Counter strategy identifier |
| `prompt_version` | VARCHAR | Compaction-prompt version |
| `is_valid` | BOOLEAN | Only valid rows may be hydrated into prompts |
| `created_at` | TIMESTAMPTZ | Server timestamp |
| `updated_at` | TIMESTAMPTZ | Updated on every write |

The redundant surrogate `id`, redundant `user_id`, free-form `summary_text`, and timestamp-derived cursor are removed. Conversation ownership is authoritative; history lookup must join or filter through the owned conversation before returning memory.

When a covered message is edited or deleted, the same transaction clears `summary_payload`, sets `last_summarized_sequence` to `NULL`, marks `is_valid=false`, increments `summary_version`, and requests a rebuild from the beginning. Deleted or stale derived text is therefore neither stored nor reused while rebuilding.

### 4.3 `conversation_summary_jobs`

Create one durable coalescing job row per conversation:

| Column | Type | Rule |
|---|---|---|
| `conversation_id` | UUID | Primary key; FK with `ON DELETE CASCADE` |
| `requested_through_sequence` | BIGINT | Latest committed assistant boundary; composite FK to the same conversation's message sequence |
| `status` | VARCHAR(16) | Check: `idle`, `pending`, `processing`, `retry`, or `dead` |
| `attempt_count` | INTEGER | Non-negative |
| `available_at` | TIMESTAMPTZ | Earliest claim time |
| `lease_token` | UUID nullable | Identifies the current worker claim |
| `lease_expires_at` | TIMESTAMPTZ nullable | Permits crash recovery |
| `last_error_code` | VARCHAR(64) nullable | Sanitized classification, never transcript or provider payload |
| `created_at` | TIMESTAMPTZ | Server timestamp |
| `updated_at` | TIMESTAMPTZ | Updated on every state transition |

Add an index on `(status, available_at)` for worker and reconciler scans.

An upsert advances `requested_through_sequence` with `GREATEST(existing, incoming)`. An unexpired processing lease remains processing while its target advances; the completing worker leaves the row pending if work remains after its captured target.

## 5. Atomic Transcript Persistence

Every message insert allocates its sequence within the message transaction. Every committed assistant message also upserts the summary job before that transaction commits.

The transaction boundary is:

1. lock/increment `conversations.next_message_sequence`;
2. insert the message with the allocated sequence;
3. for an assistant message, upsert `conversation_summary_jobs` to that sequence;
4. commit once; and
5. publish a Celery notification after commit.

The repository owns this database transaction so every assistant persistence path, including partial, stopped, resumed, and error terminal messages, advances durable work consistently. Hidden interrupt placeholders remain excluded from summary content, but their sequence may be crossed by a later valid cursor.

Publishing failure is logged with a sanitized code and does not roll back the committed message/job. The periodic reconciler discovers the database row.

## 6. Structured Conversation Memory

The compactor produces a validated schema rather than unconstrained prose:

```json
{
  "facts": [],
  "decisions": [],
  "constraints": [],
  "preferences": [],
  "open_questions": [],
  "tool_outcomes": []
}
```

Each item is a bounded string. Unknown keys, nested instructions, executable payloads, secrets, raw base64, and full tool artifacts are rejected or omitted. Attachment memory contains only safe descriptors already supported by the application.

The compaction prompt treats both previous memory and transcript messages as untrusted source data. Output is parsed and validated before persistence. Invalid, empty, or over-budget output leaves the previous valid summary intact and routes the job through retry/error handling.

Prompt hydration renders canonical JSON into a dedicated lower-priority conversation-memory message before recent transcript history. It is never concatenated into a `SystemMessage`. The wrapper identifies the payload as derived, untrusted reference data. The current user turn remains the final user message. Tool permissions, tenant checks, and HITL remain application-enforced and never depend on memory instructions.

## 7. Unified Token Accounting

### 7.1 One service

Create one `TokenCounter` abstraction used by:

- full request preflight and context-window ratios;
- history trimming;
- background and emergency compaction triggers;
- compaction output enforcement;
- tool-schema and tool-result accounting;
- attachment/image estimates;
- actual provider-usage telemetry; and
- document chunking through an explicit tokenizer strategy.

Delete independent `len(text) // 4` implementations and the separate RAG-only public token helper after callers migrate.

### 7.2 Counting strategies

The counter accepts provider, model, messages, tools, attachment metadata, and reserved output tokens.

- OpenAI models use `tiktoken.encoding_for_model` when supported, with an explicit model-family fallback.
- Gemini and Anthropic use fast conservative local estimates during normal execution and provider-native request counting when near a compaction boundary or when a background job needs an authoritative count.
- Unknown/custom models use a deliberately conservative UTF-8-byte upper estimate plus message/tool envelope overhead.
- Image estimates use provider/model metadata and dimensions when present; otherwise they use the existing conservative per-image fallback.
- Tool schemas are serialized canonically rather than through Python `str(dict)`.

The service returns both a token count and a strategy/source identifier. Provider-reported post-response usage remains authoritative for telemetry and calibration.

### 7.3 Full request budget

The request budget is:

```text
available_input = max_input_tokens - reserved_output_tokens - safety_margin_tokens
usage_ratio = counted_input_tokens / available_input
```

Counting includes system/developer instructions, conversation memory, recent history, the current turn, tool schemas, tool calls/results, attachment/image costs, and serialization overhead.

Defaults:

- compaction provider/model: `gemini` / `gemini-2.5-flash` in development; production must set both explicitly;
- background soft ratio: `0.70`;
- request hard ratio: `0.85`;
- retained recent turns: `4`;
- background message threshold: `60` eligible unsummarized messages;
- background token threshold: `18,000` counted unsummarized tokens;
- maximum compacted memory: `1,500` tokens; and
- compaction timeout: `30` seconds.

Thresholds are evaluated against the complete unsummarized window before selecting the compactable prefix. A threshold value of zero disables only that threshold. Configuration validation rejects an enabled setup in which both background thresholds are zero. The hard request budget remains active even when durable summaries are disabled.

Actual-versus-estimated deltas are recorded by provider/model and coarse content class. Tests include Thai because the current four-characters-per-token heuristic materially undercounts Thai text.

## 8. Background Compaction Worker

Use a dedicated Celery `summary` queue. Celery tasks receive only `conversation_id`; authoritative targets and state come from PostgreSQL.

Worker flow:

1. Claim an available job with a unique lease token. Multi-row reconciliation uses `FOR UPDATE SKIP LOCKED`.
2. Commit the lease transaction before any provider call.
3. Load the owned conversation, valid memory, and eligible messages through the captured target sequence.
4. Evaluate full-window message/token thresholds.
5. Select a prefix ending at a complete assistant turn while retaining the newest configured complete turns.
6. Generate and validate structured memory with the configured stable compaction model.
7. Recount the generated payload and enforce its hard token cap.
8. Persist with compare-and-swap on the base `summary_version`, base cursor, and lease token.
9. Mark the job `idle` if caught up, or `pending` if a newer target arrived.

No database row lock or transaction remains open during the model call.

Transient provider, network, rate-limit, and timeout failures use exponential backoff with jitter. Defaults are five attempts, a five-second base delay, a fifteen-minute cap, and a two-minute lease. Validation, ownership, and invariant failures use sanitized permanent error codes and become `dead`. Expired processing leases return to `retry` through reconciliation.

The compaction provider and stable model are explicit configuration. Production startup rejects a missing model/provider and rejects a model name containing `preview`. User credentials are resolved for the configured provider when policy permits; otherwise the server-managed compaction credential is used explicitly rather than silently receiving another provider's key.

## 9. Request-Path Emergency Compaction

Before model invocation, assemble the actual provider request and count it with the resolved model metadata.

- Below the soft ratio: proceed.
- At or above the soft ratio but below the hard ratio: ensure a durable job is pending and proceed if within the hard budget.
- At or above the hard ratio: synchronously compact only older persisted complete turns, with a bounded timeout, then rebuild and recount the request.
- If synchronous compaction fails or remains too large: deterministically remove the oldest complete turns until the request fits.
- If fixed system instructions plus required current-turn/tool schemas alone exceed the budget: fail with a specific safe context-budget error rather than sending an invalid provider request.

A provider context-overflow error receives one retry after aggressive reduction of both old turns and tool-result previews. Repeated overflow is surfaced; it is not retried indefinitely.

## 10. Configuration

Use one `CONVERSATION_SUMMARY_*` namespace:

- `CONVERSATION_SUMMARY_ENABLED`
- `CONVERSATION_SUMMARY_PROVIDER`
- `CONVERSATION_SUMMARY_MODEL`
- `CONVERSATION_SUMMARY_TRIGGER_MESSAGES`
- `CONVERSATION_SUMMARY_TRIGGER_TOKENS`
- `CONVERSATION_SUMMARY_SOFT_CONTEXT_RATIO`
- `CONVERSATION_SUMMARY_HARD_CONTEXT_RATIO`
- `CONVERSATION_SUMMARY_KEEP_RECENT_TURNS`
- `CONVERSATION_SUMMARY_MAX_TOKENS`
- `CONVERSATION_SUMMARY_TIMEOUT_SECONDS`
- `CONVERSATION_SUMMARY_MAX_ATTEMPTS`
- `CONVERSATION_SUMMARY_LEASE_SECONDS`
- `CONVERSATION_SUMMARY_RETRY_BASE_SECONDS`
- `CONVERSATION_SUMMARY_RETRY_MAX_SECONDS`
- `CONVERSATION_SUMMARY_RECONCILE_SECONDS`
- `CONVERSATION_SUMMARY_SAFETY_MARGIN_TOKENS`
- `CONVERSATION_SUMMARY_DEFAULT_RESERVED_OUTPUT_TOKENS`

Validation enforces non-negative integer budgets, `0 < soft_ratio < hard_ratio < 1`, positive lease/timeout/retry values, retained turns below the message threshold when that threshold is enabled, and explicit stable provider/model values in production.

There are no legacy environment aliases or compatibility properties. Deployment configuration must move directly from `MEMORY_SUMMARY_*` and `SUMMARIZATION_*` to the new names. `.env.example`, README, and operational documentation contain the exact mapping and defaults.

## 11. Observability and Operations

Emit structured logs and Prometheus metrics without transcript or summary content.

Required measurements:

- job counts by status;
- oldest pending/retry age;
- requested-versus-summarized sequence lag;
- background and emergency compaction counts;
- success, timeout, retry, lease-expiry, CAS-conflict, and dead counts;
- deterministic trim and provider-overflow-retry counts;
- compaction input/output tokens and latency;
- token estimate versus actual delta by provider/model/content class; and
- compaction-model cost metadata when provider usage exposes it.

Add a summary-health endpoint that checks queue age, dead jobs, expired leases, and sequence lag without returning tenant content or conversation identifiers. Existing Celery health remains separate.

Celery Beat reconciliation periodically selects due pending/retry rows with `FOR UPDATE SKIP LOCKED`, advances `available_at` by one dispatch-debounce interval, commits, and publishes summary tasks without taking a worker lease. A separate rate-limited backfill task requests compaction for historical conversations that already meet thresholds.

## 12. Scoped Legacy Cleanup

After the replacement is green, delete rather than wrap the legacy implementation.

Required removals:

- `app/ai/summarization_middleware.py`;
- `app/ai/conversation_summarizer.py`, replaced by the new structured compactor;
- `app/ai/memory.py` and `MemoryManager` graph fallback;
- the `MessageService` in-process summary runner, pending/active dictionaries, and `refresh_summary_after_turn` path;
- timestamp/UUID summary cursor comparison and queries;
- graph state keys `conversation_summarized`, `history_summary_updated_at`, and `summary_cursor_message_id`;
- duplicate `history_summary` plumbing, replaced with typed `conversation_memory`;
- the four-character token estimator and duplicate RAG token helper;
- tool-only overflow compaction superseded by full request budgeting;
- `MEMORY_SUMMARY_*`, `SUMMARIZATION_*`, and obsolete generic memory settings;
- obsolete imports, tests, comments, README sections, and `.env.example` entries.

Applied Alembic migrations remain unchanged. The new migration performs the forward schema transition.

Repository-wide cleanup assertions must show that removed module names, symbols, settings, and graph keys do not remain outside immutable migration history or explicit release/migration notes.

## 13. Testing Strategy

### 13.1 Unit tests

- Exact message/token trigger boundaries, including disabled individual thresholds.
- Configuration validation and production rejection of preview models.
- Structured payload validation, size limits, and prompt-injection strings.
- Token counting for English, Thai, JSON, tool schemas/results, images, and unknown models.
- Actual provider-usage extraction and estimate calibration.
- Complete-turn prefix selection and recent-turn retention.
- Budget reduction that never splits tool-call/result or user/assistant structures.
- Retry classification, backoff, lease expiry, and state transitions.

### 13.2 PostgreSQL integration tests

- Migration backfill and unique message sequences.
- Concurrent message sequence allocation.
- Atomic assistant insert plus job upsert.
- Composite cursor/target constraints and conversation cascades.
- Ownership-scoped memory lookup.
- Duplicate notification and concurrent worker claims.
- Compare-and-swap conflicts and newer-target preservation.
- Edit/delete invalidation and full rebuild.
- Reconciliation of expired processing leases.

### 13.3 End-to-end and failure tests

- Assistant persistence to Celery compaction to next-turn memory hydration.
- API crash after commit and before broker publish.
- Worker crash before and after provider response.
- Provider timeout, rate limit, invalid payload, and permanent failure.
- Background compaction below user-visible latency path.
- Emergency compaction and deterministic fallback.
- No summary/history overlap.
- Derived memory appears outside system instructions.
- Historical backfill is idempotent and rate limited.

### 13.4 Final verification

- Targeted tests for every red-green TDD increment.
- Full pytest suite.
- Ruff formatting and linting.
- Type checks configured by the repository.
- Alembic current/head and `alembic check`.
- Upgrade and downgrade on a disposable PostgreSQL database.
- Live schema/index/constraint inspection.
- Repository-wide legacy-symbol scan.
- Clean Git diff inspection with no unrelated changes.

## 14. Rollout

1. Configure the new environment names and stable compaction provider/model.
2. Apply the forward migration for sequences, summaries, and jobs.
3. Deploy API code and Celery workers with the `summary` queue and reconciler.
4. Enable durable compaction.
5. Run the rate-limited historical backfill.
6. Observe queue age, lag, retry/dead rate, token calibration, latency, and cost.
7. Complete scoped legacy deletion and run the full verification matrix again.

Because the local summary table currently contains no rows and the message volume is modest, the backfill is operationally small. The migration nevertheless validates existing data and preserves any summary rows present in other environments.

## 15. Success Criteria

The refactor is complete only when:

- every committed assistant message durably advances a coalesced database job;
- a lost broker notification is recovered automatically;
- duplicate and concurrent execution cannot regress or corrupt memory;
- trigger settings behave exactly at their documented boundaries;
- preflight token accounting covers the complete provider request and no longer uses a universal four-character heuristic;
- Thai and other multilingual inputs are handled conservatively;
- near-limit requests compact or trim before provider rejection;
- edits/deletes immediately prevent reuse of covered derived content;
- valid memory is tenant scoped, sequence based, and injected outside system instructions;
- PostgreSQL enforces one summary/job per conversation and same-conversation cursors;
- production configuration uses only `CONVERSATION_SUMMARY_*`;
- all scoped legacy files, settings, symbols, and redundant token paths are removed;
- the full verification matrix passes; and
- documentation matches the shipped architecture and operations.
