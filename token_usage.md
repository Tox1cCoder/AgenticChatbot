# Per-User Model Usage Analytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record every attributable model-provider call in a durable per-user ledger, expose trustworthy usage analytics, correct the single context-window gauge (including image generation), and add account-wide and conversation-level Streamlit views plus an AI SDK frontend contract.

**Architecture:** PostgreSQL is authoritative. A request-scoped usage context and one recorder normalize LangChain and direct-SDK calls into immutable usage events, atomically update UTC-minute rollups, and preserve application-controlled provider attempts without storing prompts or responses. SDK-internal retries are disabled so the application owns retry boundaries and can record them exactly. Authenticated APIs query only the caller's rows; Streamlit and the sidecar consume those APIs, while LangSmith remains a correlated observability system rather than an analytics dependency.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, SQLAlchemy 2.x, PostgreSQL, Alembic, Celery/Redis, LangChain/LangGraph, Google Gen AI SDK, OpenAI SDK, Streamlit, Altair, pytest.

---

## 1. Approved Product Scope

- Count every user-attributable application-controlled provider attempt: router, agents, subagents, tool loops, retries, fallbacks, title generation, follow-up suggestions, image generation/editing, image captioning, document processing, conversation compaction, and embeddings. Configure supported SDK/LangChain clients with internal retries disabled; the ledger does not claim visibility into undocumented transport retries below those clients.
- Each authenticated user sees only their own statistics. No admin role or cross-user API is introduced in this feature.
- Calls with no verified owner are recorded with `user_id = NULL`; they are excluded from user APIs and visible only through internal metrics/logging.
- Do not calculate or expose monetary cost.
- Start collecting at deployment. Do not backfill message metadata or LangSmith history.
- Keep raw per-attempt events for 90 days and aggregate buckets for two years.
- Provide totals, input/output split, hourly/daily trends, request count, success/error/cancelled rate, estimate coverage, and provider/model/operation/agent/conversation breakdowns.
- Add both a dedicated Streamlit **Usage** tab and a compact conversation-usage panel.
- Keep one context gauge. For shared-context models use `total_tokens / context_window_tokens`; for models with separate input/output limits use `max(input_tokens / max_input_tokens, output_tokens / max_output_tokens)`. Show the split, both applicable ratios, limit type, and usage source in its tooltip.
- Preserve existing AI SDK chat endpoints and stream framing. Add authenticated analytics endpoints and document them in `plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md`.

## 2. Current-Code Findings and Root Causes

1. `app/ai/token_counter.py` already extracts normalized text-model usage, and `app/ai/token_instrumentation.py` stores a per-response `token_breakdown`. That metadata describes only selected response paths and is not an immutable record of every provider attempt.
2. `app/ai/model_context.py::_select_used_tokens()` prefers `actual.total_tokens`, but the surrounding metadata does not expose a reliable input/output split to the UI. Several alternate paths attach only static context metadata.
3. `app/ai/image_generation/gemini.py` and `openai_provider.py` discard provider usage. The neutral image event vocabulary has no usage event, and `ImageGeneratorAgent._generate_images()` stops consuming once it receives the requested image count, which can skip a terminal usage-bearing chunk/event.
4. The configured `gemini-3-pro-image-preview` is absent from the context registry and is deprecated. Google lists `gemini-3-pro-image` as its replacement with a 65,536-token input limit and 32,768-token output limit.
5. OpenAI image completion events report input, output, total, and modality token details, but OpenAI does not publish a conventional context-window limit for GPT Image models. Record their tokens while leaving their gauge denominator unknown rather than inventing a limit.
6. User identity already reaches normal workflow requests through `WorkflowExecutionRequest.user_id`, but title/suggestion helpers and some worker/direct-SDK paths drop it.
7. `app/workers/cleanup_tasks.py` replaces `celery_app.conf.beat_schedule` instead of merging it, so periodic schedules declared earlier in `celery_app.py` can be lost after imports. The retention work must fix and pin this behavior.
8. A daily or hourly UTC rollup cannot reconstruct every IANA local-day boundary (for example, `Asia/Kathmandu` starts a local day at `18:15Z`) after raw events expire. Use UTC-minute rollups for two years, require minute-aligned query instants, and derive hour/day buckets from exact local boundaries at query time. Reject a requested historical boundary if `zoneinfo` resolves it with sub-minute precision rather than returning an inexact result.

Official references checked on 2026-07-20:

- Gemini token counting and `usage_metadata`: https://ai.google.dev/gemini-api/docs/generate-content/tokens
- Gemini `GenerateContentResponse.usageMetadata`: https://ai.google.dev/api/generate-content
- Gemini image generation: https://ai.google.dev/gemini-api/docs/image-generation
- Gemini 3 Pro Image limits: https://ai.google.dev/gemini-api/docs/models/gemini-3-pro-image
- Gemini deprecations: https://ai.google.dev/gemini-api/docs/deprecations
- OpenAI image-stream completion usage: https://platform.openai.com/docs/api-reference/images-streaming/image_generation/partial_image
- OpenAI GPT Image models: https://developers.openai.com/api/docs/models/gpt-image-2

## 3. Non-Goals

- Billing, currency conversion, price tables, budgets, quotas, enforcement, or alerts.
- Admin roles, organization dashboards, cross-user leaderboards, or arbitrary user-ID filters.
- Historical reconstruction.
- Persisting prompts, responses, tool arguments, images, credentials, or raw provider response bodies.
- Making LangSmith required for application analytics.
- Claiming a context-window denominator for models whose provider does not publish one.

## 4. Target Data Contract

### 4.1 Immutable event

`model_usage_events` stores one row per application-controlled provider attempt:

```text
id UUID primary key
event_key VARCHAR(160) unique, stable for idempotent retries
operation_id UUID, allocated before related retry/context-overflow/provider-fallback loops
attempt INTEGER >= 1, monotonically allocated within operation_id
user_id UUID nullable -> users.id ON DELETE CASCADE
conversation_id UUID nullable -> conversations.id ON DELETE SET NULL
request_message_id UUID nullable -> messages.id ON DELETE SET NULL
document_id UUID nullable -> documents.id ON DELETE SET NULL
correlation_id VARCHAR(128) nullable
langsmith_run_id UUID nullable
provider_request_id VARCHAR(255) nullable
provider VARCHAR(32)
model VARCHAR(255)
operation VARCHAR(64)
agent_id VARCHAR(128) nullable
status VARCHAR(16): success | error | cancelled | timeout
usage_source VARCHAR(32): provider_reported | mixed_reported_estimated | locally_estimated | unavailable
input_tokens BIGINT nullable
output_tokens BIGINT nullable
total_tokens BIGINT nullable
reasoning_tokens BIGINT nullable
cached_input_tokens BIGINT nullable
input_text_tokens BIGINT nullable
input_image_tokens BIGINT nullable
output_text_tokens BIGINT nullable
output_image_tokens BIGINT nullable
generated_images INTEGER >= 0
latency_ms BIGINT >= 0
error_code VARCHAR(128) nullable
started_at TIMESTAMPTZ
completed_at TIMESTAMPTZ
created_at TIMESTAMPTZ
```

Token columns use `NULL` for unknown and `0` only when the provider or estimator reports zero. Do not enforce `total = input + output`; reasoning and provider accounting semantics can make that equation false.

Enforce both `UNIQUE(event_key)` and `UNIQUE(operation_id, attempt)`. Derive `event_key` as `<operation_id>:<attempt>` so the same normalized command remains idempotent when replayed after a ledger-write failure.

`request_message_id` always refers to an already-persisted initiating message. Never bind the reserved assistant UUID before `MessageService` inserts that row. Provider calls without a persisted initiating message leave this field `NULL` and use their conversation/document attribution instead.

### 4.2 Long-term rollup

`model_usage_minute` stores UTC-minute aggregates keyed by:

```text
rollup_key, derived as SHA-256 over bucket_start_utc plus every dimension
bucket_start_utc, user_id, conversation_id, provider, model,
operation, agent_id, status, usage_source
```

It contains sums of every normalized token/image field, a `*_known_count` beside each nullable token sum, `request_count`, and `latency_ms_sum`. Unknown values increment the request count but not a known count; SQL aggregation may coalesce their numeric contribution to zero only because coverage remains explicit. The deterministic non-null `rollup_key` avoids PostgreSQL 14's distinct-null uniqueness behavior. Compute it from UTF-8 JSON encoding of a versioned ordered array with explicit `null` entries and an RFC 3339 UTC minute, not delimiter-joined strings. `user_id` uses `ON DELETE CASCADE`; `conversation_id` uses `ON DELETE CASCADE`, so deleting an owner or conversation removes aggregates rather than mutating a hashed dimension. The event insert and rollup upsert occur in one transaction. Minute grain plus minute-aligned API boundaries permits exact regrouping for current IANA offsets, including quarter-hour zones and 23/25-hour DST days. The service validates every resolved boundary and returns 422 if it has non-zero seconds or microseconds.

### 4.3 Context-window metadata

The existing `message_metadata.context_window` remains additive and backward compatible:

```json
{
  "provider": "gemini",
  "model": "gemini-3-pro-image",
  "context_window_tokens": null,
  "max_input_tokens": 65536,
  "max_output_tokens": 32768,
  "limit_type": "separate_io",
  "known": true,
  "source": "registry",
  "input_tokens": 120,
  "output_tokens": 2048,
  "total_tokens": 2168,
  "usage_source": "provider_reported",
  "used_tokens": 2168,
  "used_token_source": "provider_reported_total",
  "input_usage_ratio": 0.0018310546875,
  "output_usage_ratio": 0.0625,
  "usage_ratio": 0.0625,
  "usage_ratio_basis": "most_constrained_io_limit",
  "display_state": "ok"
}
```

Provider-reported input, output, and total are preserved independently; a reported total never overwrites a reported split. For `shared_context`, ratio precedence is provider-reported total, sum of known input/output, explicit estimates, then unavailable. For `separate_io`, calculate each known side only against its matching limit and use the larger ratio; do not divide a combined total by either limit. The API returns the raw ratio. Streamlit caps only the drawn gauge at 100%.

### 4.4 User analytics endpoints

```http
GET /usage/dashboard?from=2026-07-01T00:00:00%2B07:00&to=2026-08-01T00:00:00%2B07:00&bucket=day&timezone=Asia/Bangkok&conversationId=<optional UUID>
GET /usage/conversations/{conversationId}?from=<optional>&to=<optional>&bucket=day&timezone=Asia/Bangkok
```

Rules:

- `from` is inclusive and `to` is exclusive. Both must include an offset and have zero seconds/microseconds.
- Dashboard default range is the last 30 days. The conversation endpoint defaults to all retained rollups (two years). Both enforce a two-year maximum.
- `bucket=hour` is limited to 31 days and requires local top-of-hour boundaries; `bucket=day` supports two years and requires local-midnight boundaries in the requested timezone.
- `timezone` must resolve through `zoneinfo.ZoneInfo`; default is `UTC`.
- No endpoint accepts `userId`.
- Conversation endpoints and filters validate ownership before querying usage.
- Responses use the existing `ApiResponse` envelope and camelCase aliases.

## 5. File Structure

New focused units:

- `app/usage/types.py` — immutable context, normalized usage, and event command types.
- `app/usage/context.py` — async-safe `ContextVar` binding and child scopes.
- `app/usage/normalizers.py` — provider response and modality normalization.
- `app/usage/recorder.py` — sync/async attempt wrappers and failure fallback.
- `app/models/model_usage.py` — raw-event and UTC-minute SQLAlchemy models.
- `app/repositories/model_usage.py` — atomic insert/upsert, query, reconcile, and retention SQL.
- `app/services/model_usage_service.py` — ownership, aligned-range validation, timezone bucketing, and response assembly.
- `app/interfaces/model_usage_service_interface.py` — service boundary used by FastAPI injection.
- `app/schemas/model_usage.py` — API query and response models.
- `app/api/model_usage.py` — authenticated user routes.
- `app/observability/model_usage.py` — bounded-cardinality operational metrics.
- `app/workers/model_usage.py` — failed-write retry, reconciliation, and retention tasks.
- `plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md` — concise frontend handoff.

Existing files are modified only where they own call execution, identity propagation, router wiring, metadata, proxying, or UI.

---

## 6. Implementation Tasks

### Task 1: Define the Usage Domain and Async Context

**Files:**
- Create: `app/usage/__init__.py`
- Create: `app/usage/types.py`
- Create: `app/usage/context.py`
- Test: `tests/test_model_usage_context.py`

- [x] **Step 1: Write failing tests for validation and task isolation**

```python
def test_normalized_usage_preserves_unknown_instead_of_zero():
    usage = NormalizedUsage(input_tokens=None, output_tokens=0)
    assert usage.input_tokens is None
    assert usage.output_tokens == 0

@pytest.mark.asyncio
async def test_usage_context_isolated_between_concurrent_users():
    async def read_bound(user_id: UUID):
        with bind_usage_context(UsageContext(user_id=user_id)):
            await asyncio.sleep(0)
            return current_usage_context().user_id

    first, second = uuid4(), uuid4()
    assert await asyncio.gather(read_bound(first), read_bound(second)) == [first, second]
```

- [x] **Step 2: Run the focused test and confirm missing imports fail**

Run: `python -m pytest tests/test_model_usage_context.py -q`

Expected: collection fails because `app.usage` does not exist.

- [x] **Step 3: Implement immutable types and scoped binding**

```python
UsageStatus = Literal["success", "error", "cancelled", "timeout"]
UsageSource = Literal[
    "provider_reported",
    "mixed_reported_estimated",
    "locally_estimated",
    "unavailable",
]

@dataclass(frozen=True)
class UsageContext:
    user_id: UUID | None = None
    conversation_id: UUID | None = None
    request_message_id: UUID | None = None
    document_id: UUID | None = None
    correlation_id: str | None = None
    langsmith_run_id: UUID | None = None
    operation: str = "unknown"
    agent_id: str | None = None

    def child(self, **changes: Any) -> "UsageContext":
        return replace(self, **changes)

@dataclass
class UsageOperation:
    operation_id: UUID = field(default_factory=uuid4)
    _next_attempt: int = field(default=1, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def allocate_attempt(self) -> int:
        with self._lock:
            attempt = self._next_attempt
            self._next_attempt += 1
            return attempt

@dataclass(frozen=True)
class NormalizedUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    input_text_tokens: int | None = None
    input_image_tokens: int | None = None
    output_text_tokens: int | None = None
    output_image_tokens: int | None = None
    generated_images: int = 0
    source: UsageSource = "unavailable"

_usage_context: ContextVar[UsageContext] = ContextVar(
    "model_usage_context", default=UsageContext()
)
_usage_operation: ContextVar[UsageOperation | None] = ContextVar(
    "model_usage_operation", default=None
)

@contextmanager
def bind_usage_context(context: UsageContext) -> Iterator[UsageContext]:
    token = _usage_context.set(context)
    try:
        yield context
    finally:
        _usage_context.reset(token)

@contextmanager
def begin_usage_operation() -> Iterator[UsageOperation]:
    operation = UsageOperation()
    token = _usage_operation.set(operation)
    try:
        yield operation
    finally:
        _usage_operation.reset(token)
```

Validate every numeric field as a non-boolean non-negative integer in `NormalizedUsage.__post_init__`. Add tests proving nested child contexts share one operation allocator, distinct operations receive distinct IDs, and concurrent operations cannot reuse an `(operation_id, attempt)` pair.

- [x] **Step 4: Run the tests**

Run: `python -m pytest tests/test_model_usage_context.py -q`

Expected: all tests pass.

- [x] **Step 5: Commit**

```bash
git add app/usage tests/test_model_usage_context.py
git commit -m "feat: define model usage context"
```

### Task 2: Add the Raw Event and Minute Rollup Schema

**Files:**
- Create: `app/models/model_usage.py`
- Modify: `app/models/__init__.py`
- Modify: `app/models/user.py`
- Create: `app/alembic/versions/y2z3a4b5c6d7_add_model_usage_ledger.py`
- Test: `tests/test_model_usage_schema.py`
- Test: `tests/test_model_usage_migration.py`

- [x] **Step 1: Add failing schema-contract tests**

Assert exact table names, nullable token columns, unique `event_key`, unique `(operation_id, attempt)`, non-negative checks, foreign-key delete behavior, and these indexes:

```python
expected_indexes = {
    "ix_model_usage_events_user_started",
    "ix_model_usage_events_conversation_started",
    "ix_model_usage_events_provider_model_started",
    "ix_model_usage_events_operation_started",
    "ix_model_usage_minute_user_bucket",
    "ix_model_usage_minute_conversation_bucket",
}
```

The migration test must assert `down_revision == "x1y2z3a4b5c6"`, both tables are created on upgrade, and downgrade drops rollups before events.

- [x] **Step 2: Run schema tests and confirm failure**

Run: `python -m pytest tests/test_model_usage_schema.py tests/test_model_usage_migration.py -q`

Expected: imports or table assertions fail.

- [x] **Step 3: Implement SQLAlchemy models and relationships**

Use `BigInteger` for all counts, known counts, and latency, `DateTime(timezone=True)` for instants, `Date` is not used, and `server_default=func.now()` for creation time. Add a unique non-null `rollup_key = sha256(canonical_json_bytes).hexdigest()` to the rollup instead of relying on PostgreSQL 14's distinct-null uniqueness behavior. Keep `user_id` and `conversation_id` foreign keys on the minute table with `ON DELETE CASCADE`; hashed rollup dimensions must never be changed to `NULL` by FK actions.

```python
class ModelUsageEvent(Base):
    __tablename__ = "model_usage_events"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_key = Column(String(160), nullable=False, unique=True)
    operation_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    attempt = Column(Integer, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL")
    )
    request_message_id = Column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL")
    )
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"))
```

Add `usage_events = relationship("ModelUsageEvent", back_populates="user")` to `User` without eager loading.

- [x] **Step 4: Create the Alembic migration**

The migration must use explicit check constraints for attempts, tokens, image counts, and latency; create indexes after tables; and use `ON DELETE` clauses matching the ORM. It must create `model_usage_minute`, not an hourly table. Do not add data backfill SQL.

- [x] **Step 5: Run schema and migration tests**

Run: `python -m pytest tests/test_model_usage_schema.py tests/test_model_usage_migration.py tests/test_alembic_autogenerate_filters.py -q`

Expected: all tests pass.

- [x] **Step 6: Commit**

```bash
git add app/models app/alembic/versions/y2z3a4b5c6d7_add_model_usage_ledger.py tests/test_model_usage_schema.py tests/test_model_usage_migration.py
git commit -m "feat: add model usage ledger schema"
```

### Task 3: Implement Atomic Recording, Rollups, and Queries

**Files:**
- Create: `app/repositories/model_usage.py`
- Test: `tests/integration/test_model_usage_repository_postgres.py`

- [ ] **Step 1: Write PostgreSQL integration tests**

Cover seven named cases: `test_record_event_is_idempotent_and_increments_rollup_once`, `test_unknown_tokens_remain_null_while_rollup_uses_zero_for_sum`, `test_same_logical_operation_distinct_attempts_are_recorded`, `test_user_and_conversation_filters_never_cross_tenants`, `test_reconcile_minute_rebuilds_exactly_from_raw_events`, `test_rollup_fk_deletes_do_not_mutate_hashed_dimensions`, and `test_cleanup_deletes_raw_older_than_90_days_and_rollups_older_than_2_years`. Each test creates concrete users, persisted request messages, conversations, timestamps, and token counts through existing PostgreSQL fixtures, then asserts raw rows and rollup sums directly. Also assert that a reserved but unpersisted assistant UUID is rejected if accidentally supplied as `request_message_id`.

- [ ] **Step 2: Run the integration file and confirm failure**

Run: `python -m pytest tests/integration/test_model_usage_repository_postgres.py -q`

Expected: repository import fails.

- [ ] **Step 3: Implement `record_event()` as one transaction**

Use SQLAlchemy's PostgreSQL `insert(ModelUsageEvent).on_conflict_do_nothing(index_elements=["event_key"]).returning(ModelUsageEvent.id)`. Only when an event ID is returned should the same transaction execute an `insert(ModelUsageMinute).on_conflict_do_update(index_elements=["rollup_key"], set_=aggregate_increments)` statement.

```python
inserted_id = session.execute(event_insert.returning(ModelUsageEvent.id)).scalar_one_or_none()
if inserted_id is None:
    session.rollback()
    return RecordResult(inserted=False, event_id=None)
session.execute(minute_upsert)
session.commit()
return RecordResult(inserted=True, event_id=inserted_id)
```

The minute key is `started_at.astimezone(timezone.utc).replace(second=0, microsecond=0)`.

- [ ] **Step 4: Implement bounded analytics and maintenance queries**

Provide repository methods for summary totals, minute series, dimension breakdowns, latest relevant conversation event, recent-minute reconciliation, batched raw deletion, and batched rollup deletion. Every user query requires a non-null `user_id` argument and applies it in the first SQL statement.

- [ ] **Step 5: Run integration tests**

Run: `python -m pytest tests/integration/test_model_usage_repository_postgres.py -q`

Expected: all tests pass against the configured PostgreSQL test database.

- [ ] **Step 6: Commit**

```bash
git add app/repositories/model_usage.py tests/integration/test_model_usage_repository_postgres.py
git commit -m "feat: persist and aggregate model usage"
```

### Task 4: Normalize Provider Usage Without Storing Raw Payloads

**Files:**
- Create: `app/usage/normalizers.py`
- Modify: `app/ai/token_counter.py`
- Test: `tests/test_model_usage_normalizers.py`
- Test: `tests/test_token_counter.py`

- [ ] **Step 1: Write failing provider-shape tests**

Include Gemini prompt/candidate/thought/cache counts and modality arrays, OpenAI completion usage and image usage details, Anthropic cache/reasoning-compatible aliases, totals missing but input/output present, negative values, booleans, malformed objects, and fully absent usage.

```python
usage = normalize_provider_usage(
    provider="openai",
    payload={
        "usage": {
            "input_tokens": 50,
            "output_tokens": 100,
            "total_tokens": 150,
            "input_tokens_details": {"text_tokens": 10, "image_tokens": 40},
        }
    },
)
assert usage == NormalizedUsage(
    input_tokens=50,
    output_tokens=100,
    total_tokens=150,
    input_text_tokens=10,
    input_image_tokens=40,
    source="provider_reported",
)
```

- [ ] **Step 2: Run tests and confirm new cases fail**

Run: `python -m pytest tests/test_model_usage_normalizers.py tests/test_token_counter.py -q`

- [ ] **Step 3: Implement normalization**

Reuse small safe-number helpers from `TokenCounter`, extend `ReportedTokenUsage` with cached and modality details, and make `normalize_provider_usage()` return `NormalizedUsage(source="unavailable")` rather than `None`. Never retain the input payload.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_model_usage_normalizers.py tests/test_token_counter.py tests/test_context_window_message_metadata.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/usage/normalizers.py app/ai/token_counter.py tests/test_model_usage_normalizers.py tests/test_token_counter.py
git commit -m "feat: normalize provider usage metadata"
```

### Task 5: Add the Recorder, Retry Ownership, and Operational Metrics

**Files:**
- Create: `app/usage/recorder.py`
- Create: `app/observability/model_usage.py`
- Create: `app/workers/model_usage.py`
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Modify: `app/ai/agent_config.py`
- Modify: `app/ai/model_factory.py`
- Modify: `app/ai/agents/router.py`
- Modify: `app/services/rag_embedding_service.py`
- Modify: `app/services/document_processing_service.py`
- Modify: `app/services/provider_service.py`
- Modify: `app/ai/mcp_servers/form_filler_server.py`
- Modify: `app/ai/image_generation/gemini.py`
- Modify: `app/ai/image_generation/openai_provider.py`
- Modify: `app/workers/celery_app.py`
- Test: `tests/test_model_usage_recorder.py`
- Test: `tests/test_model_usage_worker_config.py`
- Test: `tests/test_model_usage_client_retry_config.py`

- [ ] **Step 1: Add failing recorder tests**

Verify success, timeout, cancellation, error re-raise, operation-scoped monotonic attempt numbers, local-estimate fallback, repository failure enqueue, broker failure logging, prompt/response exclusion from retry payloads, and non-blocking persistence from async call paths. Assert recorder wrappers execute exactly one supplied call and never add a provider retry loop.

- [ ] **Step 2: Add explicit settings**

```python
model_usage_tracking_enabled: bool = True
model_usage_ui_enabled: bool = True
model_usage_raw_retention_days: int = 90
model_usage_rollup_retention_days: int = 730
model_usage_reconcile_minutes: int = 2880
model_usage_cleanup_batch_size: int = 5000
model_usage_retry_max_attempts: int = 5
model_usage_retry_base_seconds: int = 10
model_usage_user_hash_secret: str = ""
```

Validate positive retention/batch/retry values and require `MODEL_USAGE_USER_HASH_SECRET` in production when LangSmith tracing is enabled.

- [ ] **Step 3: Disable client retries and implement one-attempt wrappers**

```python
async def record_one_async_attempt(
    self, *, call, provider, model, operation: UsageOperation, estimate=None
):
    attempt = operation.allocate_attempt()
    event_key = f"{operation.operation_id}:{attempt}"
    started = utc_now()
    try:
        response = await call()
    except asyncio.CancelledError:
        await self._persist_async(event_key, attempt, "cancelled", started=started)
        raise
    except Exception as exc:
        await self._persist_async(
            event_key, attempt, classify_status(exc),
            error_code=classify_error(exc), started=started,
        )
        raise
    usage = normalize_provider_usage(provider=provider, payload=response)
    await self._persist_async(
        event_key, attempt, "success",
        usage=usage if usage.source != "unavailable" else estimate(response),
        started=started,
    )
    return response
```

Set `max_retries=0` in every supported `ChatGoogleGenerativeAI`, `ChatOpenAI`, `OpenAI`, and `AsyncOpenAI` construction path used by instrumented calls. Construct Google Gen AI clients with `http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))`; that SDK defines attempts as including the original request, so `1` disables retry. Preserve behavior by keeping existing application retry loops and using one explicit recorder-owned loop only for direct SDK paths that currently rely solely on SDK retries. Tests monkeypatch constructors and assert these exact settings. The event definition is therefore an application-controlled provider attempt; redirects or retries hidden below the SDK transport are explicitly outside the ledger contract.

The async persistence path calls a repository method that creates and closes its own SQLAlchemy session inside `asyncio.to_thread()`; never share a request-thread `Session` with that worker thread and never block the event loop on synchronous database I/O.

- [ ] **Step 4: Implement failure fallback and metrics**

The failed-write retry payload contains only the normalized event command. Celery retry uses exponential backoff and the same `event_key`; `model_usage_retry_*` settings govern ledger-write delivery, not provider calls. Metrics use bounded labels only: provider family, operation family, status, source, and failure class. Never label with user, conversation, trace, or model IDs.

- [ ] **Step 5: Register worker imports/routes without replacing existing schedules**

Add `app.workers.model_usage` to `celery_app.conf.imports`, route retry/reconcile/cleanup tasks to the `summary` queue, and change `cleanup_tasks.py` to call `celery_app.conf.beat_schedule.update(cleanup_schedule)` with its existing three named cleanup entries.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_model_usage_recorder.py tests/test_model_usage_worker_config.py tests/test_model_usage_client_retry_config.py tests/test_celery_worker_config.py -q`

Expected: all tests pass and existing conversation-summary beat entries remain present.

- [ ] **Step 7: Commit**

```bash
git add app/usage/recorder.py app/observability/model_usage.py app/workers/model_usage.py app/core/config.py .env.example app/ai/agent_config.py app/ai/model_factory.py app/ai/agents/router.py app/services/rag_embedding_service.py app/services/document_processing_service.py app/services/provider_service.py app/ai/mcp_servers/form_filler_server.py app/ai/image_generation/gemini.py app/ai/image_generation/openai_provider.py app/workers/celery_app.py app/workers/cleanup_tasks.py tests/test_model_usage_recorder.py tests/test_model_usage_worker_config.py tests/test_model_usage_client_retry_config.py tests/test_celery_worker_config.py
git commit -m "feat: record model attempts reliably"
```

### Task 6: Wire the Recorder Through Dependency Injection

**Files:**
- Modify: `app/core/container.py`
- Modify: `app/core/dependency_injection.py`
- Modify: `app/main.py`
- Modify: `app/ai/graph.py`
- Modify: `app/ai/agents/base_agent.py`
- Modify: `app/ai/agents/router.py`
- Modify: `app/ai/agents/custom_agent.py`
- Test: `tests/test_model_usage_container.py`
- Test: `tests/test_model_usage_workflow_wiring.py`

- [ ] **Step 1: Write a failing container contract test**

Assert `model_usage_repository` is a factory, `model_usage_recorder` and metrics are singletons, and `model_usage_service` is injectable through its interface. Construct the real workflow and assert the same recorder reaches `Router`, every built-in `BaseAgent`, and a lazily created `CustomAgent`.

- [ ] **Step 2: Add providers and wiring**

```python
model_usage_repository = providers.Factory(
    ModelUsageRepository,
    session_factory=db.provided.session,
)
model_usage_recorder = providers.Singleton(
    ModelUsageRecorder,
    repository=model_usage_repository,
    settings=providers.Object(settings),
)
model_usage_service: providers.Provider[IModelUsageService] = providers.Factory(
    ModelUsageService,
    repository=model_usage_repository,
    conversation_repository=conversation_repository,
    settings=providers.Object(settings),
)
```

Add the interface mapping to `AppAutoInjector` and wire `app.api.model_usage` in `create_app()`. Add `model_usage_recorder` to `create_workflow(...)` and `MultiAgentWorkflow.__init__(...)`, then pass it explicitly to `Router`, every built-in agent constructor, and the on-demand `CustomAgent` construction path. Store it on `BaseAgent`; do not resolve the container from agent modules or introduce a mutable global recorder.

- [ ] **Step 3: Run the container test**

Run: `python -m pytest tests/test_model_usage_container.py tests/test_model_usage_workflow_wiring.py tests/test_container_import.py tests/test_graph_refactor_contract.py -q`

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add app/core/container.py app/core/dependency_injection.py app/main.py app/ai/graph.py app/ai/agents/base_agent.py app/ai/agents/router.py app/ai/agents/custom_agent.py tests/test_model_usage_container.py tests/test_model_usage_workflow_wiring.py
git commit -m "feat: wire model usage services"
```

### Task 7: Instrument Workflow, Agent Retries, Router, Titles, and Suggestions

**Files:**
- Modify: `app/services/ai_service.py`
- Modify: `app/services/message_service.py`
- Modify: `app/api/conversations.py`
- Modify: `app/ai/agents/base_agent.py`
- Modify: `app/ai/agents/router.py`
- Modify: `app/ai/graph.py`
- Modify: `app/ai/suggestion_generator.py`
- Test: `tests/test_model_usage_workflow_instrumentation.py`

- [ ] **Step 1: Write failing attribution and attempt tests**

Cover non-stream, stream, resume, router fallback, three generic provider retries, context-overflow retry, provider fallback, title generation from both API and automatic task, suggestions, concurrent users, and cancelled streams. Assert every attempt has the correct `user_id`, conversation/request-message IDs, operation ID, monotonically increasing attempt number, agent, provider, and model. Assert all retries/fallbacks for one logical invocation share one `operation_id`, while a later tool-loop invocation starts a new one.

- [ ] **Step 2: Bind usage context at AI service boundaries**

```python
context = UsageContext(
    user_id=parse_uuid(request.user_id),
    conversation_id=parse_uuid(request.conversation_id),
    request_message_id=parse_uuid(request.user_message_id),
    correlation_id=request.thread_id,
    operation="workflow",
)
with bind_usage_context(context):
    return await self.workflow.execute_request(self._to_ai_request(prepared_request))
```

For async generators, keep the context manager active for the complete `async for` body. `request.user_message_id` refers to the row persisted before workflow execution; never use the reserved `assistant_message_id` as a foreign key during generation. Resume paths rebuild ownership from authenticated arguments rather than checkpoint state.

- [ ] **Step 3: Wrap each real provider attempt in `BaseAgent._ainvoke_with_retries()`**

Create `begin_usage_operation()` before entering the existing generic retry/context-overflow/provider-fallback loops. Pass the resulting operation object through every branch and call `recorder.record_one_async_attempt()` immediately around each model invocation. Allocate attempts only when a provider call will occur. Keep current retry delays and fallback behavior; do not add a second retry loop or create a new operation ID inside a branch.

- [ ] **Step 4: Instrument direct text helpers and propagate identity**

Change title signatures to `generate_conversation_title(user_message, *, user_id, conversation_id=None)` and suggestion signatures to accept `UsageContext`. The public title route passes its authenticated UUID. `MessageService._generate_title_async()` and `_generate_and_add_suggestions()` pass the already verified user/conversation/message IDs.

- [ ] **Step 5: Instrument the router**

Wrap the existing `asyncio.to_thread()` Gemini `generate_content` call with operation `router` and agent `router`. Preserve deterministic short-circuits as zero provider events because no call occurred.

- [ ] **Step 6: Correlate LangSmith without exposing raw user IDs**

Add `metadata={"usage_operation_id": str(operation.operation_id), "usage_user_hash": hmac_sha256(secret, user_id)}` and bounded tags to the existing runnable config. Do not add email, username, conversation content, or raw UUID user tags.

- [ ] **Step 7: Run focused tests**

Run: `python -m pytest tests/test_model_usage_workflow_instrumentation.py tests/test_router.py tests/test_ai_service_initialization.py tests/test_context_overflow_retry.py tests/test_message_service_event_streaming.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add app/services/ai_service.py app/services/message_service.py app/api/conversations.py app/ai/graph.py app/ai/agents/base_agent.py app/ai/agents/router.py app/ai/suggestion_generator.py tests/test_model_usage_workflow_instrumentation.py
git commit -m "feat: attribute workflow model usage"
```

### Task 8: Correct Context Accounting and Image-Model Limits

**Files:**
- Modify: `app/ai/model_context.py`
- Modify: `app/ai/token_instrumentation.py`
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Test: `tests/test_model_context_metadata.py`
- Test: `tests/test_context_window_message_metadata.py`

- [ ] **Step 1: Write failing limit-aware gauge tests**

```python
def test_separate_io_context_uses_most_constrained_limit():
    result = build_context_window_usage(
        {
            "provider": "gemini",
            "model": "gemini-3-pro-image",
            "context_window_tokens": None,
            "max_input_tokens": 65_536,
            "max_output_tokens": 32_768,
            "limit_type": "separate_io",
            "known": True,
            "source": "registry",
        },
        NormalizedUsage(
            input_tokens=20_000,
            output_tokens=5_000,
            total_tokens=25_000,
            source="provider_reported",
        ),
    )
    assert result["input_tokens"] == 20_000
    assert result["output_tokens"] == 5_000
    assert result["used_tokens"] == 25_000
    assert result["input_usage_ratio"] == pytest.approx(20_000 / 65_536)
    assert result["output_usage_ratio"] == pytest.approx(5_000 / 32_768)
    assert result["usage_ratio"] == pytest.approx(20_000 / 65_536)
    assert result["usage_ratio_basis"] == "most_constrained_io_limit"

```

Add four more concrete tests named `test_shared_context_model_uses_reported_total`, `test_openai_image_usage_has_tokens_but_unknown_window`, `test_gemini_3_pro_image_has_65536_input_and_32768_output_limits`, and `test_ratio_is_not_clamped_in_backend`.

- [ ] **Step 2: Update the usage algorithm and its callers**

Change `build_context_window_usage()` to accept `NormalizedUsage`. Replace `_select_used_tokens()` with a function returning input, output, total, and source without forcing `total == input + output`. For `limit_type="shared_context"`, prefer a provider total only for the ratio and otherwise sum known split values or explicit estimates. For `limit_type="separate_io"`, compute input and output ratios independently and choose the larger known ratio. For `limit_type="unknown"`, retain counts but return `usage_ratio=None`. `BaseAgent._merge_context_window_usage()` converts existing `token_breakdown.actual` values to `NormalizedUsage`, so legacy persisted metadata remains readable. Attach `limit_type`, both split ratios, `usage_ratio`, and `usage_ratio_basis` to the context payload.

- [ ] **Step 3: Update the registry and default model**

Add exact/family entries for `gemini-3-pro-image`, its preview alias for persisted-history compatibility, and `gemini-3.1-flash-image`. Mark Gemini image entries `separate_io` with their documented input/output limits and no synthetic shared-context denominator. Change `image_generator_model` default and `.env.example` from `gemini-3-pro-image-preview` to `gemini-3-pro-image`. Mark `gpt-image-*` and `dall-e-*` limits `unknown`; do not add guessed denominators.

- [ ] **Step 4: Estimate response output only when reported output is absent**

Use `TokenCounter.count_text(provider, model, coerce_response_text(response.content))`; label the resulting context source `mixed_reported_estimated` or `locally_estimated`. Never overwrite provider-reported counts.

- [ ] **Step 5: Run context tests**

Run: `python -m pytest tests/test_model_context_metadata.py tests/test_context_window_message_metadata.py tests/test_ai_sdk_context_window.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/ai/model_context.py app/ai/token_instrumentation.py app/core/config.py .env.example tests/test_model_context_metadata.py tests/test_context_window_message_metadata.py
git commit -m "fix: calculate limit-aware context usage"
```

### Task 9: Preserve Gemini and OpenAI Image Usage End to End

**Files:**
- Modify: `app/ai/image_generation/models.py`
- Modify: `app/ai/image_generation/__init__.py`
- Modify: `app/ai/image_generation/gemini.py`
- Modify: `app/ai/image_generation/openai_provider.py`
- Modify: `app/ai/agents/image_generator_agent.py`
- Test: `tests/test_image_generation_providers.py`
- Test: `tests/test_model_usage_image_generation.py`

- [ ] **Step 1: Write failing stream-usage tests**

Gemini tests must put `usage_metadata` on a final chunk after the image chunk, proving the provider consumes it. OpenAI tests must attach `event.usage` to `image_generation.completed` and `image_edit.completed`. Agent tests must prove it does not break after the first `ImageFinal` before consuming usage. Add cancellation/error tests proving a started stream attempt is finalized exactly once with the terminal status even when no usage event arrives.

- [ ] **Step 2: Add a provider-neutral terminal usage event**

```python
@dataclass(frozen=True)
class ImageUsage:
    usage: NormalizedUsage
    provider_request_id: str | None = None

ImageStreamEvent = ImagePartial | ImageFinal | NarrativeDelta | ImageUsage
```

- [ ] **Step 3: Consume complete Gemini streams**

Stop emitting images after `max_images`, but continue iterating chunks. Track the latest `usage_metadata` and response ID; after exhaustion, emit exactly one `ImageUsage`. This avoids cancelling the provider stream before its terminal accounting arrives.

- [ ] **Step 4: Extract OpenAI completion-event usage**

On completed generate/edit events, emit the final image and then `ImageUsage(usage=normalize_provider_usage(provider="openai", payload=event))`. Partial events never record usage.

- [ ] **Step 5: Record the image operation once and select it for final gauge metadata**

`ImageGeneratorAgent._generate_images()` starts one recorder streaming-attempt handle immediately before each provider stream, consumes all events, captures `ImageUsage`, and finalizes that handle exactly once as success/error/cancelled/timeout. The handle owns the operation-scoped attempt number and accepts terminal usage after stream exhaustion; it does not persist a success event when the stream starts. Count final images and merge the image model's context/usage into the final response. Prompt-enhancement and acknowledgement calls remain separate operations and ledger events but do not replace the image model gauge.

- [ ] **Step 6: Run image tests**

Run: `python -m pytest tests/test_image_generation_providers.py tests/test_model_usage_image_generation.py tests/test_image_generator_harvest.py tests/test_image_preview_stream.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/ai/image_generation app/ai/agents/image_generator_agent.py tests/test_image_generation_providers.py tests/test_model_usage_image_generation.py
git commit -m "fix: retain image generation token usage"
```

### Task 10: Instrument Vision, Captioning, Embeddings, and Document Workers

**Files:**
- Modify: `app/ai/agents/chat_agent.py`
- Modify: `app/services/document_processing_service.py`
- Modify: `app/services/rag_embedding_service.py`
- Modify: `app/services/document_index_service.py`
- Modify: `app/workers/document_processor.py`
- Test: `tests/test_model_usage_direct_sdk.py`
- Test: `tests/test_model_usage_document_workers.py`
- Test: `tests/test_rag_embedding_service.py`

- [ ] **Step 1: Write failing direct-SDK tests**

Assert actual provider counts for vision and caption generation; per-batch embedding events; locally estimated embedding input when the SDK response has no usage; one event per retry attempt; document-owner attribution; query embeddings in chat context; and `user_id = NULL` when a maintenance call has no owner.

- [ ] **Step 2: Wrap synchronous SDK calls without blocking async loops further**

Use the recorder's sync wrapper inside functions already running in worker threads. For synchronous SDK calls currently made directly from async functions, retain the existing behavior in this task and wrap it; a separate async-client refactor is outside scope.

- [ ] **Step 3: Make embedding ownership explicit**

Extend `RAGEmbeddingService` in `app/services/rag_embedding_service.py` with keyword-only `usage_context: UsageContext | None = None` on document, query, and image embedding methods, and thread the argument through `DocumentIndexService`. `Document` has no `user_id`; document indexing receives a context built only after loading `document.conversation` and verifying `conversation.user_id` in `index_document_task`. Query embedding inherits the bound authenticated chat context. Local sentence-transformer calls do not create provider usage events.

- [ ] **Step 4: Bind worker ownership before caption/embed stages**

In `index_document_task`, verify the conversation owner as it already does, build `UsageContext(user_id=owner_id, conversation_id=document.conversation_id, document_id=document.id, operation="document_index")`, then wrap captioning and indexing inside `bind_usage_context(context)`. Never trust an owner passed directly in a Celery payload.

- [ ] **Step 5: Run direct-call tests**

Run: `python -m pytest tests/test_model_usage_direct_sdk.py tests/test_model_usage_document_workers.py tests/test_rag_embedding_service.py tests/test_document_processing_service.py tests/test_document_index_service.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/ai/agents/chat_agent.py app/services/document_processing_service.py app/services/rag_embedding_service.py app/services/document_index_service.py app/workers/document_processor.py tests/test_model_usage_direct_sdk.py tests/test_model_usage_document_workers.py tests/test_rag_embedding_service.py
git commit -m "feat: track document and embedding usage"
```

### Task 11: Instrument Conversation Compaction and Remaining Server Model Calls

**Files:**
- Modify: `app/ai/conversation_compactor.py`
- Modify: `app/workers/conversation_compaction.py`
- Modify: `app/ai/mcp_servers/form_filler_server.py`
- Test: `tests/test_model_usage_compaction.py`
- Test: `tests/test_model_usage_callsite_inventory.py`
- Create: `tests/fixtures/model_usage_callsite_manifest.json`

- [ ] **Step 1: Write failing compaction attribution tests**

Verify owner/conversation attribution for background and emergency compaction, provider-reported usage, timeout/error attempts, and user-credential versus server-credential calls. Credential source is not exposed in analytics.

- [ ] **Step 2: Pass conversation identity to the compaction generator**

Extend `ConversationCompactor.compact()` and its generator call with verified `user_id` and `conversation_id`; bind operation `conversation_compaction` around `_langchain_generate()` and wrap the one provider attempt.

- [ ] **Step 3: Instrument form filler as unattributed when process context is unavailable**

Wrap its Gemini call with operation `form_fill`. If the MCP transport does not carry verified user identity, record `user_id = NULL`; do not add user-controlled identity fields to tool arguments.

- [ ] **Step 4: Pin the model-call inventory**

Create an AST-based source contract test over `app/` and `client_backend/` that recognizes provider terminal calls by attribute chain, including `generate_content`, `generate_content_stream`, `embed_content`, `ainvoke`, `invoke`, `astream`, `stream`, `agenerate`, `generate`, `responses.create`, `chat.completions.create`, `images.generate`, and `images.edit`. Also inventory `ChatGoogleGenerativeAI`, `ChatOpenAI`, `genai.Client`, `OpenAI`, and `AsyncOpenAI` constructors so new clients cannot silently re-enable internal retries. Compare discovered file/function/call-chain entries with a reviewed JSON manifest whose disposition is `instrumented`, `local_model`, `non_provider_tool_dispatch`, `workflow_stream`, or `startup_validation`; this explicitly excludes current `tool.ainvoke(...)` and `graph.astream(...)` sites without mistaking them for provider calls. Fail on any new, removed, or moved callsite until the manifest and instrumentation are reviewed together. Do not treat proximity to a wrapper as proof—tests for each `instrumented` entry must name the operation and exercise the wrapper.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_model_usage_compaction.py tests/test_model_usage_callsite_inventory.py tests/test_conversation_compactor.py tests/test_conversation_compaction_tasks.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/ai/conversation_compactor.py app/workers/conversation_compaction.py app/ai/mcp_servers/form_filler_server.py tests/test_model_usage_compaction.py tests/test_model_usage_callsite_inventory.py tests/fixtures/model_usage_callsite_manifest.json
git commit -m "feat: complete provider usage coverage"
```

### Task 12: Build User-Scoped Analytics Schemas and Service

**Files:**
- Create: `app/interfaces/model_usage_service_interface.py`
- Modify: `app/interfaces/__init__.py`
- Create: `app/schemas/model_usage.py`
- Modify: `app/schemas/__init__.py`
- Create: `app/services/model_usage_service.py`
- Test: `tests/test_model_usage_service.py`

- [ ] **Step 1: Write failing range, ownership, and timezone tests**

Cover default 30 days, exclusive `to`, two-year bound, 31-day hourly bound, invalid/reversed ranges, missing offsets, non-zero seconds, hour requests not aligned to local top-of-hour, day requests not aligned to local midnight, valid and invalid IANA zones, `Asia/Kathmandu` quarter-hour boundaries, Bangkok local days, New York DST spring/fall days, a synthetic sub-minute historical boundary rejected with 422, empty results, estimate coverage, and foreign conversation IDs.

- [ ] **Step 2: Define response models**

```python
class UsageModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

class UsageTotals(UsageModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    generated_images: int = 0
    request_count: int = 0

class UsageBreakdownItem(UsageModel):
    key: str
    totals: UsageTotals

class UsageSeriesPoint(UsageModel):
    start: datetime
    end: datetime
    totals: UsageTotals

class UsageCoverage(UsageModel):
    provider_reported_requests: int = 0
    mixed_requests: int = 0
    locally_estimated_requests: int = 0
    unavailable_requests: int = 0
    requests_with_known_total: int = 0
    total_requests: int = 0
    known_total_ratio: float = 0.0

class ConversationUsageItem(UsageModel):
    conversation_id: UUID
    title: str | None = None
    totals: UsageTotals

class UsageRange(UsageModel):
    from_: datetime = Field(alias="from")
    to: datetime
    bucket: Literal["hour", "day"]
    timezone: str

class UsageDashboard(UsageModel):
    totals: UsageTotals
    outcomes: list[UsageBreakdownItem]
    series: list[UsageSeriesPoint]
    by_provider: list[UsageBreakdownItem]
    by_model: list[UsageBreakdownItem]
    by_operation: list[UsageBreakdownItem]
    by_agent: list[UsageBreakdownItem]
    top_conversations: list[ConversationUsageItem]
    coverage: UsageCoverage
    range: UsageRange
    generated_at: datetime
```

Define the conversation response with `totals`, provider/model breakdowns, `coverage`, `latest_context_window`, `range`, and `generated_at` using these same types. `total_tokens` is the sum of known event totals only; it is never synthesized from zero for unknown events. `known_total_ratio = requests_with_known_total / total_requests`, or `0.0` for an empty result. The four source counts are mutually exclusive and sum to `total_requests`. All schemas use `to_camel_case`, matching existing schema modules.

- [ ] **Step 3: Implement validation and service assembly**

Use `ZoneInfo(timezone_name)`. Interpret `from` and `to` as aware instants, convert them to the requested zone, validate local bucket alignment, resolve each local boundary back to UTC, and reject nonexistent/ambiguous user-supplied boundaries unless their numeric offset selects one unambiguously. Validate that every UTC boundary is minute-aligned. Query half-open UTC-minute rows, then regroup each minute by its local hour/day. Fill missing series buckets with zeroes, bound breakdowns to 20 rows, and return deterministic ordering (`total_tokens DESC`, then key ASC).

- [ ] **Step 4: Run service tests**

Run: `python -m pytest tests/test_model_usage_service.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/interfaces app/schemas/model_usage.py app/schemas/__init__.py app/services/model_usage_service.py tests/test_model_usage_service.py
git commit -m "feat: aggregate user usage analytics"
```

### Task 13: Expose Authenticated Usage APIs

**Files:**
- Create: `app/api/model_usage.py`
- Modify: `app/api/__init__.py`
- Modify: `app/main.py`
- Test: `tests/test_model_usage_api.py`

- [ ] **Step 1: Write failing API isolation tests**

Test missing/invalid JWT, user A seeing only A, user B receiving 404 for A's conversation, no accepted `userId` filter, query validation errors, empty success envelopes, both endpoint response shapes, and disabled UI/analytics flag behavior.

- [ ] **Step 2: Implement routes using injected identity**

```python
router = APIRouter(prefix="/usage", tags=["usage"])

@router.get("/dashboard", response_model=ApiResponse[UsageDashboard])
@AppAutoInjector.auto_inject()
def get_usage_dashboard(
    usage_service: IModelUsageService,
    user_id: UUID,
    request: UsageDashboardQuery = Depends(),
) -> ApiResponse[UsageDashboard]:
    data = usage_service.get_dashboard(user_id=user_id, query=request)
    return ApiResponse(success=True, message="Usage retrieved successfully", data=data)
```

`AppAutoInjector` injects only parameters without defaults, so both the service interface and `user_id` must remain required in the Python signature. Implement the conversation route with the same injected identity. Use existing domain exceptions for invalid range (422) and foreign/not-found conversation (404 without ownership disclosure).

- [ ] **Step 3: Register the router**

Export `model_usage_router`, wire its module, and include it exactly once without adding `/ai` aliases; the sidecar will proxy the canonical `/usage` paths.

- [ ] **Step 4: Run API tests**

Run: `python -m pytest tests/test_model_usage_api.py tests/test_exception_handler.py tests/test_private_network_cors.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/api/model_usage.py app/api/__init__.py app/main.py tests/test_model_usage_api.py
git commit -m "feat: expose user usage endpoints"
```

### Task 14: Proxy Usage APIs Through the Sidecar

**Files:**
- Modify: `client_backend/api/proxy.py`
- Test: `tests/client_backend/test_usage_proxy.py`

- [ ] **Step 1: Write failing proxy tests**

Verify query strings, bearer authentication forwarding, status/body preservation, `/usage/dashboard`, `/usage/conversations/{conversation_id}`, both plain and `/api` sidecar mounts, and no user-ID rewriting.

- [ ] **Step 2: Add explicit proxy routes**

```python
@router.get("/usage/dashboard")
async def proxy_usage_dashboard(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
):
    return await proxy_server_request(request, upstream_path="/usage/dashboard")

@router.get("/usage/conversations/{conversation_id}")
async def proxy_conversation_usage(
    conversation_id: UUID,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
):
    return await proxy_server_request(
        request, upstream_path=f"/usage/conversations/{conversation_id}"
    )
```

Reuse the existing imports from `client_backend.core.auth` and `client_backend.core.security`; do not introduce a `require_local_auth` alias.

- [ ] **Step 3: Run sidecar tests**

Run: `python -m pytest tests/client_backend/test_usage_proxy.py tests/client_backend/test_cors.py tests/client_backend/test_auth.py -q`

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add client_backend/api/proxy.py tests/client_backend/test_usage_proxy.py
git commit -m "feat: proxy usage analytics through sidecar"
```

### Task 15: Update AI SDK Metadata and Write the FE Contract

**Files:**
- Modify: `app/api/ai_sdk.py`
- Modify: `app/services/event_streaming/ai_sdk_projection.py`
- Create: `plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md`
- Test: `tests/test_model_usage_ai_sdk_contract.py`
- Test: `tests/test_ai_sdk_v6_stream_contract.py`

- [ ] **Step 1: Write failing compatibility tests**

Assert no new mandatory stream event, unchanged terminal framing, final/history metadata containing the additive context split, unknown fields remaining ignorable, and contract examples validating against Pydantic response schemas.

- [ ] **Step 2: Preserve additive metadata in final and history paths**

Ensure metadata scrubbing does not drop `context_window.input_tokens`, `output_tokens`, `total_tokens`, `usage_source`, `limit_type`, `input_usage_ratio`, `output_usage_ratio`, `usage_ratio`, `usage_ratio_basis`, or model-limit fields. Do not duplicate account-wide dashboard data into message streams.

- [ ] **Step 3: Write the concise frontend contract**

The contract must include:

1. Both usage endpoints, JWT requirement, query limits, inclusive/exclusive range semantics, and error statuses.
2. Exact JSON examples and TypeScript types for totals, series, breakdowns, coverage, conversation summary, and context metadata.
3. Chart mapping: stacked input/output area or bar series, outcome rate, ranked provider/model/operation/agent lists, and top conversations.
4. Refresh behavior: fetch dashboard on entry/filter change; refetch conversation summary after AI SDK `finish`; do not poll during generation.
5. Context gauge formula and unknown-limit behavior.
6. Empty/loading/error states and the rule that all unknown future fields are ignored.

- [ ] **Step 4: Run contract tests**

Run: `python -m pytest tests/test_model_usage_ai_sdk_contract.py tests/test_ai_sdk_v6_stream_contract.py tests/test_ai_sdk_assistant_ui_compat.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/api/ai_sdk.py app/services/event_streaming/ai_sdk_projection.py plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md tests/test_model_usage_ai_sdk_contract.py tests/test_ai_sdk_v6_stream_contract.py
git commit -m "docs: define frontend usage analytics contract"
```

### Task 16: Add the Streamlit Usage Dashboard and Conversation Panel

**Files:**
- Modify: `demo.py`
- Test: `tests/test_demo_usage_dashboard.py`
- Test: `tests/test_context_window_message_metadata.py`

- [ ] **Step 1: Write failing UI helper tests**

Test API query construction, response normalization, compact token formatting, shared-context and separate-I/O gauge tooltips, 100% visual cap with uncapped backend ratio, unknown denominator, reported/estimated badge text, empty data, chart-frame construction, and conversation-panel refresh after completion.

- [ ] **Step 2: Add authenticated API helpers**

```python
def get_usage_dashboard(*, start, end, bucket, timezone_name, conversation_id=None):
    zone = ZoneInfo(timezone_name)
    start_at = align_usage_boundary(start, bucket=bucket, zone=zone)
    end_at = align_usage_boundary(end, bucket=bucket, zone=zone)
    params = {
        "from": start_at.isoformat(),
        "to": end_at.isoformat(),
        "bucket": bucket,
        "timezone": timezone_name,
    }
    if conversation_id:
        params["conversationId"] = conversation_id
    endpoint = f"/usage/dashboard?{urlencode(params)}"
    return make_api_request("GET", endpoint, use_cache=True)
```

Import `urlencode` from `urllib.parse` and `ZoneInfo` from `zoneinfo`, and use the existing authenticated `make_api_request()` response-envelope helper. `align_usage_boundary()` converts day-picker dates to local midnight and hour inputs to local top-of-hour; the UI's inclusive end date becomes the next local midnight because the API `to` value is exclusive. Increase the GET cache TTL to 30 seconds only for usage helpers through a dedicated `_cached_usage_get_request()` so chat/history caching behavior remains unchanged; clear that cache after a completed turn.

- [ ] **Step 3: Implement `render_usage_view()`**

Add a **Usage** tab after Models. Render date range, hour/day selector, timezone selector defaulting to the local zone when discoverable, optional conversation filter, total/input/output/request/image cards, stacked input/output trend, outcome chart, breakdown tables/charts, top conversations, coverage caption, and generated-at timestamp.

Use Altair only through Streamlit's installed dependency. Bound chart rows to API output; do not perform unbounded client-side history fetches.

- [ ] **Step 4: Add the compact conversation panel**

In `render_chat_view()`, fetch `/usage/conversations/{id}` without a range for retained two-year cumulative input/output/total, requests, models, and latest gauge. Endpoint failure must not block chat rendering.

- [ ] **Step 5: Correct the existing circle tooltip and CSS**

For shared context format `2.2k / 65.5k (3%) · input 120 · output 2.0k · provider reported`. For separate I/O format `6% limiting · input 120 / 65.5k (0.2%) · output 2.0k / 32.8k (6%) · provider reported`. Keep `ok/warn/danger/unknown` states, visually cap fill at 100%, and preserve accessible `title`/`aria-label` text.

- [ ] **Step 6: Run UI tests**

Run: `python -m pytest tests/test_demo_usage_dashboard.py tests/test_context_window_message_metadata.py tests/test_demo_stream_rendering.py tests/test_ai_sdk_context_window.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add demo.py tests/test_demo_usage_dashboard.py tests/test_context_window_message_metadata.py
git commit -m "feat: add Streamlit usage analytics"
```

### Task 17: Add Reconciliation, Retention, and Health Surfaces

**Files:**
- Modify: `app/workers/model_usage.py`
- Modify: `app/workers/celery_app.py`
- Modify: `app/workers/cleanup_tasks.py`
- Modify: `app/api/health.py`
- Modify: `app/observability/model_usage.py`
- Test: `tests/test_model_usage_retention.py`
- Test: `tests/test_model_usage_health.py`

- [ ] **Step 1: Write failing maintenance tests**

Test reconciliation of the most recent 2,880 complete UTC minutes, exclusion of the active partial minute until safely recomputed, idempotent rebuilds, 5,000-row cleanup batches, 90/730-day cutoffs, both existing schedule dictionaries surviving module import order, and health classification from recorder failures/unattributed rate/rollup lag.

- [ ] **Step 2: Implement periodic tasks**

```python
celery_app.conf.beat_schedule.update({
    "reconcile-model-usage": {
        "task": "app.workers.model_usage.reconcile_model_usage_task",
        "schedule": crontab(minute=15),
        "options": {"queue": "summary"},
    },
    "cleanup-model-usage": {
        "task": "app.workers.model_usage.cleanup_model_usage_task",
        "schedule": crontab(minute=40, hour=3),
        "options": {"queue": "summary"},
    },
})
```

Cleanup loops in bounded batches and stops when a batch deletes fewer rows than the configured size.

Define named schedule dictionaries in both `celery_app.py` and `cleanup_tasks.py` and apply them with `celery_app.conf.beat_schedule.update(...)`; neither module may assign a replacement dictionary. The worker-config test imports the modules in both orders and asserts the union of pre-existing conversation jobs, cleanup jobs, reconciliation, and usage retention jobs.

- [ ] **Step 3: Add internal health and metrics endpoints**

Add `/health/model-usage` and `/metrics/model-usage`, following the existing conversation-compaction pattern. Health responses expose aggregate counts/lag only and no tenant identifiers.

- [ ] **Step 4: Run maintenance tests**

Run: `python -m pytest tests/test_model_usage_retention.py tests/test_model_usage_health.py tests/test_celery_worker_config.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/workers/model_usage.py app/workers/celery_app.py app/workers/cleanup_tasks.py app/api/health.py app/observability/model_usage.py tests/test_model_usage_retention.py tests/test_model_usage_health.py tests/test_celery_worker_config.py
git commit -m "feat: operate model usage retention"
```

### Task 18: Document Operations and Run Full Verification

**Files:**
- Modify: `README.md`
- Create: `docs/operations/model-usage-analytics.md`
- Modify: `Chatbot API.postman_collection.json`
- Test: `tests/test_model_usage_docs.py`

- [ ] **Step 1: Write a failing documentation contract test**

Assert README configuration names, endpoint paths, retention semantics, no-cost statement, Postman requests, operational runbook sections, and the frontend contract link.

- [ ] **Step 2: Write operational documentation**

Document migration/rollback order, deployment flags, start-clean behavior, retention schedules, health/metrics interpretation, unattributed-call investigation, replay-safe failed writes, query limits, data classification, user deletion behavior, and rollback with collection disabled before schema downgrade.

- [ ] **Step 3: Add Postman requests**

Add dashboard and conversation-usage requests using existing bearer-token and conversation variables. Include a Bangkok daily example and an hourly 7-day example.

- [ ] **Step 4: Run focused documentation tests**

Run: `python -m pytest tests/test_model_usage_docs.py -q`

Expected: all tests pass.

- [ ] **Step 5: Run formatting and static checks**

Run: `python -m ruff check app client_backend tests demo.py`

Expected: exit code 0.

Run: `python -m ruff format --check app client_backend tests demo.py`

Expected: exit code 0.

- [ ] **Step 6: Run the complete test suite**

Run: `python -m pytest -q`

Expected: exit code 0 with no failures.

- [ ] **Step 7: Run migration smoke checks in a PostgreSQL test environment**

Run: `python -m alembic upgrade head`

Expected: migration `y2z3a4b5c6d7` applies successfully.

Run: `python -m alembic downgrade x1y2z3a4b5c6`

Expected: usage rollup and event tables are removed without affecting existing tables.

Run: `python -m alembic upgrade head`

Expected: migration reapplies successfully.

- [ ] **Step 8: Perform manual smoke verification**

1. Start PostgreSQL, Redis, Qdrant, the FastAPI server, Celery workers/beat, sidecar, and Streamlit.
2. Sign in as user A; send a text request with a tool loop, create an image, and upload/index a document.
3. Confirm one row per provider attempt, including router/helper/image/embedding calls and correct retry/fallback linkage.
4. Confirm the image turn shows Gemini's single gauge based on the more constrained 65,536-input/32,768-output ratio with both values in the tooltip; an OpenAI GPT Image turn shows token totals with an unknown denominator.
5. Confirm `/usage/dashboard` and the Streamlit Usage tab agree on totals and filters.
6. Sign in as user B and confirm none of user A's totals, conversations, or breakdowns appear.
7. Disable LangSmith and confirm analytics remain functional.
8. Disable `MODEL_USAGE_UI_ENABLED` and confirm collection continues while usage routes/UI are unavailable.
9. Simulate repository failure and confirm chat succeeds, a normalized retry is queued, and replay does not double-count.

- [ ] **Step 9: Commit**

```bash
git add README.md docs/operations/model-usage-analytics.md Chatbot\ API.postman_collection.json tests/test_model_usage_docs.py
git commit -m "docs: operate per-user model usage analytics"
```

---

## 7. Acceptance Criteria

- Every application-controlled provider attempt in the reviewed callsite manifest creates exactly one raw event or one observable ledger-write retry-exhaustion metric; supported clients have SDK-internal retries disabled.
- User rows always carry verified ownership; unattributed rows never appear in authenticated analytics.
- A provider response with reported input/output preserves both values and does not replace them with estimates.
- Retries, fallbacks, timeouts, errors, and cancellations are individually visible without double counting.
- Gemini and OpenAI image usage survives streaming; requested image count does not truncate terminal usage metadata.
- `gemini-3-pro-image` displays one gauge based on its documented separate 65,536-input and 32,768-output limits; GPT Image usage is counted but its denominator remains unknown until officially published.
- The single gauge uses shared-total or most-constrained separate-I/O semantics according to `limit_type`; the API returns the split, applicable ratios, source, basis, and uncapped limiting ratio.
- Within the raw-retention window, dashboard totals from minute rollups equal an independent sum of matching raw events; older ranges use minute rollups alone and never double count raw plus aggregate rows.
- Daily charts are correct in `Asia/Bangkok` and across DST transitions in `America/New_York`.
- Raw events older than 90 days and rollups older than 730 days are removed in bounded, observable jobs.
- `/usage/*` endpoints never accept arbitrary user identity and pass cross-user isolation tests.
- Existing `/api/chat/{conversationId}` and `/ai/chat/{conversationId}` clients continue to work unchanged.
- Streamlit renders the account dashboard, compact conversation panel, corrected gauge, and graceful empty/error states.
- `plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md` is sufficient for the frontend team to implement the same UI without reading backend code.
- Full pytest, Ruff, and migration smoke checks pass.

## 8. Rollout and Rollback

1. Deploy schema and code with `MODEL_USAGE_TRACKING_ENABLED=false` and `MODEL_USAGE_UI_ENABLED=false`.
2. Enable collection in staging, verify unattributed-call rate and callsite inventory for text, tool, image, document, and compaction paths.
3. Enable the usage API/UI in staging and compare raw-event sums with dashboard totals.
4. Enable collection in production while keeping UI disabled for one observation window.
5. Enable the user UI after recorder failures, rollup lag, and unattributed rate remain within operational thresholds established from staging.
6. To roll back, disable UI first, then collection, allow queued normalized writes to drain, deploy the previous code, and only then downgrade the migration if data removal is intended.

No historical backfill runs at any stage.

---

## 9. Progress Log and Design Decisions

Implementation progress and decisions made during execution. Updated after each task.

### Execution decisions (pre-flight, 2026-07-20)

- **D1:** Implementer subagents edit and self-test only; the controller session verifies, reviews, and commits (user's global delegation rules override the skill's default of subagent commits).
- **D2:** The local global-guard hook blocks both reading and writing `.env.example`. Tasks 5 and 8 record their exact required `.env.example` additions in this section for manual application; commits exclude `.env.example`.
- **D3:** PostgreSQL integration tests run against a dedicated `chatbot_test` database (PostgreSQL 18.3, created 2026-07-20) via `.superpowers/sdd/run_pg_tests.py`, which derives `TEST_DATABASE_URL` from app settings in-process so credentials are never printed or persisted.
- **D4:** Verified the live alembic head is `x1y2z3a4b5c6`, matching the plan's expected `down_revision` for migration `y2z3a4b5c6d7`.
- **D5:** Test commands run as `.venv/Scripts/python.exe -m pytest ...` from Git Bash (plan's `python -m pytest` assumes an activated venv).

### Task progress

- **Task 1 — complete (2026-07-20).** Commits `3659e07` + `3f83da1`. 21 tests green (`-W error`), ruff clean. Review round 1 raised two Important findings; both fixed and re-review Approved.
  - Decision: `generated_images` is validated as a required non-negative int (`None` rejected with `TypeError`), unlike the nine genuinely-Optional token fields where `None` means "unknown".
  - Decision: CPython's GIL makes a pair-uniqueness stress test (32 threads × 200 allocations, tiny switch interval) unable to detect a lockless `allocate_attempt`; mutual exclusion is instead proven deterministically by `test_allocate_attempt_serializes_concurrent_callers`, which holds `operation._lock` and asserts the allocator blocks. Break-the-code verified (test fails immediately without the lock).

- **Task 2 — complete (2026-07-20).** Commits `1d4dc13` + `2ac5ca5`. 24 tests green against live PostgreSQL, dev-DB schema-contract tests green, ruff clean. Review round 1 Needs-fixes; re-review Approved.
  - Decision: `rollup_key` is the primary key of `model_usage_minute` (no surrogate id) — matches the contract's "keyed by rollup_key" and the existing business-key-PK pattern (`ConversationMemorySummary`).
  - Decision: non-negative CHECK constraints cover the minute table's sums/known-counts too, not just the events table.
  - Decision: `model_usage_minute` gained a conventional `created_at` column beyond §4.2's explicit list.
  - Defect caught by controller verification: PostgreSQL's 63-char identifier limit broke `alembic upgrade head` (constraint name was 66 chars). Minute-table check names now use `ck_mu_minute_<col>_nonneg`; a guard test asserts every constraint/index is named and ≤63 chars, and a live-PostgreSQL test runs the migration's real `upgrade()`/`downgrade()` in an isolated scratch schema.
  - Review fix: ORM now names PK/FK constraints identically to the migration (`pk_/fk_` names), enforced by a parity test; migration ondelete checks are column-bound (events: user→CASCADE, others→SET NULL; minute: both→CASCADE).
  - Migration `y2z3a4b5c6d7` was applied to the dev database on 2026-07-20 (schema-first rollout per §8; collection stays disabled until Task 5's flags exist).
