# Model-usage analytics operations

This runbook covers the PostgreSQL model-call ledger, UTC-minute rollups,
authenticated usage APIs, background maintenance, and aggregate operational
surfaces. The ledger is authoritative for usage counts. LangSmith and provider
dashboards can help investigate calls, but neither is an analytics dependency.

The complete list of all 17 `MODEL_USAGE_*` settings, their defaults, ranges,
and cross-field validation rules lives in the
[README configuration reference](../../README.md#model-usage-analytics). Keep
that reference and `.env.example` aligned instead of creating a second settings
table here.

## Schema-first deployment

The current Alembic chain is:

```text
x1y2z3a4b5c6
  -> y2z3a4b5c6d7  model_usage_events and model_usage_minute
  -> z3a4b5c6d7e8  device-scoped HITL policy
  -> a4b5c6d7e8f9  model-usage timestamp indexes
  -> b5c6d7e8f9a0  tool-approval schema repair (current head)
```

Deploy schema before code, then enable collection before presentation:

For a database created by the original `6c6598a9eb26`, the current head also
normalizes tool-approval enum labels. Drain old API/worker approval writers and
apply `b5c6d7e8f9a0` together with code that persists `DecisionType.value`; do
not leave an older uppercase-mapped process running during this repair. The
ordinary schema-first sequence below applies once that compatibility boundary
has been handled.

1. Back up PostgreSQL, confirm Redis and the `summary` worker are reachable,
   validate the 17 settings, and begin with both feature flags disabled.
2. Run `.venv\Scripts\python.exe -m alembic upgrade head`. Migration
   `y2z3a4b5c6d7` is additive and creates empty tables. No historical backfill
   is performed: analytics start clean with provider attempts recorded after
   tracking is enabled.
3. Deploy compatible API, worker, beat, sidecar, and UI code while
   `MODEL_USAGE_TRACKING_ENABLED=false` and `MODEL_USAGE_UI_ENABLED=false`.
4. Start the `summary` worker and Celery beat. Enable
   `MODEL_USAGE_TRACKING_ENABLED=true` across API and worker processes, confirm
   new ledger rows and rollups, then enable `MODEL_USAGE_UI_ENABLED=true`.
5. Verify the capability, dashboard, health, and metrics surfaces and compare a
   bounded sample of provider attempts with ledger rows.

The two flags deliberately control different behavior:

- `MODEL_USAGE_TRACKING_ENABLED` controls event collection, failed-write
  replay, reconciliation, and cleanup. Turning it off stops those behaviors.
- `MODEL_USAGE_UI_ENABLED` controls presentation. When false, dashboard and
  conversation usage routes return `404` and the Streamlit usage views stay
  hidden. The authenticated capability route still returns `enabled: false`.
  Tracking can and normally should remain enabled while presentation is off.

Changing either flag requires restarting every process that loads settings.
Avoid expecting old calls to appear after a rollout: there is no historical
backfill from messages, traces, or provider billing data.

## Workers, schedules, and retention

Run a worker that consumes the `summary` queue and run one beat scheduler:

```powershell
.venv\Scripts\celery.exe -A app.workers.celery_app:celery_app worker -Q summary
.venv\Scripts\celery.exe -A app.workers.celery_app:celery_app beat
```

Celery uses UTC. Reconciliation runs hourly at minute `:15`; cleanup runs daily
at `03:40`. Failed-write retry, reconciliation, and cleanup all use the
`summary` queue. Monitor its depth and oldest task age so compaction work cannot
silently starve usage maintenance.

Reconciliation reads only complete UTC minutes: the current partial minute is
excluded. It rebuilds the trailing `MODEL_USAGE_RECONCILE_MINUTES` from raw
events exactly, in transactions no larger than
`MODEL_USAGE_RECONCILE_CHUNK_MINUTES`. Writers take a shared per-minute
PostgreSQL advisory lock; the reconciler takes the corresponding exclusive
locks in timestamp order before atomically replacing each chunk. This makes
overlap safe without allowing a write to disappear between the raw scan and
rollup replacement. Do not disable or bypass the advisory lock semantics in an
operational repair script.

Default cleanup retains 90 days of raw events and 730 days of minute rollups,
deleting at most 5,000 rows per committed batch. The reconciliation window must
remain strictly shorter than raw retention, calculated in minutes; otherwise a
gap could no longer be repaired from authoritative raw events. Rollup retention
must remain at least raw retention. Cleanup boundaries are age-based UTC
instants, while API buckets are rendered in the requested local timezone.

## Authenticated API and sidecar routes

Every usage route is authenticated and derives the user from the session or
bearer token. A client must never send a user ID.

| Surface | Routes | Behavior |
|---|---|---|
| Canonical API | `/usage/capabilities`, `/usage/dashboard`, `/usage/conversations/{conversation_id}` | Bearer-authenticated. Conversation reads require ownership. |
| Sidecar compatibility routes | `/usage/capabilities`, `/usage/dashboard`, `/usage/conversations/{conversation_id}` | Require the local sidecar session and proxy the canonical route. |
| Sidecar `/api` aliases | `/api/usage/capabilities`, `/api/usage/dashboard`, `/api/usage/conversations/{conversation_id}` | Same session and upstream behavior as the plain sidecar routes. |

The capability route is authenticated even when the UI feature is disabled.
It reports only whether analytics presentation is enabled. Dashboard and
conversation responses contain user-scoped analytics and must not be cached
across authenticated identities.

The precise camelCase response, error, message-metadata, rendering, and refresh
rules for an AI SDK frontend are in
[`plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md`](../../plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md).
That contract is additive: do not invent a streaming usage event or copy
dashboard aggregates into message metadata.

## Query boundaries, timezones, and DST

`from` and `to` are optional as a pair. When supplied, `from` is inclusive and
`to` is exclusive; both must be RFC 3339 timestamps with a numeric UTC offset,
zero seconds/microseconds, and an offset valid for the requested IANA timezone.
`bucket` is `day` or `hour`, and `timezone` defaults to `UTC`.

- The dashboard defaults to 30 local calendar days. A conversation defaults to
  730 days. Any explicit range is limited to 730 days.
- Hourly ranges are limited to 31 days and must start and end at a local top of
  hour. Daily ranges must use local-midnight boundaries.
- The service resolves local boundaries through the IANA timezone. Around DST
  gaps and folds, the numeric offset must identify a real, unambiguous instant.
  Do not create local buckets by adding a fixed 24 hours: a local day can be 23,
  24, or 25 hours, and some zones have non-hour transitions.
- Use the public `conversationId` query alias on the dashboard. The conversation
  detail route uses the path ID. Never send `conversation_id`, `userId`, or
  `user_id` as analytics query parameters.

The Postman collection includes a Bangkok daily range and a seven-day Bangkok
hourly range. Adjust the fixed example dates while preserving alignment and
the encoded `+07:00` offset.

## Health and metrics interpretation

`GET /health/model-usage` refreshes a database-backed snapshot over configured
complete UTC minutes. `GET /metrics/model-usage` refreshes the same snapshot
and renders Prometheus text. Both are aggregate-only operational surfaces:
they contain no tenant, user, conversation, model, request, trace, prompt, or
response dimensions. Do not use them to reconstruct dashboard analytics.

Interpret the health payload and matching gauges together:

- `raw_event_count` and `rollup_request_count` must match. Any nonzero
  `rollup_gap_count` is unhealthy and means reconciliation or write atomicity
  needs investigation.
- `unattributed_rate` is the fraction of durable attempts without a verified
  owner. A value above
  `MODEL_USAGE_HEALTH_UNATTRIBUTED_DEGRADED_RATIO` degrades health. Some
  explicit maintenance calls can be unattributed; a sustained increase in
  user-facing traffic is not expected.
- `rollup_lag_minutes` compares the latest raw-event and rollup minute. Values
  above the degraded or unhealthy thresholds indicate the summary queue or
  reconciler is late.
- Recent persistence failures are recorded in a content-free shared Redis
  failure store. `persistence_failure_store_available=false` means the
  deployment-wide count is unavailable, not zero, and health is degraded.
  `process_persistence_failure_count` is only the serving process's bounded
  fallback signal.
- `model_usage_health_snapshot_timestamp_seconds` is the freshness marker for
  the last successful durable refresh. Alert when its age exceeds the health
  scrape objective. On refresh failure `/health/model-usage` returns `503`;
  `/metrics/model-usage` marks status unhealthy but deliberately leaves the
  timestamp unchanged so stale data cannot look fresh.

### Investigation

1. Confirm PostgreSQL, Redis, Celery beat, and at least one `summary` worker are
   healthy. Check queue depth, oldest task age, worker restarts, and the most
   recent `reconcile-model-usage` execution.
2. If the Redis store is unavailable, check the configured Redis URL,
   credentials, network path, timeout, TTL, and key expiry. Restore availability
   before treating `persistence_failure_count` as deployment-wide.
3. For a raw/rollup gap or rollup lag, compare only complete UTC minutes within
   raw retention. Run the normal reconciliation task and verify the gap closes;
   its chunk transactions and advisory locks make replay safe.
4. For persistence failures, inspect aggregate persistence metrics and
   content-free worker logs for `failure_class`, retry attempt, and `event_key`.
   Check the failed-write queue before retry limits expire. Never add provider
   payloads, prompt text, response text, or tenant IDs to metrics labels.
5. For a high unattributed rate, group a protected database sample by operation,
   provider, model, source, and call path. Verify user context propagation in
   auxiliary, document, and maintenance workers. Do not expose row-level
   identifiers through the health endpoint.
6. Re-run health after at least one complete minute and confirm the freshness
   timestamp, gap, lag, failure-store availability, and unattributed rate all
   recover.

## Failed writes and replay safety

A provider response is not failed merely because analytics persistence fails.
The recorder queues a content-free serialized command for retry with exponential
backoff (`MODEL_USAGE_RETRY_BASE_SECONDS`) up to
`MODEL_USAGE_RETRY_MAX_ATTEMPTS`. This retry repeats the ledger write, never the
provider call.

Each normalized provider attempt has a stable `operation_id` and attempt number.
Its unique `event_key` is `operation_id:attempt`. Both inline persistence and a
later replay use that key; conflict handling is idempotent, so duplicate Celery
delivery cannot increment the minute rollup twice. Keep the same event key when
manually re-enqueuing a captured command. Creating a new key converts a replay
into a new event and will double-count usage.

Before a rollback or maintenance stop, disable collection on API producers,
keep a compatible tracking-enabled `summary` worker alive long enough to drain
failed-write retries, and only then disable tracking on that worker. Preserve
content-free retry payloads until they are stored or have reached the documented
terminal retry policy.

## Data classification and deletion

Treat model-usage data as internal, user-scoped telemetry. The ledger stores
user-scoped identifiers, provider and model names, operation/agent/status/source
dimensions, token and image usage, latency, timestamps, bounded error codes, and
provider or trace correlation IDs. Correlated tracing may also handle keyed user
hashes. Access to rows, hashes, and authenticated aggregates must follow the same
controls as conversation metadata.

No prompt or response content, tool arguments, raw provider payloads, API keys,
or provider credentials are stored in the ledger, rollups, Redis failure keys,
retry payloads, health output, or metric labels. This boundary is mandatory;
do not weaken it for debugging.

### Queued failed-write lifecycle

“Content-free” does not mean anonymous. A failed-write payload can include user,
conversation, message, and document IDs when the call was attributable. It also
contains the operation ID and attempt, correlation and LangSmith run IDs,
provider/model/status/timing dimensions, a bounded error code, and normalized
usage. Classify the broker payload as user-linked telemetry even though it has no
prompt, response, tool argument, or raw provider payload.

The initial Celery `.delay(payload)` call and the retry task set no
analytics-specific expiry or TTL. The Redis failure-store TTL controls only
aggregate health counters; it does not expire queued commands. Likewise, the
broker visibility timeout is not a privacy TTL: it governs redelivery of an
unacknowledged task. A payload remains subject to the broker's configured
lifecycle until a worker acknowledges it or an operator purges it. On failure,
the worker schedules exponential countdown retries up to
`MODEL_USAGE_RETRY_MAX_ATTEMPTS`; exhaustion records a dropped outcome and the
task fails. The application configures no dedicated dead-letter queue. If the
deployment adds a DLQ or failed-task archive, apply the same retention and
deletion controls there.

Deleting a user cascades to both raw `model_usage_events` and aggregated
`model_usage_minute` rows through PostgreSQL foreign keys. Conversation deletion
removes conversation rollups and clears the nullable conversation reference on
raw events while the user exists.

Before account deletion, block new authenticated work for the user, then drain or
purge that user's queued, scheduled, reserved, and deployment-DLQ failed-write
payloads. Draining requires a tracking-enabled, schema-compatible `summary`
worker; a selective purge must use broker tooling that preserves other users'
tasks. After the database delete, any racing replay that still carries deleted
foreign keys is rejected as `ModelUsageReferenceError` and retried until the
configured limit. PostgreSQL reference validation means it cannot recreate the
user or tenant data, but the rejected payload still retains identifiers until it
is acknowledged, exhausted, or purged.

LangSmith is an external data store. The application database cascade does not
erase external traces. Honor the configured LangSmith retention/deletion policy
through the provider project/API for every project used by the deployment. Build
the authorized deletion inventory before the database cascade, then locate and
verify traces by keyed user hash or correlation metadata without putting raw IDs,
hashes, or correlation values into tickets, logs, dashboards, or metrics.

### Account-deletion verification

- **PostgreSQL:** verify `model_usage_events` and `model_usage_minute` have no
  rows for the deleted user, and verify the user row itself is absent.
- **Broker:** inspect queued, scheduled, and reserved tasks plus the broker and
  any deployment-managed DLQ or failed-task archive. Confirm no failed-write
  payload remains for the deletion inventory; do not copy identifiers into the
  audit record.
- **LangSmith:** in every configured project, verify the provider project/API
  retention or deletion result using the ephemeral keyed hash/correlation
  inventory. Record only the outcome and policy reference, not tenant-linked
  values.

## Rollback

Rollback is feature-flag first. A normal rollback leaves the additive usage
tables and timestamp indexes in place:

1. Set `MODEL_USAGE_UI_ENABLED=false` everywhere and verify dashboard and
   conversation routes are hidden.
2. Set `MODEL_USAGE_TRACKING_ENABLED=false` and restart API producers to disable
   new collection on API producers.
3. Keep a schema-compatible, tracking-enabled current-release `summary` worker
   running while it can drain failed-write retries. Then disable tracking on that
   worker and stop it.
4. The normal rollback path performs no schema action. If an optional index-only
   downgrade is required, perform it now, using the current release's migration
   artifacts before any old code is deployed:

   ```powershell
   .venv\Scripts\python.exe -m alembic downgrade z3a4b5c6d7e8
   ```

   From the current head, this first applies migration `b5c6d7e8f9a0`'s safe
   no-op downgrade, preserving repaired tool-approval schema and data, and then
   removes only migration `a4b5c6d7e8f9`'s timestamp indexes. It does not remove
   the usage ledger. Skip this step for the normal path.
5. Deploy the previous application, worker, sidecar, and UI release only after
   the optional current-artifact migration step has completed or been skipped.
   Keep presentation and tracking disabled until compatibility is verified.
6. Test the table migration only in an isolated scratch database: bring the
   scratch schema to `x1y2z3a4b5c6`, upgrade to `y2z3a4b5c6d7`, verify empty
   usage tables, then downgrade to `x1y2z3a4b5c6` and discard the database.

Do not blindly downgrade production from head to `x1y2z3a4b5c6`: that also
reverts the later device-scoped schema. If production must remove usage tables
after later migrations have shipped, plan coordinated maintenance and a new
forward migration that removes or decouples the schema safely. Take a backup,
stop producers, drain retries, and verify no supported code still references the
tables before any destructive schema change.

## No monetary cost tracking

The system records provider attempts, token/image quantities, coverage, and
context-window metadata. It performs no price lookup, currency conversion,
invoice reconciliation, or monetary aggregation. No monetary cost tracking is
provided, and token totals must not be presented as billed spend. Use the
provider's billing system for monetary cost analysis.
