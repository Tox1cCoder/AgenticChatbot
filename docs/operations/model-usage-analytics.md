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
  -> a4b5c6d7e8f9  model-usage timestamp indexes (current head)
```

Deploy schema before code, then enable collection before presentation:

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

Deleting a user cascades to both raw `model_usage_events` and aggregated
`model_usage_minute` rows through PostgreSQL foreign keys. Conversation deletion
removes conversation rollups and clears the nullable conversation reference on
raw events while the user exists. Include both analytics tables when verifying
account deletion and retention audits.

## Rollback

Rollback ordering is the reverse of enablement:

1. Set `MODEL_USAGE_UI_ENABLED=false` everywhere and verify dashboard and
   conversation routes are hidden.
2. Set `MODEL_USAGE_TRACKING_ENABLED=false` on API producers and restart them to
   disable new collection. Keep a schema-compatible, tracking-enabled `summary`
   worker running while it drains failed-write retries; then disable tracking on
   that worker. Finish this sequence before rolling back code/schema.
3. Roll back application, worker, sidecar, and UI code to a schema-compatible
   release. In a normal rollback, leave the additive usage tables in place.
4. From the current head, `.venv\Scripts\python.exe -m alembic downgrade
   z3a4b5c6d7e8` removes only the latest timestamp-index migration. This is an
   index rollback, not a removal of the usage ledger.
5. Test the table migration only in an isolated scratch database: bring the
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
