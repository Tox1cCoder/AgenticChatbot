# Conversation compaction operations guide

This runbook covers the PostgreSQL-backed conversation-memory compactor. The API
persists coalesced jobs, the `summary` Celery queue performs bounded model calls,
and Celery Beat recovers missed publications and schedules historical backfill.
Compacted content and tenant identifiers must never be placed in logs or metric
labels.

## Production configuration

Production requires an explicit provider and a stable model; model names
containing `preview` are rejected. Configure a credential for exactly the
selected provider. The service may use an eligible user credential, otherwise
it requires that provider's server-managed credential and never substitutes a
key belonging to another provider.

| Variable | Default | Purpose |
|---|---:|---|
| `CONVERSATION_SUMMARY_ENABLED` | `true` | Enable durable and emergency compaction. |
| `CONVERSATION_SUMMARY_PROVIDER` | `gemini` | Provider used only for compaction. |
| `CONVERSATION_SUMMARY_MODEL` | `gemini-2.5-flash` | Stable compaction model. |
| `CONVERSATION_SUMMARY_TRIGGER_MESSAGES` | `60` | Pending-message threshold; zero disables this trigger only. |
| `CONVERSATION_SUMMARY_TRIGGER_TOKENS` | `18000` | Pending-token threshold; zero disables this trigger only. |
| `CONVERSATION_SUMMARY_SOFT_CONTEXT_RATIO` | `0.70` | Queue durable work at or above this request ratio. |
| `CONVERSATION_SUMMARY_HARD_CONTEXT_RATIO` | `0.85` | Run bounded emergency handling at or above this ratio. |
| `CONVERSATION_SUMMARY_KEEP_RECENT_TURNS` | `4` | Complete recent turns retained outside compacted memory. |
| `CONVERSATION_SUMMARY_MAX_TOKENS` | `1500` | Positive compacted-memory output cap. |
| `CONVERSATION_SUMMARY_TIMEOUT_SECONDS` | `30` | Provider timeout. |
| `CONVERSATION_SUMMARY_MAX_ATTEMPTS` | `5` | Attempts before a job becomes dead. |
| `CONVERSATION_SUMMARY_LEASE_SECONDS` | `120` | Worker ownership lease. |
| `CONVERSATION_SUMMARY_RETRY_BASE_SECONDS` | `5` | Initial retry delay. |
| `CONVERSATION_SUMMARY_RETRY_MAX_SECONDS` | `900` | Retry-delay cap. |
| `CONVERSATION_SUMMARY_RECONCILE_SECONDS` | `60` | Reconciliation cadence and publication debounce. |
| `CONVERSATION_SUMMARY_SAFETY_MARGIN_TOKENS` | `1024` | Input-budget safety reserve. |
| `CONVERSATION_SUMMARY_DEFAULT_RESERVED_OUTPUT_TOKENS` | `4096` | Output space reserved when model metadata is unavailable. |

At least one background trigger must remain positive while compaction is
enabled. Keep-recent turns must be lower than the enabled message threshold,
and the ratios must satisfy `0 < soft < hard < 1`.

## Deploy and start

1. Set the variables above and validate the selected provider credential.
2. Back up PostgreSQL and stop old API and worker processes.
3. Apply the schema transition:

   ```bash
   alembic upgrade x1y2z3a4b5c6
   alembic current
   alembic check
   ```

4. Deploy the API and start a worker consuming the dedicated queue:

   ```bash
   celery -A app.workers.celery_app:celery_app worker -Q summary --loglevel=INFO
   ```

   `python -m app.workers.start_worker` also starts the parse, index, and
   summary workers together. Size summary concurrency for provider limits; the
   job lease and compare-and-swap checks provide correctness, not throughput.

5. Start exactly one scheduler instance:

   ```bash
   celery -A app.workers.celery_app:celery_app beat --loglevel=INFO
   ```

Beat publishes reconciliation every configured interval and an hourly
backfill batch. Reconciliation recovers expired leases and notifications lost
between a database commit and broker publication.

## Historical backfill

The scheduled `backfill_conversation_summaries_task` scans at most 100
candidates per hourly batch. It publishes one
`compact_backfill_conversation_task` per conversation; that child task is
rate-limited to 10 per minute. Database target coalescing makes repeated scans
and duplicate delivery idempotent.

To request a bounded batch manually:

```bash
celery -A app.workers.celery_app:celery_app call app.workers.conversation_compaction.backfill_conversation_summaries_task --args='[100]' --queue=summary
```

Run smaller batches first, then watch queue age, retry/dead counts, provider
rate limits, latency, and cost before increasing throughput.

## Health, dashboards, and alerts

- `GET /health/conversation-compaction` returns aggregate status, job counts,
  oldest actionable age, expired-lease count, and sequence lag. It returns no
  conversation identifiers or tenant content.
- `GET /metrics/conversation-compaction` exposes the dedicated Prometheus
  registry. Existing `/health/celery` remains a separate broker/worker check.

Dashboard at minimum:

- `conversation_compaction_job_count` by status;
- `conversation_compaction_oldest_actionable_age_seconds`;
- `conversation_compaction_sequence_lag` for maximum and total lag;
- operation rate/outcome and `conversation_compaction_duration_seconds`;
- input/output-token histograms and estimate delta;
- deterministic trims, provider-overflow retries, and reported cost.

Alert immediately when health is `unhealthy`, any dead job exists, or an
expired lease persists through two reconciliation intervals. Alert as degraded
when actionable age exceeds `max(2 * RECONCILE_SECONDS, LEASE_SECONDS)`, maximum
sequence lag exceeds the message trigger, retry rate rises continuously for 10
minutes, or p95 compaction latency approaches the provider timeout. Investigate
credential, rate-limit, and validation error classes before replaying work.

## Credential rotation

For credential rotation, install and validate the replacement key for the
configured provider, restart API and summary workers so cached provider clients
are replaced, execute one canary compaction, then revoke the old key. Never log
keys or provider responses. A provider change is a configuration rollout: set
both provider and stable model, provision that provider's credential, canary,
and only then remove the former credential.

## Rollback limitations

Setting `CONVERSATION_SUMMARY_ENABLED=false` stops worker, reconciler, backfill,
and emergency compaction entry points without deleting schema or durable state.
Use that as the first rollback action. Do not deploy code from before this
migration against the upgraded schema unless that release is known to tolerate
the new constraints.

A schema downgrade must be done only after all new API, worker, and Beat
processes are stopped and a database backup is verified:

```bash
alembic downgrade w7x8y9z0a1b2
```

The downgrade restores the prior schema but cannot preserve job leases,
structured compacted-memory fidelity, or generated message sequences. Forward
recovery requires applying `alembic upgrade x1y2z3a4b5c6` again and rerunning
the idempotent backfill.

## Rollout checklist

1. Validate new variables and a stable provider/model in staging.
2. Back up and migrate PostgreSQL; inspect the current Alembic revision.
3. Deploy API, summary workers, and one Beat scheduler.
4. Enable compaction for a canary and verify memory hydration on the next turn.
5. Start bounded historical backfill.
6. Observe health, lag, retries/dead jobs, token calibration, latency, and cost.
7. Expand traffic only while the summary queue remains healthy.

## Release-only configuration mapping

The following names are migration guidance only; they are not runtime aliases.

<!-- legacy-map:start -->
| Removed name | Replacement or action |
|---|---|
| `MEMORY_SUMMARY_MIN_UNSUMMARIZED_MESSAGES` | `CONVERSATION_SUMMARY_TRIGGER_MESSAGES` |
| `MEMORY_SUMMARY_MIN_UNSUMMARIZED_TOKENS` | `CONVERSATION_SUMMARY_TRIGGER_TOKENS` |
| `MEMORY_SUMMARY_KEEP_MESSAGES` | `CONVERSATION_SUMMARY_KEEP_RECENT_TURNS` (convert messages to complete turns) |
| `MEMORY_SUMMARY_MAX_TOKENS` | `CONVERSATION_SUMMARY_MAX_TOKENS` |
| `MEMORY_SUMMARY_TIMEOUT_SECONDS` | `CONVERSATION_SUMMARY_TIMEOUT_SECONDS` |
| `ENABLE_SUMMARIZATION` | `CONVERSATION_SUMMARY_ENABLED` |
| `SUMMARIZATION_TRIGGER_MESSAGES` | `CONVERSATION_SUMMARY_TRIGGER_MESSAGES` |
| `SUMMARIZATION_TRIGGER_TOKENS` | `CONVERSATION_SUMMARY_TRIGGER_TOKENS` |
| `SUMMARIZATION_TRIGGER_FRACTION` | split into soft and hard context ratios |
| `SUMMARIZATION_MODEL_CONTEXT_SIZE` | remove; model context comes from the registry |
| `SUMMARIZATION_KEEP_MESSAGES` | `CONVERSATION_SUMMARY_KEEP_RECENT_TURNS` |
| `SUMMARIZATION_MODEL` | set both `CONVERSATION_SUMMARY_PROVIDER` and `CONVERSATION_SUMMARY_MODEL` |
| `SUMMARIZATION_MAX_SUMMARY_TOKENS` | `CONVERSATION_SUMMARY_MAX_TOKENS` |
| `SUMMARIZATION_TIMEOUT_SECONDS` | `CONVERSATION_SUMMARY_TIMEOUT_SECONDS` |
<!-- legacy-map:end -->
