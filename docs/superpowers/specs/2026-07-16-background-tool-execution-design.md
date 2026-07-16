# Durable Background Tool Execution Design

## Summary

Long-running server-side tools need a durable execution path that returns control
to the chat turn quickly. This design adds an explicit registry of background
tool adapters, a PostgreSQL-owned job state machine, Celery dispatch, progress
and result retrieval, cooperative cancellation, and reconciliation after worker
or broker failure.

The first phase supports server-side `internal` and `server_mcp` capabilities
only when an application-owned adapter is registered for them. It does not
serialize arbitrary LangChain tool objects, run arbitrary model-selected tools
inside Celery, or add client-side MCP/skill job execution.

## Relationship to Interactive Tool Policy

This is the follow-up to `plans/tool-execution-policy.md`. The interactive policy
remains responsible for the short enqueue/status/result/cancel tool calls. The
background worker does not bypass that policy; it is a separate, durable
execution environment with its own adapter-declared worker timeout, retry, and
cancellation contract.

The implementation order is:

1. Complete the origin-aware interactive tool execution policy.
2. Add this durable background job substrate.
3. Migrate selected registered server tools one at a time.
4. Design client-side background execution separately if production use cases
   justify a sidecar job/progress/cancellation protocol.

## Goals

- Return a compact durable job handle within the normal interactive timeout.
- Keep PostgreSQL authoritative for job ownership, lifecycle, leases, progress,
  cancellation requests, retries, and terminal outcomes.
- Use Celery/Redis only as a dispatch hint; committed jobs remain recoverable if
  publishing fails.
- Permit only application-registered background adapters with validated,
  versioned payloads.
- Prevent duplicate Celery delivery from producing duplicate claims.
- Bound duplicate side-effect risk through lease heartbeats, adapter-declared
  retry safety, and external idempotency keys where supported.
- Enforce user/conversation ownership on enqueue, status, result, and cancel.
- Preserve the original approved payload for deferred mutations.
- Store compact results inline and large results through the existing tool-result
  blob pattern.
- Expose sanitized progress and failure codes without leaking worker exceptions,
  credentials, local paths, or raw provider payloads.

## Non-Goals

- Do not run arbitrary tool objects or import paths supplied by the model.
- Do not persist unvalidated raw tool arguments.
- Do not persist secret values; adapters store credential/provider references.
- Do not promise exactly-once external side effects.
- Do not kill Python threads or arbitrary third-party SDK calls.
- Do not add client-side MCP or client-side skill background execution.
- Do not add WebSocket push notifications or automatic chat polling in phase one.
- Do not replace existing document-processing or conversation-compaction job
  tables with the generic background job table.

## Chosen Architecture

### Why an Explicit Registry

A generic “run any tool in Celery” wrapper is unsafe and unreliable: LangChain
tool objects are not durable worker payloads, arbitrary arguments may contain
secrets, worker imports may differ from request-process bindings, and tools have
different retry/cancellation semantics.

The registry makes each supported background capability an application-owned
adapter. Registration is code reviewable, testable, and independent of remote
MCP metadata. An adapter owns validation, durable encoding, execution, progress,
result serialization, error classification, cancellation checkpoints, and retry
safety for exactly one versioned capability.

### Existing Patterns Reused

- `ConversationSummaryJob` and `ConversationCompactionRepository` provide the
  PostgreSQL lease, compare-and-swap, `SKIP LOCKED`, reconciliation, and
  Redis/Celery-as-hint pattern.
- `ToolResultBlob` and `ToolResultBlobService` provide owner-scoped large-result
  storage and retrieval.
- Existing Celery configuration supplies worker lifecycle, soft/hard time limits,
  late acknowledgement behavior, and queue management.
- Existing tool context, HITL gate, and canonical tool identity provide the
  initiating user, conversation, tool call, and approval scope.

These patterns are copied by contract, not coupled through the conversation
compaction model or repository.

## Background Adapter Contract

Each registry entry has one stable `adapter_key` and `payload_version`:

```python
class BackgroundToolAdapter(Protocol):
    adapter_key: str
    payload_version: int
    tool_origin: Literal["internal", "server_mcp"]
    qualified_tool_id: str
    retry_safe: bool
    mutation: bool
    cancellation_supported: bool
    worker_timeout_seconds: int
    max_attempts: int

    def validate_and_encode(
        self,
        arguments: Mapping[str, Any],
        context: BackgroundEnqueueContext,
    ) -> dict[str, Any]:
        pass

    async def execute(
        self,
        payload: Mapping[str, Any],
        context: BackgroundJobExecutionContext,
    ) -> Any:
        pass

    def serialize_result(self, result: Any) -> BackgroundSerializedResult:
        pass

    def classify_error(self, exception: BaseException) -> BackgroundJobError:
        pass
```

The implementation uses an abstract base class or protocol plus concrete
adapters; the exact typing form does not change the contract.

Registry invariants:

- `adapter_key` is unique and application-owned.
- `(tool_origin, qualified_tool_id)` maps to at most one active adapter version.
- Payload versions are positive integers and are resolved explicitly by the
  worker; unknown versions fail with `PAYLOAD_VERSION_UNSUPPORTED`.
- `max_attempts > 1` is rejected unless `retry_safe=true`.
- Mutation adapters require a persisted approval snapshot whose payload hash
  matches the encoded job payload.
- Adapters cannot request disabled worker time limits.

`validate_and_encode()` returns a JSON-safe payload containing only fields the
worker needs. Secret-bearing integrations store stable credential references or
provider connection identifiers; they never store passwords, API keys, access
tokens, refresh tokens, cookies, or authorization headers in the job row.

For each migrated capability, the registry builds a model-callable
`StructuredTool` under the approved exposed name. That wrapper only validates and
enqueues; the worker adapter is never bound to the model. Wrapper metadata carries
the canonical tool identity, `background_enqueue=true`, and the adapter mutation
flag so the existing interactive policy and HITL gate run before job creation.

## Durable Job Model

Create `background_tool_jobs` with these fields:

### Identity and Ownership

- `id`: UUID primary key.
- `user_id`: required FK to `users`, indexed, cascade on user deletion.
- `conversation_id`: required FK to `conversations`, indexed, cascade on
  conversation deletion.
- `tool_call_id`: required bounded string from the initiating model tool call.
- `idempotency_key`: required bounded string, unique with `user_id`.
- `tool_origin`: `internal` or `server_mcp`.
- `qualified_tool_id`: canonical source identity.
- `exposed_tool_name`: model-callable name used for diagnostics.
- `adapter_key`: registry key.
- `payload_version`: positive integer.

### Durable Input and Authorization

- `request_payload`: JSONB containing adapter-approved data.
- `request_payload_sha256`: SHA-256 of canonical JSON.
- `mutation`: boolean copied from the registry.
- `mutation_approved`: boolean; false for read-only adapters, true only when the
  existing HITL/policy path approved the exact payload.
- `approved_arguments_sha256`: nullable SHA-256 of the canonical model tool
  arguments accepted by the HITL/policy gate.
- `approval_source`: nullable bounded code such as `human`, `policy`, or
  `hitl_disabled`; it contains no user-entered decision text.
- `approved_at`: nullable timestamp.

The worker verifies adapter identity, payload version, canonical payload hash,
and mutation approval before execution. It never trusts Celery task arguments
for any of these fields; the task carries only `job_id`.

### Approval Binding

The existing HITL gate runs before the enqueue wrapper, so approval evidence must
survive into deferred job creation without trusting a wrapper-supplied boolean.
Extend scoped tool context with application-owned approval evidence containing
`tool_call_id`, canonical argument SHA-256, approval source, and approval time.

For a mutation enqueue:

1. The HITL/policy gate hashes the same normalized tool-call argument object it
   presents for approval.
2. The enqueue service independently canonicalizes the received raw arguments
   and requires its hash to equal the scoped approval evidence for the same tool
   call id.
3. The adapter validates and encodes those arguments into its versioned durable
   payload.
4. The service stores both `approved_arguments_sha256` and
   `request_payload_sha256`, plus `mutation_approved=true` and the bounded
   approval source.
5. The worker verifies the stored request payload hash and requires all mutation
   approval fields before executing.

Read-only adapters store no approval evidence. Approval evidence cannot be
provided through model arguments, Celery messages, or remote MCP metadata.

### Lifecycle and Lease

- `status`: `queued`, `running`, `retry`, `cancel_requested`, `succeeded`,
  `failed`, `cancelled`, or `expired`.
- `attempt_count`: non-negative integer.
- `max_attempts`: integer from one through five, copied from the adapter at
  enqueue time.
- `available_at`: next eligible claim time.
- `lease_token`: nullable UUID.
- `lease_expires_at`: nullable timestamp.
- `heartbeat_at`: nullable timestamp.
- `cancel_requested_at`: nullable timestamp.
- `started_at`, `completed_at`, `expires_at`: nullable timestamps.
- `created_at`, `updated_at`: required timestamps.

### Progress and Outcome

- `progress_percent`: integer from zero through one hundred.
- `progress_stage`: nullable application-owned bounded code.
- `progress_message`: nullable sanitized user-facing text, maximum 240
  characters.
- `result_payload`: nullable JSONB for compact JSON-safe results.
- `result_preview`: nullable sanitized text preview.
- `result_blob_id`: nullable FK to `tool_result_blobs` for large/full text.
- `error_code`: nullable bounded application-owned code.
- `error_message`: nullable sanitized user-facing text, maximum 240 characters.

Indexes support `(status, available_at)`, `(user_id, created_at)`, and
`(conversation_id, created_at)`. Check constraints enforce statuses, progress,
attempt ranges, and terminal timestamp/result consistency.

## Idempotent Enqueue

`execute_tool_calls()` already owns `tool_call_id`; background execution adds it
to the scoped tool context before invoking a background enqueue wrapper.

The default idempotency key is:

```text
background-tool:{user_id}:{conversation_id}:{tool_call_id}
```

The enqueue transaction:

1. Resolves the adapter from application-owned canonical identity.
2. Confirms the user owns the conversation.
3. Validates and encodes arguments.
4. Computes canonical payload JSON and SHA-256.
5. Confirms mutation approval for the exact canonical argument hash when
   required and records the encoded payload hash separately.
6. Inserts the job with the unique user/idempotency key.
7. On conflict, loads the existing job and requires the same conversation,
   adapter, canonical tool identity, and payload hash. A mismatch returns
   `IDEMPOTENCY_CONFLICT`; an exact match returns the existing handle.
8. Commits before publishing a Celery hint.
9. Suppresses and logs sanitized broker publication errors because reconciliation
   can publish the committed queued job later.

Direct service callers without a model `tool_call_id` must provide an explicit
idempotency key. The public model-facing enqueue wrapper never accepts an
idempotency key from model arguments.

## Job Handle and Internal Tools

An enqueue call returns compact JSON:

```json
{
  "status": "accepted",
  "job_id": "uuid",
  "job_status": "queued",
  "progress_percent": 0,
  "poll_after_seconds": 5,
  "hint": "The job is running in the background. Do not poll repeatedly in this turn."
}
```

Add three internal tools:

- `background_job_status(job_id)`: returns status, sanitized progress, timestamps,
  and whether cancellation is available.
- `background_job_result(job_id)`: returns the compact result or a preview plus
  owner-scoped blob reference after success; returns a compact lifecycle response
  before success.
- `background_job_cancel(job_id)`: marks queued/retry jobs cancelled immediately
  and running jobs `cancel_requested`.

Every operation derives `user_id` from tool context and uses owner-filtered
repository methods. Job existence is not disclosed across users. Conversation
ownership is checked at enqueue and retained on every read/cancel query.

The tools instruct the model not to busy-poll during one chat turn. Phase one
does not schedule an automatic continuation or send a completion notification;
the user or a later model turn asks for status/result.

`background_job_cancel` is marked as a mutation and uses the existing HITL gate.
Status and result tools are read-only. A migrated mutation adapter likewise marks
its enqueue wrapper as a mutation, so approval occurs before the exact payload is
persisted for deferred execution.

Equivalent authenticated HTTP endpoints are provided for UI use:

- `GET /background-tool-jobs/{job_id}`
- `GET /background-tool-jobs/{job_id}/result`
- `POST /background-tool-jobs/{job_id}/cancel`

Responses use the repository’s owner-filtered projection and never expose
request payloads, payload hashes, lease tokens, adapter internals, or raw errors.

## Worker, Lease, and Progress Protocol

The Celery task receives only a job UUID. The worker:

1. Claims an eligible job with `FOR UPDATE SKIP LOCKED`, assigns a random lease
   token, increments `attempt_count`, and commits before external work.
2. Loads the detached job input in a new transaction and verifies the adapter,
   payload version/hash, mutation approval, and lease.
3. Executes with the adapter’s hard Celery time limit and an application-level
   timeout shorter than that hard limit.
4. Renews the lease periodically while execution is active.
5. Exposes `report_progress(percent, stage, message)` and
   `raise_if_cancel_requested()` through `BackgroundJobExecutionContext`.
6. Throttles progress writes to at most once per second unless the percentage or
   stage reaches a terminal boundary.
7. Commits result/error state only with a compare-and-swap on job id, `running`
   or `cancel_requested` status, and the current lease token.
8. Discards late results after lease loss or cancellation and records a sanitized
   operational log.

The worker never holds a database transaction open during external work.

## Cancellation Semantics

Cancellation is cooperative:

- `queued` or `retry` transitions directly to `cancelled`.
- `running` transitions to `cancel_requested`; the current lease remains intact.
- The adapter checks cancellation at documented safe checkpoints.
- A worker observing cancellation stops and commits `cancelled` with no result.
- If external work completes after cancellation was requested, the result CAS
  sees `cancel_requested` and commits `cancelled`, discarding the late result.
- Celery revoke/terminate is not part of correctness and is not exposed to users.
- An adapter that cannot check cancellation declares cancellation unavailable;
  the cancel endpoint returns a conflict without changing state.

Cancellation does not claim to undo an external side effect that already
occurred.

## Retry and Duplicate-Side-Effect Safety

The database claim prevents two healthy workers from owning a live lease. Lease
heartbeats reduce false expiry during long work, but no distributed design can
guarantee exactly-once external effects after a process or network partition.

Therefore:

- Non-retry-safe adapters have `max_attempts=1`.
- An expired lease for a non-retry-safe adapter becomes `failed` with
  `WORKER_LEASE_LOST`; reconciliation does not run it again.
- Retry-safe adapters may transition from expired `running` to `retry` with
  bounded exponential backoff and jitter.
- Provider adapters that support external idempotency receive the stable job id
  or adapter-derived idempotency token on every attempt.
- Registry tests reject `max_attempts > 1` for unsafe adapters.

This provides at-least-once dispatch and single-live-lease execution without
misrepresenting external side effects as exactly once.

## Reconciliation and Expiry

Periodic Celery Beat tasks:

- publish due `queued` and `retry` jobs whose `available_at` has passed;
- recover expired leases according to adapter retry safety;
- finalize stale `cancel_requested` jobs whose worker lease expired as
  `cancelled` without replay;
- expire terminal jobs after the configured retention period;
- soft-delete or detach result blobs when their job expires.

Reconciliation operates in bounded batches and rate limits publication. Broker
failure never rolls back committed state.

Default operational settings:

- queue: `background_tools`
- claim lease: 60 seconds, renewed every 20 seconds
- progress write minimum interval: 1 second
- maximum attempts: 5, with adapter defaults normally 1
- maximum worker execution: 30 minutes
- terminal job retention: 7 days
- failed/cancelled job retention: 7 days
- default status `poll_after_seconds`: 5

All settings are positive and bounded by Pydantic validation.

## Result Storage

`BackgroundSerializedResult` contains a JSON-safe compact payload, sanitized
preview, optional full text/content type, and optional structured rich-result
metadata.

- Compact results within the configured byte threshold stay in `result_payload`.
- Larger/full results use `ToolResultBlobService` with the original user,
  conversation, tool call id, and tool name, then store `result_blob_id` and a
  preview on the job.
- Add a `ToolResultBlobRepository.create_in_session()` path. Blob insertion and
  the lease-token-guarded job success update occur in one database transaction;
  a failed result CAS rolls back the blob insertion and cannot leave an
  accessible orphan.
- Model-facing result tools use the same preview/blob-reference contract as
  existing large tool results.

## Error Contract

Adapters return bounded application error codes. The worker maps infrastructure
failures to:

- `PAYLOAD_VERSION_UNSUPPORTED`
- `PAYLOAD_HASH_MISMATCH`
- `MUTATION_APPROVAL_MISSING`
- `ADAPTER_NOT_REGISTERED`
- `WORKER_TIMEOUT`
- `WORKER_LEASE_LOST`
- `CANCELLED`
- `TRANSIENT_DEPENDENCY_FAILURE`
- `BACKGROUND_TOOL_FAILED`

User-facing messages are sanitized and capped at 240 characters. Raw exceptions
and provider responses appear only in protected logs under existing redaction
rules; they are not persisted on the job or returned by tools/API endpoints.

## Security Boundaries

- Only code-registered adapters can create executable jobs.
- Celery messages contain only job ids.
- Repository reads and mutations require job id plus owner id.
- Request payloads use adapter allowlists and versioned schemas.
- Secret values are prohibited from durable payloads and progress/result text.
- Mutation approval is bound to the canonical encoded payload hash.
- Workers re-check the persisted approval snapshot before side effects.
- Status/result projections exclude payload, hash, approval internals, lease
  fields, and raw errors.
- Cancellation is owner-scoped and cannot target another user’s jobs.

## Observability and Health

Structured metrics use bounded labels only: adapter key, tool origin, status,
outcome/error class, and retry-safe class. Never label by user, conversation,
job id, qualified tool id, arguments, payload data, or provider response.

Required metrics:

- enqueue count and enqueue-to-claim latency;
- current jobs by status;
- running lease age and expired lease count;
- execution duration by adapter/outcome;
- progress update count;
- cancellation request/completion count;
- retry and terminal failure count;
- reconciliation publish/recovery count;
- queue lag and oldest due job age.

Add separate `/health/background-tools` and `/metrics/background-tools` endpoints.
Health reports aggregate status and counts only.

## Rollout

1. Add schema, repository state machine, registry, and service with a test-only
   deterministic adapter.
2. Add worker, lease heartbeat, progress, cancellation, retry, reconciliation,
   and expiry.
3. Add internal status/result/cancel tools and authenticated API projections.
4. Add observability, health, and operational documentation.
5. Migrate one read-only retry-safe server tool as the first production adapter.
6. Observe queue lag, lease loss, retry, cancellation, and result retention.
7. Migrate mutation adapters only after payload-bound approval tests pass.

## Testing Strategy

- SQL contract tests for columns, FKs, checks, indexes, unique idempotency, and
  cascades.
- PostgreSQL integration tests for concurrent enqueue, conflict detection,
  `SKIP LOCKED`, lease ownership, heartbeat, lease loss, CAS terminal writes,
  cancellation races, retry safety, and reconciliation.
- Registry tests for duplicate identity, payload versions, timeout bounds,
  retry safety, mutation approval, and forbidden secret-bearing payload fields.
- Service tests for owner filtering, conversation ownership, exact idempotency,
  broker failure after commit, compact job handles, and payload hashing.
- Worker tests for success, progress throttling, timeout, retry, unsafe lease
  loss, cancellation checkpoints, late result discard, blob offload, and raw
  exception sanitization.
- Tool/API tests for cross-user 404 behavior, lifecycle projections, cancellation
  conflicts, result availability, and absence of private fields.
- Celery configuration tests for the dedicated queue, Beat reconciliation, soft
  and hard limits, late acknowledgements, and worker startup.
- Observability tests for bounded labels and content-free health/metrics output.

## Acceptance Criteria

- Enqueue returns a durable job handle without waiting for background completion.
- PostgreSQL is authoritative and a broker publication failure cannot lose a
  committed job.
- Only registered server adapters execute; arbitrary tool objects and imports are
  impossible.
- Repeated enqueue with the same idempotency key and payload returns the same job;
  mismatched payload returns `IDEMPOTENCY_CONFLICT`.
- Every read/cancel/result operation is owner-filtered and cross-user job ids are
  not disclosed.
- Mutation jobs execute only with a payload-hash-matched approval snapshot.
- A live lease has one owner; late workers cannot commit progress or outcomes.
- Unsafe adapters never retry after lease loss; retry-safe adapters remain within
  their persisted attempt limit.
- Progress is sanitized, bounded, and write-throttled.
- Queued jobs cancel immediately; running jobs use cooperative cancellation and
  discard late results.
- Large results reuse owner-scoped tool result blob storage.
- Reconciliation recovers due safe jobs and finalizes abandoned cancellation
  without replaying unsafe work.
- Health and metrics contain no user content or unbounded identifiers.
- Client-side MCP and skill execution remain outside phase one.
