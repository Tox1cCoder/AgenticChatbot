# Tool execution policy operations guide

This runbook covers the interactive tool execution guardrail used by internal,
server MCP, client MCP, and client skill tools. The resolver produces one
immutable policy for a call, and the runner uses that policy for every ordinary
retry and server-MCP reconnect attempt.

## Identity and matching

The canonical exact identity is the pair `tool_origin` plus
`qualified_tool_id`. A qualified id by itself is never a policy or trust key,
because a server MCP tool and a client MCP tool can publish the same id.

Allowed origins are `internal`, `server_mcp`, `client_mcp`, and
`client_skill`. Aliasing changes only the exposed name; the source name and
canonical qualified id remain stable.

Matching applies every applicable rule from least to most specific:

1. origin;
2. origin plus exposed tool name;
3. origin plus server name and source tool name;
4. origin plus qualified tool id.

At one specificity, more than one match is a configuration error. More-specific
fields override earlier fields, except every matching `max_timeout_seconds`
accumulates as a minimum cap. The global interactive maximum is the final cap,
so an exact rule cannot bypass a broad incident limit.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `TOOL_EXECUTION_TIMEOUT` | `30` | Default server soft timeout for unknown interactive tools. |
| `TOOL_EXECUTION_POLICIES` | `{}` | JSON object of named deployment policy rules. |
| `TOOL_EXECUTION_MAX_INTERACTIVE_TIMEOUT_SECONDS` | `120` | Final wall-clock cap for an interactive call. |
| `TOOL_EXECUTION_CANCELLATION_GRACE_SECONDS` | `2` | Cleanup window between soft cancellation and hard abandonment. |
| `TOOL_EXECUTION_CLIENT_EXECUTION_GRACE_SECONDS` | `2` | Amount subtracted from server soft timeout for client execution. |
| `TOOL_EXECUTION_CLIENT_RESPONSE_GRACE_SECONDS` | `1` | Amount subtracted from server soft timeout for bridge response waiting. |

Environment values use JSON. Keep the object on one line in `.env` files:

```dotenv
TOOL_EXECUTION_POLICIES={"client-default-cap":{"match":{"tool_origin":"client_mcp"},"max_timeout_seconds":60},"desktop-start-process":{"match":{"tool_origin":"client_mcp","qualified_tool_id":"desktop_commander::start_process"},"timeout_seconds":45,"hard_timeout_seconds":47,"total_timeout_seconds":47,"max_attempts":1,"retry_safe":false,"idempotent":false,"trust_mcp_metadata":false,"timeout_hint":"The process exceeded its interactive budget. Ask before repeating it."}}
```

Dictionary keys such as `client-default-cap` are diagnostic names. They are
recorded in artifacts and logs but do not participate in matching.

Unknown tools receive one attempt by default. `max_attempts` can exceed one
only after the resolved policy declares `retry_safe=true` or `idempotent=true`.
A transient failure is not enough on its own: automatic and model-facing retry
permission also require a safe-repeat policy.

## Metadata trust

Application execution metadata is trusted only under
`metadata.application_execution_policy` on application-created internal tools.
Remote MCP metadata is diagnostic-only unless an exact origin plus qualified-id
rule sets `trust_mcp_metadata=true`. Even then, only the allowlisted fields are
normalized: `idempotentHint`, execution timeout, retry safety, and a bounded
timeout hint. Remote metadata cannot change identity, hard or total deadlines,
attempt budgets, cancellation capability, or outer-timeout enforcement.

Legacy top-level `execution_timeout_seconds` metadata is ignored for every
origin.

## Deadline ordering and cancellation

For client-runtime tools, deadlines always satisfy:

```text
client execution < bridge response < server soft <= server hard <= total
```

With default settings this is `28 < 29 < 30 <= 32 <= 32` seconds. The sidecar
therefore stops its operation before the server stops waiting for the response,
and the server bridge still responds before the outer tool runner cancels.
Configurations too short to preserve strict ordering fail closed.

At the soft deadline the runner cancels its asyncio task. It waits only through
the hard deadline or remaining total budget. For native async tools this is
cooperative cancellation. For thread-backed and client-runtime work it is
`abandon_only`: cancellation stops waiting but does not claim that a worker
thread, sidecar operation, or remote provider call stopped.

## Safe rollout

1. Deploy the resolver and observe default policy snapshots and
   `tool_execution_attempt` events before adding behavioral overrides.
2. Add diagnostic-only rules containing just `match`. Confirm the expected
   `policy_config_keys` appear for canary traffic.
3. Add broad caps first, starting below the global maximum only where observed
   latency supports it.
4. Add exact timeout or retry rules one tool at a time. Grant multiple attempts
   only after confirming idempotency or explicit retry safety.
5. Enable MCP metadata trust only on an exact origin plus qualified-id rule and
   inspect the resulting `metadata_trusted` artifact field.
6. Expand traffic while watching timeout phases, attempt counts, session/network
   classes, and abandoned work.

Attempt logs contain identity, policy, timing, classification, and cancellation
fields. They intentionally omit arguments, results, exception messages, secrets,
and sidecar detail. Final artifacts preserve a bounded policy snapshot and at
most five attempt records for incident analysis.

## Incident response

To shorten one tool during an incident, add or edit its exact rule and lower
`timeout_seconds`, `hard_timeout_seconds`, and `total_timeout_seconds` while
preserving the cancellation grace. To cap a whole origin, add a broad
`max_timeout_seconds` rule; accumulated caps ensure exact rules cannot override
it. Reduce `max_attempts` to `1` to stop automatic repeats.

Do not set a cap at or below the cancellation grace. Invalid ordering or unsafe
retry configuration fails closed during policy resolution. After changing
rules, verify `policy_config_keys`, resolved deadlines, and `auto_retry_allowed`
in canary artifacts before expanding the change.

## Mutating calls and durable receipts

Ordinary retries are a policy decision. A *mutation* is not: repeating one can
charge a card twice or send a second message, and no timeout setting can undo
that. Mutating tool calls therefore run through
`ToolExecutionReceiptService`, which is inserted **after** authorization and
approval and **before** the provider call — a receipt for a call the user was
never allowed to make would make the refusal unretryable.

The state machine per execution key:

| Row state | What happens on the next attempt |
|---|---|
| no row | reserve, invoke, complete in the same breath as the effect |
| `completed` | return the recorded result; the provider is not called |
| `failed` | the provider never accepted it, so invoking again is safe |
| `reserved` + provider deduplicates | retry under the **same** execution key |
| `reserved` + provider does not | becomes `outcome_unknown`; the caller gets `MutationOutcomeUnknown` |
| `outcome_unknown` | never retried, by this turn or any later one |

The execution key is derived from `(thread_id, dispatch_id, task_id,
tool_call_id)`. **A model can neither supply nor read it**: a key the model
could choose is a key it could reuse to replay someone else's effect, or vary
to force a duplicate. `provider_idempotency` describes a provider capability
and is deliberately not part of the identity, so learning that a provider
deduplicates does not move a call to a different row.

The key is offered to the provider only when the resolved adapter declares it
honours one (`idempotency_key` in its signature, or `**kwargs`). A provider
that silently ignores the key would accept a duplicate while looking
deduplicated.

`max_attempts > 1` and receipts are complementary, not redundant.
`max_attempts` governs retries *within* one process while the reservation is
still held; receipts govern what happens after that process is gone. Raising
`max_attempts` on a mutating tool without `retry_safe=true` or
`idempotent=true` still fails closed during policy resolution.

Reconciling `outcome_unknown` rows is an operator task with no code path:
see "Reconciling `outcome_unknown`" in
[`routing-v2-rollout.md`](routing-v2-rollout.md).

## There is no unbounded exception any more

`_DISABLE_OUTER_TIMEOUT_ALLOWLIST` is **empty**. Every interactive tool call is
bounded, with no identity exempt.

It previously held one entry, `internal::dispatch_subagents`, because that tool
ran an entire fan-out inside a single interactive call: its inner worker and
provider operations carried the real budgets, and the outer call had to stay
open to collect their results. In routing-v2 there is no such call.
`dispatch_subagents` is bound to the Planning model as a *schema only* — the
server reads the proposal, validates it in full, and fans out as parent-graph
topology — and its function body raises `DispatchControlSchemaExecuted` so that
a wiring bug which routed it through the common tool pipeline fails loudly
instead of quietly executing. `TOOL_STAGE_NODES` is empty, so no parent-level
tool stage resolves a policy for it at all.

The **check** was kept and only the entry removed. A tool that still claims
`disable_outer_timeout` — from deployment config or from trusted application
metadata — is now refused during policy resolution and never invoked. Failing
closed is deliberate: an unbounded interactive call nobody reviewed is worse
than a timeout. `tests/test_routing_legacy_removal.py` asserts the allowlist
stays empty so the grant cannot creep back.

An unbounded mode was never generic. Interactive tools that
cannot finish inside a bounded policy must either return an existing
provider-owned task handle immediately or remain disabled. A generic job store,
polling API, progress stream, retention policy, and remote cancellation
protocol are outside this feature. The approved follow-up design is documented
in
[`docs/superpowers/specs/2026-07-16-background-tool-execution-design.md`](../superpowers/specs/2026-07-16-background-tool-execution-design.md).

## Verification

Run the focused policy and runtime tests after configuration or code changes:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_policy.py tests/test_tool_error_policy.py tests/test_tool_execution_recovery.py tests/test_tool_execution_rendering.py tests/test_client_invocation_isolation.py tests/test_skills_tool.py tests/test_tool_execution_receipt_service.py tests/test_mcp_adapter_utils.py tests/client_backend/test_runtime_bridge.py -q
```

The durable half needs a dedicated PostgreSQL database, because the atomicity
the design rests on is a database guarantee — a read-then-write repository or a
missing unique index passes every fake-backed test and duplicates real effects
in production:

```powershell
$env:TEST_DATABASE_URL='postgresql+psycopg://<user>:<password>@localhost:5432/chatbot_test'
.venv\Scripts\python.exe -m pytest tests/integration/test_tool_execution_receipt_repository_postgres.py -q
```

Then run the full suite and compilation checks before deployment.
