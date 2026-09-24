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

## Client MCP servers on the device

The sidecar keeps one long-lived session per enabled MCP server, so a server
keeps its state between calls: a process Desktop Commander starts can be read,
written to, and stopped by later calls. A timed-out call leaves the session
open. A server that exits while idle is restarted for the next call, and that
call is sent to the new server, because the request never reached the old one.
A connection that closes *during* a call is reported, not repeated, since the
call may already have had its effect. Reloading MCP servers or stopping the
sidecar ends every server and, through its kill-on-close job, every process
those servers started.

Desktop Commander is recognized by `desktop-commander` in its command or
arguments and is launched hardened (`client_backend/services/desktop_commander_policy.py`):

- an unpinned or ranged `@wonderwhy-er/desktop-commander` spec runs the pinned
  `PINNED_VERSION`; an exact version the user configured is kept. Bump the pin
  deliberately. The first launch of a new version is a full `npx` install, and
  an interrupted install leaves a broken `npm-cache/_npx/<hash>` folder that
  fails every later launch with `ENOENT package.json` until it is removed;
- `DESKTOP_COMMANDER_DISABLE_TELEMETRY=1` unless the user set it, and
  `--no-onboarding`;
- `set_config_value`, `get_recent_tool_calls`, `get_usage_stats`, and
  `give_feedback_to_desktop_commander` are not offered and cannot be called;
- every tool not on the read-only list, including tools a later release adds,
  is published with `mutation: true`. The server gates mutations for approval
  unless the user's per-server or per-tool rule for that device says otherwise,
  and runs them through mutation receipts.

Desktop Commander's `allowedDirectories` and `blockedCommands` live in
`~/.claude-server-commander/config.json`, which every Desktop Commander client
of that OS user shares. The sidecar does not write it, and neither setting
confines shell commands.

A sidecar running as administrator does not register its tools unless
`CLIENT_ALLOW_ELEVATED_RUNTIME=true`, because every dispatched command would
run with administrator rights.

## Client tool requests on the bridge

The sidecar runs each request in its own task, up to
`CLIENT_MAX_CONCURRENT_TOOL_CALLS` (default 4) at once. A request's budget
starts when it arrives, so time spent waiting for a slot counts: a request
whose deadline passes before it gets a slot is answered `TIMEOUT_NOT_STARTED`
and never runs. Before this, requests ran one at a time with a budget that
started late, so a queued command could still run after the server had
reported it timed out.

When the server stops waiting -- its response deadline passes, or the user
presses Stop -- it withdraws the request. A request still queued is removed
(Redis `LREM`, or skipped by the in-memory store) and never reaches the
device; one the device already has is followed by a `cancel` message. A
cancelled skill command's process tree is killed. A cancelled MCP call only
stops being waited for, because MCP gives the sidecar no way to stop work a
server has begun; a Desktop Commander process started that way keeps running
and can be read or stopped by later calls.

Sidecar error codes say whether the tool could have run:

| Code | Ran? |
|---|---|
| `SESSION_REQUEST_REJECTED` | No: the request no longer matched the session catalog. |
| `TIMEOUT_NOT_STARTED` | No: the deadline passed while waiting for a slot. |
| `TIMEOUT_CLIENT_EXECUTION` | Maybe: the device stopped waiting mid-call. |
| `TOOL_CONNECTION_LOST` | Maybe: the MCP server exited mid-call. |
| `DEVICE_DISCONNECTED` | Maybe: the device dropped off the bridge. |

Those three "maybe" codes, and any timeout, mark the error artifact
`outcome_unknown`. For a mutation the receipt becomes `outcome_unknown`, the
model is told not to retry, and the turn is not offered Continue. Other
failures are recorded as `failed`, which a replay may retry.

Results are capped on the device before they cross the bridge, with budgets
the server sends in each request:

| Setting (server) | Default | Applies to |
|---|---:|---|
| `CLIENT_RUNTIME_MAX_TOOL_RESULT_SIZE_BYTES` | 1 MiB | Text and structured content, as sent. The middle is cut and marked; the beginning and end are kept. |
| `CLIENT_RUNTIME_MAX_TOOL_RESULT_MEDIA_BYTES` | 5 MiB | Decoded images and audio in one result. Media over the budget is dropped whole with a note, never cut. |

The text budget bounds transport and storage, not what the model reads:
output over `TOOL_RESULT_OFFLOAD_THRESHOLD_CHARS` (16,000) is offloaded to a
blob with a 4,000-character preview either way. The server repeats the cap for
an older sidecar that does not apply it. Both sides accept WebSocket messages
up to 16 MiB (`RUNTIME_MAX_MESSAGE_BYTES`, matching uvicorn's `ws_max_size`
default); tool arguments larger than that are refused before dispatch with an
error the model can act on, instead of dropping the device's connection.

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
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_policy.py tests/test_tool_error_policy.py tests/test_tool_execution_recovery.py tests/test_tool_execution_rendering.py tests/test_client_invocation_isolation.py tests/test_skills_tool.py tests/test_tool_execution_receipt_service.py tests/test_mcp_adapter_utils.py tests/client_backend/test_runtime_bridge.py tests/client_backend/test_local_mcp_manager_sessions.py tests/client_backend/test_desktop_commander_policy.py tests/test_client_mcp_mutation_approval.py tests/client_backend/test_runtime_bridge_dispatch.py tests/test_client_runtime_cancel.py tests/test_client_tool_results.py tests/test_runtime_result_limits.py -q
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
