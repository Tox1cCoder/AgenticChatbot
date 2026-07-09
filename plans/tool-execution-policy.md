# Tool Execution Policy Design

## Context

The chatbot currently protects tool calls with a generic per-tool timeout in
`app/ai/tool_execution.py`. The default comes from
`settings.tool_execution_timeout` and is 30 seconds. When a tool exceeds that
budget, `app/ai/tool_error_policy.py` emits a compact model-facing payload such
as:

```json
{"status":"error","error_type":"timeout","retryable":true,"message":"Tool timed out after 30s.","hint":"Retry only if the operation is likely safe; otherwise adjust inputs, use another available tool, discover a better tool, or ask the user."}
```

This behavior is intentional as a guardrail, but it is too coarse for production
operations. Some tools legitimately need more time, some tools hang, and some
tools fail internally but do not surface their exception before the outer timeout
fires. A permanent fix must distinguish those cases rather than raising the
global timeout for every tool. The current compact payload also treats timeout as
a retryable failure class; this design narrows the future model-facing
`retryable` flag so it means safe to repeat the call under the resolved policy.

## Goals

- Keep the default interactive timeout conservative for unknown tools.
- Allow known slow tools to declare a justified larger budget.
- Give deployment configuration final authority to cap or override tool-declared
  budgets, so operators can respond to production incidents without code
  changes.
- Treat third-party MCP metadata as advisory unless the tool origin is trusted or
  the deployment explicitly allows that metadata to drive policy.
- Require long-running tools to use a background/progress workflow instead of
  blocking the chat turn.
- Detect and report timeout cause with enough detail for operators to diagnose
  the specific tool and policy involved.
- Preserve safe retry behavior: model-facing retry guidance and automatic
  retries are allowed only for tools that explicitly declare retry-safe or
  idempotent behavior, or for narrowly allowlisted internal discovery tools.
- Propagate resolved tool budgets to every client-side runtime dispatch path so
  server and sidecar expectations match.
- Keep model-facing errors compact while preserving detailed diagnostics in
  artifacts and logs.

## Non-Goals

- Do not globally raise `tool_execution_timeout` as the primary fix.
- Do not blindly retry unknown tools, because they may have side effects.
- Do not add tool-specific prompt hacks for individual integrations.
- Do not remove the existing guardrail for hung tools.
- Do not let arbitrary third-party tool metadata disable or raise server-side
  execution guardrails.
- Do not build a full job queue system for every tool in this change. Only tools
  classified as long-running should move to a background pattern.

## Proposed Architecture

Add a tool execution policy layer between `execute_tool_calls()` and
`invoke_tool_with_policy()`.

The policy layer resolves a `ToolExecutionPolicy` for every tool call with this
deterministic flow:

1. Start from conservative defaults for unknown tools.
2. Apply trusted first-party metadata, such as `execution_timeout_seconds`,
   `retry_safe`, `idempotent`, `execution_mode`, or cancellation support.
3. Apply deployment configuration overrides and caps keyed by
   `qualified_tool_id`, `tool_origin`, `server_name`, and tool name. Config has
   final operational authority over metadata.
4. Enforce invariant safety rules, such as no disabled outer timeout for
   interactive tools, valid hard timeout ordering, and configured maximum
   interactive budgets.

Metadata is trusted when it is attached by this application, by first-party
internal tools, or by an explicitly allowlisted origin. Metadata received from a
third-party MCP server may be recorded for diagnostics, but it must not raise or
disable guardrails unless configuration explicitly permits it.

Config matching should prefer the most specific identifier first:

1. `qualified_tool_id`, for example `desktop_commander::start_process`
2. `tool_origin` plus `server_name` plus source tool name
3. `tool_origin` plus exposed tool name
4. `tool_origin` defaults

The resolved policy is then used for timeout enforcement, retry decisions,
client runtime dispatch, artifact metadata, and structured logs.

## Configuration Shape

Add a structured configuration map for operational overrides. The exact Pydantic
field name can be chosen during implementation, but it should model this shape:

```json
{
  "tool_execution_policies": {
    "desktop_commander::start_process": {
      "match": {"qualified_tool_id": "desktop_commander::start_process"},
      "timeout_seconds": 45,
      "hard_timeout_seconds": 60,
      "max_timeout_seconds": 60,
      "execution_mode": "interactive",
      "expected_duration_class": "slow",
      "retry_safe": false,
      "idempotent": false,
      "trust_metadata": false,
      "allow_disable_outer_timeout": false,
      "timeout_hint": "The process did not finish in the interactive budget. Ask before retrying."
    }
  }
}
```

Override validation rules:

- `timeout_seconds` and `hard_timeout_seconds` must be positive when set.
- `hard_timeout_seconds` must be greater than or equal to `timeout_seconds`.
- `max_timeout_seconds` caps both metadata and config-derived soft timeouts.
- `allow_disable_outer_timeout` is valid only for first-party orchestration tools
  and should fail closed for unknown or third-party tools.
- `trust_metadata` can be enabled per exact `qualified_tool_id` or per trusted
  origin, but broad origin trust should be avoided for third-party MCP servers.

## Policy Fields

Each resolved policy should include:

- `timeout_seconds`: the soft interactive budget before the tool result becomes
  a timeout for the model.
- `hard_timeout_seconds`: final budget for cleanup/cancellation when the runtime
  can support it.
- `execution_mode`: `interactive` or `background`.
- `expected_duration_class`: `fast`, `slow`, or `long_running`.
- `retry_safe`: whether repeating the call after a transient failure is
  explicitly safe.
- `idempotent`: whether duplicate calls with the same arguments should have the
  same durable effect.
- `cancellation`: `cooperative`, `killable`, or `abandon_only`.
- `policy_source`: `config` when any config override or cap changes
  enforcement, `metadata` when trusted metadata changes defaults, otherwise
  `default`.
- `policy_config_key`: the config key that matched, when a config override was
  applied.
- `metadata_trusted`: whether tool metadata was allowed to affect enforcement.
- `outer_timeout_disabled`: whether the server-side timeout wrapper is disabled.
  This is allowed only for explicitly allowlisted first-party orchestration
  tools and must not be available to unknown or third-party tools.
- `client_dispatch_timeout_seconds`: the timeout propagated to client runtime
  dispatch, when applicable.
- `auto_retry_allowed`: whether this exact call may be automatically retried
  after a transient failure.
- `timeout_hint`: short recovery guidance used in the compact model-facing
  payload.

Unknown tools default to:

- `timeout_seconds = settings.tool_execution_timeout`
- `hard_timeout_seconds = timeout_seconds`
- `execution_mode = interactive`
- `expected_duration_class = fast`
- `retry_safe = false`
- `idempotent = false`
- `cancellation = abandon_only` unless the invocation adapter can prove
  cooperative cancellation
- `policy_source = default`
- `metadata_trusted = false` for third-party metadata
- `outer_timeout_disabled = false`

## Runtime Behavior

### Normal Interactive Tools

Fast tools use the default 30 second budget. If they fail before the timeout,
the real exception is classified and returned. If they exceed the budget, the
timeout payload includes the tool name, origin, elapsed time, timeout budget,
matched policy source, and config key in artifacts/logs.

### Slow But Valid Tools

Known slow tools can opt into a longer `timeout_seconds` through trusted metadata
or configuration. Deployment configuration can always cap or shorten the budget.
This is explicit and auditable. The global default remains unchanged for all
other tools.

### Long-Running Tools

Tools expected to take longer than an interactive turn should use
`execution_mode = background`. A background tool returns a task id, status, and
progress/read-result affordance quickly. It must not block inside
`execute_tool_calls()` until the full job completes.

`execution_timeout_seconds = None` is not a general long-running-tool mechanism.
It remains available only for explicitly allowlisted first-party orchestration
tools, such as existing subagent dispatch behavior that already enforces inner
budgets. New long-running tools should use the background contract instead of
disabling the outer timeout.

### Hung Tools

If a tool exceeds its soft timeout, the model gets a compact timeout error. The
artifact records whether cancellation was attempted and whether it succeeded.
Cancellation mode must be derived from the runtime path:

- Native async tool calls are `cooperative` when cancellation propagates through
  the coroutine.
- Sync tools executed through `asyncio.to_thread()` are `abandon_only`, because
  the server cannot kill the underlying worker thread.
- Client runtime dispatch is `abandon_only` from the server perspective unless
  the sidecar protocol later adds explicit cancellation acknowledgement.
- A future process-backed runtime can declare `killable` only when the process
  can be terminated and cleanup is confirmed.

### Errors That Masquerade As Timeouts

If a tool starts background work, opens a subprocess, or calls an external API,
it must surface known error states before the timeout whenever possible. The
policy layer should preserve the real terminal exception when it is available.
For client runtime tools and client-side skill activation, sidecar error context
must flow through `RuntimeErrorContext` as structured data, not just a flattened
`RuntimeError` string. The classifier should map sidecar errors to validation,
permission, network, session, or unknown before falling back to timeout.

### Retry Semantics

The compact model-facing `retryable` field should mean "safe to repeat this
same tool call automatically or by the model." It should be `true` only when the
failure is transient and the resolved policy has `retry_safe`, `idempotent`, or
an internal allowlist grant.

Artifacts and logs may also record `failure_retryable`, meaning the failure
class appears transient, and `auto_retry_allowed`, meaning policy permits an
automatic retry. This avoids telling the model to repeat side-effecting tools
after a timeout or network failure.

## Client Runtime Propagation

Client-local MCP tools are built in `app/ai/client_runtime_tools.py` and
dispatched through `ClientDeviceService.dispatch_tool_call()`. Client-side skill
activation in `app/ai/skills_tool.py` uses the same dispatch service. Today,
both paths use `settings.client_runtime_ws_timeout_seconds` when sending the
request to the sidecar, while the outer server tool policy may time out sooner.

The new policy should propagate the resolved timeout to client dispatch so the
server-side soft timeout and sidecar-side request timeout are aligned. The
sidecar may still apply its own local hard timeout as a defense-in-depth limit.

Implementation should expose the current resolved policy through a scoped
execution context around `invoke_tool()`, so existing tool call signatures do not
need to change. Client runtime wrappers and `activate_skill` should read that
context and pass `client_dispatch_timeout_seconds` to
`ClientDeviceService.dispatch_tool_call()`. If no policy context exists, they
fall back to `settings.client_runtime_ws_timeout_seconds` for direct/manual
dispatch use.

For client runtime tools, the sidecar request timeout should be the resolved
soft timeout. The server may use a short hard-timeout grace window around the
dispatch call so a sidecar timeout result with `RuntimeErrorContext` can be
classified before the server reports an outer timeout.

## Observability

Every tool attempt should emit structured log/artifact fields:

- `tool_name`
- `tool_origin`
- `qualified_tool_id` when available
- `execution_mode`
- `expected_duration_class`
- `policy_source`
- `policy_config_key`
- `metadata_trusted`
- `timeout_seconds`
- `hard_timeout_seconds`
- `client_dispatch_timeout_seconds`
- `elapsed_ms`
- `attempts`
- `failure_retryable`
- `auto_retry_allowed`
- `error_type`
- `retryable`
- `was_cancelled`
- `cancellation`
- `timeout_phase`: `soft_timeout`, `hard_timeout`, or `none`

Model-facing ToolMessages remain compact JSON. Full details stay in artifacts
and logs.

## Rollout

1. Add `app/ai/tool_execution_policy.py` with the policy dataclass, config
   matching, trusted-metadata checks, safety validation, and scoped current-policy
   context.
2. Add resolver tests for defaults, trusted metadata, untrusted metadata,
   config-derived overrides, config caps over metadata, and disabled-timeout
   allowlisting.
3. Route normal invocation through the resolved policy and update
   `tool_error_policy.py` so compact `retryable` reflects safe-to-repeat policy.
4. Route MCP reconnect retry through the same resolved policy instead of the
   global timeout helper.
5. Add artifact/log fields for timeout diagnostics, policy source, retry
   decisions, elapsed time, and cancellation mode.
6. Propagate the resolved budget to client runtime MCP tools and
   `activate_skill`.
7. Preserve structured `RuntimeErrorContext` through client runtime dispatch and
   classify it before falling back to timeout or unknown errors.
8. Convert selected known slow tools to explicit metadata/config policies.
9. Identify tools that should move to background execution and migrate them one
   at a time.

## Testing Strategy

- Unit test default policy resolution for unknown tools.
- Unit test trusted metadata overrides for longer timeouts.
- Unit test that untrusted third-party metadata cannot raise, disable, or mark
  retry-safe policy without config.
- Unit test disabled outer timeout only for explicitly allowlisted first-party
  orchestration tools.
- Unit test config overrides for specific tool names/origins.
- Unit test config caps overriding trusted metadata.
- Unit test that unknown side-effect tools are not automatically retried.
- Unit test that retry-safe tools can retry transient failures.
- Unit test that compact model-facing `retryable` is false for transient
  failures on unsafe tools while artifact diagnostics still record the transient
  failure class.
- Async test that a slow tool returns a structured timeout with policy metadata.
- Async test that a long-running/background tool returns quickly.
- Client runtime test that the resolved timeout is passed to
  `dispatch_tool_call()`.
- Client runtime test that `activate_skill` receives the resolved timeout.
- Client runtime test that `RuntimeErrorContext` is classified without losing
  structured sidecar error details.
- Reconnect regression test that MCP reconnect retry uses the resolved policy
  and records policy diagnostics.
- Regression test that existing compact model-facing error payloads remain small.

## Acceptance Criteria

- The global default remains 30 seconds for unknown interactive tools.
- A known slow tool can exceed 30 seconds only through an explicit policy.
- Deployment config can cap or override any tool-declared timeout.
- Third-party metadata cannot disable the outer timeout or raise the budget
  unless config explicitly trusts it.
- A hung tool produces structured timeout diagnostics with policy metadata.
- Side-effecting tools are not retried unless explicitly retry-safe or
  idempotent.
- Compact model-facing retry guidance does not instruct the model to repeat an
  unsafe side-effecting call.
- Client-side MCP tool dispatch and client-side skill activation receive the
  resolved timeout budget.
- Long-running tools have a documented path to return task/progress state instead
  of blocking the chat turn.
- Focused tool execution and client runtime tests pass.
