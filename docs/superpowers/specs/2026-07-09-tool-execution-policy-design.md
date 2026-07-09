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
global timeout for every tool.

## Goals

- Keep the default interactive timeout conservative for unknown tools.
- Allow known slow tools to declare a justified larger budget.
- Require long-running tools to use a background/progress workflow instead of
  blocking the chat turn.
- Detect and report timeout cause with enough detail for operators to diagnose
  the specific tool and policy involved.
- Preserve safe retry behavior: automatic retries only for tools that explicitly
  declare retry-safe or idempotent behavior.
- Propagate resolved tool budgets to client-side runtime tools so server and
  sidecar expectations match.
- Keep model-facing errors compact while preserving detailed diagnostics in
  artifacts and logs.

## Non-Goals

- Do not globally raise `tool_execution_timeout` as the primary fix.
- Do not blindly retry unknown tools, because they may have side effects.
- Do not add tool-specific prompt hacks for individual integrations.
- Do not remove the existing guardrail for hung tools.
- Do not build a full job queue system for every tool in this change. Only tools
  classified as long-running should move to a background pattern.

## Proposed Architecture

Add a tool execution policy layer between `execute_tool_calls()` and
`invoke_tool_with_policy()`.

The policy layer resolves a `ToolExecutionPolicy` for every tool call from these
sources, in priority order:

1. Tool metadata, such as `execution_timeout_seconds`, `retry_safe`, or
   `idempotent`.
2. Server configuration overrides keyed by tool origin/name, for deployments that
   need operational tuning without code changes.
3. Conservative defaults for unknown tools.

The resolved policy is then used for timeout enforcement, retry decisions,
client runtime dispatch, artifact metadata, and structured logs.

## Policy Fields

Each resolved policy should include:

- `timeout_seconds`: the soft interactive budget before the tool result becomes
  a timeout for the model.
- `hard_timeout_seconds`: final budget for cleanup/cancellation when the runtime
  can support it.
- `execution_mode`: `interactive` or `background`.
- `expected_duration_class`: `fast`, `slow`, or `long_running`.
- `retry_safe`: whether automatic retry is allowed for retryable failures.
- `idempotent`: whether repeating the call should be safe.
- `cancellation`: `cooperative`, `killable`, or `abandon_only`.
- `policy_source`: `metadata`, `config`, or `default`.
- `timeout_hint`: short recovery guidance used in the compact model-facing
  payload.

Unknown tools default to:

- `timeout_seconds = settings.tool_execution_timeout`
- `hard_timeout_seconds = timeout_seconds`
- `execution_mode = interactive`
- `expected_duration_class = fast`
- `retry_safe = false`
- `idempotent = false`
- `cancellation = cooperative`
- `policy_source = default`

## Runtime Behavior

### Normal Interactive Tools

Fast tools use the default 30 second budget. If they fail before the timeout,
the real exception is classified and returned. If they exceed the budget, the
timeout payload includes the tool name, origin, elapsed time, timeout budget, and
policy source in artifacts/logs.

### Slow But Valid Tools

Known slow tools can opt into a longer `timeout_seconds` through metadata or
configuration. This is explicit and auditable. The global default remains
unchanged for all other tools.

### Long-Running Tools

Tools expected to take longer than an interactive turn should use
`execution_mode = background`. A background tool returns a task id, status, and
progress/read-result affordance quickly. It must not block inside
`execute_tool_calls()` until the full job completes.

### Hung Tools

If a tool exceeds its soft timeout, the model gets a compact timeout error. The
artifact records whether cancellation was attempted and whether it succeeded.
If the runtime cannot truly cancel the work, the artifact should mark
`cancellation = abandon_only` so operators know the underlying work may still be
running.

### Errors That Masquerade As Timeouts

If a tool starts background work, opens a subprocess, or calls an external API,
it must surface known error states before the timeout whenever possible. The
policy layer should preserve the real terminal exception when it is available.
For client runtime tools, sidecar error context should continue flowing through
`RuntimeErrorContext` and should be classified as validation, permission,
network, session, or unknown before falling back to timeout.

## Client Runtime Propagation

Client-local MCP tools are built in `app/ai/client_runtime_tools.py` and
dispatched through `ClientDeviceService.dispatch_tool_call()`. Today, those
tools use `settings.client_runtime_ws_timeout_seconds` when sending the request
to the sidecar, while the outer server tool policy may time out sooner.

The new policy should propagate the resolved timeout to client dispatch so the
server-side soft timeout and sidecar-side request timeout are aligned. The
sidecar may still apply its own local hard timeout as a defense-in-depth limit.

## Observability

Every tool attempt should emit structured log/artifact fields:

- `tool_name`
- `tool_origin`
- `qualified_tool_id` when available
- `execution_mode`
- `expected_duration_class`
- `policy_source`
- `timeout_seconds`
- `hard_timeout_seconds`
- `elapsed_ms`
- `attempts`
- `error_type`
- `retryable`
- `was_cancelled`
- `cancellation`
- `timeout_phase`: `soft_timeout`, `hard_timeout`, or `none`

Model-facing ToolMessages remain compact JSON. Full details stay in artifacts
and logs.

## Rollout

1. Add the policy resolver and tests for default, metadata, and config-derived
   policies.
2. Route tool invocation through the resolved policy.
3. Add artifact/log fields for timeout diagnostics.
4. Propagate the resolved budget to client runtime tool dispatch.
5. Convert selected known slow tools to explicit metadata/config policies.
6. Identify tools that should move to background execution and migrate them one
   at a time.

## Testing Strategy

- Unit test default policy resolution for unknown tools.
- Unit test metadata overrides for longer timeouts and disabled outer timeout
  where already justified by orchestration tools.
- Unit test config overrides for specific tool names/origins.
- Unit test that unknown side-effect tools are not automatically retried.
- Unit test that retry-safe tools can retry transient failures.
- Async test that a slow tool returns a structured timeout with policy metadata.
- Async test that a long-running/background tool returns quickly.
- Client runtime test that the resolved timeout is passed to
  `dispatch_tool_call()`.
- Regression test that existing compact model-facing error payloads remain small.

## Acceptance Criteria

- The global default remains 30 seconds for unknown interactive tools.
- A known slow tool can exceed 30 seconds only through an explicit policy.
- A hung tool produces structured timeout diagnostics with policy metadata.
- Side-effecting tools are not retried unless explicitly retry-safe or
  idempotent.
- Client-side tool dispatch receives the resolved timeout budget.
- Long-running tools have a documented path to return task/progress state instead
  of blocking the chat turn.
- Focused tool execution and client runtime tests pass.
