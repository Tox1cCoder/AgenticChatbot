# Tool Execution Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current permissive per-tool timeout metadata handling with a deterministic, origin-aware execution policy that enforces bounded interactive deadlines, safe retry semantics, aligned client-runtime deadlines, structured error classification, and auditable attempt diagnostics.

**Architecture:** Resolve one immutable `ToolExecutionPolicy` from application-owned identity, trusted local metadata, advisory MCP metadata, and layered deployment rules. Execute every attempt through a two-phase deadline runner, share one cumulative deadline across ordinary and MCP reconnect attempts, and expose the resolved policy through a scoped context to client-runtime wrappers. Keep model-facing errors compact while recording policy and attempt history in artifacts and structured logs.

**Tech Stack:** Python 3.11+, asyncio, contextvars, Pydantic v2 / pydantic-settings, LangChain tools, MCP adapters, FastAPI runtime bridge, pytest / pytest-asyncio.

---

## Verified Current State

The implementation already has several pieces that this work must replace or preserve:

- `settings.tool_execution_timeout` defaults to 30 seconds.
- `_resolve_tool_timeout_seconds()` accepts top-level
  `tool.metadata["execution_timeout_seconds"]`; `None` disables the outer
  timeout without validating tool origin.
- `invoke_tool_with_policy()` uses `asyncio.wait_for()` and permits automatic
  retries only when `retry_safe`, `idempotent`, or the internal `tool_search`
  allowlist says the call is safe.
- Timeout and network classifiers set model-facing `retryable=true` even when
  the tool itself is unsafe to repeat.
- The MCP reconnect branch repeats the tool outside the normal retry-safety gate
  and uses the global timeout rather than the tool-specific timeout.
- Client MCP wrappers and `activate_skill` send
  `settings.client_runtime_ws_timeout_seconds` (60 seconds by default), while
  the outer server execution timeout defaults to 30 seconds.
- `RuntimeErrorContext` already crosses the WebSocket protocol, but client tool
  wrappers flatten it into a `RuntimeError` string before classification.
- MCP server tools and client tools can share the same bare
  `server_name::tool_name` qualified id. `qualified_tool_id` is therefore not a
  globally unique policy key.
- MCP adapter metadata mixes standard annotations and remote `_meta` with
  application-attached identity fields in one `tool.metadata` dictionary.
- Sync tools run through `asyncio.to_thread()`; timing out the awaiting task does
  not stop the worker thread.

These are the supported reasons for this change. Claims about a particular
integration hiding an internal exception require a separate incident trace or
reproduction and are not an acceptance dependency for this plan.

## Scope

### In Scope

- Origin-aware tool identity and deterministic policy matching.
- Field-level metadata trust and normalization.
- Deployment overrides, caps, and validation.
- Soft cancellation and bounded hard cleanup deadlines.
- A cumulative deadline and attempt budget shared by every retry path.
- Safe model-facing and automatic retry semantics.
- Client execution, bridge-response, and server deadlines in strict order.
- Typed preservation and classification of `RuntimeErrorContext`.
- Attempt-level structured logs and final artifact history.
- A narrowly allowlisted compatibility path for `dispatch_subagents`, whose
  inner worker/provider operations already have their own budgets.

### Explicitly Deferred

A generic background job system is a separate feature. This plan does not add a
job store, task-status schema, polling tool, progress stream, result retention,
or remote cancellation protocol. Tools that cannot return inside an interactive
policy must either:

1. return a provider-owned task handle immediately using an already-existing
   integration contract, or
2. remain disabled/not migrated until the background-execution feature is
   designed and implemented.

`execution_mode="background"` and `expected_duration_class="long_running"` are
not accepted configuration values in this change. This prevents an undefined
mode from silently bypassing the interactive guardrail.

The approved server-side follow-up design is documented in
`docs/superpowers/specs/2026-07-16-background-tool-execution-design.md`.

### Non-Goals

- Do not globally raise the default 30-second soft timeout.
- Do not trust an entire `tool.metadata` dictionary based on one trusted field.
- Do not use bare `qualified_tool_id` as a policy or trust key.
- Do not retry unknown or side-effecting tools by default.
- Do not add integration-specific prompt instructions.
- Do not claim that cancellation kills a worker thread or remote sidecar call.
- Do not build the deferred generic background job system in this change.

## Policy Contract

### Canonical Tool Identity

Every resolver input is normalized to:

```python
@dataclass(frozen=True)
class ToolIdentity:
    tool_origin: str
    qualified_tool_id: str
    exposed_tool_name: str
    source_tool_name: str
    server_name: str | None
```

Allowed origins are `internal`, `server_mcp`, `client_mcp`, and
`client_skill`. The canonical exact key is the tuple
`(tool_origin, qualified_tool_id)`, never the qualified id alone.

Identity ownership rules:

- `clone_mcp_tool()` overwrites, rather than `setdefault()`-merges,
  `tool_origin`, `server_name`, and `qualified_tool_id` with
  application-derived values.
- Client runtime wrappers use the application-built `ClientRuntimeToolSpec`.
- Internal tools fall back to `internal::{source_tool_name}` when they do not
  declare an application-owned qualified id.
- Aliases change `exposed_tool_name` only. `source_tool_name` and the canonical
  qualified id remain stable.

### Metadata Trust

Application-owned execution metadata lives only under:

```python
metadata["application_execution_policy"]
```

This namespace is trusted only for `internal` tools created by this application.
Server and client MCP tools use deployment configuration by default.

When an exact deployment rule for `(tool_origin, qualified_tool_id)` sets
`trust_mcp_metadata=true`, the resolver may normalize only this allowlist:

- MCP annotation `idempotentHint` -> `idempotent`
- `_meta.execution_timeout_seconds` -> `timeout_seconds`
- `_meta.retry_safe` -> `retry_safe`
- `_meta.timeout_hint` -> `timeout_hint`, trimmed to 240 characters

All other remote fields remain diagnostic-only. Remote metadata may never set
`disable_outer_timeout`, `hard_timeout_seconds`, `total_timeout_seconds`,
`max_attempts`, cancellation capability, or origin/identity.

Client catalog entries do not gain execution metadata in this change. Client
tools receive non-default policy only through deployment configuration.

### Deployment Configuration

Add these Pydantic models before `Settings` in `app/core/config.py`:

```python
class ToolExecutionPolicyMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_origin: Literal["internal", "server_mcp", "client_mcp", "client_skill"]
    qualified_tool_id: str | None = None
    server_name: str | None = None
    source_tool_name: str | None = None
    exposed_tool_name: str | None = None


class ToolExecutionPolicyOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    match: ToolExecutionPolicyMatch
    timeout_seconds: float | None = Field(default=None, gt=0)
    hard_timeout_seconds: float | None = Field(default=None, gt=0)
    total_timeout_seconds: float | None = Field(default=None, gt=0)
    max_timeout_seconds: float | None = Field(default=None, gt=0)
    max_attempts: int | None = Field(default=None, ge=1, le=5)
    retry_safe: bool | None = None
    idempotent: bool | None = None
    trust_mcp_metadata: bool = False
    disable_outer_timeout: bool = False
    timeout_hint: str | None = Field(default=None, max_length=240)
```

`ToolExecutionPolicyMatch` accepts exactly one of four shapes: origin only;
origin + exposed name; origin + server name + source name; or origin + qualified
id. Mixed shapes and partial server/source pairs are validation errors. This
keeps specificity deterministic and prevents a rule from changing meaning when
new identity fields become available.

Add these settings:

```python
tool_execution_policies: dict[str, ToolExecutionPolicyOverride] = Field(default_factory=dict)
tool_execution_max_interactive_timeout_seconds: float = Field(default=120.0, gt=0)
tool_execution_cancellation_grace_seconds: float = Field(default=2.0, ge=0)
tool_execution_client_execution_grace_seconds: float = Field(default=2.0, gt=0)
tool_execution_client_response_grace_seconds: float = Field(default=1.0, gt=0)
```

Matching applies all matching rules from least to most specific:

1. origin default: `tool_origin`
2. origin + exposed tool name
3. origin + server name + source tool name
4. origin + qualified tool id

At each specificity, more than one matching rule is a configuration error. Rule
dictionary keys are diagnostic names, not identity. All matching
`max_timeout_seconds` values accumulate as a minimum wall-clock cap, so a broad
operational cap cannot be bypassed by a more-specific override. For an enabled
outer timeout, resolution preserves `soft < hard <= total <= accumulated cap`;
the soft timeout is shortened when necessary to retain the configured
cancellation grace. A cap that is not larger than the cancellation grace is
invalid. The global maximum interactive timeout participates as the final
wall-clock cap.

`trust_mcp_metadata=true` is valid only on an exact origin + qualified-id rule.
`disable_outer_timeout=true` is valid only for the exact
`("internal", "internal::dispatch_subagents")` identity, which is also checked
against a code-owned allowlist. Invalid configuration fails closed during
settings validation.

Example:

```json
{
  "tool_execution_policies": {
    "client-mcp-default-cap": {
      "match": {"tool_origin": "client_mcp"},
      "max_timeout_seconds": 60
    },
    "desktop-start-process": {
      "match": {
        "tool_origin": "client_mcp",
        "qualified_tool_id": "desktop_commander::start_process"
      },
      "timeout_seconds": 45,
      "hard_timeout_seconds": 47,
      "total_timeout_seconds": 47,
      "max_attempts": 1,
      "retry_safe": false,
      "idempotent": false,
      "trust_mcp_metadata": false,
      "timeout_hint": "The process did not finish in the interactive budget. Ask before repeating it."
    }
  }
}
```

### Resolved Policy

```python
@dataclass(frozen=True)
class ToolExecutionPolicy:
    identity: ToolIdentity
    timeout_seconds: float
    hard_timeout_seconds: float
    total_timeout_seconds: float
    max_attempts: int
    retry_safe: bool
    idempotent: bool
    metadata_trusted: bool
    outer_timeout_disabled: bool
    cancellation: str
    client_execution_timeout_seconds: float | None
    client_response_timeout_seconds: float | None
    policy_source: str
    policy_config_keys: tuple[str, ...]
    timeout_hint: str
```

Default unknown interactive tools resolve to:

- `timeout_seconds = settings.tool_execution_timeout` (30 seconds by default)
- `hard_timeout_seconds = timeout_seconds + cancellation grace`
- `total_timeout_seconds = hard_timeout_seconds`
- `max_attempts = 1`
- `retry_safe = false`
- `idempotent = false`
- `metadata_trusted = false`
- `outer_timeout_disabled = false`
- `cancellation = cooperative` for native async invocation and `abandon_only`
  for thread or client-runtime invocation
- `policy_source = default`

Explicit retry-safe policies may set `max_attempts > 1`, but
`total_timeout_seconds` remains the wall-clock ceiling for the entire tool call.
The resolver rejects `max_attempts > 1` unless `retry_safe` or `idempotent` is
true after trust and override resolution.

For client tools, deadlines must satisfy:

```text
client execution timeout
    < server bridge response timeout
    < server soft timeout
    <= server hard timeout
    <= total timeout
```

The default client deadlines are derived by subtracting the configured grace
values from the server soft timeout. A client policy that is too short to
preserve strict ordering fails validation instead of collapsing the deadlines.

## Runtime Contract

### Two-Phase Attempt Deadline

Each attempt follows this state machine:

1. Compute the remaining cumulative time.
2. Start exactly one `asyncio.Task` for the selected invocation adapter.
3. Wait until the earlier of the policy soft deadline and cumulative deadline.
4. If completed, preserve the real result or exception.
5. At the soft deadline, call `task.cancel()` and record
   `cancellation_attempted=true`.
6. Wait only through the earlier of the hard deadline and cumulative deadline.
7. If cancellation completes, consume the task result and record whether it
   ended with `CancelledError`, a late success, or a terminal exception. A
   non-cancellation terminal exception observed before the hard deadline is
   classified as the real failure; a cancellation or late success remains a
   timeout because the soft budget was already exceeded.
8. If it does not complete, attach a done callback that consumes future
   exceptions and return a hard-timeout result. For thread-backed work, record
   `abandon_only`; never claim the underlying thread stopped.

`timeout_phase=soft_timeout` means cancellation completed inside the grace
window. `timeout_phase=hard_timeout` means the runner stopped waiting at the
hard deadline. `timeout_phase` is diagnostic: a terminal exception surfaced
during soft-timeout cleanup retains its real `error_type`; cancellation, late
success, and hard-deadline abandonment are model-facing timeout failures.

### Retry and Reconnect

- `failure_retryable` describes whether the failure class is transient.
- `policy_retry_allowed` is true only when the policy is retry-safe or
  idempotent, or when the exact internal `tool_search` allowlist grants it.
- `auto_retry_allowed` is
  `failure_retryable and policy_retry_allowed and attempts < max_attempts and cumulative_time_remaining`.
- Model-facing `retryable` uses `failure_retryable and policy_retry_allowed`.
  It does not depend on whether the server has attempts remaining because the
  model may choose a corrected call in a later round.
- Every ordinary retry and MCP reconnect retry increments the same attempt
  counter and consumes the same cumulative deadline.
- MCP reconnect happens only for a session failure when policy permits a repeat.
  Reconnection itself is bounded by the remaining cumulative deadline. Only a
  `server_mcp` identity uses `MCPManager.reconnect_and_get_tool()`; client and
  internal session failures never enter the server-MCP reconnect path.

### Client Runtime Errors

Add a typed `ClientRuntimeToolError` carrying a `RuntimeErrorContext`. Client MCP
wrappers and `activate_skill` raise it for sidecar failures rather than flattening
the response. Classification uses code families before message text:

- `INVALID_*`, `VALIDATION_*` -> `validation`
- `PERMISSION_*`, `DENIED_*` -> `permission`
- `DEVICE_*`, `SESSION_*`, `CATALOG_*` -> `session`
- `NETWORK_*`, `UNAVAILABLE_*` -> `network`
- `TIMEOUT_*` -> `timeout`
- missing or unknown code -> `unknown`

The raw context remains artifact-only. Model-facing messages use compact,
sanitized summaries and never expose device ids, local paths, secrets, or raw
sidecar detail.

### Attempt Diagnostics

Emit one structured log event per attempt and attach a bounded
`attempt_history` list to the final artifact. Each record contains:

```python
{
    "attempt": 1,
    "tool_name": "client__desktop_commander__start_process",
    "tool_origin": "client_mcp",
    "qualified_tool_id": "desktop_commander::start_process",
    "policy_source": "config",
    "policy_config_keys": ["client-mcp-default-cap", "desktop-start-process"],
    "metadata_trusted": False,
    "timeout_seconds": 45.0,
    "hard_timeout_seconds": 47.0,
    "total_timeout_seconds": 47.0,
    "elapsed_ms": 45012,
    "failure_retryable": True,
    "policy_retry_allowed": False,
    "auto_retry_allowed": False,
    "error_type": "timeout",
    "retryable": False,
    "cancellation": "abandon_only",
    "cancellation_attempted": True,
    "cancellation_completed": False,
    "timeout_phase": "hard_timeout"
}
```

Keep at most five attempt records; `max_attempts` is also capped at five. Full
exception text and `RuntimeErrorContext.detail` remain subject to existing
artifact truncation/offload behavior.

## File Map

- Create `app/ai/tool_execution_policy.py`: identity extraction, metadata
  normalization, layered resolver, scoped policy context, and validation.
- Create `app/ai/client_runtime_errors.py`: typed sidecar exception and code
  classification helpers.
- Modify `app/core/config.py`: policy configuration models and settings.
- Modify `app/core/mcp_adapter_utils.py`: application-owned MCP identity fields.
- Modify `app/ai/tool_execution.py`: two-phase runner, cumulative retry loop,
  unified reconnect path, and attempt diagnostics.
- Modify `app/ai/tool_error_policy.py`: separate failure transience from safe
  repeatability and classify typed client errors.
- Modify `app/ai/client_runtime_tools.py`: scoped deadlines and typed errors.
- Modify `app/ai/skills_tool.py`: scoped deadlines and typed errors.
- Modify `app/services/client_device_service.py`: separate sidecar execution and
  server response deadlines.
- Modify `app/services/client_runtime_store.py`: wait using the response deadline.
- Modify `app/schemas/runtime_protocol.py`: allow positive float execution
  deadlines.
- Modify `client_backend/services/runtime_bridge.py`: preserve float execution
  deadlines instead of truncating to integers.
- Modify `client_backend/services/local_mcp_manager.py`: accept float timeouts.
- Modify `app/ai/planning_subagents.py`: application-owned identity and the one
  approved outer-timeout exception.
- Create `tests/test_tool_execution_policy.py`: resolver and configuration tests.
- Extend focused execution, runtime, and skill tests listed in the tasks below.

## Implementation Tasks

### Task 1: Add policy configuration and canonical identity

**Files:**
- Create: `app/ai/tool_execution_policy.py`
- Modify: `app/core/config.py`
- Modify: `app/core/mcp_adapter_utils.py`
- Test: `tests/test_tool_execution_policy.py`
- Test: `tests/test_mcp_adapter_utils.py`

- [x] **Step 1: Write failing configuration and identity tests**

Add tests proving:

```python
def test_exact_policy_identity_includes_origin():
    server = ToolIdentity(
        tool_origin="server_mcp",
        qualified_tool_id="desktop_commander::start_process",
        exposed_tool_name="start_process",
        source_tool_name="start_process",
        server_name="desktop_commander",
    )
    client = ToolIdentity(
        tool_origin="client_mcp",
        qualified_tool_id="desktop_commander::start_process",
        exposed_tool_name="client__desktop_commander__start_process",
        source_tool_name="start_process",
        server_name="desktop_commander",
    )
    assert (server.tool_origin, server.qualified_tool_id) != (
        client.tool_origin,
        client.qualified_tool_id,
    )


def test_bare_qualified_id_rule_is_rejected():
    with pytest.raises(ValidationError):
        ToolExecutionPolicyOverride(
            match={"qualified_tool_id": "desktop_commander::start_process"},
            timeout_seconds=45,
        )


def test_clone_mcp_tool_overwrites_remote_identity_fields():
    tool = SimpleNamespace(
        name="start_process",
        args_schema={"type": "object", "properties": {}},
        metadata={
            "tool_origin": "internal",
            "server_name": "forged",
            "qualified_tool_id": "forged::tool",
        },
    )
    cloned = clone_mcp_tool(tool, server_name="trusted_config_name")
    assert cloned.metadata["tool_origin"] == "server_mcp"
    assert cloned.metadata["server_name"] == "trusted_config_name"
```

- [x] **Step 2: Run the tests and verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_policy.py tests/test_mcp_adapter_utils.py -q
```

Expected: failures because the policy models, `ToolIdentity`, and exact-key
validation do not exist and MCP identity still uses `setdefault()`.

- [x] **Step 3: Implement the configuration models and identity extractor**

Implement the contracts from “Deployment Configuration” and “Canonical Tool
Identity.” Expose `resolve_tool_identity(tool, *, exposed_tool_name)`,
`policy_match_specificity(match)`, and
`matching_policy_rules(identity, rules)`. Their return types are respectively
`ToolIdentity`, `int`, and a least-to-most-specific list of
`(config_key, ToolExecutionPolicyOverride)` tuples.

The implementation must reject same-specificity ambiguity and sort valid rules
from least to most specific. Change MCP identity assignment from `setdefault()`
to application-owned assignment.

- [x] **Step 4: Run the focused tests**

Run the Task 1 command again. Expected: all selected tests pass.

- [x] **Step 5: Commit**

```powershell
git add app/ai/tool_execution_policy.py app/core/config.py app/core/mcp_adapter_utils.py tests/test_tool_execution_policy.py tests/test_mcp_adapter_utils.py
git commit -m "feat: add origin-aware tool policy identity"
```

### Task 2: Resolve trusted metadata, layered overrides, and safety caps

**Files:**
- Modify: `app/ai/tool_execution_policy.py`
- Modify: `app/ai/planning_subagents.py`
- Test: `tests/test_tool_execution_policy.py`
- Test: `tests/test_planning_subagents.py`

- [x] **Step 1: Write failing resolver tests**

Cover these exact cases:

- `test_unknown_tool_uses_single_attempt_bounded_default` asserts the configured
  default soft timeout, soft timeout plus cancellation grace as the hard
  timeout, the hard timeout as total timeout, and `max_attempts == 1`.
- `test_untrusted_mcp_meta_cannot_raise_timeout_or_enable_retry` supplies remote
  `_meta` requesting `None`/larger timeout and retry safety, then asserts every
  request is ignored.
- `test_exact_rule_may_trust_allowlisted_mcp_fields` uses an exact origin +
  qualified-id rule and asserts only the documented four remote fields apply.
- `test_remote_meta_cannot_disable_outer_timeout` asserts remote requests for
  disabled, hard, total, attempt, and cancellation fields never apply.
- `test_broad_cap_still_limits_more_specific_override` applies a 60-second
  origin cap plus a 90-second exact timeout and asserts 60 seconds wins.
- `test_retry_attempts_require_retry_safe_or_idempotent` asserts a policy with
  `max_attempts=2` and both safety flags false is rejected.
- `test_only_dispatch_subagents_can_disable_outer_timeout` checks one accepted
  exact internal identity plus rejected internal, server, and client identities.
- `test_same_specificity_matches_fail_closed` creates two matching exact rules
  and asserts a configuration error names both diagnostic keys.

Assertions must include resolved timeout, hard timeout, total timeout,
`max_attempts`, metadata trust, config keys, and outer-timeout state.

- [x] **Step 2: Run the resolver tests and verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_policy.py tests/test_planning_subagents.py -q
```

Expected: the new resolver cases fail because metadata normalization and policy
resolution are not implemented.

- [x] **Step 3: Implement the resolver and scoped context**

Expose `resolve_tool_execution_policy(tool, *, exposed_tool_name,
invocation_kind) -> ToolExecutionPolicy`,
`tool_policy_context(policy) -> ContextManager[ToolExecutionPolicy]`, and
`get_current_tool_policy() -> ToolExecutionPolicy | None`. Restrict
`invocation_kind` to `native_async`, `sync_thread`, or `client_runtime`.

Use a `ContextVar` token and `reset(token)` so nested and concurrent executions
do not leak policy state. Normalize only the allowlisted metadata keys. Apply
rules in the specified order, then accumulated caps, then invariant validation.

Replace the legacy subagent metadata with:

```python
metadata={
    "tool_origin": "internal",
    "qualified_tool_id": "internal::dispatch_subagents",
    "application_execution_policy": {"disable_outer_timeout": True},
}
```

Keep a code-owned exact allowlist containing only that identity.

- [x] **Step 4: Run the resolver and subagent tests**

Run the Task 2 command again. Expected: all selected tests pass.

- [x] **Step 5: Commit**

```powershell
git add app/ai/tool_execution_policy.py app/ai/planning_subagents.py tests/test_tool_execution_policy.py tests/test_planning_subagents.py
git commit -m "feat: resolve trusted tool execution policies"
```

### Task 3: Add the two-phase deadline runner

**Files:**
- Modify: `app/ai/tool_execution.py`
- Test: `tests/test_tool_execution_recovery.py`

- [x] **Step 1: Write failing async and thread cancellation tests**

Add these tests:

- `test_soft_timeout_cancels_cooperative_async_tool` sets an event in the tool's
  `CancelledError` handler and asserts the event, `soft_timeout`, and completed
  cancellation.
- `test_hard_timeout_stops_waiting_for_cancellation_suppressing_tool` catches the
  first cancellation and waits on a release event; assert the runner returns at
  the hard deadline with incomplete cancellation.
- `test_sync_thread_timeout_is_recorded_as_abandon_only` blocks a worker thread
  on an event; assert the runner returns first, marks `abandon_only`, and does not
  claim underlying cancellation.
- `test_abandoned_task_exception_is_consumed` releases abandoned work with a
  terminal exception and asserts the event loop exception handler receives no
  “Task exception was never retrieved” context.

The hard-timeout test must use an event to release the test coroutine after the
runner returns so the suite does not leak live work. The thread test must prove
the call returns before the worker finishes and later consumes the worker result.

- [x] **Step 2: Run the four tests and verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_recovery.py -k "soft_timeout or hard_timeout or abandon_only or abandoned_task" -q
```

Expected: failures because execution still uses one `asyncio.wait_for()` call and
does not expose cancellation diagnostics.

- [x] **Step 3: Implement invocation selection and the deadline runner**

Add immutable result types:

```python
@dataclass(frozen=True)
class AttemptOutcome:
    result: Any = None
    exception: BaseException | None = None
    elapsed_ms: int = 0
    cancellation_attempted: bool = False
    cancellation_completed: bool = False
    timeout_phase: str = "none"
```

Implement `invoke_tool_attempt(tool, tool_args, *, policy,
remaining_total_seconds) -> AttemptOutcome` as an async function.

Use `asyncio.create_task()` plus `asyncio.wait()` for both phases. Attach a
done-callback that calls `task.exception()` for work abandoned at the hard
deadline and catches `asyncio.CancelledError` from that inspection. Preserve
real terminal exceptions observed before the hard deadline. Do not report
`cancellation_completed=true` for the underlying thread when cancellation only
stopped awaiting `asyncio.to_thread()`.

- [x] **Step 4: Run all tool execution recovery tests**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_recovery.py -q
```

Expected: all tests pass.

- [x] **Step 5: Commit**

```powershell
git add app/ai/tool_execution.py tests/test_tool_execution_recovery.py
git commit -m "feat: enforce two-phase tool deadlines"
```

### Task 4: Unify retry, model retryability, and MCP reconnect

**Files:**
- Modify: `app/ai/tool_execution.py`
- Modify: `app/ai/tool_error_policy.py`
- Test: `tests/test_tool_error_policy.py`
- Test: `tests/test_tool_execution_recovery.py`

- [x] **Step 1: Write failing retry-budget tests**

Add these cases:

- `test_transient_failure_on_unsafe_tool_is_not_model_retryable` classifies a
  network failure, passes `policy_retry_allowed=False`, and asserts the compact
  payload contains `retryable=false`.
- `test_failure_retryable_is_preserved_for_artifact_diagnostics` asserts the
  same artifact has `failure_retryable=true` and
  `policy_retry_allowed=false`.
- `test_retries_share_one_total_deadline` uses two slow transient attempts and
  asserts the second receives only remaining time and wall-clock duration stays
  inside the total deadline plus scheduler tolerance.
- `test_unsafe_session_failure_does_not_reconnect_and_repeat` asserts one
  invocation and zero reconnect calls.
- `test_safe_session_reconnect_counts_as_next_attempt` asserts two attempts, one
  reconnect, and the same policy/config identity on both attempt records.

For the unsafe session case, assert the tool invocation count is exactly one and
`reconnect_and_get_tool()` is not called. For the safe case, assert the final
attempt count is two and elapsed wall time remains below the total deadline plus
a small scheduler tolerance.

- [x] **Step 2: Run focused retry tests and verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_error_policy.py tests/test_tool_execution_recovery.py -k "retry or reconnect or total_deadline" -q
```

Expected: failures because model retryability still reflects only failure class
and reconnect bypasses the policy loop.

- [x] **Step 3: Separate failure classification from repeat policy**

Replace the overloaded summary flag with:

```python
@dataclass(frozen=True)
class ToolErrorSummary:
    error_type: str
    failure_retryable: bool
    message: str
    hint: str
    attempts: int
```

Make `build_tool_error_payloads()` accept `policy_retry_allowed` and derive:

```python
model_retryable = summary.failure_retryable and policy_retry_allowed
```

Record both values in artifacts. Keep the compact model payload keys unchanged
apart from the corrected boolean meaning.

- [x] **Step 4: Move reconnect into the cumulative attempt loop**

Remove the separate reconnect execution branch. On a session failure, reconnect
only when `auto_retry_allowed` is true, bound reconnect with the remaining total
deadline, replace the tool in `tool_map`, and continue the same loop. Preserve
one final artifact and one `attempt_history` list.

- [x] **Step 5: Run the complete error-policy and recovery suites**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_error_policy.py tests/test_tool_execution_recovery.py -q
```

Expected: all tests pass.

- [x] **Step 6: Commit**

```powershell
git add app/ai/tool_execution.py app/ai/tool_error_policy.py tests/test_tool_error_policy.py tests/test_tool_execution_recovery.py
git commit -m "fix: unify safe tool retries and reconnects"
```

### Task 5: Order client execution, bridge, and outer deadlines

**Files:**
- Modify: `app/ai/client_runtime_tools.py`
- Modify: `app/ai/skills_tool.py`
- Modify: `app/services/client_device_service.py`
- Modify: `app/services/client_runtime_store.py`
- Modify: `app/schemas/runtime_protocol.py`
- Modify: `client_backend/services/runtime_bridge.py`
- Modify: `client_backend/services/local_mcp_manager.py`
- Test: `tests/test_client_invocation_isolation.py`
- Test: `tests/test_skills_tool.py`
- Test: `tests/client_backend/test_runtime_bridge.py`

- [x] **Step 1: Write failing deadline propagation tests**

Add tests asserting:

```python
assert dispatch_kwargs["execution_timeout_seconds"] == 28.0
assert dispatch_kwargs["response_timeout_seconds"] == 29.0
assert request.timeout_seconds == 28.0
assert store_dispatch_timeout == 29.0
```

Use a 30-second resolved server soft timeout and configured two-second execution
grace plus one-second response grace. Add an `activate_skill` case with the same
assertions and a no-policy-context case that preserves the existing
`client_runtime_ws_timeout_seconds` fallback.

- [x] **Step 2: Run the client deadline tests and verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_client_invocation_isolation.py tests/test_skills_tool.py tests/client_backend/test_runtime_bridge.py -k "timeout or deadline or dispatch" -q
```

Expected: failures because dispatch currently accepts one integer timeout and
ignores resolved policy context.

- [x] **Step 3: Split execution and response deadlines**

Change `ClientDeviceService.dispatch_tool_call()` to keep its current identity,
argument, binding, and mutation parameters, replace `timeout_seconds` with the
required float parameters `execution_timeout_seconds` and
`response_timeout_seconds`, and retain `dict[str, Any]` as its async return type.

Put `execution_timeout_seconds` into `ToolDispatchRequest.timeout_seconds` and
pass `response_timeout_seconds` only to the server runtime store wait. Change the
protocol field and sidecar call paths to positive floats; remove the integer cast
in `RuntimeBridgeService._execute_tool_request()`.

- [x] **Step 4: Read scoped policy in both client wrappers**

When policy context exists, use its client execution and response fields. When
it does not exist, pass the current WebSocket timeout for both values so direct
and manual dispatch behavior remains compatible. Give the generated
`activate_skill` tool application-owned identity metadata with
`tool_origin="client_skill"` and
`qualified_tool_id="client_skill::activate"` so its policy cannot fall back to
an unrelated internal-tool identity.

- [x] **Step 5: Run the three client suites**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_client_invocation_isolation.py tests/test_skills_tool.py tests/client_backend/test_runtime_bridge.py -q
```

Expected: all tests pass.

- [x] **Step 6: Commit**

```powershell
git add app/ai/client_runtime_tools.py app/ai/skills_tool.py app/services/client_device_service.py app/services/client_runtime_store.py app/schemas/runtime_protocol.py client_backend/services/runtime_bridge.py client_backend/services/local_mcp_manager.py tests/test_client_invocation_isolation.py tests/test_skills_tool.py tests/client_backend/test_runtime_bridge.py
git commit -m "fix: order client runtime execution deadlines"
```

### Task 6: Preserve and classify structured client runtime errors

**Files:**
- Create: `app/ai/client_runtime_errors.py`
- Modify: `app/ai/client_runtime_tools.py`
- Modify: `app/ai/skills_tool.py`
- Modify: `app/ai/tool_error_policy.py`
- Test: `tests/test_client_invocation_isolation.py`
- Test: `tests/test_skills_tool.py`
- Test: `tests/test_tool_error_policy.py`

- [ ] **Step 1: Write failing structured-error tests**

Cover validation, permission, session, network, timeout, unknown, and missing
error context. Assert that raw `detail` is present in artifact diagnostics but
absent from model-facing content.

```python
context = RuntimeErrorContext(
    message="local path was denied",
    code="PERMISSION_DENIED",
    detail={"path": "C:/private/secret.txt"},
)
error = ClientRuntimeToolError(context)
summary = classify_tool_error(error, tool_name="read_file", timeout_seconds=30, attempts=1)
assert summary.error_type == "permission"
assert "secret.txt" not in summary.message
```

- [ ] **Step 2: Run focused structured-error tests and verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_client_invocation_isolation.py tests/test_skills_tool.py tests/test_tool_error_policy.py -k "error_context or structured_runtime or sidecar" -q
```

Expected: failures because wrappers still flatten or stringify sidecar context.

- [ ] **Step 3: Implement the typed error and classifier mapping**

Define:

```python
class ClientRuntimeToolError(RuntimeError):
    def __init__(self, context: RuntimeErrorContext):
        super().__init__(context.message)
        self.context = context
```

Add a constructor that safely converts response dictionaries to
`RuntimeErrorContext`, falling back to code `UNKNOWN_RUNTIME_ERROR`. Raise this
error from both client wrappers on unsuccessful sidecar responses. Classify code
families exactly as specified in “Client Runtime Errors.”

- [ ] **Step 4: Run all three suites**

Run the Task 6 command without `-k`. Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add app/ai/client_runtime_errors.py app/ai/client_runtime_tools.py app/ai/skills_tool.py app/ai/tool_error_policy.py tests/test_client_invocation_isolation.py tests/test_skills_tool.py tests/test_tool_error_policy.py
git commit -m "fix: preserve structured client runtime errors"
```

### Task 7: Add policy and attempt diagnostics to every final artifact

**Files:**
- Modify: `app/ai/tool_execution.py`
- Modify: `app/ai/tool_error_policy.py`
- Test: `tests/test_tool_execution_recovery.py`
- Test: `tests/test_tool_execution_rendering.py`

- [ ] **Step 1: Write failing observability tests**

Add one failure case and one fail-then-success case. Assert:

```python
assert artifact["policy"]["tool_origin"] == "internal"
assert artifact["policy"]["timeout_seconds"] == 30.0
assert len(artifact["attempt_history"]) == 2
assert artifact["attempt_history"][0]["error_type"] == "network"
assert artifact["attempt_history"][1]["error_type"] is None
assert "attempt_history" not in json.loads(output["content"])
```

Capture logs and assert one `tool_execution_attempt` event per attempt with no
raw tool arguments or `RuntimeErrorContext.detail`.

- [ ] **Step 2: Run observability tests and verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_recovery.py tests/test_tool_execution_rendering.py -k "attempt_history or policy_diagnostics or attempt_log" -q
```

Expected: failures because current artifacts record only final error summary and
successful retries lose prior attempt information.

- [ ] **Step 3: Attach bounded diagnostics**

Add a JSON-safe policy snapshot and at most five attempt records to both success
and error artifacts. Emit structured logs with:

```python
logger.info(
    "tool_execution_attempt",
    extra={"tool_execution": make_json_safe(attempt_record)},
)
```

Do not log tool arguments, raw results, secrets, or sidecar detail. Keep the
model payload below the existing compact-payload test threshold.

- [ ] **Step 4: Run rendering and recovery suites**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_recovery.py tests/test_tool_execution_rendering.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add app/ai/tool_execution.py app/ai/tool_error_policy.py tests/test_tool_execution_recovery.py tests/test_tool_execution_rendering.py
git commit -m "feat: record tool policy attempt diagnostics"
```

### Task 8: Remove legacy timeout trust paths and document operations

**Files:**
- Modify: `app/ai/tool_execution.py`
- Modify: `README.md`
- Create: `docs/operations/tool-execution-policy.md`
- Test: `tests/test_tool_execution_policy.py`
- Test: `tests/test_conversation_compaction_docs.py` only if its documentation inventory requires the new operations page

- [ ] **Step 1: Write a failing legacy-metadata regression test**

```python
def test_legacy_top_level_timeout_metadata_is_ignored_for_non_internal_tool():
    tool = SimpleNamespace(
        name="remote_tool",
        metadata={
            "tool_origin": "server_mcp",
            "server_name": "remote",
            "qualified_tool_id": "remote::remote_tool",
            "execution_timeout_seconds": None,
        },
    )
    policy = resolve_tool_execution_policy(
        tool,
        exposed_tool_name="remote_tool",
        invocation_kind="native_async",
    )
    assert policy.outer_timeout_disabled is False
    assert policy.timeout_seconds == settings.tool_execution_timeout
```

- [ ] **Step 2: Run the regression test and verify its state**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_policy.py -k legacy_top_level -q
```

Expected before cleanup: fail if any compatibility path still trusts top-level
third-party timeout metadata.

- [ ] **Step 3: Delete obsolete helpers and document operator behavior**

Remove `_resolve_tool_timeout_seconds()` and the now-unused
`tool_execution_max_retries` setting after every caller uses the resolved
policy. Document:

- canonical origin + qualified-id matching;
- least-to-most-specific layering and cap behavior;
- JSON environment configuration example;
- safe rollout procedure starting with diagnostic-only rules;
- how to shorten/cap a policy during an incident;
- why client deadlines are strictly ordered;
- why `dispatch_subagents` is the only disabled-outer-timeout exception;
- why generic long-running/background jobs are outside this feature.

- [ ] **Step 4: Run documentation and policy tests**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_policy.py tests/test_conversation_compaction_docs.py -q
```

Expected: all selected tests pass. If the documentation inventory test is not
generic and does not cover this page, it must still pass unchanged.

- [ ] **Step 5: Commit**

```powershell
git add app/ai/tool_execution.py README.md docs/operations/tool-execution-policy.md tests/test_tool_execution_policy.py
git commit -m "docs: document tool execution policy operations"
```

### Task 9: Run focused and full verification

**Files:**
- No production changes expected
- Modify tests only if verification reveals a genuine missing regression, using a separate red-green commit

- [ ] **Step 1: Run the focused policy and runtime suite**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tool_execution_policy.py tests/test_tool_error_policy.py tests/test_tool_execution_recovery.py tests/test_tool_execution_rendering.py tests/test_client_invocation_isolation.py tests/test_skills_tool.py tests/test_planning_subagents.py tests/test_mcp_adapter_utils.py tests/client_backend/test_runtime_bridge.py -q
```

Expected: all selected tests pass with no leaked-task or unhandled-exception
warnings.

- [ ] **Step 2: Run the full test suite**

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Expected: all repository tests pass. Environment-dependent integration tests may
skip only through their existing skip conditions; new failures are not accepted.

- [ ] **Step 3: Verify source compilation**

```powershell
.venv\Scripts\python.exe -m compileall -q app client_backend tests
```

Expected: exit code 0 with no syntax errors.

- [ ] **Step 4: Inspect the final diff and policy invariants**

```powershell
git diff --check
git status --short
```

Confirm from the diff that:

- no bare qualified-id trust lookup remains;
- no top-level third-party timeout metadata can disable the outer guardrail;
- reconnect has no independent invocation path;
- client execution and response deadlines are distinct;
- model-facing `retryable` requires both a transient failure and safe-repeat policy;
- no generic background job implementation entered this change.

- [ ] **Step 5: Commit any verification-only regression tests**

If Step 2 exposed a missing regression and production behavior was corrected via
red-green TDD, commit that isolated correction with its exact files. If no files
changed, do not create an empty commit.

## Acceptance Criteria

- Unknown interactive tools retain a 30-second soft timeout and one attempt by
  default.
- Every policy match includes `tool_origin`; a bare qualified id cannot grant
  trust, retries, a larger timeout, or disabled timeout enforcement.
- Application-owned identity cannot be overwritten by MCP metadata.
- Remote metadata is advisory unless an exact origin + qualified-id deployment
  rule trusts the allowlisted fields.
- Deployment rules can override timeouts and broad rules can cap more-specific
  overrides.
- Soft timeout initiates cancellation; the hard timeout bounds cleanup waiting;
  the total timeout bounds the whole call across all attempts.
- Thread-backed and client-runtime calls never claim that server cancellation
  stopped underlying work.
- Ordinary retry and MCP reconnect use the same safety gate, attempt counter, and
  cumulative deadline.
- Side-effecting tools expose model-facing `retryable=false` unless policy
  explicitly declares safe repeatability.
- Client execution timeout is strictly less than server bridge response timeout,
  which is strictly less than the server soft timeout.
- Client MCP tools and `activate_skill` use resolved deadlines and preserve the
  existing fallback outside policy context.
- Structured sidecar error codes are classified without exposing raw detail to
  the model.
- Every attempt emits a sanitized structured log; final artifacts preserve a
  bounded attempt history on both failure and eventual success.
- `dispatch_subagents` is the only code-allowlisted disabled-outer-timeout tool.
- No undefined `background` execution mode or generic job system is introduced.
- Focused tests, the full suite, compilation, and `git diff --check` pass.

---

## Progress Log & Design Decisions (execution record)

### Task 1 — complete (commits a81df99..c443031, review Approved)

46 new tests green; ruff clean on touched files. Reviewer verified both `clone_mcp_tool` call sites tolerate setdefault→overwrite.

Design decisions:
- `resolve_tool_identity` defaults a missing `tool_origin` to `internal` (matches the plan's internal fallback; an unrecognized origin string can never match a `Literal`-constrained deployment rule, so it cannot gain policy trust).
- Match-shape validation (exactly one of four shapes, partial server/source pairs rejected) lives on the `ToolExecutionPolicyMatch` Pydantic model; same-specificity ambiguity raises `AmbiguousToolExecutionPolicyError` from `matching_policy_rules`, failing closed on the first ambiguous bucket.
- `disable_outer_timeout` code-owned allowlist enforcement deferred to Task 2 (needs the resolver).

### Task 2 — complete (commits c443031..1520b74, review Needs-fixes → re-review Approved)

Resolver + scoped ContextVar policy context + dispatch_subagents metadata migration; 119 policy/subagent tests green, 186 across the wider regression sweep.

Design decisions:
- `policy_source` values: `default` | `config` | `metadata` | `config+metadata` (deterministic, recorded in artifacts).
- Temporary compat bridge in `tool_execution._resolve_tool_timeout_seconds`: honors `application_execution_policy.disable_outer_timeout` for ONLY the exact `("internal", "internal::dispatch_subagents")` identity so production dispatch_subagents keeps its unbounded timeout until Task 8 migrates the runner onto the resolver. Bridge + tests are removed/replaced in Task 8.
- Review finding (fixed): `max_timeout_seconds` now min()-accumulates across ALL matching rules plus the global interactive cap — last-write-wins would have let a specific override bypass a broad operational cap.
- Plan-text conflict resolved: `tool_execution_cancellation_grace_seconds` is `ge=0` per plan, but strict `soft < hard` cannot hold at grace=0 (default derivation is `hard = soft + grace`). Adopted invariant: `soft <= hard`, `hard - soft >= grace` — strict inequality guaranteed whenever grace > 0; `soft == hard` permitted only in the degenerate grace=0 configuration. Pinned by tests.
- Internal `application_execution_policy` trust accepts the full override-shaped field set (plan trusts the namespace for internal tools); only `disable_outer_timeout` has a real caller today.

### Task 3 — complete (commits 4faab1f..58f574b, spec Approved; quality With-fixes → re-review Approved)

Two-phase attempt deadline runner + cancellation diagnostics; 23 recovery tests and 57 policy tests green in the final controller sweep, with Ruff and `git diff --check` clean. The implementation reuses the existing invocation adapter and does not rewire the retry loop (Task 4 scope).

Design decisions:
- A timeout outcome carries built-in `TimeoutError`, consistent with the existing execution/classification path (`asyncio.TimeoutError` is an alias on the supported Python runtime).
- Caller cancellation during either wait cancels the child task, attaches guarded terminal-state consumption, and immediately re-raises the parent's `CancelledError`; cleanup never adds an unbounded await to an already-cancelled request.
- `abandon_only` remains truthful for thread-backed work: cancelling the awaitable never claims that the underlying thread stopped.
- `None` and positive infinity mean an unbounded cumulative remainder; NaN, negative infinity, zero, and negative finite values fail closed as exhausted.
- Review finding (fixed): exhausted cumulative budgets are rejected before task creation. Creating a task and then calling `asyncio.wait(timeout=0)` let an immediate tool execute after deadline exhaustion (reproduced 100/100 before the fix).
- The sync-thread regression test waits for worker startup under a generous guard before awaiting the deadline outcome, avoiding a flaky assumption that a cold Windows executor starts inside 50 ms.

### Task 4 — complete (commits 1616862..601f376, controller review Approved)

35 complete error-policy and recovery tests green; `git diff --check` clean.

Design decisions:
- Failure transience and safe repeatability are now distinct; model-facing `retryable` requires both.
- Ordinary retries and server-MCP reconnects share one attempt loop and cumulative deadline.
- `dispatch_subagents` remains unbounded through the unified runner via the exact code-owned allowlist while the legacy bridge awaits Task 8 removal.
- Subagent review was not used because the active workspace policy requires explicit user authorization for delegation; the controller reviewed the exact Task 4 diff and focused suite before proceeding.

### Task 5 — complete (commit ae88c0c, controller review Approved)

29 planned client/runtime tests green; 53 tests green including the adjacent multi-sidecar gateway suite; Ruff and `git diff --check` clean.

Design decisions:
- Corrected the stale Task 5 example from 27/29 to 28/29. The accepted resolver contract independently subtracts the two-second execution grace and one-second response grace from the 30-second server soft timeout.
- Client MCP wrappers and `activate_skill` read the scoped policy and retain the prior WebSocket timeout for both values when invoked outside policy context.
- `activate_skill` now carries application-owned `client_skill` identity metadata.
- A full call-site audit found `DeviceRuntimeGateway` outside the original Task 5 file map. Its manual single timeout now maps to both deadlines, preserving that compatibility API.
- Float deadlines are preserved through the protocol, bridge, runtime store, and local MCP manager without integer truncation.
