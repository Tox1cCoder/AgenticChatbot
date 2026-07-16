# Tool Policy and Repository Remediation Design

**Date:** 2026-07-16

**Status:** Approved for implementation planning

## Goal

Close every issue identified by the tool-execution-policy review: repair the
policy trust and validation boundaries, make client deadlines end-to-end,
correct diagnostics and documentation, resolve all seven full-suite failures,
and eliminate all repository-wide Ruff violations without changing unrelated
behavior.

## Scope

The remediation includes:

- application-owned identity for every policy-match field;
- strict, finite validation for deployment and trusted internal policy values;
- startup validation for policy configurations that can be proven invalid
  without a runtime tool instance;
- an end-to-end client-side execution deadline;
- consistent reconnect-failure diagnostics and sanitized policy errors;
- removal or migration of the unused legacy retry helper;
- alignment of the operations guide and implementation for the one permitted
  disabled outer timeout;
- root-cause fixes for the seven currently failing repository tests; and
- all 95 Ruff findings reported by `ruff check app client_backend tests`.

The work does not add background execution, new retry modes, or unrelated
feature refactors.

## Remediation Strategy

Work proceeds in three isolated stages so behavioral changes and mechanical
cleanup remain attributable.

### Stage 1: Tool policy correctness

`clone_mcp_tool()` will overwrite `source_tool_name` alongside origin, server,
and qualified identity. Trusted internal execution metadata will be validated
through a strict model that shares the deployment policy's numeric and attempt
constraints. All deadline inputs will reject NaN and infinity.

Settings validation will reject statically detectable ambiguity, invalid
disabled-timeout authority, non-finite values, impossible global grace/cap
ordering, and reversed client grace ordering. Runtime identity resolution will
reject unrecognized origins rather than silently skipping origin rules.

The disabled outer-timeout exception will require all of:

1. the exact `internal::dispatch_subagents` canonical identity;
2. trusted `application_execution_policy` metadata on that tool; and
3. the code-owned allowlist.

Deployment rules and remote metadata will not be able to grant the exception.
The operations guide will describe this same rule.

Policy-resolution failures that nevertheless occur dynamically will produce a
fixed model-facing configuration-error message. Full exception information and
diagnostic rule keys remain restricted to server logs and artifacts.

### Stage 2: Runtime deadline and diagnostics

The client runtime bridge will enforce `ToolDispatchRequest.timeout_seconds`
around the entire execution operation, including manager initialization,
session setup, tool discovery, invocation, and failure reload. Timeout results
will use a typed `TIMEOUT_*` runtime error context. A timed-out operation will no
longer keep the bridge receive loop waiting for completion; any uncooperative
work will be abandoned with its terminal state consumed.

The server's existing response and outer deadlines remain independently
ordered:

```text
client execution < bridge response < server soft <= server hard <= total
```

Reconnect failure will produce a terminal diagnostic record matching the
returned error. Attempt history remains capped at five, and each emitted record
continues to omit arguments, results, raw exception messages, and runtime
details.

The unused `should_auto_retry_tool()` helper will be removed with its obsolete
tests so no legacy top-level metadata trust path remains available for reuse.

### Stage 3: Repository health

Each of the seven full-suite failures will be fixed from its demonstrated root
cause:

- normalize the live document-upload response contract used by the integration
  client;
- isolate Brave default-setting tests from developer environment secrets;
- make FastAPI route-introspection tests tolerate the installed router wrapper
  representation while still verifying registered paths;
- align server MCP configuration with the declared global allowlist and remove
  machine-specific server entries;
- align non-widget artifact expectations with the current full-output contract;
- preserve existing integration skip behavior and avoid introducing local
  machine dependencies.

Ruff cleanup follows functional green tests. Auto-fixable imports and formatting
are applied mechanically, while long strings are wrapped without content
changes. `B904`, `B017`, `SIM102`, and `E402` findings receive explicit small
edits that preserve exception semantics, test specificity, and import-time
setup requirements. Lint cleanup is split into focused batches so a behavior
regression can be traced to a small diff.

## Components and Responsibilities

- `app/ai/tool_execution_policy.py`: canonical identity validation, trusted
  metadata normalization, layered resolution, and runtime policy invariants.
- `app/core/config.py`: strict deployment models and startup-validatable
  cross-field checks.
- `app/core/mcp_adapter_utils.py`: application ownership of all server MCP
  identity fields.
- `app/ai/tool_execution.py`: sanitized configuration failures, cumulative
  attempts, and reconnect diagnostics.
- `app/ai/tool_error_policy.py`: canonical resolved-policy retry behavior only.
- `client_backend/services/runtime_bridge.py`: whole-operation client deadline
  and typed timeout response.
- `client_backend/services/local_mcp_manager.py`: MCP execution implementation
  governed by the bridge deadline rather than a partial inner-only guarantee.
- targeted API, configuration, and tests associated with the seven full-suite
  failures;
- repository files currently reported by Ruff, changed only as necessary to
  satisfy configured rules.

## Error Handling and Safety

Invalid deployment configuration fails during `Settings` construction whenever
the invalidity is independent of a concrete tool. Dynamic ambiguity or identity
errors fail the individual call closed and expose only a fixed safe message to
the model.

Client deadline expiry is classified as transient timeout failure, but automatic
retry still requires the resolved safe-repeat policy. Cancellation never claims
that a remote operation or worker thread stopped when only the local wait ended.

No tool arguments, results, local paths, device identifiers, raw sidecar detail,
or deployment rule names enter model-facing policy errors or attempt logs.

## Testing Strategy

Every behavioral correction uses red-green TDD. New regression coverage will
include:

- forged MCP `source_tool_name` cannot match a source-specific rule;
- malformed, non-finite, or over-limit internal metadata is rejected;
- non-finite deployment deadlines and impossible startup policy combinations
  are rejected;
- config-only disabled outer timeout is rejected;
- unknown runtime origins fail closed;
- policy-resolution failures are sanitized for the model;
- initialization, session creation, discovery, invocation, and reload all fall
  under the client execution deadline;
- reconnect failure history matches the returned terminal error;
- each currently failing repository test has a focused passing regression.

Verification gates, in order:

1. focused tests for each red-green cycle;
2. the 213-test policy/runtime matrix plus new regressions;
3. tests covering the seven previous full-suite failures;
4. `ruff check app client_backend tests` with zero findings;
5. `python -m compileall -q app client_backend tests`;
6. the complete `pytest -q` suite with no failures;
7. `git diff --check` and a final worktree audit.

## Delivery

Functional policy changes, full-suite root-cause fixes, and lint-only cleanup
will remain separated in the implementation history. Documentation and the
original execution plan will be updated only after behavior and verification
results are known, so recorded acceptance status reflects fresh evidence.
