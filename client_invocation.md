# Client Tool Invocation Isolation — Spec & Implementation Plan

**Branch:** `Thai-Postgre-FastAPI` (or feature branch off it)
**Status:** Draft for review
**Date:** 2026-06-10

---

## 1. Specification

### 1.1 Problem Statement

When a user chats from one sidecar client (machine B), tool calls can still be
dispatched to and executed by the sidecar process on a *different* machine
(machine A). By design, each connected client must have its own independent
tool list: a chat turn must only see and invoke the tools of the client the
message was sent from, and must not know that other clients' tools exist.

### 1.2 Root Cause Analysis

The per-turn binding/dispatch pipeline is correctly device-scoped
(`(user_id, device_id, session_id, catalog_version)` at every cache layer).
The bug is in **which `device_id` a turn ends up carrying**. Three confirmed
defects chain together:

| # | Defect | Location | Effect |
|---|--------|----------|--------|
| D1 | `device_id` is only written into graph state when the request has one. With a checkpointer, the previous turn's value **persists** in `state["device_id"]`. | `app/ai/graph.py:518-519` (`_build_initial_state_from_request`) | A conversation started on machine A keeps binding and dispatching to A on every later turn whose request lacks a `device_id` (e.g., client B's runtime bridge not yet connected — `client_backend/api/messages.py:135-139` explicitly proceeds "without device context"). |
| D2 | The client proxy only adds its own `device_id` when the payload has none; a `device_id`/`deviceId` already present (stale value persisted by the UI, or replayed from another machine) is forwarded **unmodified and unvalidated**. | `client_backend/api/common.py:43-59` (`add_device_context`) | A chat from machine B can carry machine A's `device_id`; the server trusts it. |
| D3 | The dispatch guard is vacuous: `context_device_id = str(ctx.device_id or bound_device_id)` falls back to the bound device when context is missing, so the subsequent `context_device_id != bound_device_id` check can never fire in exactly the stale-state cases it should catch. | `app/ai/client_runtime_tools.py:196-237` (`_build_tool` / `_dispatch_client_tool`) | No last line of defense; a tool bound under a stale device dispatches silently. |
| D4 | (Latent, same-machine case) `generate_device_identifier()` hashes only machine attributes (hostname, arch, CPU, MAC), despite the docstring promising "unique per installation". Two installations on one machine collapse into one device record and share one request queue. | `client_backend/core/security.py:113-143`; uniqueness constraint at `app/models/client_device.py:35-64` | Whichever process polls first executes the other's tool calls; catalogs overwrite each other (last sync wins). |

Secondary risk (audit, not confirmed firing): `get_loaded_client_tools()` falls
back to a cross-device scan of `_client_tool_scopes` when `device_id` or
`session_id` is missing (`app/ai/deferred_tool_state.py:520-559`).

### 1.3 Requirements

Decisions confirmed with the product owner (2026-06-10):

- **FR-1 Strict per-client isolation.** Each chat turn binds ONLY the tools of
  the client the message was sent from. Tools loaded by another client earlier
  in the same conversation are invisible and un-invokable for this turn. A turn
  with no connected client binds no client tools at all.
- **FR-2 No cross-client knowledge.** The model must not be able to discover
  (via tool_search, catalogs, or stale checkpoint state) that another client's
  tools exist.
- **FR-3 Per-installation device identity.** Two client installations on the
  same machine are separate, independent devices. The identifier must be
  random, generated once, persisted in the installation's config directory, and
  stable across restarts.
- **FR-4 Graceful unavailability.** If the model calls a client tool whose
  client is not the originating one or is disconnected, return a clear tool
  error result to the model so it can explain the situation; the conversation
  continues. Never crash the turn; never silently dispatch elsewhere.
- **NFR-1** No regression to single-client flows, HITL interrupt/resume, or
  server-side MCP tools.
- **NFR-2** Existing devices re-register cleanly if their identifier changes
  (new device row; orphaned rows are inert and may be cleaned up later).

### 1.4 Out of Scope

- Cryptographic proof of request origin (signing chat requests with the
  runtime session). Noted as a future hardening option.
- Cross-client tool sharing or "conversation-sticky" routing (explicitly
  rejected in requirements).
- UI changes beyond what the proxy fix requires.

---

## 2. Design

### 2.1 Chosen Approach: Defense in Depth, Three Layers

**Layer 1 — The proxy asserts its own identity (fixes D2).**
`add_device_context` always overwrites `device_id`/`deviceId` with the local
bridge's registered device id. If an incoming payload carried a *different*
device id, log a warning (this is exactly the bug signature). If the bridge is
not connected, strip the keys and forward without device context — the server
then binds no client tools (FR-1), which is correct and visible, rather than
silently using another machine.

**Layer 2 — The server treats `device_id` as per-turn, validated input
(fixes D1, hardens D2).**
- `_build_initial_state_from_request` writes `initial_state["device_id"] =
  request.device_id` **unconditionally** (including `None`), so the checkpoint
  channel is overwritten every turn and can never leak a previous client.
- Before the value enters graph state, validate it: the device must belong to
  the requesting user and have an active runtime session
  (`get_active_client_runtime_session` already implements this check). Invalid
  or inactive → treat as `None` and log; never fall back to a different device.

**Layer 3 — Dispatch guard actually guards (fixes D3, implements FR-4).**
In `_dispatch_client_tool`, remove the `or bound_device_id` fallback. If the
tool context has no `device_id`, or it differs from `bound_device_id`, return a
tool **error result** ("This tool belongs to a client that is not connected to
this chat session.") instead of raising `RuntimeError`, so the model can
recover (FR-4). The same graceful error applies when `dispatch_tool_call`
finds no active session (client disconnected mid-conversation).

**Identity fix (fixes D4, implements FR-3).**
`generate_device_identifier()` becomes: read `device_identity.json` (or similar)
from the client's config/app-data directory; if absent, generate `uuid4().hex`,
write it, return it. The machine-hash code is deleted. Existing installations
re-register as new devices on first run (NFR-2).

### 2.2 Alternatives Considered

1. **Server-only fix (validate + reset state, no proxy change).** Insufficient:
   a UI that persists `deviceId` across machines would still pass validation if
   machine A's session happens to be active — the request *looks* legitimate.
   The proxy is the only component that knows where the request physically
   originated.
2. **Conversation-sticky routing** (dispatch back to whichever client loaded
   the tool). Rejected by requirements — violates strict per-client isolation
   and surprises users when a tool runs on a machine they are not at.
3. **Session-signed requests** (proxy attaches its runtime `session_id`; server
   verifies the `(device_id, session_id)` pair against `ClientRuntimeStore`).
   Strongest origin guarantee, but adds payload/contract changes across three
   components. Deferred; Layers 1–3 close the observed bug without contract
   changes.

### 2.3 Data Flow After Fix

```
UI → client_backend proxy
        └─ add_device_context: ALWAYS device_id := own registered id (or none)
     → server /ai/chat | /messages/stream
        └─ validate device_id: owned by user + active session, else None
     → graph initial state
        └─ device_id channel overwritten EVERY turn (None included)
     → per-turn tool binding (already device-scoped)
     → model calls tool
        └─ dispatch guard: ctx.device_id must equal bound_device_id, else tool error
     → request queue for exactly that device_id
```

---

## 3. Implementation Plan

### Phase 1 — Server: per-turn device authority

- [x] **T001** `app/ai/graph.py` — in `_build_initial_state_from_request`,
  set `initial_state["device_id"] = request.device_id` unconditionally
  (drop the `is not None` guard at lines 518-519). Audit the file for any
  other state-construction path with the same conditional pattern.
  *Done when:* a turn without `device_id` observably clears the checkpointed
  value (covered by T010). ✅ Done 2026-06-11; T010 failed before the fix
  (stale device retained in checkpoint), passes after. Audit: the only other
  `device_id` state writes in graph.py are per-turn subagent state built from
  the parent turn's state — correct by construction.
- [x] **T002** Add request-level validation where `WorkflowExecutionRequest`
  is built (message service / chat entry points): if `device_id` is present
  but `get_active_client_runtime_session(user_id, device_id)` returns `None`,
  replace it with `None` and log a warning including both ids.
  *Done when:* an invalid/foreign/inactive `device_id` results in a turn with
  no client tools, never a different device's tools. ✅ Done 2026-06-11 as
  `MessageService._validate_request_device_id`, called from
  `_build_user_message_workflow_request` (the single place a request
  `device_id` enters workflow execution; the `ai_service.get_bot_response_sync`
  path never carries one).
- [x] **T003** `app/ai/deferred_tool_state.py:520-559` — audit and constrain
  the no-filter fallback in `get_loaded_client_tools`: every caller on the
  binding/execution path must pass `device_id` + `session_id`; the
  cross-device scan must not be reachable from binding or execution. Either
  remove the fallback or restrict it to read-only/diagnostic callers.
  *Done when:* grep shows all binding/execution callers pass both ids, and a
  unit test proves missing-id lookups return `[]` (or are impossible).
  ✅ Done 2026-06-11: cross-device scan deleted; missing/unresolvable scope
  returns `[]`. All three production callers (`tool_execution.py`,
  `base_agent.py`, `custom_agent.py`) already pass both ids.

### Phase 2 — Server: dispatch guard (FR-4)

- [x] **T004** `app/ai/client_runtime_tools.py` (`_build_tool`) — remove the
  `ctx.device_id or bound_device_id` fallback. Missing context device or
  mismatch with `bound_device_id` → return a tool error string result (not an
  exception) explaining the tool's client is not part of this chat session.
  ✅ Done 2026-06-11; D3 regression test proved the old guard silently
  dispatched on missing context before the fix.
- [x] **T005** Same file / `ClientDeviceService.dispatch_tool_call` path —
  when no active session exists for the bound device at dispatch time
  (client disconnected), return the same style of graceful tool error result.
  *Done when:* T012 passes — the model receives the error text and the turn
  completes normally. ✅ Done 2026-06-11 (see implementation log).

### Phase 3 — Client proxy: assert own identity

- [x] **T006** `client_backend/api/common.py` (`add_device_context`) — always
  overwrite `device_id` and remove any `deviceId` key. If the incoming value
  differs from the bridge's registered id, log a warning with both values.
  If the bridge is not connected, strip both keys.
  *Done when:* T013 passes for all four cases (absent, same, foreign, bridge
  down). ✅ Done 2026-06-11; tests in
  `tests/client_backend/test_device_context.py` (8 cases incl. camelCase,
  both-keys, and no-mutation), foreign/strip cases verified failing first.
- [x] **T007** Audit all `add_device_context` call sites
  (`client_backend/api/messages.py` and others) — confirm every chat /
  resume-interrupt path goes through the updated function; no route forwards
  raw payload device ids. ✅ Done 2026-06-11: all 7 chat/stream/resume/stop
  routes in messages.py call it; no `proxy_server_request` caller enables
  `inject_device_context`; remaining proxied routes don't target endpoints
  that consume a body device_id (and T002 validation backstops them).
  Custom-agents proxy uses query-param `deviceId` (not JSON body) — its
  intentional usage is unaffected by the body rewrite.

### Phase 4 — Per-installation device identity (FR-3)

- [x] **T008** `client_backend/core/security.py` — replace
  `generate_device_identifier()` machine-hash with persisted identity:
  read identifier from a file in the client config directory (reuse the
  existing settings/app-data location used by `client_settings`); on first
  run generate `uuid4().hex`, write with restrictive permissions, return it.
  Delete the hostname/MAC hashing code. ✅ Done 2026-06-11:
  `device_identity.json` under `client_settings.profile_root` (overridable
  `config_dir` param keeps tests hermetic); `os.chmod 0o600` best-effort.
- [x] **T009** Migration note in README/client docs: existing installs will
  re-register as a new device on next start; stale device rows are inert.
  ✅ Done 2026-06-11 — "Device identity" subsection added to README's
  Client Runtime Bridge section.

### Phase 5 — Tests (behavior-level; see Testing Strategy)

- [x] **T010** Regression for D1: turn 1 with device A → turn 2 with
  `device_id=None` in the same thread binds **zero** client tools and the
  graph state's `device_id` is cleared.
  ✅ `tests/test_client_invocation_isolation.py` (verified failing pre-fix).
- [x] **T011** Isolation: devices A and B both connected, same conversation;
  a turn originating from B binds only B's catalog; dispatch lands on B's
  queue; A's loaded tools are not visible in the bound tool list.
  ✅ Same file; passed pre-fix too (binding layer was already device-scoped,
  as the root-cause analysis predicted).
- [x] **T012** FR-4: model calls a client tool after its device disconnected
  (and the mismatch case ctx.device ≠ bound.device) → tool error result
  returned, turn completes, no dispatch enqueued.
  ✅ 4 tests (mismatch, missing-context, disconnect, dispatch race); all
  verified failing pre-fix — the missing-context case dispatched silently.
- [x] **T013** Proxy: `add_device_context` absent/same/foreign/disconnected
  cases (foreign id is replaced + warning logged; disconnected strips keys).
  ✅ `tests/client_backend/test_device_context.py`, 8 cases.
- [x] **T014** Identity: two distinct config directories yield two distinct
  persisted identifiers; same directory yields a stable identifier across
  calls; file created on first use.
  ✅ `tests/client_backend/test_device_identity.py`, 5 cases.
- [x] **T015** Run existing suites `tests/test_client_tool_isolation.py`,
  `tests/test_multi_sidecar_hardening.py`, `tests/test_client_tool_scope.py`,
  `tests/client_backend/` — all green.
  ✅ All green 2026-06-11 (`tests/client_backend/test_live_server_integration.py`
  excluded: it requires a live canonical server on :8000 and fails with
  connection errors on HEAD too — environmental, not a regression).

### Suggested order / parallelism

T001–T003 (server state) and T006–T007 (proxy) are independent — can be done
in parallel. T004–T005 depends on nothing else but tests for it (T012) need
T002's validation in place to assert end-to-end behavior. T008 is fully
independent.

---

## 4. Testing Strategy

- **Behavioral, not structural:** tests simulate two device sessions via
  `ClientRuntimeStore` (in-memory) and assert which tools are *bound* and
  where dispatch requests are *enqueued* — not internal call patterns.
- **Edge cases covered:** missing `device_id`, foreign `device_id`, device
  disconnect between binding and dispatch, HITL resume from a different
  device (existing validation at `app/services/message_service.py:1250-1252`
  must keep rejecting), reconnect with new `session_id` (cache keys include
  session + catalog version — existing tests).
- **Verify tests fail first:** T010–T012 must fail against current code
  (they encode the bug) before the fixes land.

## 5. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LangGraph channel semantics: writing `None` must actually overwrite the checkpointed value. | T010 asserts it directly; if `None` writes are dropped by the channel, use an explicit sentinel/reducer for the `device_id` channel instead. |
| Frontend UI that intentionally sets `deviceId` for some feature breaks when the proxy overwrites it. | Repo-wide audit during T007; warning logs make any such case visible immediately. |
| Identifier change orphans existing device rows / re-prompts device trust. | Accepted (NFR-2); document in T009. |
| Tool-error wording leaks other clients' existence. | Error text mentions only "this chat session's client", never other devices (FR-2). |

---

## 6. Implementation Log (decisions made during implementation)

**Environment.** Tests run with `.conda/python.exe -m pytest` (Python 3.14,
pytest 9.0.2); the `.venv` runtime env has no pytest installed.

**Phase 1 (2026-06-11):**
- New tests live in `tests/test_client_invocation_isolation.py` (T010, T011,
  T002 ×3, T003). All bug-encoding tests were verified to fail against the
  pre-fix code before the fixes landed (TDD per Testing Strategy).
- T010 verified the LangGraph channel risk directly: a real
  `StateGraph(GraphState)` + `InMemorySaver` confirms a `None` write DOES
  overwrite the checkpointed `device_id` (LastValue channel) — no sentinel
  /reducer needed.
- T002 validation lives in `MessageService` (not the API layer) so every
  chat entry point — streaming, non-streaming, AI SDK — funnels through one
  choke point: `_build_user_message_workflow_request`. With the runtime
  bridge disabled, any `device_id` is dropped too; that is correct because
  no client tools can exist without the bridge.
- T003: chose "remove the fallback" over "restrict to diagnostic callers"
  for `get_loaded_client_tools` — no production caller needed the scan.
  `get_all_loaded_tool_names` keeps its device-filtered enumeration: it has
  zero production callers (tests only), i.e. it is the read-only/diagnostic
  case the plan allows.
- Existing test `test_broad_lookup_returns_both_devices`
  (tests/test_multi_sidecar_hardening.py) asserted the removed cross-device
  scan; rewritten as `test_broad_lookup_without_scope_returns_nothing` to
  encode the new FR-2 contract.

**Phase 2 (2026-06-11):**
- All four T012 tests verified failing first: mismatch + disconnect raised
  `RuntimeError`; the missing-context case **silently dispatched** (D3
  confirmed live, not just theoretical).
- Guard failures return module-level error constants
  (`_ERR_TOOL_NOT_THIS_SESSION`, `_ERR_CLIENT_DISCONNECTED`,
  `_ERR_CLIENT_RECONNECTED`) prefixed "Tool unavailable:"; FR-2 wording
  mentions only "this chat session's client". Device ids go to server logs
  (`logger.warning`) for audit — logs are not model-visible.
- Session-changed (reconnect) also degrades to an error result rather than a
  raise — same family as disconnect; the next turn rebinds against the new
  session/catalog automatically.
- Scope decision: the `dispatch_tool_call` await is wrapped in
  `except RuntimeError` (T005 covers the guard→dispatch race), but a
  *successful* dispatch whose response has `success=False` still raises as
  before — that is a real client-side tool failure, and
  `execute_tool_calls`'s catch-all already renders it as an error output
  with artifact metadata (NFR-1: rendering behavior preserved).
- Service-level raises inside `ClientDeviceService.dispatch_tool_call`
  remain — they are the contract for direct service callers; the tool
  closure is the layer that converts them into model-readable results.

**Phase 3 (2026-06-11):**
- `add_device_context` pops BOTH `device_id` and `deviceId` first, then
  re-stamps `device_id` only when the bridge is registered — one code path
  handles all four T013 cases (absent / same / foreign / bridge down).
  Foreign values are logged with the local id; bridge-down + incoming keys
  also warns (it is the same replay signature).
- Audit note: custom-agents proxy intentionally forwards `deviceId` as a
  *query parameter* for CRUD scoping; `add_device_context` rewrites JSON
  bodies only, so that feature is untouched (closes the "UI intentionally
  sets deviceId" risk from §5).

**Phase 4 (2026-06-11):**
- `generate_device_identifier(config_dir=None)` gained an optional explicit
  directory parameter so T014 tests run against `tmp_path` instead of the
  real profile root; production callers (only
  `RuntimeBridgeService.__init__`) call it with no args → profile root.
- Corrupt/empty identity files regenerate with a logged warning (NFR-2:
  re-registration is accepted, never a crash). Extra edge test added beyond
  the plan: corrupt-file regeneration.
- `os.chmod(0o600)` is best-effort under `contextlib.suppress(OSError)` —
  Windows ACLs ignore POSIX modes; the profile dir already holds equally
  sensitive material (`.local_session_secret`) with the same exposure.

**Phase 5 / T015 final verification (2026-06-11):**
- Full repo suite (`tests/` minus `test_live_server_integration.py`):
  **1065 passed, 0 failed**. The excluded file needs a live canonical
  server on `127.0.0.1:8000` (all 7 failures are
  "Cannot connect to server"; identical on unmodified HEAD).
- Ruff on every touched file: 0 new findings (8 pre-existing findings
  verified present on HEAD at shifted line numbers; left untouched to keep
  the diff in scope).
