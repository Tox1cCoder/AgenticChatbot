# Resilient HITL and Local Skill Secrets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make duplicate HITL approval submissions safe and recoverable for both Streamlit and AI SDK clients, terminate failed claimed resumes, and provide clear, device-local skill credential management.

**Architecture:** `HITLInterruptRepository.try_transition_to_resolving()` remains the only authority that can claim a graph resume. A claimed run reaches one terminal durable state—`resolved`, `failed`, or `expired`—so every client can reconcile with an owner-filtered state read rather than replaying a tool call. Canonical JSON errors retain `code`; the internal SSE, AI SDK SSE, and local sidecar project the same domain identity in their protocol-specific shapes.

**Tech Stack:** FastAPI, SQLAlchemy/PostgreSQL, Alembic, Pydantic v2, SSE, local FastAPI sidecar, Streamlit, pytest.

---

## Scope and non-goals

### In scope

- Add `failed` as a durable terminal HITL lifecycle state, including PostgreSQL enum migration and all claimed-resume failure/cancellation paths.
- Add an authenticated, ownership-filtered `GET /hitl/interrupts/{interrupt_id}` endpoint and local sidecar proxy.
- Preserve a canonical error code in JSON envelopes, internal SSE, and AI SDK SSE.
- Make Streamlit reconciliation uncached, no-replay, and terminal-state aware.
- Specify the equivalent no-replay contract for the AI SDK frontend.
- Display a skill's readiness/setup state and manage explicit, encrypted, local secret bindings without exposing values.

### Explicit non-goals

- Do not relax the first-write-wins `pending -> resolving` transition or issue a second resume request during reconciliation.
- Do not infer credential names from `SKILL.md`, source code, or prose; the user supplies the skill author's documented environment-variable name.
- Do not synchronize secrets to the canonical server, return secret values, or key them by conversation.
- Do not add polling sleeps, a Streamlit component, a new secret store, or a private skill manifest.

## File structure and responsibilities

| File | Change | Responsibility |
|---|---|---|
| `app/models/hitl_interrupt.py` | Modify | Add `FAILED` to the durable Python enum. |
| `app/alembic/versions/w7x8y9z0a1b2_add_failed_hitl_interrupt_status.py` | Create | Add `failed` to PostgreSQL's `hitl_interrupt_status` enum. |
| `app/repositories/hitl_interrupt.py` | Modify | Provide ownership-safe reads and conditional expiry/failure transitions. |
| `app/services/message_service.py` | Modify | Mark a claimed interrupt failed on terminal stream errors, cancellation, or unexpected exit. |
| `app/schemas/hitl.py` | Modify | Publish the narrow lifecycle response including `failed`. |
| `app/api/hitl.py` | Modify | Serve the owner-filtered lifecycle state. |
| `app/core/dependency_injection.py` | Modify | Auto-inject `HITLInterruptRepository` into the lifecycle route. |
| `app/schemas/responses/api_response.py` | Modify | Serialize the optional canonical domain `code`. |
| `app/services/event_streaming/internal_sse.py` | Modify | Retain typed V3 error metadata in the internal stream projection. |
| `app/api/messages.py` | Modify | Serialize known pre-stream/internal errors for Streamlit SSE. |
| `app/services/event_streaming/ai_sdk_v6.py` | Modify | Serialize known errors as AI SDK `errorText` + `statusCode`/`errorCode`. |
| `client_backend/api/messages.py` | Modify | Preserve upstream error identity in each sidecar SSE dialect. |
| `client_backend/api/proxy.py` | Modify | Proxy the lifecycle read without adding device context. |
| `app/ui/hitl_recovery.py` | Create | Pure error-code extraction and lifecycle-to-UI classification. |
| `demo.py` | Modify | Perform uncached Streamlit reconciliation, lock submissions, and render terminal/reconciling UI. |
| `client_backend/api/skills.py` | Modify | Return safe readiness/setup values in skill summaries. |
| `plans/AI_SDK_FE_CONTRACT.md` | Modify | Give the web team the lifecycle endpoint, typed AI SDK error shape, and no-replay procedure. |
| `README.md` | Modify | Document durable recovery and local credential boundaries. |
| `tests/test_hitl_api.py` | Modify | Cover owner filtering, expiry, `failed`, and the public field allowlist. |
| `tests/test_message_service_event_streaming.py` | Modify | Cover failed lifecycle transitions after a claim. |
| `tests/test_message_stream_errors.py` | Create | Cover the canonical JSON/internal-SSE error contract. |
| `tests/test_ai_sdk_v6_stream_contract.py` | Modify | Cover typed AI SDK stream errors. |
| `tests/test_hitl_ui_recovery.py` | Create | Cover pure error extraction and every durable state classification. |
| `tests/client_backend/test_messages.py` | Modify | Cover typed internal and AI SDK sidecar errors. |
| `tests/client_backend/test_hitl_proxy.py` | Modify | Cover the lifecycle proxy route. |
| `tests/client_backend/test_skills_api.py` | Modify | Cover safe readiness/setup fields. |
| `tests/test_hitl_demo_panel.py` | Modify | Keep static seams for Streamlit controls and secret widgets. |

## Public contracts

### Durable interrupt state

`GET /hitl/interrupts/{interrupt_id}` is authenticated and returns a record only when `HITLInterrupt.user_id == current_user_id`. A missing or foreign ID returns the same normal `404` error envelope with code `INTERRUPT_NOT_FOUND`.

```json
{
  "success": true,
  "message": "HITL interrupt state retrieved",
  "data": {
    "interruptId": "interrupt-123",
    "conversationId": "8b24b5d4-6d5c-4d89-8ad4-1b33a9ccf2d4",
    "status": "failed",
    "expiresAt": "2026-07-14T10:00:00Z",
    "updatedAt": "2026-07-14T09:45:00Z"
  }
}
```

`status` is exactly one of `pending`, `resolving`, `resolved`, `failed`, or `expired`. The response never includes decisions, tool arguments, session/device identifiers, user identifiers, secret data, or resolution source.

Lifecycle rules:

- `pending -> resolving` is still the conditional first-write-wins update.
- `resolving -> resolved` occurs only after completion or a persisted follow-up interrupt.
- `resolving -> failed` occurs if the claimed resume emits an error, raises, is cancelled, or finishes without `complete`/`interrupt`.
- `pending -> expired` occurs only when the stored expiry is due; an expired record is never claimed.
- `failed`, `resolved`, and `expired` are terminal and cannot be resumed.

### Error contracts

Canonical non-streaming API error envelopes include the optional domain code:

```json
{
  "success": false,
  "code": "INTERRUPT_ALREADY_RESOLVED",
  "message": "This interrupt has already been resolved."
}
```

Known errors use these stream shapes:

```json
// Internal Streamlit SSE (canonical API and sidecar)
{
  "type": "error",
  "error": "This interrupt has already been resolved.",
  "status_code": 409,
  "error_code": "INTERRUPT_ALREADY_RESOLVED"
}

// AI SDK UI Message Stream SSE (canonical API and sidecar)
{
  "type": "error",
  "errorText": "This interrupt has already been resolved.",
  "statusCode": 409,
  "errorCode": "INTERRUPT_ALREADY_RESOLVED"
}
```

Unknown exceptions retain the existing minimal `error`/`errorText` event. A post-claim continuation error uses `INTERRUPT_FAILED` with `status_code`/`statusCode` `500`, after the durable record is marked `failed`.

### Reconciliation contract

Every client sets an interrupt-ID-scoped submission lock before opening a resume stream. It disables every control that can call resume. Only `INTERRUPT_ALREADY_RESOLVED` and `INTERRUPT_CONFLICT` are duplicate conflicts; both trigger one lifecycle read and never a replay.

| State result | Required UI behavior |
|---|---|
| `pending` | Remove the lock and suppression marker; restore the approval form. |
| `resolving` | Clear decisions, retain a suppression marker, and show a non-submittable processing state with one manual status check. |
| `resolved` | Clear paused UI, retain suppression for the stale interrupt, refresh page-one/history, and return to normal chat. |
| `failed` / `expired` / state unavailable | Clear paused UI, retain suppression, explain that a new message is required, and return to normal chat. |

Unrecognized errors—including expiry, device/session/catalog/tool mismatches, incomplete decisions, and `INTERRUPT_FAILED`—stay visible as failures. They are never rendered as successful completion.

### Credential behavior

- The user enters an environment-variable identifier and a password-style value; the name must come from skill documentation.
- `POST /skills/{name}/secrets` clears the password widget after success. `GET` lists configured names only; `DELETE` removes one named binding.
- Existing `SkillSecretStore` encryption and profile scoping (`profile root / server hash / user / skills / secrets.json`) remain the only secret implementation.

## Implementation tasks

### Task 1: Add a terminal failed lifecycle and safe lifecycle-state API

**Files:**

- Modify: `app/models/hitl_interrupt.py:15-23`
- Create: `app/alembic/versions/w7x8y9z0a1b2_add_failed_hitl_interrupt_status.py`
- Modify: `app/repositories/hitl_interrupt.py:50-141,197-222`
- Modify: `app/services/message_service.py:1220-1460,1668-1861`
- Modify: `app/schemas/hitl.py:1-28`
- Modify: `app/core/dependency_injection.py:15-180`
- Modify: `app/api/hitl.py:1-57`
- Modify: `tests/test_hitl_api.py`
- Modify: `tests/test_message_service_event_streaming.py`
- Modify: `tests/test_client_tool_isolation.py`

- [x] **Step 1: Write failing lifecycle tests.**

  Extend `tests/test_hitl_api.py` with a fixture helper that creates the user, owned `Conversation`, and `HITLInterrupt`, then deletes interrupts before conversation/user teardown. Add the following assertions in addition to the existing settings tests:

  ```python
  def test_get_interrupt_state_returns_only_public_failed_lifecycle_fields(api):
      client, user_id, sf = api
      interrupt_id = _create_interrupt(sf, user_id, status=HITLInterruptStatus.FAILED)

      response = client.get(f"/hitl/interrupts/{interrupt_id}")

      assert response.status_code == 200
      assert response.json()["data"]["status"] == "failed"
      assert set(response.json()["data"]) == {
          "interruptId", "conversationId", "status", "expiresAt", "updatedAt"
      }
  ```

  Add a repository test that `mark_failed()` changes only a `RESOLVING` row, and a message-service stream test whose claimed source yields `make_event("error", sequence=1, data={"error": "upstream failure"})`. Assert `mark_failed(interrupt_id, resolution_source="stream_error")` is called before the terminal error event. Add a cancellation test that closes a claimed source and asserts `resolution_source="client_disconnect"`.

- [x] **Step 2: Run the focused tests and confirm the new status is unavailable.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_hitl_api.py tests/test_message_service_event_streaming.py tests/test_client_tool_isolation.py
  ```

  Expected: FAIL because `FAILED`, `mark_failed`, and the state route do not exist.

- [x] **Step 3: Add the enum value and PostgreSQL migration.**

  Add `FAILED = "failed"` after `RESOLVING` in `HITLInterruptStatus`. Create a migration whose revision follows current head `v1w2x3y4z5a6`:

  ```python
  revision = "w7x8y9z0a1b2"
  down_revision = "v1w2x3y4z5a6"

  def upgrade() -> None:
      op.execute("ALTER TYPE hitl_interrupt_status ADD VALUE IF NOT EXISTS 'failed'")

  def downgrade() -> None:
      # PostgreSQL does not support removing an enum label safely.
      pass
  ```

  Keep the existing SQLAlchemy enum type name `hitl_interrupt_status`; do not create a second enum type.

- [x] **Step 4: Add conditional lifecycle repository methods.**

  Implement the following methods. All writes must commit and return `True` only when one row changed:

  ```python
  def mark_failed(self, interrupt_id: str, *, resolution_source: str) -> bool:
      now = datetime.now(timezone.utc)
      with self.session_factory() as db:
          result = db.execute(
              update(HITLInterrupt)
              .where(
                  HITLInterrupt.id == interrupt_id,
                  HITLInterrupt.status == HITLInterruptStatus.RESOLVING,
              )
              .values(
                  status=HITLInterruptStatus.FAILED,
                  resolution_source=resolution_source,
                  resolved_at=now,
                  updated_at=now,
              )
          )
          db.commit()
          return result.rowcount == 1

  def expire_pending_if_due_for_user(self, interrupt_id: str, user_id: UUID) -> bool:
      now = datetime.now(timezone.utc)
      with self.session_factory() as db:
          result = db.execute(
              update(HITLInterrupt)
              .where(
                  HITLInterrupt.id == interrupt_id,
                  HITLInterrupt.user_id == user_id,
                  HITLInterrupt.status == HITLInterruptStatus.PENDING,
                  HITLInterrupt.expires_at <= now,
              )
              .values(status=HITLInterruptStatus.EXPIRED, resolution_source="timeout", updated_at=now)
          )
          db.commit()
          return result.rowcount == 1
  ```

  Also add `get_by_id_for_user(interrupt_id, user_id)`. Keep `mark_expired()` for existing cleanup callers. In `_validate_and_claim_interrupt_resume()`, reject failed records before the existing resolved/resolving branch:

  ```python
  if record.status == HITLInterruptStatus.FAILED:
      raise CustomHTTPException(
          status_code=http_status.HTTP_409_CONFLICT,
          detail="This approval cannot be resumed after a failed continuation. Please send a new message.",
          error_code="INTERRUPT_FAILED",
      )
  if record.status in (HITLInterruptStatus.RESOLVED, HITLInterruptStatus.RESOLVING):
      raise CustomHTTPException(
          status_code=http_status.HTTP_409_CONFLICT,
          detail="This interrupt has already been resolved.",
          error_code="INTERRUPT_ALREADY_RESOLVED",
      )
  ```

- [x] **Step 5: Close every claimed-resume lifecycle branch.**

  Add a private best-effort helper in `MessageService`:

  ```python
  def _mark_claimed_interrupt_failed(self, interrupt_id: str | None, source: str) -> None:
      if self.hitl_interrupt_repository and interrupt_id:
          with contextlib.suppress(Exception):
              self.hitl_interrupt_repository.mark_failed(
                  interrupt_id, resolution_source=source
              )
  ```

  In `resume_message_creation_stream()`, call it before every post-claim terminal error yield (`stream_error`), after an unexpected exhausted source (`stream_incomplete`), in the `CancelledError`/`GeneratorExit` path (`client_disconnect`), and in the outer exception path (`stream_exception`). Add `"error_code": "INTERRUPT_FAILED"` and `"status_code": 500` to every post-claim V3 error event. Do not call it on validation failures before the claim.

  Preserve the existing `mark_resolved()` calls for `complete`. For a nested `interrupt`, persist the next paused message first, then mark the prior record resolved; if persistence raises, mark the prior record failed and emit `INTERRUPT_FAILED`.

- [x] **Step 6: Add the state schema, DI binding, and route.**

  Define:

  ```python
  class HitlInterruptStateResponse(_CamelModel):
      interrupt_id: str
      conversation_id: UUID
      status: Literal["pending", "resolving", "resolved", "failed", "expired"]
      expires_at: datetime
      updated_at: datetime
  ```

  Add `HITLInterruptRepository: container_ref.hitl_interrupt_repository` to `AppAutoInjector.setup_wiring_map`. The route must use `get_by_id_for_user()`, call `expire_pending_if_due_for_user()` only after an owned pending read, re-read the owned row, and map only the five public fields. A missing first or second read raises `CustomHTTPException(404, "HITL interrupt not found.", "INTERRUPT_NOT_FOUND")`.

- [x] **Step 7: Run lifecycle tests and migration verification.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m alembic upgrade head
  .venv\Scripts\python.exe -m pytest -q tests/test_hitl_api.py tests/test_message_service_event_streaming.py tests/test_client_tool_isolation.py tests/test_hitl_policy.py
  ```

  Expected: migration succeeds; all tests pass.

- [x] **Step 8: Commit the lifecycle/API change.**

  ```powershell
  git add app/models/hitl_interrupt.py app/alembic/versions/w7x8y9z0a1b2_add_failed_hitl_interrupt_status.py app/repositories/hitl_interrupt.py app/services/message_service.py app/schemas/hitl.py app/core/dependency_injection.py app/api/hitl.py tests/test_hitl_api.py tests/test_message_service_event_streaming.py tests/test_client_tool_isolation.py
  git commit -m "fix(hitl): finalize failed claimed resumes"
  ```

### Task 2: Preserve canonical codes across JSON, internal SSE, and AI SDK SSE

**Files:**

- Modify: `app/schemas/responses/api_response.py:16-24`
- Modify: `app/utils/exception_handler.py:19-93`
- Modify: `app/api/messages.py:30-106`
- Modify: `app/services/event_streaming/internal_sse.py:65-79`
- Modify: `app/services/event_streaming/ai_sdk_v6.py:60-105,360-395`
- Modify: `client_backend/api/messages.py:30-90`
- Create: `tests/test_message_stream_errors.py`
- Create: `tests/test_exception_handler.py`
- Modify: `tests/test_ai_sdk_v6_stream_contract.py`
- Modify: `tests/client_backend/test_messages.py`

- [x] **Step 1: Write failing transport-contract tests.**

  Add a full FastAPI exception-handler test:

  ```python
  def test_custom_http_error_envelope_retains_domain_code():
      app = FastAPI()
      register_exception_handlers(app)

      @app.get("/conflict")
      async def conflict():
          raise CustomHTTPException(409, "Already resolved", "INTERRUPT_ALREADY_RESOLVED")

      response = TestClient(app).get("/conflict")
      assert response.json() == {
          "success": False,
          "code": "INTERRUPT_ALREADY_RESOLVED",
          "message": "Already resolved",
      }
  ```

  Add adapter tests for a V3 error with snake-case fields and for a source that raises `CustomHTTPException`. Assert the internal projection emits `status_code`/`error_code`, while the AI SDK projection emits `statusCode`/`errorCode` and retains `errorText`. Add sidecar tests whose internal and AI SDK source iterators raise `ServerAPIError("Server error: 409", status_code=409, detail={"code": "INTERRUPT_CONFLICT", "message": "Interrupt was claimed by a concurrent request."})`.

- [x] **Step 2: Run the tests and confirm code metadata is dropped.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_exception_handler.py tests/test_message_stream_errors.py tests/test_ai_sdk_v6_stream_contract.py tests/client_backend/test_messages.py
  ```

  Expected: FAIL because `ApiResponse` discards `code` and the adapters emit only text.

- [x] **Step 3: Serialize the canonical envelope code.**

  Add the field to `ApiResponse`:

  ```python
  code: str | None = Field(None, description="Stable machine-readable error code")
  ```

  Keep `exclude_none=True` in exception handlers so successful responses and errors without a code remain backward compatible. Do not modify the sidecar's success envelope unless it has a code to forward; `proxy_server_request()` will relay the canonical JSON unchanged.

- [x] **Step 4: Add protocol-specific stream serializers.**

  In `app/api/messages.py`, use one `_stream_error_event(exc)` helper for producer and generator exceptions. It emits `error`, `status_code`, and `error_code` for `CustomHTTPException`.

  In `internal_sse.py`, when mapping `V3StreamEvent(type="error")`, copy only known optional `status_code`/`error_code` keys into the public error dict. In `ai_sdk_v6.py`, add helpers that map an exception or V3 error data to:

  ```python
  {"type": "error", "errorText": message, "statusCode": status_code, "errorCode": error_code}
  ```

  Include `statusCode` and `errorCode` only when present. Route both `iter_sse()`'s outer exception and `_error()` through these helpers. Preserve the existing `text-end`, `reasoning-end`, `finish-step`, `finish`, and `[DONE]` sequence.

- [x] **Step 5: Preserve the same identity at the sidecar boundary.**

  Replace the two opaque `str(exc)` branches in `_build_sse_response()` with:

  ```python
  def _upstream_stream_error_event(exc: Exception, *, ai_sdk: bool) -> dict[str, Any]:
      text_key = "errorText" if ai_sdk else "error"
      status_key = "statusCode" if ai_sdk else "status_code"
      code_key = "errorCode" if ai_sdk else "error_code"
      event: dict[str, Any] = {"type": "error", text_key: str(exc)}
      if isinstance(exc, ServerAPIError):
          if exc.status_code is not None:
              event[status_key] = exc.status_code
          detail = exc.detail if isinstance(exc.detail, dict) else {}
          event[text_key] = str(detail.get("message") or str(exc))
          code = detail.get("code")
          if code:
              event[code_key] = str(code)
      return event
  ```

  Never serialize `exc.detail` wholesale. Normal upstream SSE events already carry their metadata and must pass through unchanged.

- [x] **Step 6: Run all transport regressions.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_exception_handler.py tests/test_message_stream_errors.py tests/test_ai_sdk_v6_stream_contract.py tests/test_message_service_event_streaming.py tests/client_backend/test_messages.py tests/client_backend/test_server_api.py tests/client_backend/test_sse_keepalive.py
  ```

  Expected: PASS; AI SDK streams still end in `[DONE]` and internal streams retain heartbeat behavior.

- [x] **Step 7: Commit the transport change.**

  ```powershell
  git add app/schemas/responses/api_response.py app/utils/exception_handler.py app/api/messages.py app/services/event_streaming/internal_sse.py app/services/event_streaming/ai_sdk_v6.py client_backend/api/messages.py tests/test_exception_handler.py tests/test_message_stream_errors.py tests/test_ai_sdk_v6_stream_contract.py tests/client_backend/test_messages.py
  git commit -m "fix(hitl): preserve typed resume failures across clients"
  ```

### Task 3: Add the lifecycle proxy and pure reconciliation policy

**Files:**

- Modify: `client_backend/api/proxy.py:35-50`
- Create: `app/ui/hitl_recovery.py`
- Create: `tests/test_hitl_ui_recovery.py`
- Modify: `tests/client_backend/test_hitl_proxy.py`

- [x] **Step 1: Write failing pure-policy and proxy tests.**

  ```python
  def test_extracts_codes_from_stream_and_fastapi_error_envelopes():
      assert extract_error_code({"errorCode": "INTERRUPT_CONFLICT"}) == "INTERRUPT_CONFLICT"
      assert extract_error_code({"detail": {"code": "INTERRUPT_CONFLICT"}}) == "INTERRUPT_CONFLICT"

  def test_classifies_every_lifecycle_state():
      assert reconciliation_action("pending") == "restore_form"
      assert reconciliation_action("resolving") == "show_processing"
      assert reconciliation_action("resolved") == "refresh_history"
      assert reconciliation_action("failed") == "require_new_message"
      assert reconciliation_action("expired") == "require_new_message"
      assert reconciliation_action(None) == "require_new_message"
  ```

  Add a sidecar route test for `GET /hitl/interrupts/int-1`; assert it forwards exactly that path and no `params_override`/device context.

- [x] **Step 2: Run the focused tests.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_hitl_ui_recovery.py tests/client_backend/test_hitl_proxy.py
  ```

  Expected: FAIL because the module, classifier, and route do not exist.

- [x] **Step 3: Implement the pure helpers and sidecar route.**

  Create `app/ui/hitl_recovery.py` with these stable outputs:

  ```python
  from collections.abc import Mapping
  from typing import Any, Literal

  _DUPLICATE_CODES = frozenset({"INTERRUPT_ALREADY_RESOLVED", "INTERRUPT_CONFLICT"})

  def extract_error_code(payload: Mapping[str, Any]) -> str | None:
      for key in ("error_code", "errorCode", "code"):
          value = payload.get(key)
          if value not in (None, ""):
              return str(value)
      detail = payload.get("detail")
      return extract_error_code(detail) if isinstance(detail, Mapping) else None

  def is_recoverable_resume_conflict(payload: Mapping[str, Any]) -> bool:
      return extract_error_code(payload) in _DUPLICATE_CODES

  def should_suppress_pending_interrupt(
      interrupt_id: str | None, reconciling_interrupt_id: str | None
  ) -> bool:
      return bool(interrupt_id and interrupt_id == reconciling_interrupt_id)

  def reconciliation_action(status: str | None) -> Literal[
      "restore_form", "show_processing", "refresh_history", "require_new_message"
  ]:
      return {
          "pending": "restore_form",
          "resolving": "show_processing",
          "resolved": "refresh_history",
      }.get(status, "require_new_message")
  ```

  Keep `should_suppress_pending_interrupt()` equality-based. Add the sidecar `@router.get("/hitl/interrupts/{interrupt_id}")` route using `proxy_server_request(request, upstream_path=f"/hitl/interrupts/{interrupt_id}")` and normal local-session authentication.

- [x] **Step 4: Run focused policy/proxy tests.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_hitl_ui_recovery.py tests/client_backend/test_hitl_proxy.py
  ```

  Expected: PASS.

- [x] **Step 5: Commit the shared recovery foundation.**

  ```powershell
  git add app/ui/hitl_recovery.py client_backend/api/proxy.py tests/test_hitl_ui_recovery.py tests/client_backend/test_hitl_proxy.py
  git commit -m "feat(hitl): add shared lifecycle reconciliation policy"
  ```

### Task 4: Make the Streamlit approval flow no-replay and terminal-aware

**Files:**

- Modify: `demo.py:18-35,2781-2979,3500-3565,7332-7701,7904-7927`
- Modify: `tests/test_hitl_demo_panel.py`

- [x] **Step 1: Extend the static guard for the required seams.**

  Assert that the source imports `extract_error_code`, `is_recoverable_resume_conflict`, `reconciliation_action`, and `should_suppress_pending_interrupt`; calls `make_api_request("GET", f"/hitl/interrupts/{interrupt_id}", use_cache=False)` for lifecycle state; uses `hitl_resume_inflight_`; handles `failed`; and renders `statusCode`/`errorCode`-compatible errors.

- [x] **Step 2: Run the guard before implementation.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py tests/test_hitl_ui_recovery.py
  ```

  Expected: FAIL on the new assertions.

- [x] **Step 3: Add an uncached GET option and lifecycle helper.**

  Change the request signature without changing existing callers, then make this one conditional change so the already-existing direct `requests.Session.request` branch handles uncached GETs:

  ```diff
  -def make_api_request(method: str, endpoint: str, data: dict | None = None) -> dict:
  +def make_api_request(
  +    method: str, endpoint: str, data: dict | None = None, *, use_cache: bool = True
  +) -> dict:
  -    if method == "GET" and data is None:
  +    if method == "GET" and data is None and use_cache:
  ```

  The only executable behavior change is adding `and use_cache` to the existing GET-cache condition; do not add a second HTTP client or duplicate its established error/authentication handling.

  Then define:

  ```python
  def get_hitl_interrupt_state(interrupt_id: str) -> dict[str, Any] | None:
      response = make_api_request(
          "GET", f"/hitl/interrupts/{interrupt_id}", use_cache=False
      )
      return response.get("data") if response else None
  ```

  In `make_streaming_request()` parse the error JSON and add `status_code` plus `error_code=extract_error_code(payload)` to the internal error event. Use `_extract_api_error_message()` for its toast; do not echo error details wholesale.

- [x] **Step 4: Add interrupt-ID-scoped locks and reconciliation.**

  Add `_hitl_resume_lock_key()`, `_hitl_reconciliation_key()`, and `_clear_interrupt_ui_state()`. `_clear_interrupt_ui_state()` removes decisions, edit keys, the lock, `pending_interrupt`, and `interrupt_conversation_id`, then sets `conversation_messages_page = 0`; it deliberately does not remove the suppression marker.

  `_reconcile_interrupt()` must call the uncached lifecycle helper exactly once and branch on `reconciliation_action(status)`:

  - `restore_form`: remove lock and suppression marker.
  - `show_processing`: remove local decisions, store this interrupt as the suppression marker, and leave no enabled resume control.
  - `refresh_history`: clear paused UI, retain the marker, and set the notice `"Approval completed elsewhere; conversation refreshed."`.
  - `require_new_message`: clear paused UI, retain the marker, and set the notice `"This approval can no longer be resumed. Send a new message."`.

  It never calls `/messages/resume-interrupt`.

- [x] **Step 5: Lock the form before any resume call.**

  In `render_interrupt_approval_ui()`, render the suppression panel before actions. It exposes only **Check status** and **Return to chat**; both preserve the marker, and neither sends a resume request. Otherwise, calculate `resume_inflight` from the interrupt-ID lock and pass `disabled=resume_inflight` to **Submit Decisions**, **Approve All**, and **Cancel All**. Set the lock immediately before `_submit_interrupt_decisions()`.

  In `_submit_interrupt_decisions()`, retain the complete error event. A duplicate error calls `_reconcile_interrupt()` and reruns. An `INTERRUPT_FAILED` event or any event whose lifecycle read classifies as `require_new_message` clears paused UI, shows its actual error text, and does not restore the form. Pre-claim actionable errors remain visible; if their lifecycle state is still `pending`, only then restore the form.

- [x] **Step 6: Suppress stale rehydration after every terminal state.**

  In the paused-message recovery loop, derive `recovered_interrupt_id` and call `should_suppress_pending_interrupt(recovered_interrupt_id, marker)` before assigning `pending_interrupt`. Preserve rehydration for a different follow-up interrupt. Clear the suppression marker only when the lifecycle read returns `pending` for that exact ID or the session is reset/logout.

- [x] **Step 7: Run the Streamlit regression set.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py tests/test_hitl_ui_recovery.py tests/test_hitl_ui_decisions.py tests/test_hitl_decision_mapping.py tests/test_hitl_interrupt_payload_recovery.py
  ```

  Expected: PASS.

- [x] **Step 8: Commit the Streamlit recovery flow.**

  ```powershell
  git add demo.py tests/test_hitl_demo_panel.py
  git commit -m "fix(hitl): reconcile terminal approval outcomes in Streamlit"
  ```

### Task 5: Show skill readiness and manage encrypted local credentials

**Files:**

- Modify: `client_backend/api/skills.py:43-59`
- Modify: `demo.py:3550-3565,3890-3908,7139-7303`
- Modify: `tests/client_backend/test_skills_api.py`
- Modify: `tests/test_hitl_demo_panel.py`
- Reuse unchanged: `client_backend/services/skill_runtime/secrets.py`
- Reuse unchanged: `tests/client_backend/test_skill_secrets.py`

- [x] **Step 1: Write failing readiness and UI seam tests.**

  ```python
  def test_skill_summary_exposes_safe_setup_status(monkeypatch):
      monkeypatch.setattr(
          skills_api, "SkillRuntimeManager", lambda: _ReadinessManager("not_ready", "setup_required")
      )
      summary = skills_api._skill_summary(_Skill("python-skill", command_capable=False))
      assert summary["commandCapable"] is False
      assert summary["runtimeStatus"] == "not_ready"
      assert summary["setupStatus"] == "setup_required"
      assert "runtimeRoot" not in summary
  ```

  Add static assertions for password input, all three `/skills/{name}/secrets` verbs, value-key clearing, user-and-skill-scoped keys, and readiness guidance.

- [x] **Step 2: Run the tests and confirm fields/widgets are absent.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/client_backend/test_skills_api.py tests/test_hitl_demo_panel.py
  ```

  Expected: FAIL on `setupStatus` and credential UI seams.

- [x] **Step 3: Return only safe readiness fields.**

  Evaluate readiness once in `_skill_summary()` and add `setupStatus: readiness.setup_status`. Keep `commandCapable: readiness.status == "ready"`. Do not return runtime roots, command paths, setup logs, secret bindings, or secret values.

- [x] **Step 4: Add names-only credential helpers and widgets.**

  Add `get_skill_secrets`, `set_skill_secret`, and `delete_skill_secret` around the existing sidecar routes. In every skill card, render a **Local credentials** expander keyed by `current_user_id` plus `skill_name`. Use `st.text_input("Secret value", type="password", key=value_key)`, clear `value_key` immediately after a successful save, display only `f"{secret_name} configured"`, and remove `skill_secret_` state keys in `_clear_skill_hitl_session_state()` during logout. Never include the value in a toast, success text, cache key, test assertion, or log.

- [x] **Step 5: Render runtime guidance before approval controls.**

  Render `Command runtime: Ready` for ready skills; the instruction-only explanation for `instruction_only`; otherwise show `Command runtime is not ready (<setupStatus>)`. Keep the approval radio solely inside the ready branch for `skill::<name>::run_skill_command`.

- [x] **Step 6: Run the complete secret/runtime regression set.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/client_backend/test_skills_api.py tests/client_backend/test_skill_secrets.py tests/client_backend/test_skill_execution_engine.py tests/client_backend/test_skill_runtime_manager.py tests/test_hitl_demo_panel.py
  ```

  Expected: PASS.

- [x] **Step 7: Commit the skills-tab changes.**

  ```powershell
  git add client_backend/api/skills.py demo.py tests/client_backend/test_skills_api.py tests/test_hitl_demo_panel.py
  git commit -m "feat(skills): show readiness and manage local credentials"
  ```

### Task 6: Update the AI SDK frontend contract and operational documentation

**Files:**

- Modify: `plans/AI_SDK_FE_CONTRACT.md:3-10,177-289,452-501`
- Modify: `README.md`

- [x] **Step 1: Update the frontend endpoint and error contract.**

  Add `GET /hitl/interrupts/{interruptId}` to the endpoint table. Document its authenticated, owner-filtered response, all five status values, `404 INTERRUPT_NOT_FOUND`, and the instruction to use `cache: "no-store"` for reconciliation reads.

  Replace the untyped AI SDK error example with the optional `statusCode`/`errorCode` fields. State that duplicate conflicts are delivered as a terminal `200` SSE stream when discovered during stream iteration; JSON validation failures may still be ordinary non-stream HTTP errors. Retain the existing AI SDK terminal sequence.

- [x] **Step 2: Replace the duplicate-resume guidance with no-replay behavior.**

  In the Resume section, require an interrupt-ID-scoped lock before POST. For `INTERRUPT_ALREADY_RESOLVED` or `INTERRUPT_CONFLICT`, require one state GET and map `pending`, `resolving`, `resolved`, `failed`, and `expired` exactly as the public reconciliation table specifies. State explicitly that `failed` and `expired` require a new chat message and neither client may POST resume again.

- [x] **Step 3: Document Streamlit/operator behavior in the README.**

  Explain first-write-wins resume, the terminal `failed` state, manual status checks without polling, stale-paused-message suppression, and local credential boundaries (manual names, encrypted user/device-local values, no server synchronization or chat-history exposure).

- [x] **Step 4: Review both documents for contract consistency.**

  Verify these exact pairs agree: `failed` lifecycle status; `status_code`/`error_code` vs `statusCode`/`errorCode`; `INTERRUPT_FAILED`; duplicate-code allowlist; uncached state reads; and no-replay behavior.

- [x] **Step 5: Commit contract documentation.**

  ```powershell
  git add plans/AI_SDK_FE_CONTRACT.md README.md
  git commit -m "docs(hitl): define no-replay recovery for all clients"
  ```

### Task 7: Run the integrated regression and manual acceptance checks

**Files:**

- No production files; run verification after all prior tasks.

- [x] **Step 1: Run lint and format checks.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m ruff check app/models/hitl_interrupt.py app/repositories/hitl_interrupt.py app/services/message_service.py app/schemas/hitl.py app/api/hitl.py app/api/messages.py app/services/event_streaming/internal_sse.py app/services/event_streaming/ai_sdk_v6.py app/ui/hitl_recovery.py app/schemas/responses/api_response.py app/utils/exception_handler.py client_backend/api/messages.py client_backend/api/proxy.py client_backend/api/skills.py demo.py tests/test_hitl_api.py tests/test_message_service_event_streaming.py tests/test_message_stream_errors.py tests/test_ai_sdk_v6_stream_contract.py tests/test_hitl_ui_recovery.py tests/test_exception_handler.py tests/client_backend/test_messages.py tests/client_backend/test_hitl_proxy.py tests/client_backend/test_skills_api.py
  .venv\Scripts\python.exe -m ruff format --check app/models/hitl_interrupt.py app/repositories/hitl_interrupt.py app/services/message_service.py app/schemas/hitl.py app/api/hitl.py app/api/messages.py app/services/event_streaming/internal_sse.py app/services/event_streaming/ai_sdk_v6.py app/ui/hitl_recovery.py app/schemas/responses/api_response.py app/utils/exception_handler.py client_backend/api/messages.py client_backend/api/proxy.py client_backend/api/skills.py demo.py tests/test_hitl_api.py tests/test_message_service_event_streaming.py tests/test_message_stream_errors.py tests/test_ai_sdk_v6_stream_contract.py tests/test_hitl_ui_recovery.py tests/test_exception_handler.py tests/client_backend/test_messages.py tests/client_backend/test_hitl_proxy.py tests/client_backend/test_skills_api.py
  ```

  Expected: both commands exit `0`.

- [x] **Step 2: Run the cross-layer regression matrix.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_hitl_api.py tests/test_message_service_event_streaming.py tests/test_message_stream_errors.py tests/test_ai_sdk_v6_stream_contract.py tests/test_exception_handler.py tests/test_hitl_demo_panel.py tests/test_hitl_ui_recovery.py tests/test_hitl_ui_decisions.py tests/test_hitl_decision_mapping.py tests/test_hitl_interrupt_payload_recovery.py tests/test_hitl_policy.py tests/test_hitl_gate_policy.py tests/test_client_tool_isolation.py tests/client_backend/test_hitl_proxy.py tests/client_backend/test_messages.py tests/client_backend/test_server_api.py tests/client_backend/test_sse_keepalive.py tests/client_backend/test_skills_api.py tests/client_backend/test_skill_secrets.py tests/client_backend/test_skill_hitl.py tests/client_backend/test_skill_execution_engine.py tests/client_backend/test_skill_runtime_manager.py
  ```

  Expected: PASS. If a test fails, stop and use `superpowers:systematic-debugging` before changing the implementation.

- [ ] **Step 3: Perform manual acceptance with a disposable user.**

  1. Trigger a mutation-gated tool, approve in two Streamlit tabs, and confirm exactly one tool execution/resume occurs.
  2. Force the winning resumed graph to emit an error; confirm the durable row becomes `failed`, both clients clear the approval UI, and a new message is required.
  3. Repeat the duplicate scenario through `/ai/resume-interrupt`; confirm the terminal AI SDK error includes `statusCode` and `errorCode`, then reconcile from the lifecycle endpoint without another POST.
  4. Let an approval expire and change a client-device session/catalog; confirm each remains visibly actionable and no duplicate resume is attempted.
  5. Save `DEMO_TOKEN` for a ready skill, switch conversations, verify only `DEMO_TOKEN configured` is displayed, then sign in as another user and verify the binding is absent.

- [x] **Step 4: Commit final verification documentation if it changed.**

  ```powershell
  git status --short
  ```

  Expected: only intentionally uncommitted implementation work remains.

## Implementation Progress

- 2026-07-14 — Task 7, Step 1 corrective verification: User authorized resolving the committed lint/format baseline. Root cause: `demo.py` contains intentionally long embedded HTML/CSS/JavaScript strings plus one late import; eight listed files had formatter drift. Applied the project formatter to the exact Task 7 file list, moved the `stream_markdown` import into the module import block, and added a file-scoped Ruff `E501` exemption for generated UI content only. Verification: the exact Task 7 Ruff check reported `All checks passed!`; the exact format check reported `24 files already formatted`. Decision: preserve all generated UI string content while making the lint boundary explicit, rather than mechanically splitting embedded browser code.
- 2026-07-14 — Task 7, Step 2 re-verification: Re-ran the exact cross-layer regression matrix after lint/format corrections; all 178 tests passed in 9.82 seconds. The only output was the pre-existing `langchain-community` deprecation warning. Decision: formatting/import changes do not alter the resilient-HITL or credential behavior covered by the matrix.
- 2026-07-14 — Task 7, Step 4 final state: `git diff --check` was clean; staged-only formatting/import cleanup was committed as `4e5411d` (`style: satisfy integration lint checks`). Fresh `git status --short` reports only this implementation-progress document as uncommitted. Decision: retain the log outside the source-formatting commit so it remains the final audit record; Step 3 remains blocked pending an interactive authenticated local browser session and healthy service dependencies.

- 2026-07-14 — Task 7, Step 4: `git status --short` reported exactly one uncommitted file: `M docs/superpowers/specs/2026-07-14-resilient-hitl-and-skill-secrets-implementation-plan.md`. This is the required verification-progress record; no implementation files are modified. Decision: per the verification assignment, do not stage this documentation or create a commit, so the commit-specific checkbox remains unchecked.
- 2026-07-14 — Task 7, Step 3: Manual acceptance was attempted using local-only, read-only checks; no user, secret, approval, resume request, tool invocation, or external action was created. `GET` to local API (`:8000`), sidecar (`:8100`), and Streamlit (`:8501`) endpoints succeeded. Interactive acceptance is blocked because the session has no in-app browser surface (`agent.browsers.list()` returned `[]`); additionally, `/health/all` is `degraded` because Qdrant health errors with `'CollectionInfo' object has no attribute 'vectors_count'`, and sidecar `/health` is `degraded` with `server_connected: false`. Automated coverage that passed in Step 2 includes: duplicate/no-replay and terminal UI recovery (`test_hitl_demo_panel.py`, `test_hitl_ui_recovery.py`); failed durable resumes and typed terminal events (`test_message_service_event_streaming.py`, `test_message_stream_errors.py`); AI SDK resume route/stream behavior (`test_ai_sdk_v6_stream_contract.py`, `tests/client_backend/test_messages.py`, `test_server_api.py`); expiry/device/catalog isolation (`test_hitl_api.py`, `test_client_tool_isolation.py`, `test_hitl_policy.py`, `test_hitl_gate_policy.py`); and scoped secrets/approval-before-resolution (`test_skill_secrets.py`, `test_skills_api.py`, `test_skill_hitl.py`, `test_skill_execution_engine.py`). Decision: do not fabricate the five interactive acceptance outcomes; leave Step 3 unchecked pending a disposable authenticated UI session with a healthy connected sidecar and Qdrant check.
- 2026-07-14 — Task 7, Step 2: The exact cross-layer pytest matrix passed: 178 passed in 9.86s. The only output aside from progress was the pre-existing `langchain-community` deprecation warning from `app/services/document_processing_service.py:19`. Decision: functional regression coverage is green; this does not override the independent Step 1 Ruff/format blocker.
- 2026-07-14 — Task 7, Step 1: The exact Ruff check failed with 47 `demo.py` findings: one `E402` late module import at lines 2338–2340 and 46 `E501` lines over 100 characters. The exact formatter check failed: it would reformat `app/models/hitl_interrupt.py`, `app/repositories/hitl_interrupt.py`, `app/services/message_service.py`, `client_backend/api/skills.py`, `demo.py`, `tests/client_backend/test_skills_api.py`, `tests/test_hitl_api.py`, and `tests/test_message_service_event_streaming.py` (the remaining 16 listed files were formatted). Systematic-debugging Phase 1 evidence: `git status --short` showed only this plan document modified; representative lint lines blame to commits `92392bad` (2025-12-08), `325e98be` (2026-02-24), `83f3941b` (2026-04-06), `06cdb578` (2026-05-21), and `ed3593bd` (2026-06-23), so the failures are committed baseline state rather than Task 7 changes. Decision: preserve the verification-only/no-production-edit scope and leave Step 1 unchecked; these failures prevent a clean Task 7 quality claim.
- 2026-07-14 — Task 1, Step 1: Added lifecycle API, conditional repository, claimed-stream failure/cancellation, and failed-resume rejection regressions. Verification: `.venv\Scripts\python.exe -m pytest -q` against the five new tests returned the expected 5 failures: absent `FAILED`, absent `mark_failed`, and unclosed claimed-failure paths. Decision: API fixtures create owned conversation-backed interrupts and delete interrupts before conversations/users; stream tests observe lifecycle ordering without testing persistence internals.
- 2026-07-14 — Task 1, Step 2: Ran `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_api.py tests/test_message_service_event_streaming.py tests/test_client_tool_isolation.py`; result was the expected 5 new failures with 17 existing tests passing. The failures confirm that the status, repository transition, failed-resume rejection, route, and stream terminal handling are not yet implemented.
- 2026-07-14 — Task 1, Step 3: Added `HITLInterruptStatus.FAILED` and migration `w7x8y9z0a1b2`. Verification: enum assertion and `.venv\Scripts\python.exe -m alembic heads` both exited 0; Alembic reported `w7x8y9z0a1b2 (head)`. Decision: retain the existing `hitl_interrupt_status` type and make the downgrade intentionally forward-only.
- 2026-07-14 — Task 1, Step 4: Added conditional `mark_failed`, owner-filtered conditional expiry, and owner-filtered read methods; failed claimed resumes now return `409 INTERRUPT_FAILED` before resolved/resolving handling. Verification: the failed-resume regression passed and a repository API-surface assertion exited 0. Decision: defer the database-backed transition assertion until Step 7 applies the required enum migration, preserving the prescribed migration order.
- 2026-07-14 — Task 1, Step 5: Added best-effort claimed-failure handling for stream errors, incomplete streams, cancellation, outer exceptions, and nested-interrupt persistence failures; all post-claim V3 error events now carry `INTERRUPT_FAILED`/500. Verification: `.venv\Scripts\python.exe -m pytest -q` for the claimed upstream-error, client-disconnect, and successful-resume tests passed 3/3. Decision: persist a nested paused message before resolving the prior record; a persistence exception marks that prior record failed with the typed terminal event.
- 2026-07-14 — Task 1, Step 6: Added the five-field camel-case lifecycle schema, repository DI binding, and owner-filtered state route with owned-pending-only expiry/re-read. Verification: existing HITL settings API regressions passed 3/3 and a route/schema assertion exited 0. Decision: use the same `404 INTERRUPT_NOT_FOUND` for absent initial and post-expiry reads, and keep the database-backed failed-state route assertion for Step 7 after migration.
- 2026-07-14 — Task 1, Step 7: Ran `.venv\Scripts\python.exe -m alembic upgrade head` successfully, then `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_api.py tests/test_message_service_event_streaming.py tests/test_client_tool_isolation.py tests/test_hitl_policy.py`; all 33 tests passed (one pre-existing third-party deprecation warning). Decision: the route/repository lifecycle tests now execute against the migrated PostgreSQL enum.
- 2026-07-14 — Task 1, Step 8: Staged only the ten specified Task 1 implementation/test files, verified the staged diff with `git diff --cached --check`, and committed `da7b1ce` (`fix(hitl): finalize failed claimed resumes`). Decision: leave this implementation-progress log unstaged as requested; no Task 2 work was started.
- 2026-07-14 — Task 1 corrective review P1 #1: Root cause was post-claim setup (`_get_conversation_context`, custom-agent validation/audit/state resolution) executing before the stream `try`, so exceptions escaped without a terminal transition. Added the minimal setup-exception regression; RED: the test raised `RuntimeError("setup failed")`; GREEN: the same command passed after moving only post-claim setup under the existing terminal-error handler. Decision: retain pre-claim validation outside the handler, but treat all work after a successful claim as `stream_exception` and emit `INTERRUPT_FAILED`/500. Review loop: P1 #1 green; P1 #2 pending.
- 2026-07-14 — Task 1 corrective review P1 #2: Root cause was resolving the claimed interrupt before awaiting completed-message persistence; a persistence exception then left the conditional `mark_failed` transition ineligible. Added the stateful completion-persistence regression; RED: state remained `resolved`; GREEN: the same command passed after persisting first and resolving only after success. Decision: completion persistence is the durable success boundary; its exception remains eligible for `RESOLVING -> FAILED` with `stream_exception` and a typed error. Review loop: P1 #1/P1 #2 green; P1 #3 pending.
- 2026-07-14 — Task 1 corrective review P1 #3: Root cause was `_persist_interrupt_bot_message` swallowing follow-up `HITLInterruptRepository.create` failures, so the nested caller saw apparent success and resolved the prior record. Added the durable-follow-up creation regression; RED: the prior state became `resolved` after the helper logged the failure; GREEN: the same command passed after allowing only claimed nested callers to require durable-creation propagation. Decision: preserve best-effort behavior for existing non-nested callers, but raise into the nested terminal handler when `require_durable_interrupt=True`; emit its typed failure using a fresh message ID to avoid colliding with a previously persisted paused message. Review loop: P1 #1/P1 #2/P1 #3 green; full Task 1 verification pending.
- 2026-07-14 — Task 1 corrective review final verification: `.venv\Scripts\python.exe -m alembic upgrade head` succeeded and the required lifecycle suite (`tests/test_hitl_api.py`, `tests/test_message_service_event_streaming.py`, `tests/test_client_tool_isolation.py`, `tests/test_hitl_policy.py`) passed 36/36, with only the pre-existing third-party deprecation warning. Review loop: all three P1 corrections verified green; corrective commit pending.
- 2026-07-14 — Task 1 quality review Important #1: Diagnostic comparison confirmed `MessageRepository.create` commits the paused message before follow-up interrupt creation; a durable-create failure therefore left a soft-recoverable orphan while nested SSE advertised the old ID but carried a new error message. Extended the nested durable-failure regression with explicit, distinct paused/error IDs; RED: it failed because `event.message_id` was the old reserved ID; GREEN: the same command passed after soft-deleting the paused message before propagation and advertising `error_message.id`. Decision: only required durable nested persistence deletes its committed paused placeholder; the successful nested path remains persist-next-before-resolve-prior. Review loop: Important #1 green; Important #2 pending.
- 2026-07-14 — Task 1 quality review Important #2: Diagnostic comparison found raw-error and nested-failure branches already call `clear_paused_for_conversation`, while incomplete, client-disconnect, and outer-exception terminal branches only marked the row failed and retained the paused lock. Added focused registry-recorder assertions; RED: all three tests failed with no cleanup calls; GREEN: all three passed after adding the same cleanup at each terminal branch. Decision: release paused generation locks before durable failure marking for every post-claim terminal failure path; successful complete/nested behavior remains unchanged. Review loop: Important #1/Important #2 green; Minor #3 pending.
- 2026-07-14 — Task 1 quality review Minor #3: RED: `.venv\Scripts\python.exe -m ruff check app/api/hitl.py app/core/dependency_injection.py` reported two I001 import-order errors. GREEN: the same command reported `All checks passed!` after alphabetizing the new HITL imports. Decision: retain only formatter-compliant ordering; no behavior changed. Review loop: Important #1/Important #2/Minor #3 green; Minor #4 pending.
- 2026-07-14 — Task 1 quality review Minor #4: Added foreign-owned expired-row and owned-expired-row lifecycle API regressions. RED: the foreign test returned 404 but the lightweight test app dropped the `CustomHTTPException` domain code because shared envelope serialization is Task 2 scope; the owned lazy-expiry test already passed. GREEN: both passed after the test harness mirrored the normal CustomHTTPException code envelope without changing Task 2 production response serialization. Decision: verify Task 1 route behavior through the code carried by the raised domain exception; foreign reads return 404 and leave even expired foreign records pending, while owned pending records lazily expire and expose only the five public fields. Review loop: all quality findings green; full Task 1 verification pending.
- 2026-07-14 — Task 1 quality review final verification: Ruff on all corrective-pass Task 1 files passed after wrapping two test-only E501 lines. `.venv\Scripts\python.exe -m alembic upgrade head` succeeded; the required lifecycle suite passed 39/39 with only the pre-existing third-party deprecation warning. Review loop: Important #1/#2 and Minor #3/#4 all green; narrow corrective commit pending.
- 2026-07-14 — Task 2, Step 1: Added the canonical JSON exception-envelope regression; internal V3-to-SSE and source-exception projections; AI SDK V3/source-exception projections; and internal/AI SDK sidecar structured-upstream-error regressions. Focused RED verification produced the expected 7 failures: JSON omitted `code`; internal and AI SDK adapters omitted status/code metadata; and sidecar errors flattened `ServerAPIError` to `str(exc)`. Decision: assert exact public envelopes so unknown detail fields cannot leak, and retain `[DONE]` assertions for AI SDK terminal compatibility.
- 2026-07-14 — Task 2, Step 2: Ran `.venv\Scripts\python.exe -m pytest -q tests/test_exception_handler.py tests/test_message_stream_errors.py tests/test_ai_sdk_v6_stream_contract.py tests/client_backend/test_messages.py`; result was the expected 7 new contract failures and 16 existing tests passing (plus the pre-existing third-party deprecation warning). Decision: the failures cleanly identify the required implementation seams without unrelated regressions.
- 2026-07-14 — Task 2, Step 3: Added optional `ApiResponse.code`; `.venv\Scripts\python.exe -m pytest -q tests/test_exception_handler.py` passed 1/1. Decision: `register_exception_handlers()` already passes its canonical codes and serializes with `exclude_none=True`, so no handler change was needed; successful and code-less envelopes remain backward compatible.
- 2026-07-14 — Task 2, Step 4: Added one internal stream exception projector for producer/generator failures, whitelisted `status_code`/`error_code` in the internal V3 adapter, and dedicated AI SDK exception/V3-data projectors with camel-case optional metadata. Verification: `.venv\Scripts\python.exe -m pytest -q tests/test_message_stream_errors.py tests/test_ai_sdk_v6_stream_contract.py` passed 17/17 (plus the pre-existing third-party deprecation warning). Decision: metadata is omitted when absent, while AI SDK terminal sequencing remains centralized in the existing termination paths.
- 2026-07-14 — Task 2, Step 5: Replaced sidecar's opaque exception branches with one dialect-aware `ServerAPIError` projector. Verification: `.venv\Scripts\python.exe -m pytest -q tests/client_backend/test_messages.py` passed 5/5. Decision: use only `detail.message` and `detail.code` when `detail` is a mapping; never forward arbitrary detail content, and leave ordinary upstream SSE events unchanged.
- 2026-07-14 — Task 2, Step 6: Ran `.venv\Scripts\python.exe -m pytest -q tests/test_exception_handler.py tests/test_message_stream_errors.py tests/test_ai_sdk_v6_stream_contract.py tests/test_message_service_event_streaming.py tests/client_backend/test_messages.py tests/client_backend/test_server_api.py tests/client_backend/test_sse_keepalive.py`; all 42 tests passed. AI SDK terminal coverage still asserts `[DONE]`, while sidecar keepalive and internal service-stream coverage remain green. The only warning was the pre-existing third-party `langchain-community` deprecation warning.
- 2026-07-14 — Task 2, pre-commit quality check: Ruff initially reported two E501 test assertions and formatter drift from LF patch sections in CRLF files. Root cause: formatting only; no behavioral defect. After formatter normalization, `ruff check` passed, `ruff format --check` reported all 10 Task 2 files formatted, and the full Task 2 transport suite passed again (42/42, same pre-existing third-party warning).
- 2026-07-14 — Task 2, pre-commit compatibility correction: Read-only review identified that FastAPI response-model serialization emitted the newly optional `code` as `null` on ordinary success envelopes, unlike the explicit `exclude_none=True` exception-handler path. RED: a minimal `response_model=ApiResponse` route returned `code: null` while the typed exception envelope remained correct. GREEN: added an `ApiResponse` wrap serializer that removes only absent `code`; the focused success/error contracts passed 2/2. Decision: preserve the established `data`/`error` null fields and omit only the newly introduced field, retaining backwards compatibility without weakening typed error serialization.
- 2026-07-14 — Task 2, Step 7: Staged only the nine changed Task 2 implementation/test files (the listed exception-handler file required no edit), verified `git diff --cached --check`, and committed `54b8631` (`fix(hitl): preserve typed resume failures across clients`). Final verification before staging: Ruff check/format passed and the complete Task 2 transport suite passed 43/43, with only the pre-existing third-party deprecation warning. Decision: leave this shared implementation-progress log unstaged as requested; no Task 3 work was started.
- 2026-07-14 — Task 2, post-commit corrective review: Root cause was the pre-existing `CustomHTTPException` handler fallback `exc.error_code or "custom_error"`, which became externally visible once `ApiResponse.code` was serialized. RED: the new no-domain-code FastAPI route emitted `code: "custom_error"`. GREEN: changed the handler to pass only `exc.error_code`; `exclude_none=True` now omits the field while typed codes remain intact. Focused exception-handler verification passed 3/3. Decision: `custom_error` is not a canonical domain code, so code-less custom errors stay backward compatible rather than receiving a synthetic machine contract.
- 2026-07-14 — Task 2, post-commit corrective commit: Verified Ruff check/format, then the complete Task 2 transport suite passed 44/44 with only the pre-existing third-party deprecation warning. Staged only `app/utils/exception_handler.py` and `tests/test_exception_handler.py`, verified `git diff --cached --check`, and committed `284d6f3` (`fix(hitl): omit synthetic custom error codes`). Decision: retain the original Task 2 commit and add this narrow corrective commit rather than rewriting history.
- 2026-07-14 — Task 3, Step 1: Added pure recovery-policy tests for canonical/nested error-code extraction, duplicate-conflict classification, lifecycle actions, and exact interrupt-ID suppression, plus the sidecar lifecycle-read proxy regression. Verification: `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_ui_recovery.py tests/client_backend/test_hitl_proxy.py` failed as expected during collection with `ModuleNotFoundError: app.ui.hitl_recovery`. Decision: keep reconciliation as a dependency-free UI policy module and verify the proxy never opts into device-specific parameter overrides.
- 2026-07-14 — Task 3, Step 2: Ran `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_ui_recovery.py tests/client_backend/test_hitl_proxy.py`; it failed as expected with the missing `app.ui.hitl_recovery` module. Decision: the focused command is intentionally shared by the policy and proxy tests so the new public imports fail before any implementation can mask the absence of the required recovery foundation.
- 2026-07-14 — Task 3, Step 3: Added `app/ui/hitl_recovery.py` with recursive canonical-code extraction, the two-code duplicate allowlist, equality-based suppression, and the stable lifecycle-action mapping; added authenticated `GET /hitl/interrupts/{interrupt_id}` sidecar forwarding with no device override. Verification: `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_ui_recovery.py tests/client_backend/test_hitl_proxy.py` passed 12/12. Decision: unknown, failed, and expired statuses share `require_new_message`, while only pending/resolving/resolved map to recoverable in-place UI actions.
- 2026-07-14 — Task 3, Step 4: Re-ran `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_ui_recovery.py tests/client_backend/test_hitl_proxy.py`; all 12 tests passed. Decision: retain the focused suite as the handoff proof because it exercises both framework-independent recovery decisions and the authenticated sidecar route boundary.
- 2026-07-14 — Task 3, Step 5: Ran Ruff check, Ruff format check, the focused 12-test suite, and `git diff --cached --check`; all passed before staging only the four Task 3 files. Committed `cfbf35f` (`feat(hitl): add shared lifecycle reconciliation policy`), then re-ran the focused suite (12/12), `git show --check`, and `git status --short`; the commit contains exactly the four Task 3 files and this plan remains the only unstaged file. Decision: keep the implementation-progress document deliberately outside the commit so subsequent tasks can continue recording progress without rewriting Task 3 history.

- 2026-07-14 — Task 4, Step 1: Added a static Streamlit recovery guard for the shared policy imports, exact uncached lifecycle helper call, interrupt-scoped lock prefix, terminal failed state, and typed internal error fields. Verification: `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py` produced the expected one failure because `demo.py` does not yet import the recovery helpers. Decision: use AST to make the lifecycle request expression exact while keeping the remaining guard independent of Streamlit runtime setup.
- 2026-07-14 — Task 4, Step 2: Ran `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py tests/test_hitl_ui_recovery.py`; result was the expected 1 new static-guard failure with 14 tests passing. Decision: this establishes the recovery policy module itself is green and isolates the pre-implementation gap to `demo.py`.
- 2026-07-14 — Task 4, Step 3: Added opt-out caching to the existing GET condition, an exact uncached lifecycle helper, and typed HTTP SSE errors that parse only the JSON envelope/message. Verification: the static suite advanced to the expected Step 4-only missing `hitl_resume_inflight_` assertion; Ruff also identified the three recovery imports as temporarily unused, alongside pre-existing full-file lint violations. Decision: route uncached GET through the existing authenticated request branch and use `_extract_api_error_message()` so raw error details never reach the toast.
- 2026-07-14 — Task 4, Step 4: Added interrupt-scoped lock/marker keys, a marker-preserving paused-UI cleanup helper, and one-read lifecycle reconciliation with each policy action and exact terminal notices. Verification: `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py tests/test_hitl_ui_recovery.py` passed 15/15. Decision: processing retains the paused interrupt plus suppression marker while terminal outcomes remove the paused UI and keep the marker; reconciliation never opens a resume stream.
- 2026-07-14 — Task 4, Step 5: Added a suppression-only status/chat panel, interrupt-scoped disabled submit controls, and pre-submit locks; resume errors now retain the full event for duplicate reconciliation, terminal cleanup, and actual-message display. Verification: `.venv\Scripts\python.exe -m py_compile demo.py` and `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py tests/test_hitl_ui_recovery.py` both passed (15 tests). Decision: preserve local choices for an actionable pre-claim error while clearing them only after non-error completion or terminal lifecycle handling.
- 2026-07-14 — Task 4, Step 6: Recovery now derives each paused interrupt ID and suppresses only an exact terminal marker before assigning paused UI; marker clearing is restricted to exact-ID pending reconciliation or session reset. Verification: `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py tests/test_hitl_ui_recovery.py tests/test_hitl_ui_decisions.py tests/test_hitl_decision_mapping.py tests/test_hitl_interrupt_payload_recovery.py` passed 35/35. Decision: a new follow-up interrupt with a different ID still rehydrates normally, while terminal notices render after stale paused UI is skipped.
- 2026-07-14 — Task 4, Step 7: Re-ran the required Streamlit regression command; all 35 tests passed in 5.35 seconds. Decision: preserve the focused suite as the Task 4 handoff proof because it spans the static UI seams, pure lifecycle policy, approval decision UI, decision mapping, and live interrupt payload recovery.
- 2026-07-14 — Task 4, pre-commit typed-stream correction: A new 409 response regression failed as expected because `requests.Response` is falsey for HTTP errors, so `if http_error.response` erased its status. GREEN: changed only that guard to `is not None`; the focused event test passed 1/1, and the full Task 4 regression suite then passed 36/36. Decision: retain the response JSON's canonical code and public message while projecting the actual HTTP status to the internal SSE event.
- 2026-07-14 — Task 4, Step 8: Staged only `demo.py` and `tests/test_hitl_demo_panel.py`, verified `git diff --cached --check`, and committed the final reviewed change as `1abdb3d` (`fix(hitl): reconcile terminal approval outcomes in Streamlit`). Decision: leave this implementation-progress log unstaged as requested; no Task 5 work was started.
- 2026-07-14 — Task 4, post-commit review correction: Review found terminal error branches cleared paused state but returned during the prior render, delaying access to the new-message input. RED: the new terminal-resume regression recorded the actual error but no rerun. GREEN: terminal branches now persist that actual notice and rerun after cleanup; Ruff checks, compilation, and the required Streamlit suite passed 37/37. Decision: a refreshed normal-chat view immediately displays the real terminal error and exposes the message composer without replaying the approval.
- 2026-07-14 — Task 4, strict-review correction #1: Root cause: every non-`INTERRUPT_FAILED` error unconditionally called `_reconcile_interrupt()`, so a pre-claim device mismatch read its still-pending row and restored the form. RED: the new mismatch regression recorded one reconciliation call. GREEN: only `is_recoverable_resume_conflict()` errors reconcile; other visible failures release their local lock without a lifecycle read or replay. Verification: `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py -k nonduplicate_resume_error` passed 1/1. Decision: preserve the paused approval and actual failure message for a nonduplicate pre-claim error instead of treating it as a duplicate conflict.
- 2026-07-14 — Task 4, strict-review correction #2: Root cause: `reset_conversation_state()` removed only the reconciliation marker/notice and left paused approval, decision/edit, and ID-scoped lock state in the session. RED: the reset regression retained `pending_interrupt`. GREEN: reset now invokes the existing all-lock paused-UI cleanup before generic conversation reset state. Verification: `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py -k reset_conversation_clears_all_interrupt` passed 1/1. Decision: reset and logout cannot rehydrate a stale approval or transfer its controls to another conversation/user.
- 2026-07-14 — Task 4, strict-review correction #3: Root cause: `resume_inflight` was calculated after the per-tool decision/edit controls, leaving them active while the batch controls were disabled. RED: AST guard found the assignment after the tool loop. GREEN: calculate the lock before any approval control and pass it to change, approve, edit, reject, and edit-form save/cancel controls. Verification: `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py -k approval_controls_are_disabled` passed 1/1. Decision: every control capable of changing decisions is frozen for the exact interrupt as soon as a resume is in flight.
- 2026-07-14 — Task 4, strict re-review correction #4: Root cause: an `INTERRUPT_EXPIRED` 410 took the generic visible-error path, which removed only the lock and left paused state eligible to replay. RED: the expiry regression had no rerun and retained `pending_interrupt`. GREEN: known failed/expired/not-found terminal codes (and 410 responses) clear paused UI, preserve the exact suppression marker, store the actual error notice, and rerun without lifecycle reconciliation or resume replay. Verification: `.venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py -k "expired_resume_error or approval_controls_are_disabled"` passed 2/2. Decision: only a known duplicate conflict reconciles; durable-terminal preclaim errors return immediately to normal chat with their real message.
- 2026-07-14 — Task 4, strict re-review correction #5: Root cause: the editable tool-arguments textarea remained mutable despite all surrounding action buttons respecting `resume_inflight`. RED: the static control guard found no textarea `disabled` argument. GREEN: the textarea now receives `disabled=resume_inflight`; the guard verifies it along with every decision-changing button. Verification: the same focused 2-test command passed. Decision: edits cannot race an already-started resume, including text already open in the edit form.
- 2026-07-14 — Task 4, quality-review runtime matrix: Added parameterized duplicate-conflict coverage through `_submit_interrupt_decisions()` and the real reconciliation helper for pending, resolving, resolved, failed, expired, and unavailable lifecycle reads. RED: failed/expired/unavailable correctly cleared UI but the duplicate branch overwrote the required terminal notice with `"Already resolved"`. GREEN: retain the lifecycle helper's exact notice; the matrix passed 6/6 and verifies one uncached state read, one initial resume request only, no replay, plus lock/marker/paused-UI outcomes per state. Decision: duplicate conflict text is not a replacement for the durable lifecycle result presented to the user.

- 2026-07-14 — Task 5, Steps 1–2: Added the safe-readiness API regression and static credential/runtime UI seams. RED verification with `.venv\Scripts\python.exe -m pytest -q tests/client_backend/test_skills_api.py tests/test_hitl_demo_panel.py` produced the expected two failures: `setupStatus` was absent and the credential helpers/widgets did not exist. Decision: static UI tests inspect only names, API verbs, password input, user-and-skill-scoped widget keys, value clearing, logout cleanup, and readiness guidance; they never embed a credential value.
- 2026-07-14 — Task 5, Steps 3–5: `_skill_summary()` now evaluates readiness once and exposes only `setupStatus` alongside the existing safe runtime status/capability fields. The Skills tab now wraps the established sidecar names-only credential routes in user-and-skill-scoped widgets, clears the password key from a Streamlit button callback only after a successful save, and removes all `skill_secret_` state on logout. Decision: callback-based saving permits immediate password-field clearing before the widget is instantiated for the rerun; configured-name/status text never contains a secret value. Ready, instruction-only, and not-ready runtime guidance renders before the ready-only approval radio.
- 2026-07-14 — Task 5, Step 6: `.venv\Scripts\python.exe -m pytest -q tests/client_backend/test_skills_api.py tests/client_backend/test_skill_secrets.py tests/client_backend/test_skill_execution_engine.py tests/client_backend/test_skill_runtime_manager.py tests/test_hitl_demo_panel.py` completed with 56 passed in 3.22s. Follow-up focused verification after the static-test refactor completed with 25 passed; `ruff check` with the repository’s pre-existing `demo.py` E402/E501 exclusions reported no further findings, and `git diff --check` reported no whitespace errors. Decision: retain the existing encrypted `SkillSecretStore` and route implementation unchanged; Task 5 only consumes their names-only contract.
- 2026-07-14 — Task 5, Step 7: Re-ran the complete Task 5 secret/runtime suite with 56 passed in 3.35s, then staged only `client_backend/api/skills.py`, `demo.py`, `tests/client_backend/test_skills_api.py`, and `tests/test_hitl_demo_panel.py`; `git diff --cached --check` was clean. Committed `4c9eff0` (`feat(skills): show readiness and manage local credentials`). Decision: leave this implementation-progress log unstaged, preserving the sole dirty plan file and not starting Task 6.
- 2026-07-14 — Task 5 post-commit quality correction: Root cause was the `make_api_request()` `unauthenticated` envelope path resetting auth/login state directly, unlike explicit logout, so user-and-skill-scoped `skill_secret_` widget state could outlive an expired session. RED: a behavioral POST-envelope regression left both credential name/value keys present; a companion ordinary-failed-save characterization retained the value-key identity. GREEN: invoke the existing `_clear_skill_hitl_session_state()` immediately before the unauthenticated auth reset; the focused pair passed 2/2 and the full Task 5 suite passed 58/58. Decision: cleanup is terminal-auth-transition-only; ordinary failed saves retain their password field so a user can correct and retry without re-entering it.
- 2026-07-14 — Task 5 second post-commit quality correction: The actual sidecar `require_local_session` failure is HTTP 401, which `make_api_request()` handled in its earlier `HTTPError` return path and therefore never reached the corrected envelope branch. RED: a behavioral 401 regression retained the expired auth/user state and both `skill_secret_` keys; added runtime save-success coverage confirms the value key clears only when the POST helper returns success, while the failed-save test preserves it. GREEN: route both envelope and 401 auth failures through `_transition_to_login()`, which clears user-scoped widget state before resetting auth/login fields; the focused set passed 4/4 and the full Task 5 suite passed 60/60. Decision: use one terminal-auth-transition helper so HTTP and envelope paths cannot diverge; ordinary failed saves continue to preserve the password widget for retry.
- 2026-07-14 — Task 5 third post-commit quality correction: Cached GET calls return from the `status_code >= 400` branch before reaching either prior auth transition, and `get_skill_secrets()` uses that cached path. RED: a behavioral cached-GET 401 regression retained the expired auth/user state and both credential widget keys; the combined auth/save focused set failed 1/5. GREEN: call `_transition_to_login()` in the cached 401 branch before its early return; all three auth forms plus the save-success/failure contracts passed 5/5 and the full Task 5 suite passed 61/61. Decision: every make-api-request 401 exit uses the one shared terminal transition; no Task 6 work started.

- 2026-07-14 — Task 6, Step 1: Updated the AI SDK contract endpoint table and lifecycle-read section to specify authenticated owner filtering, the five lifecycle statuses, HTTP `404 INTERRUPT_NOT_FOUND`, and client `cache: "no-store"`. The AI SDK error example and prose now specify optional camel-case `statusCode`/`errorCode` as the projection of internal `status_code`/`error_code`; duplicate conflicts discovered after streaming starts remain terminal `200` SSE responses, while pre-stream validation stays JSON HTTP. Decision: retain the established terminal AI SDK sequence and expose no decisions, arguments, device/session data, user IDs, or secrets from the reconciliation read.
- 2026-07-14 — Task 6, Step 2: Replaced ambiguous duplicate guidance with first-write-wins recovery: an interrupt-ID lock precedes POST, only `INTERRUPT_ALREADY_RESOLVED` and `INTERRUPT_CONFLICT` trigger exactly one uncached state GET, and no POST is replayed automatically. The public table now gives separate `pending`, `resolving`, `resolved`, `failed`, and `expired` actions, plus the unavailable/404 fallback. Decision: `pending` may deliberately restore the form after its one read, whereas `failed` and `expired` retain exact-ID stale-message suppression, require a new message, and can never be resumed again.
- 2026-07-14 — Task 6, Step 3: Added README operator guidance for durable first-write-wins recovery, terminal `failed`, one manual non-polling status action, and exact-ID stale paused-message suppression. Documented local skill credentials as manually named, encrypted per-skill/per-user/per-machine values that return names only, never synchronize, and never enter chat history. Decision: distinguish internal snake-case stream metadata from the AI SDK camel-case projection without changing either wire contract.
- 2026-07-14 — Task 6, Step 4: Ran `git diff --check` and an exact-pair consistency scan for lifecycle states, owner-filtered 404, uncached reads, snake/camel typed errors, the two-code duplicate allowlist, no automatic replay, failed/expired terminal handling, manual checks, stale suppression, and local credential isolation; every check passed. Decision: document unavailable lifecycle reads as a recovery fallback, not a sixth persisted lifecycle status.
- 2026-07-14 — Task 6, Step 5: Re-ran the documentation consistency checks, staged only `plans/AI_SDK_FE_CONTRACT.md` and `README.md`, verified the staged diff is whitespace-clean, and committed `docs(hitl): define no-replay recovery for all clients`. Decision: leave this implementation-progress log unstaged so it remains the sole working-tree change for later task records.

## Plan self-review

- **Coverage:** Tasks 1–4 close the failed-lifecycle, lost-code, stale-cache, and UI-state gaps; Task 5 preserves the approved secret design; Task 6 gives the external web client the same recovery contract; Task 7 verifies every boundary.
- **Safety:** No task relaxes the atomic claim, replays a resume, exposes secret values, or removes device/session/catalog validation.
- **Database compatibility:** The `failed` enum label is introduced by a PostgreSQL-safe, forward-only Alembic migration before application code depends on it.
- **Contract consistency:** Internal SSE uses snake_case metadata; AI SDK uses camelCase metadata; canonical JSON uses `code`; all map to the same domain codes and durable state endpoint.
