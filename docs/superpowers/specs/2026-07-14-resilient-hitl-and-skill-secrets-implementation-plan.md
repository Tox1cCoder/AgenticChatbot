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

- [ ] **Step 1: Write failing lifecycle tests.**

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

- [ ] **Step 2: Run the focused tests and confirm the new status is unavailable.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_hitl_api.py tests/test_message_service_event_streaming.py tests/test_client_tool_isolation.py
  ```

  Expected: FAIL because `FAILED`, `mark_failed`, and the state route do not exist.

- [ ] **Step 3: Add the enum value and PostgreSQL migration.**

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

- [ ] **Step 4: Add conditional lifecycle repository methods.**

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

- [ ] **Step 5: Close every claimed-resume lifecycle branch.**

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

- [ ] **Step 6: Add the state schema, DI binding, and route.**

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

- [ ] **Step 7: Run lifecycle tests and migration verification.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m alembic upgrade head
  .venv\Scripts\python.exe -m pytest -q tests/test_hitl_api.py tests/test_message_service_event_streaming.py tests/test_client_tool_isolation.py tests/test_hitl_policy.py
  ```

  Expected: migration succeeds; all tests pass.

- [ ] **Step 8: Commit the lifecycle/API change.**

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

- [ ] **Step 1: Write failing transport-contract tests.**

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

- [ ] **Step 2: Run the tests and confirm code metadata is dropped.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_exception_handler.py tests/test_message_stream_errors.py tests/test_ai_sdk_v6_stream_contract.py tests/client_backend/test_messages.py
  ```

  Expected: FAIL because `ApiResponse` discards `code` and the adapters emit only text.

- [ ] **Step 3: Serialize the canonical envelope code.**

  Add the field to `ApiResponse`:

  ```python
  code: str | None = Field(None, description="Stable machine-readable error code")
  ```

  Keep `exclude_none=True` in exception handlers so successful responses and errors without a code remain backward compatible. Do not modify the sidecar's success envelope unless it has a code to forward; `proxy_server_request()` will relay the canonical JSON unchanged.

- [ ] **Step 4: Add protocol-specific stream serializers.**

  In `app/api/messages.py`, use one `_stream_error_event(exc)` helper for producer and generator exceptions. It emits `error`, `status_code`, and `error_code` for `CustomHTTPException`.

  In `internal_sse.py`, when mapping `V3StreamEvent(type="error")`, copy only known optional `status_code`/`error_code` keys into the public error dict. In `ai_sdk_v6.py`, add helpers that map an exception or V3 error data to:

  ```python
  {"type": "error", "errorText": message, "statusCode": status_code, "errorCode": error_code}
  ```

  Include `statusCode` and `errorCode` only when present. Route both `iter_sse()`'s outer exception and `_error()` through these helpers. Preserve the existing `text-end`, `reasoning-end`, `finish-step`, `finish`, and `[DONE]` sequence.

- [ ] **Step 5: Preserve the same identity at the sidecar boundary.**

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

- [ ] **Step 6: Run all transport regressions.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_exception_handler.py tests/test_message_stream_errors.py tests/test_ai_sdk_v6_stream_contract.py tests/test_message_service_event_streaming.py tests/client_backend/test_messages.py tests/client_backend/test_server_api.py tests/client_backend/test_sse_keepalive.py
  ```

  Expected: PASS; AI SDK streams still end in `[DONE]` and internal streams retain heartbeat behavior.

- [ ] **Step 7: Commit the transport change.**

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

- [ ] **Step 1: Write failing pure-policy and proxy tests.**

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

- [ ] **Step 2: Run the focused tests.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_hitl_ui_recovery.py tests/client_backend/test_hitl_proxy.py
  ```

  Expected: FAIL because the module, classifier, and route do not exist.

- [ ] **Step 3: Implement the pure helpers and sidecar route.**

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

- [ ] **Step 4: Run focused policy/proxy tests.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_hitl_ui_recovery.py tests/client_backend/test_hitl_proxy.py
  ```

  Expected: PASS.

- [ ] **Step 5: Commit the shared recovery foundation.**

  ```powershell
  git add app/ui/hitl_recovery.py client_backend/api/proxy.py tests/test_hitl_ui_recovery.py tests/client_backend/test_hitl_proxy.py
  git commit -m "feat(hitl): add shared lifecycle reconciliation policy"
  ```

### Task 4: Make the Streamlit approval flow no-replay and terminal-aware

**Files:**

- Modify: `demo.py:18-35,2781-2979,3500-3565,7332-7701,7904-7927`
- Modify: `tests/test_hitl_demo_panel.py`

- [ ] **Step 1: Extend the static guard for the required seams.**

  Assert that the source imports `extract_error_code`, `is_recoverable_resume_conflict`, `reconciliation_action`, and `should_suppress_pending_interrupt`; calls `make_api_request("GET", f"/hitl/interrupts/{interrupt_id}", use_cache=False)` for lifecycle state; uses `hitl_resume_inflight_`; handles `failed`; and renders `statusCode`/`errorCode`-compatible errors.

- [ ] **Step 2: Run the guard before implementation.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py tests/test_hitl_ui_recovery.py
  ```

  Expected: FAIL on the new assertions.

- [ ] **Step 3: Add an uncached GET option and lifecycle helper.**

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

- [ ] **Step 4: Add interrupt-ID-scoped locks and reconciliation.**

  Add `_hitl_resume_lock_key()`, `_hitl_reconciliation_key()`, and `_clear_interrupt_ui_state()`. `_clear_interrupt_ui_state()` removes decisions, edit keys, the lock, `pending_interrupt`, and `interrupt_conversation_id`, then sets `conversation_messages_page = 0`; it deliberately does not remove the suppression marker.

  `_reconcile_interrupt()` must call the uncached lifecycle helper exactly once and branch on `reconciliation_action(status)`:

  - `restore_form`: remove lock and suppression marker.
  - `show_processing`: remove local decisions, store this interrupt as the suppression marker, and leave no enabled resume control.
  - `refresh_history`: clear paused UI, retain the marker, and set the notice `"Approval completed elsewhere; conversation refreshed."`.
  - `require_new_message`: clear paused UI, retain the marker, and set the notice `"This approval can no longer be resumed. Send a new message."`.

  It never calls `/messages/resume-interrupt`.

- [ ] **Step 5: Lock the form before any resume call.**

  In `render_interrupt_approval_ui()`, render the suppression panel before actions. It exposes only **Check status** and **Return to chat**; both preserve the marker, and neither sends a resume request. Otherwise, calculate `resume_inflight` from the interrupt-ID lock and pass `disabled=resume_inflight` to **Submit Decisions**, **Approve All**, and **Cancel All**. Set the lock immediately before `_submit_interrupt_decisions()`.

  In `_submit_interrupt_decisions()`, retain the complete error event. A duplicate error calls `_reconcile_interrupt()` and reruns. An `INTERRUPT_FAILED` event or any event whose lifecycle read classifies as `require_new_message` clears paused UI, shows its actual error text, and does not restore the form. Pre-claim actionable errors remain visible; if their lifecycle state is still `pending`, only then restore the form.

- [ ] **Step 6: Suppress stale rehydration after every terminal state.**

  In the paused-message recovery loop, derive `recovered_interrupt_id` and call `should_suppress_pending_interrupt(recovered_interrupt_id, marker)` before assigning `pending_interrupt`. Preserve rehydration for a different follow-up interrupt. Clear the suppression marker only when the lifecycle read returns `pending` for that exact ID or the session is reset/logout.

- [ ] **Step 7: Run the Streamlit regression set.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py tests/test_hitl_ui_recovery.py tests/test_hitl_ui_decisions.py tests/test_hitl_decision_mapping.py tests/test_hitl_interrupt_payload_recovery.py
  ```

  Expected: PASS.

- [ ] **Step 8: Commit the Streamlit recovery flow.**

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

- [ ] **Step 1: Write failing readiness and UI seam tests.**

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

- [ ] **Step 2: Run the tests and confirm fields/widgets are absent.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/client_backend/test_skills_api.py tests/test_hitl_demo_panel.py
  ```

  Expected: FAIL on `setupStatus` and credential UI seams.

- [ ] **Step 3: Return only safe readiness fields.**

  Evaluate readiness once in `_skill_summary()` and add `setupStatus: readiness.setup_status`. Keep `commandCapable: readiness.status == "ready"`. Do not return runtime roots, command paths, setup logs, secret bindings, or secret values.

- [ ] **Step 4: Add names-only credential helpers and widgets.**

  Add `get_skill_secrets`, `set_skill_secret`, and `delete_skill_secret` around the existing sidecar routes. In every skill card, render a **Local credentials** expander keyed by `current_user_id` plus `skill_name`. Use `st.text_input("Secret value", type="password", key=value_key)`, clear `value_key` immediately after a successful save, display only `f"{secret_name} configured"`, and remove `skill_secret_` state keys in `_clear_skill_hitl_session_state()` during logout. Never include the value in a toast, success text, cache key, test assertion, or log.

- [ ] **Step 5: Render runtime guidance before approval controls.**

  Render `Command runtime: Ready` for ready skills; the instruction-only explanation for `instruction_only`; otherwise show `Command runtime is not ready (<setupStatus>)`. Keep the approval radio solely inside the ready branch for `skill::<name>::run_skill_command`.

- [ ] **Step 6: Run the complete secret/runtime regression set.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m pytest -q tests/client_backend/test_skills_api.py tests/client_backend/test_skill_secrets.py tests/client_backend/test_skill_execution_engine.py tests/client_backend/test_skill_runtime_manager.py tests/test_hitl_demo_panel.py
  ```

  Expected: PASS.

- [ ] **Step 7: Commit the skills-tab changes.**

  ```powershell
  git add client_backend/api/skills.py demo.py tests/client_backend/test_skills_api.py tests/test_hitl_demo_panel.py
  git commit -m "feat(skills): show readiness and manage local credentials"
  ```

### Task 6: Update the AI SDK frontend contract and operational documentation

**Files:**

- Modify: `plans/AI_SDK_FE_CONTRACT.md:3-10,177-289,452-501`
- Modify: `README.md`

- [ ] **Step 1: Update the frontend endpoint and error contract.**

  Add `GET /hitl/interrupts/{interruptId}` to the endpoint table. Document its authenticated, owner-filtered response, all five status values, `404 INTERRUPT_NOT_FOUND`, and the instruction to use `cache: "no-store"` for reconciliation reads.

  Replace the untyped AI SDK error example with the optional `statusCode`/`errorCode` fields. State that duplicate conflicts are delivered as a terminal `200` SSE stream when discovered during stream iteration; JSON validation failures may still be ordinary non-stream HTTP errors. Retain the existing AI SDK terminal sequence.

- [ ] **Step 2: Replace the duplicate-resume guidance with no-replay behavior.**

  In the Resume section, require an interrupt-ID-scoped lock before POST. For `INTERRUPT_ALREADY_RESOLVED` or `INTERRUPT_CONFLICT`, require one state GET and map `pending`, `resolving`, `resolved`, `failed`, and `expired` exactly as the public reconciliation table specifies. State explicitly that `failed` and `expired` require a new chat message and neither client may POST resume again.

- [ ] **Step 3: Document Streamlit/operator behavior in the README.**

  Explain first-write-wins resume, the terminal `failed` state, manual status checks without polling, stale-paused-message suppression, and local credential boundaries (manual names, encrypted user/device-local values, no server synchronization or chat-history exposure).

- [ ] **Step 4: Review both documents for contract consistency.**

  Verify these exact pairs agree: `failed` lifecycle status; `status_code`/`error_code` vs `statusCode`/`errorCode`; `INTERRUPT_FAILED`; duplicate-code allowlist; uncached state reads; and no-replay behavior.

- [ ] **Step 5: Commit contract documentation.**

  ```powershell
  git add plans/AI_SDK_FE_CONTRACT.md README.md
  git commit -m "docs(hitl): define no-replay recovery for all clients"
  ```

### Task 7: Run the integrated regression and manual acceptance checks

**Files:**

- No production files; run verification after all prior tasks.

- [ ] **Step 1: Run lint and format checks.**

  Run:

  ```powershell
  .venv\Scripts\python.exe -m ruff check app/models/hitl_interrupt.py app/repositories/hitl_interrupt.py app/services/message_service.py app/schemas/hitl.py app/api/hitl.py app/api/messages.py app/services/event_streaming/internal_sse.py app/services/event_streaming/ai_sdk_v6.py app/ui/hitl_recovery.py app/schemas/responses/api_response.py app/utils/exception_handler.py client_backend/api/messages.py client_backend/api/proxy.py client_backend/api/skills.py demo.py tests/test_hitl_api.py tests/test_message_service_event_streaming.py tests/test_message_stream_errors.py tests/test_ai_sdk_v6_stream_contract.py tests/test_hitl_ui_recovery.py tests/test_exception_handler.py tests/client_backend/test_messages.py tests/client_backend/test_hitl_proxy.py tests/client_backend/test_skills_api.py
  .venv\Scripts\python.exe -m ruff format --check app/models/hitl_interrupt.py app/repositories/hitl_interrupt.py app/services/message_service.py app/schemas/hitl.py app/api/hitl.py app/api/messages.py app/services/event_streaming/internal_sse.py app/services/event_streaming/ai_sdk_v6.py app/ui/hitl_recovery.py app/schemas/responses/api_response.py app/utils/exception_handler.py client_backend/api/messages.py client_backend/api/proxy.py client_backend/api/skills.py demo.py tests/test_hitl_api.py tests/test_message_service_event_streaming.py tests/test_message_stream_errors.py tests/test_ai_sdk_v6_stream_contract.py tests/test_hitl_ui_recovery.py tests/test_exception_handler.py tests/client_backend/test_messages.py tests/client_backend/test_hitl_proxy.py tests/client_backend/test_skills_api.py
  ```

  Expected: both commands exit `0`.

- [ ] **Step 2: Run the cross-layer regression matrix.**

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

- [ ] **Step 4: Commit final verification documentation if it changed.**

  ```powershell
  git status --short
  ```

  Expected: only intentionally uncommitted implementation work remains.

## Plan self-review

- **Coverage:** Tasks 1–4 close the failed-lifecycle, lost-code, stale-cache, and UI-state gaps; Task 5 preserves the approved secret design; Task 6 gives the external web client the same recovery contract; Task 7 verifies every boundary.
- **Safety:** No task relaxes the atomic claim, replays a resume, exposes secret values, or removes device/session/catalog validation.
- **Database compatibility:** The `failed` enum label is introduced by a PostgreSQL-safe, forward-only Alembic migration before application code depends on it.
- **Contract consistency:** Internal SSE uses snake_case metadata; AI SDK uses camelCase metadata; canonical JSON uses `code`; all map to the same domain codes and durable state endpoint.
