# Stop / Interrupt Streaming Chat (Backend + Streamlit Demo)

## Summary
Add a "Stop generating" control to the Streamlit UI (`demo.py`) and backend support to reliably cancel in-flight streaming responses without UI lag/reload artifacts or cancellation/disconnect errors being persisted as failures.

This plan targets the SSE chat path used by the demo:
- Backend SSE: `POST /messages/stream` -> `app/api/messages.py#create_message_stream`
- Streaming core: `app/services/message_service.py#MessageService.create_message_stream` -> `app/services/ai_service.py#AIService.generate_bot_response_stream` -> `app/ai/graph.py#MultiAgentWorkflow.execute_stream`
- Frontend: `demo.py` "Use streaming endpoint for real-time response" loop calling `make_streaming_request("/messages/stream", ...)`

## Goals
- Stop/interrupt an in-flight assistant response from the demo UI with one click.
- Stop should work even when the stream is "idle" (e.g., long tool run with no tokens).
- Avoid UI jank: no full-page "reset", no expensive re-fetch loops; keep chat history stable.
- Backend must not persist stop/disconnect as an error message (avoid "Error generating response: ..." artifacts).

## Non-goals (for first iteration / MVP)
- Guaranteed termination of every long-running external tool call (best-effort cancellation only).
- Cross-process cancellation coordination when running multiple Uvicorn workers (in-memory registry won't work across processes).
- A full "resume generation from partial" feature.

## Current Behavior (what's blocking "Stop" today)
### Frontend (Streamlit)
- `demo.py#make_streaming_request()` (line ~1737) uses `requests.post(..., stream=True, timeout=(10, 900))` and blocks on `response.iter_lines()`.
- During long tool runs (no SSE output), the UI can't reliably interrupt the Python script execution because the script is blocked in a long-running IO read.
- Streamlit's execution model reruns the entire script on any widget click; the only practical way to "signal" the running streaming loop is by forcing the HTTP connection to yield frequently (heartbeats) or by closing the connection.

### Backend
- Streaming is request-scoped; no explicit cancel API exists.
- `MessageService.create_message_stream()` (line ~418) wraps generation in a broad `except Exception` (line ~672) which persists the error as a bot message via `_create_bot_response_message`. This can persist disconnect-related exceptions (e.g., `ConnectionResetError`) as user-visible "errors".
- `asyncio.CancelledError` is special:
  - Python 3.11+ (including the current environment's Python 3.14): it inherits from `BaseException` (not caught by `except Exception`).
  - Python 3.10: it inherits from `Exception` (and can be incorrectly persisted by broad handlers).
  Either way, cancellation must be handled explicitly and never stored as a "failed response".
- The SSE generator in `app/api/messages.py` has no disconnect detection (`Request.is_disconnected()` is not used) and no structured cancellation behavior.
- No heartbeat/keepalive events are emitted, so clients can see long "silence" during tool execution.
- Precedent: the AI SDK endpoint (`app/api/ai_sdk.py`, ~line 946) handles cancellation correctly via `except asyncio.CancelledError: return`.

## Proposed Design

### UX (Streamlit)
- When streaming starts: show a "Stop generating" button next to/above the input area.
- While streaming: keep rendering tokens as today; additionally keep a lightweight status line ("Thinking...", "Running tool ...", "Stopping...").
- On Stop:
  - Immediately stop waiting for more tokens.
  - Keep the partial assistant text that has already streamed.
  - Re-enable input without reloading conversation list/messages unnecessarily.

### Backend API + Cancellation Model

#### 1) Add an in-flight generation registry (server-side)
Implement a small, best-effort in-memory registry keyed by a stable id (recommended: the persisted user message id created at stream start).

Entry shape (conceptual):
- `conversation_id`, `user_id` (for access checks)
- `cancel_event: asyncio.Event`
- `done: asyncio.Future[MessageRead | None]` (completed when generation ends or is cancelled)
- `partial_text: str` (accumulated server-side)
- (optional) `partial_thinking: str` (only if you intentionally persist/show "thinking" in production)
- bookkeeping: `started_at`, `last_event_at`

Where to register:
- Right after the user message is persisted in `MessageService.create_message_stream()` (it already yields `user_message_created` with the message payload).

Cleanup:
- Always remove the entry on `complete`, `interrupt`, `error`, or cancellation/disconnect.

Safety nets:
- Use `cachetools.TTLCache(maxsize=1000, ttl=600)` to prevent unbounded growth if cleanup fails.
- Production hardening option: store the registry in Redis (see "Production hardening" section).

Best-practice note:
- Add a stable linkage in assistant message metadata, e.g. `{ "reply_to_user_message_id": "<uuid>" }`, so `/messages/stop` can be idempotent and the UI can append the correct assistant message without full refreshes.

#### 2) Emit heartbeat events during streaming silence
Add a heartbeat SSE event at a fixed interval while a stream is active and no other events are flowing.

Why:
- Keeps the Streamlit script from blocking for up to the 900s read timeout during tool execution.
- Gives Streamlit frequent "yield points" so a Stop click can interrupt promptly.
- Keeps the SSE connection alive through proxies/load balancers with idle timeouts.

Where:
- Implement at the API layer (`app/api/messages.py`) using an `asyncio.Queue` + producer task pattern so heartbeats can be emitted even while the producer is blocked running a tool.

Heartbeat payload:
- `{"type": "heartbeat"}`

Interval:
- MVP: 1.0s (good for Streamlit responsiveness).
- Production: make configurable (e.g. env/config), default 10-15s to reduce chatter.

Reference sketch:
```python
async def event_generator_with_heartbeat():
    queue: asyncio.Queue = asyncio.Queue()

    async def producer():
        try:
            async for event in message_service.create_message_stream(message_data, user_id):
                await queue.put(event)
        finally:
            await queue.put(None)  # sentinel

    task = asyncio.create_task(producer())
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\\n\\n"
                continue

            if event is None:
                break

            yield f"data: {json.dumps(event)}\\n\\n"
            if event.get('type') in ('complete', 'error', 'interrupt'):
                break
    finally:
        task.cancel()
```

Frontend handling:
- Ignore heartbeats entirely (no-op). Optionally update a timer/status label every N heartbeats.

#### 3) Add `POST /messages/stop` endpoint
Purpose: allow the UI to explicitly request cancellation and receive the best available "final" assistant message. This is the Phase 2 cleanup call after the HTTP disconnect (Phase 1) already signaled cancellation.

Request schema (add to `app/schemas/message.py`):
```python
class StopGenerationRequest(BaseModel):
    conversation_id: UUID = Field(..., description="Conversation ID")
    user_message_id: UUID = Field(..., description="User message ID from user_message_created event")
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
```

Response (recommendation for consistency with the rest of the API):
- Wrap in `ApiResponse[...]`, returning either:
  - `data = MessageRead` when an assistant message was persisted (partial or final), OR
  - a small response model like `{ "status": "not_inflight" | "cancellation_requested" }`.

Idempotency:
- Calling stop multiple times is safe.
- If the generation already completed, stop returns success with `status="not_inflight"` (UI can refresh once).
- If a partial/final assistant message already exists for `reply_to_user_message_id`, return it instead of creating another.

Timeouts:
- `POST /messages/stop` should not hang indefinitely. If `done` isn't resolved quickly (2-5s), return `status="cancellation_requested"` and let the UI refresh messages after a short delay.

Access validation:
- Use `conversation_validation_utils.validate_conversation_access(user_id, conversation_id)` (same pattern as other endpoints).
- Use the same DI style (`@AppAutoInjector.auto_inject()` + `IMessageService`).

#### 4) Treat cancellation/disconnect as non-error
Two-phase cancellation model:
1. Phase 1 - HTTP disconnect (immediate): Streamlit button click -> script rerun -> drops the `/messages/stream` connection. Server should treat disconnect as cancellation and avoid persisting error artifacts.
2. Phase 2 - explicit cleanup (next rerun): Streamlit calls `POST /messages/stop` -> triggers best-effort cancellation, persists partial text if available, clears registry entry, returns final state to UI.

In `MessageService.create_message_stream()`:
- Accumulate `partial_text` as token events are streamed (mirrors the frontend).
- Add a dedicated `except (asyncio.CancelledError, GeneratorExit):` handler before the existing `except Exception`:
  - cancel `title_task` (if running)
  - store any `partial_text` into the registry entry
  - resolve `done` future
  - do not persist an "error" assistant message
  - exit cleanly

In `app/api/messages.py`:
- Accept `Request` so we can check `await request.is_disconnected()` and/or treat write failures as disconnect.
- Add `except asyncio.CancelledError: return` in the API generator (matching `ai_sdk.py`).
- (Optional hardening) also treat Starlette disconnect exceptions as non-errors (e.g., `ClientDisconnect`) if they appear in your stack.

#### 5) Planning pause enums (optional)
Avoid mixing "user stopped generation" into planning pause reasons. Prefer metadata keys like:
`{ "stopped": true, "partial": true, "stop_reason": "user_requested" }`.

### Frontend Implementation (demo.py)

#### State additions (session_state)
Add minimal keys to track in-flight streaming:
- `stream_inflight`: bool
- `stream_conversation_id`: str
- `stream_user_message_id`: str (from `user_message_created` event)
- `stream_partial_text`: str
- `stream_partial_thinking`: str (optional; consider not persisting in production)
- `stream_selected_agent`: str | None
- `stream_stop_requested`: bool

#### Streaming loop changes
- On `user_message_created`: store `stream_user_message_id`.
- On `token`/`thinking`: update `stream_partial_*` in `st.session_state` (not only local vars) so the partial survives reruns.
- Handle `heartbeat`: ignore (no-op).

Best-practice note:
- Ensure the streaming `requests` response is closed even on interruption (use a context manager and/or `finally: response.close()` in `make_streaming_request()`).

#### Stop button flow (two-phase, no lag/reload)
Streamlit reruns the script on button clicks. Clicking Stop should:
- set `stream_stop_requested = True`
- trigger rerun (this drops the stream connection)

On the next rerun:
- if `stream_stop_requested` and `stream_user_message_id` present:
  - call `POST /messages/stop` with `conversationId` + `userMessageId`
  - if response includes an assistant message, append it to `st.session_state.messages` (avoid a full `load_messages_page(1)` refresh)
  - if response indicates `not_inflight`, do a single `load_messages_page(1)` to sync
  - display `stream_partial_text` from session state as a visual preview while the stop call completes
  - clear inflight state + stop flag

Performance guardrails:
- Avoid `load_messages_page(1)` on every stop click; only as a fallback.
- Optional: throttle UI markdown updates (e.g., update bubble every 50-100ms) if per-token rendering becomes costly.

## Step-by-step Implementation Plan

### Phase 1 - Backend primitives
- [x] Add `GenerationRegistry` (module-level singleton) backed by `cachetools.TTLCache(maxsize=1000, ttl=600)` for in-flight streams.
  - **Implemented**: `app/services/generation_registry.py` with `InflightEntry` dataclass and `GenerationRegistry` class.
- [x] (Best practice) Add/update message-service interface methods in `app/interfaces/message_service_interface.py` for:
  - `create_message_stream(...)`
  - `resume_message_creation(...)`
  - new: `stop_message_generation(...)` (or equivalent)
- [x] Update `MessageService.create_message_stream()` to:
  - accumulate `partial_text` server-side
  - register/unregister in-flight entries in the registry
  - handle `asyncio.CancelledError` and `GeneratorExit` separately (no error persistence)
  - **Design Decision**: `CancelledError` & `GeneratorExit` caught in a dedicated handler before the `except Exception` block; partial text is persisted with `{"stopped": true, "partial": true, "stop_reason": "disconnect"}` metadata. Each event type updates `inflight.partial_text` and `inflight.touch()` for tracking.
- [x] Refactor `/messages/stream` SSE generator to use `asyncio.Queue` + producer task pattern with heartbeat injection (configurable interval; MVP 1.0s).
  - **Design Decision**: `HEARTBEAT_INTERVAL_SECONDS = 1.0` constant at module level in `app/api/messages.py`. Producer task drains service-layer generator into queue; consumer emits heartbeats on timeout.
- [x] Accept `Request` parameter in the stream endpoint for disconnect detection.
  - **Implemented**: `await request.is_disconnected()` checked in the consumer loop.
- [x] Add `except asyncio.CancelledError: return` to the API-layer generator (matching `ai_sdk.py` pattern).
- [x] Add `StopGenerationRequest` schema to `app/schemas/message.py`.
  - **Also added**: `StopGenerationResponse` schema for consistency.
- [x] Add `POST /messages/stop` endpoint with access validation and idempotent behavior.
  - **Implemented**: Uses `IMessageService.stop_message_generation()` with DI auto-injection. Returns `ApiResponse` with `{status, message}`.

### Phase 2 - Streamlit Stop UX
- [x] Add in-flight state fields to `st.session_state` (`stream_inflight`, `stream_conversation_id`, `stream_user_message_id`, `stream_partial_text`, `stream_partial_thinking`, `stream_selected_agent`).
  - **Design Decision**: Removed `stream_stop_requested` from original plan. Streamlit's rerun-on-button-click naturally drops the HTTP connection (Phase 1); detection of `stream_inflight == True` on the next rerun triggers Phase 2 cleanup. This is simpler and avoids race conditions in the Streamlit execution model.
- [x] Add "Stop generating" button visible only during streaming.
  - **Design Decision**: Button is rendered once via `st.empty().button()` BEFORE the streaming loop starts, not inside the per-event loop. This avoids duplicate widget key errors and works with Streamlit's execution model (button click triggers a script rerun automatically).
- [x] Persist partial token/thinking into `st.session_state` during the stream (not only local vars).
- [x] Add `heartbeat` handling in `make_streaming_request()` loop (ignore / no-op).
  - **Implemented**: Heartbeat events (`type == "heartbeat"`) are silently consumed with `continue`.
- [x] On stop rerun: call `POST /messages/stop`, update local message list without `load_messages_page(1)` unless fallback needed.
  - **Design Decision**: `_handle_stop_rerun()` is a module-level function (not nested inside `render_chat_view()`) to avoid coupling with the nested `load_messages_page()`. Fallback uses `conversation_messages_page = 0` reset instead of calling `load_messages_page(1)` directly.
- [x] Ensure stopping clears spinners/status and re-enables input.
- [x] Display accumulated partial text preview while the stop call completes.
  - **Implemented**: Shows `partial_preview + " *(stopped)*"` in a chat_message container.
- [x] Ensure `response.close()` is called in `make_streaming_request()` `finally` block.

### Phase 3 - Polish + edge cases
- [x] Handle "stop before first token" cleanly (no empty assistant bubble).
  - **Implemented**: Backend only persists partial text if `partial.strip()` is non-empty. Frontend only shows preview if `partial_preview` is non-empty.
- [x] If a HITL `interrupt` arrives, keep current behavior (approval UI) and clear in-flight stop state.
  - **Implemented**: `_clear_inflight_state()` called in the interrupt handler branch.
- [x] If the stream errors after stop is requested, prefer showing "Stopped" rather than surfacing the error toast.
  - **Implemented**: `_clear_inflight_state()` called in the error handler branch.
- [ ] Optional: add a "Stop (and discard)" vs "Stop (keep partial)" UX toggle (default keep partial).
  - **Deferred**: Default is keep partial per plan. Can be added later.

### Phase 4 - Validation checklist
- [ ] Manual: start a long tool call (MCP) and verify Stop works within ~1s (MVP).
- [ ] Manual: stop mid-token stream; partial assistant message remains visible.
- [ ] Manual: stop immediately after send; no bogus error message is persisted.
- [ ] Manual: normal completion unchanged; no extra UI lag from heartbeat events.
- [ ] (Production) Add at least one automated test for "disconnect/cancel does not persist error message" and for `/messages/stop` idempotency.

## Acceptance Criteria
- Clicking "Stop generating" stops the response promptly (< 2s) and leaves the UI usable (input enabled).
- No cancellation/disconnect-related errors are stored in the DB as bot messages.
- Partial assistant text (if any) is persisted with metadata like `{ "stopped": true, "partial": true }`.
- The demo UI does not noticeably re-render/reload conversations when stopping.
- Streaming still works for normal (non-stopped) paths, including HITL interrupts.
- Heartbeat events keep the SSE connection alive during long tool runs without causing UI churn.
- The `/messages/stop` endpoint is idempotent.

## Production hardening (recommended follow-up)
- Multi-worker: move registry/cancel signaling to Redis (or enforce single-worker for SSE endpoints).
- Config: make heartbeat interval and stop timeouts configurable (env/settings).
- Observability: log cancellations as info (not errors) and add basic metrics (stops, disconnects, time-to-stop).
- Security: rate-limit stop calls per user/conversation if exposed publicly.
- Data policy: reconsider streaming/persisting "thinking" content; keep it behind a debug flag if needed.

## Risks / Tradeoffs
- In-memory cancellation registry won't work across multiple server processes; if multi-worker is needed, move registry to Redis (the codebase already uses `redis_client` in `MessageService`).
- Some tool calls may not be interruptible; cancellation is best-effort and may only stop token streaming.
- Heartbeat frequency too high increases SSE traffic; keep it configurable and prefer ~10-15s in production if Streamlit responsiveness isn't the primary concern.
- The `asyncio.Queue` + producer task pattern adds indirection; careful `finally` cleanup is essential to avoid leaked tasks.
- Cancellation semantics are subtle; ensure cancellation/disconnect is not persisted as an "error message" and does not create duplicate assistant messages.

## Resolved Questions
- Persist partial by default? -> Yes, persist by default with `{ "stopped": true, "partial": true }` metadata. Discarding loses potentially useful context. The UI can display a visual indicator for partial messages.
- AI SDK parity? -> Not for v1. The AI SDK path (`app/api/ai_sdk.py`) already handles `CancelledError` correctly. Add registry/stop support later if needed.
- Heartbeat interval? -> MVP 1.0s for Streamlit; production default 10-15s (configurable).

## Remaining Open Questions
- Should the partial message have a visual badge in the UI (e.g., "(stopped)" suffix or a subtle indicator)?
- Should we cap the maximum generation time server-side (e.g., 5 minutes) as an independent safety net?

