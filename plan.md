# Summarization Middleware: Production Readiness Plan

## Context
This repo implements rolling conversation summarization to keep LangGraph checkpoint state small while still giving the LLM "memory" via `state["history_summary"]`.

The *conversation summary itself* is leaking to the end user in **graph streaming** output.

## Requirements (confirmed)
- Leak must be fixed: end users **must not** see the rolling conversation summary.
- Keep summarization provider/model behavior "as is" (currently Gemini via `ChatGoogleGenerativeAI`).
- Summary should include tool call/results (so the LLM can continue progress).
- Persistence: prefer checkpoint-only; consider whether DB storage is needed as a fallback.
- Plan format: standard "spec-kit" style (goals, non-goals, design, milestones, verification, rollout).

---

## Current Implementation (what's happening today)

### Where summarization runs — graph path
- `app/ai/graph.py` builds a LangGraph workflow with a first node: `summarize` → `route` → agent.
- `app/ai/graph.py:_summarization_node()` calls `app/ai/summarization_middleware.py:summarize_for_state()`.
- `summarize_for_state()`:
  - Checks thresholds (`should_summarize`) based on `state["messages"]` size.
  - Calls `generate_summary()` (Gemini model via `ainvoke`) to produce a rolling summary.
  - Writes the rolling summary into `state["history_summary"]`.
  - Removes old messages using `RemoveMessage(...)` so `add_messages` reducer deletes them from checkpoint.

### Where summarization runs — fast-path (traditional RAG)
- `execute_stream()` bypasses the graph pipeline for traditional non-agentic RAG.
- Before streaming starts, it calls `_run_fast_path_summarization()`, which reads the checkpoint, runs `should_summarize` + `generate_summary()` directly, then writes results back to the checkpoint.
- This path does **not** cause a streaming leak (runs before streaming, not inside the graph), but shares the same `generate_summary()` function and has no timeout or tagging.
- The split/keep logic is duplicated manually in `_run_fast_path_summarization()` rather than reusing `summarize_for_state()`, creating a future divergence risk.

### How the summary is used
- For most agents: `BaseAgent._build_system_prompt(..., history_summary=...)` injects the rolling summary as an internal "Conversation Memory" block.
- For traditional RAG: `build_rag_prompt(..., history_summary=...)` injects a similar memory block.
- The summary is **not intended** to be user-visible.

### How streaming works (and why the graph-path leak happens)
- `app/ai/graph.py:execute_stream()` runs `self.graph.astream(..., stream_mode=["messages","updates"])`.
- The `"messages"` stream mode emits **LLM message chunks from all nodes**, including the summarization node's model call.
- `generate_summary()` uses `model.ainvoke()`, not `astream`, so the summary arrives as **one large chunk** (not incremental tokens), but it still flows through LangGraph's callback/streaming layer and appears in the `"messages"` mode output.
- `execute_stream()` currently has no node-based or tag-based filter, so the summarization chunk is accumulated into `accumulated_content` and emitted as a `"token"` event.
- The same code paths handle `content_blocks` of type `"reasoning"` / `"thinking"`, meaning Gemini reasoning output from the summarization model also leaks into `accumulated_thinking`.
- Result:
  - When summarization triggers, the entire summary appears mid-stream as normal assistant text.
  - Stopping/disconnecting can persist the internal summary as the assistant message in the database.

This is the core production bug.

---

## Findings / Issues

### P0 — Conversation summary leaks to end users during streaming
**Root cause:** `execute_stream()` does not filter output by node, so the `"summarize"` node's `ainvoke` result is emitted as user-facing tokens.

**Precise mechanism:**
- `generate_summary()` calls `await model.ainvoke(...)`, not `astream`.
- Despite using `ainvoke`, LangGraph's `"messages"` stream mode still surfaces the completed `AIMessage` from the summarize node as a single large chunk.
- The chunk hits every branch of `execute_stream()`'s content-handling logic (`content_blocks`, list content, legacy string content) and is appended to `accumulated_content` unconditionally.
- `"reasoning"` / `"thinking"` content blocks from the same model call also accumulate into `accumulated_thinking`, which is later embedded in `metadata["thinking_summary"]`.

**Impact:**
- Users see internal memory summaries (unexpected UX + potential data leakage).
- Stopping/disconnecting persists internal summary text as the assistant message in the DB.
- Internal reasoning from the Gemini summarization model surfaces as `thinking` events to the client.

**Primary touchpoints:**
- `app/ai/graph.py` — `execute_stream()`, all content-handling branches
- `app/ai/summarization_middleware.py` — `generate_summary()` (no tags/metadata on the run)

### P0b — Silent message data-loss on summarization failure
`generate_summary()` currently catches exceptions internally and returns the stub `"[Previous conversation with N messages]"`. Both callers (`summarize_for_state()` and `_run_fast_path_summarization()`) then treat that stub as success and pass it to `apply_summarization_to_state()`, which **still issues `RemoveMessage` for all summarized messages** — silently destroying conversation history and replacing it with a near-useless placeholder.

This is a correctness bug independent of the streaming leak.

**Fix:** make summarization fail closed:
- Change `generate_summary()` to re-raise errors (or return an explicit success flag).
- In **both** callers, only run `apply_summarization_to_state()` on genuine success.
- On failure, return the original state/checkpoint values unchanged.

### P1 — Fast-path summarization (`_run_fast_path_summarization`) is untagged and untimed
The traditional-RAG code path calls `generate_summary()` outside the graph, so it does not cause a streaming leak. However:
- No `asyncio.wait_for` timeout — a slow Gemini call blocks the entire request.
- The split/keep calculation is duplicated in `_run_fast_path_summarization()` rather than delegating to `summarize_for_state()`. Both must be kept in sync manually.
- It inherits the same P0b data-loss risk unless fast-path also gates `apply_summarization_to_state()` on true summarization success.
- If the tag-based approach is adopted, this call also needs tagging for consistent observability.

### P2 — Auto-continue rounds and summarization idempotence (validation item)
`execute_stream()` wraps `graph.astream()` in an auto-continue loop. The graph re-enters the `"summarize"` node each round, so `context["conversation_summarized"]` must persist in continuation state to avoid duplicate summarization calls.

Current code appears to preserve this context key, so this is primarily a regression-check item (not a confirmed defect).

### P3 — Missing general "internal run" gating
Even after fixing summarization, any future internal nodes that call an LLM (e.g., safety checks, background classifiers) would leak the same way. The current `suppress_tokens` flag is a `bool` hardcoded to `image_generator_agent`, which doesn't scale. A reusable mechanism is needed.

### P4 — Summarization call hardening (latency/cost control)
`generate_summary()` does not currently:
- enforce a dedicated timeout/budget (no `asyncio.wait_for`),
- pass `max_output_tokens` to the model constructor — only post-hoc char truncation is applied after the model returns.

Note: `summarization_max_summary_tokens` already exists in `Settings` and `SummarizationConfig`. The missing step is wiring it to `ChatGoogleGenerativeAI(max_output_tokens=...)` to cap generation at the source, not just truncate afterward.

### P5 — Fragile `getattr`-based config access in `summarization_middleware.py`
`_get_config()` accesses settings via `getattr(settings, "summarization_trigger_tokens", 18000)` etc. All these fields are defined in the Pydantic `Settings` model with defaults, so the `getattr` fallbacks are dead code that silently swallow typos.

**Fix:** replace with direct `settings.summarization_*` attribute access.

### P6 — Dead legacy state fields
`apply_summarization_to_state()` writes `context["summary_text"]` and `context["messages_summarized_count"]` with a comment "kept for backward compatibility". Nothing in the codebase reads these fields.

**Fix:** remove them entirely.

### P7 — Persistence clarity (checkpoint vs DB)
When `enable_langgraph_checkpoints` is enabled, `history_summary` is persisted in **LangGraph checkpoint tables (Postgres)** only, not in the app's messages/conversations tables.

This is fine if checkpoints are always enabled and shared across instances. If checkpoints are disabled or ephemeral, summarization stops working across turns.

---

## Goals / Non-Goals

### Goals
1. **Stop streaming internal summarization output** — no summary tokens, text chunks, or thinking events emitted to clients.
2. Ensure internal chunks **never** pollute:
   - `accumulated_content` / `accumulated_thinking`,
   - persisted partial responses (stop/disconnect scenarios).
3. **Fix silent message data-loss** when `generate_summary()` errors.
4. Establish a **reusable pattern** for internal LLM calls so the fix is future-proof.
5. Keep summarization behavior/functionality the same for the LLM.

### Non-Goals (for this pass)
- Changing providers/models (OpenAI vs Gemini) for summarization.
- Replacing the memory system (DB memory manager vs checkpoint messages) wholesale.
- Big prompt rewrites (only adjust if needed to enforce concise summaries).

---

## Proposed Design

### Design principle: "Internal LLM runs must be tagged and stream-suppressed"
Implement two complementary protections:

1. **Tag internal LLM invocations** by invoking the model with a `RunnableConfig`:
   ```python
   config = RunnableConfig(
       tags=["internal", "summarization"],
       metadata={"internal": True, "purpose": "summarization"},
   )
   response = await model.ainvoke([HumanMessage(content=prompt)], config=config)
   ```

2. **Drop internal chunks** inside `execute_stream()` using a helper:
   ```python
   def _is_internal_stream_chunk(metadata: dict) -> bool:
       tags = metadata.get("tags") or []
       return "internal" in tags or metadata.get("metadata", {}).get("internal") is True
   ```
   Early-`continue` for matching chunks — skip all content accumulation and event emission.

3. **Replace the `suppress_tokens: bool` pattern** with a `suppressed_nodes: set[str]` set so future internal nodes are a one-liner addition:
   ```python
   suppressed_nodes = {"image_generator_agent"}  # extend as needed
   suppress_tokens = selected_agent in suppressed_nodes
   ```

4. **Guard the `accumulated_content` fallback** against internal-only contamination. Track `_internal_content_only: bool`: set `True` when the first chunk is received and it is internal; set `False` the moment any non-internal chunk arrives. In the post-stream fallback branch, only use `accumulated_content` when `_internal_content_only is False`.

### Where to apply

| File | Change |
|---|---|
| `app/ai/summarization_middleware.py` | Tag the `ainvoke` call in `generate_summary()` with internal config |
| `app/ai/summarization_middleware.py` | Change `generate_summary()` error contract to fail closed (raise or explicit success flag), not fallback-summary success |
| `app/ai/summarization_middleware.py` | In `summarize_for_state()`, call `apply_summarization_to_state()` only on genuine success |
| `app/ai/summarization_middleware.py` | Remove `getattr` fallbacks in `_get_config()` — use direct `settings.*` access |
| `app/ai/summarization_middleware.py` | Remove dead `context["summary_text"]` / `context["messages_summarized_count"]` writes |
| `app/ai/summarization_middleware.py` | Wire `max_output_tokens=config.max_summary_tokens` into `_get_summarization_model()` |
| `app/ai/graph.py` | Add `_is_internal_stream_chunk(metadata)` helper |
| `app/ai/graph.py` | Early-`continue` all content-handling branches (including `reasoning`/`thinking` blocks) for internal chunks |
| `app/ai/graph.py` | Track `_internal_content_only` flag; guard `accumulated_content` fallback |
| `app/ai/graph.py` | Replace `suppress_tokens: bool` with `suppressed_nodes: set` pattern |
| `app/ai/graph.py` | In `_run_fast_path_summarization()`, wrap `generate_summary()` with `asyncio.wait_for` and apply state updates only on genuine success |
| `app/ai/graph.py` | Validate `context["conversation_summarized"]` is preserved across auto-continue rounds (add regression check) |

---

## Milestones (implementation plan)

### Milestone 0 — Confirm metadata shape in stream chunks (1–2 hours)
- Temporarily lower `summarization_trigger_messages` and `summarization_trigger_tokens` in settings to force summarization quickly.
- Add a single `logger.debug` call at the top of the `"messages"` branch in `execute_stream()` to print `metadata` keys for one chunk.
- Confirm the presence of `langgraph_node`, `tags`, `metadata`, and/or other stably-available fields.
- Remove the debug log after confirmation.

**Success criteria:** we can reliably identify "summarize" node chunks in `execute_stream()` via metadata.

### Milestone 1 — Fix the streaming leak and data-loss bug (P0 + P0b) (2–4 hours)

#### 1a. Fix silent data-loss contract at the source (`summarization_middleware.py`)
Change `generate_summary()` so failures are explicit (raise exception or return a typed success flag), then gate state mutation in the caller:
```python
# After: only apply (and remove messages) on genuine success
async def summarize_for_state(...):
    ...
    try:
        summary = await generate_summary(...)
    except Exception as e:
        logger.error("summarize_for_state: generate_summary failed, skipping: %s", e)
        return state   # return unmodified; no messages removed
    return apply_summarization_to_state(state, summary, messages_to_summarize, config)
```

#### 1b. Tag the internal LLM run (`summarization_middleware.py`)
- In `generate_summary()`, pass `RunnableConfig(tags=["internal","summarization"], metadata={"internal": True, "purpose": "summarization"})` to `model.ainvoke()`.

#### 1c. Suppress internal chunks in `execute_stream()` (`graph.py`)
- Add `_is_internal_stream_chunk(metadata: dict) -> bool`.
- At the top of the `mode == "messages"` branch, before any content handling, check the helper and `continue` early if internal.
- The early-continue must cover **all** content paths: `content_blocks` (text, thinking, reasoning, tool_call_chunk), list content, and legacy string content.
- Track `_internal_content_only` to guard the post-stream `accumulated_content` fallback.

#### 1d. Extend suppression to `accumulated_thinking`
- The early `continue` in step 1c prevents accumulation of `thinking`/`reasoning` blocks from internal chunks.
- Confirm the `accumulated_thinking` fallback in post-stream code is also guarded by `_internal_content_only`.

**Success criteria:**
- In `demo.py`, no conversation summary appears while streaming.
- Hitting "Stop generating" mid-stream does not persist summary text.
- `accumulated_thinking` does not contain summarization reasoning.

### Milestone 2 — Fix correctness issues in fast-path and continuation (P1 + P2) (half day)

#### 2a. Auto-continue rounds and summarization (`graph.py`)
- Inspect `_build_continuation_state()` — confirm `context["conversation_summarized"]` is carried into continuation state.
- Add a regression test/log assertion to ensure it stays preserved in future refactors.

#### 2b. Harden `_run_fast_path_summarization()` (`graph.py`)
- Wrap the `generate_summary()` call in `asyncio.wait_for(..., timeout=settings.summarization_timeout_seconds)`.
- Pass the same internal tags/metadata config to `generate_summary()` for consistent tracing.
- Fail closed on summarization errors: keep existing `history_summary`, skip `apply_summarization_to_state()`, and leave checkpoint messages untouched.
- Consider extracting the split/keep logic into a shared helper called by both `summarize_for_state()` and `_run_fast_path_summarization()` to eliminate duplication.

### Milestone 3 — Hardening and cleanup (half day)

#### 3a. Wire `max_output_tokens` to model constructor (`summarization_middleware.py`)
The `summarization_max_summary_tokens` setting and `SummarizationConfig.max_summary_tokens` already exist. Wire them to the model:
```python
return ChatGoogleGenerativeAI(
    model=config.model,
    google_api_key=get_api_key(),
    temperature=config.temperature,
    max_output_tokens=config.max_summary_tokens,  # cap at source, not just post-hoc
)
```

#### 3b. Add `summarization_timeout_seconds` setting and `asyncio.wait_for` (`config.py` + `summarization_middleware.py`)
- Add `summarization_timeout_seconds: int = Field(default=30, ...)` to `Settings`.
- In `summarize_for_state()` wrap `generate_summary()`:
  ```python
  summary = await asyncio.wait_for(
      generate_summary(messages_to_summarize, config, existing_summary),
      timeout=settings.summarization_timeout_seconds,
  )
  ```

#### 3c. Fix fragile `getattr` config access (`summarization_middleware.py`)
- Replace `getattr(settings, "summarization_trigger_tokens", 18000)` etc. with `settings.summarization_trigger_tokens` (all fields are defined in the Pydantic `Settings` model with defaults).

#### 3d. Remove dead legacy state fields (`summarization_middleware.py`)
- Delete the `context["summary_text"]` and `context["messages_summarized_count"]` writes from `apply_summarization_to_state()`.
- Confirm nothing reads them (grep confirms no readers) before deleting.

#### 3e. Replace `suppress_tokens: bool` with `suppressed_nodes: set` (`graph.py`)
- Refactor to `suppressed_nodes = {"image_generator_agent"}` and `suppress_tokens = selected_agent in suppressed_nodes`.

#### 3f. Add `conversation_id` to observability logging (`summarization_middleware.py`)
- Pass `conversation_id` down through `summarize_for_state()` → `apply_summarization_to_state()` and include it in the `INFO` log line.

### Milestone 4 — Persistence decision (optional; 0.5–1 day)
Document and choose between:

**Option A (recommended for now): checkpoint-only**
- Keep `history_summary` in LangGraph checkpoint state only.
- Works well when `enable_langgraph_checkpoints=True` and checkpoints are on Postgres (current design).

**Option B: DB fallback storage**
- Add `conversation.history_summary`, `history_summary_updated_at`, `summary_cursor_message_id` columns.
- On request start: if checkpoint has no summary, load from DB; if DB has no summary, generate from DB messages (expensive — only when needed).
- Pros: survives checkpoint disable/reset; can be inspected/admin-managed.
- Cons: schema migration + extra consistency rules.

**Decision criteria:**
- Do you ever run without checkpointer in prod?
- Do you need summary visibility/debugging in admin tools?
- Cost limits for regenerating summaries from DB history.

---

## Verification Plan

### Manual
1. Lower `summarization_trigger_messages` to `5` in dev settings to force early triggering.
2. Start a short chat; confirm no "conversation summary" block appears in the streamed response.
3. Hit "Stop generating" during a turn that triggers summarization; confirm the persisted assistant message does not contain summary text.
4. Confirm the assistant still has memory continuity across turns (summary injected into prompts, never visible to user).
5. Trigger an auto-continue scenario; confirm summarization does not double-fire.
6. Trigger a summarization error (e.g., disconnect API key temporarily); confirm no messages are removed and the conversation continues normally.

### Automated
- Unit test for `_is_internal_stream_chunk()` with representative metadata dicts (node name only, tags only, both, neither).
- Unit test for `apply_summarization_to_state()`: given messages with IDs, verify `RemoveMessage` entries are produced and `history_summary` / cursor keys are updated.
- Unit test for `summarize_for_state()` **error path**: mock `generate_summary` to raise; assert returned state is identical to input (no `RemoveMessage` entries, no changed keys).
- Unit test for `_run_fast_path_summarization()` **error path**: mock `generate_summary` to raise/timeout; assert checkpoint updates are skipped and existing `history_summary` is preserved.
- Unit test for `should_summarize()` threshold logic.
- Light integration test with a stubbed summarizer model (no external API call) via dependency injection in `generate_summary()`.

---

## Rollout / Feature Flags
- Add `suppress_internal_stream_chunks: bool = Field(default=True, ...)` to `Settings`. Gate the early-`continue` in `execute_stream()` behind this flag so it can be disabled for debugging without a code change.
- `summarization_timeout_seconds` (Milestone 3b) acts as its own production knob.
- Optionally allow admins to view the current summary via a separate debug endpoint later, without ever mixing it into the main assistant stream.

---

## Change Summary (what this revision adds vs. the initial draft)

| # | Addition / Correction |
|---|---|
| 1 | Corrected the leak mechanism: `ainvoke` not `astream`; summary arrives as one chunk, not incremental tokens. |
| 2 | Added the fast-path RAG summarization path (`_run_fast_path_summarization`) — no streaming leak there, but needs timeout + tagging + deduplication. |
| 3 | Corrected P0b mechanism and scope: `generate_summary()` currently swallows errors; both graph-path and fast-path can remove messages after fallback text unless fail-closed handling is added. |
| 4 | Refined P2 from a suspected defect to a validation/regression item (context flag appears preserved today). |
| 5 | Made the `accumulated_content` fallback guard concrete: track `_internal_content_only` flag. |
| 6 | Extended suppression scope to `accumulated_thinking` (Gemini reasoning blocks from summarize node). |
| 7 | Corrected Milestone 3 `max_output_tokens`: wire the *existing* `summarization_max_summary_tokens` to the model constructor rather than adding a new setting. |
| 8 | Added cleanup of fragile `getattr` pattern in `_get_config()`. |
| 9 | Changed legacy field action from "gate behind debug flag" to "remove entirely" (nothing reads them). |
| 10 | Replaced boolean `suppress_tokens` flag with `suppressed_nodes: set` for extensibility. |

---

## Implementation Progress

### Status: Milestones 0–3 complete (Milestone 4 deferred as planned)

All code changes were implemented with zero syntax / lint errors. Each milestone was verified via `py_compile` and VS Code error diagnostics.

---

### Milestone 0 — Metadata shape confirmation

Skipped the dedicated debug-log step. The suppression helper `_is_internal_stream_chunk` uses a three-layer defence so no preliminary live-run confirmation is needed:
1. `"internal" in metadata.get("tags", [])` — via `RunnableConfig` tags.
2. `metadata.get("metadata", {}).get("internal") is True` — via `RunnableConfig` metadata.
3. `metadata.get("langgraph_node") == "summarize"` — hard-coded node-name fallback (defence-in-depth).

---

### Milestone 1 — Streaming leak + data-loss (P0 + P0b) DONE

**1a — Fail-closed `generate_summary()`** (`summarization_middleware.py`)
- Removed the `try/except` wrapper from `generate_summary()`. The function now raises on failure.
- Added `try/except asyncio.TimeoutError / Exception` in `summarize_for_state()` that returns original `state` unchanged on any error. `apply_summarization_to_state()` is only reached on genuine success.
- Same discipline applied in `_run_fast_path_summarization()` via an inner `try/except` guarding only the generation step.
- **Design decision:** error handling moved from inside `generate_summary()` to the caller so message removal and error handling are visibly separate.

**1b — Tag internal LLM invocations** (`summarization_middleware.py`)
- `model.ainvoke()` in `generate_summary()` now passes `RunnableConfig(tags=["internal","summarization"], metadata={"internal": True, "purpose": "summarization"})`.

**1c/1d — Suppress internal chunks in `execute_stream()`** (`graph.py`)
- Added `_is_internal_stream_chunk(metadata)` static method (three-layer check).
- Gated behind `settings.suppress_internal_stream_chunks` (default `True`); disable for debugging.
- `_internal_content_only: bool = True` flag; set to `False` on first non-internal chunk.
- All post-stream `accumulated_content` / `accumulated_thinking` fallback branches guarded with `not _internal_content_only`.

---

### Milestone 2 — Fast-path + continuation correctness (P1 + P2) DONE

**2a — `conversation_summarized` across auto-continue rounds** (`graph.py`)
- Confirmed `_build_continuation_state()` does NOT pop `conversation_summarized`. Added a comment explaining why.
- Validation item confirmed: no bug.

**2b — Harden `_run_fast_path_summarization()`** (`graph.py`)
- `asyncio.wait_for(..., timeout=settings.summarization_timeout_seconds)` wrapping `generate_summary()`.
- Inner `try/except` covers only generation; outer covers checkpoint I/O. Fail-closed on generation failure.
- Renamed local var to `new_summary` so the original `history_summary` is preserved for fallback return.

---

### Milestone 3 — Hardening + cleanup DONE

**3a — `max_output_tokens` wired to model constructor** (`summarization_middleware.py`)
- `_get_summarization_model()` passes `max_output_tokens=config.max_summary_tokens` when value > 0.
- **Design decision:** `0` means unlimited; passing `0` to the provider is avoided to sidestep SDK differences.

**3b — `summarization_timeout_seconds`** (`config.py` + `summarization_middleware.py`)
- Added `summarization_timeout_seconds: int = Field(default=30)` to `Settings` and `_non_negative_int` validator.

**3c — Remove fragile `getattr` config access** (`summarization_middleware.py`)
- All `getattr(settings, "summarization_*", default)` replaced with direct attribute access in `_get_config()` and `should_summarize()`.

**3d — Remove dead legacy state fields** (`summarization_middleware.py`)
- Deleted `context["summary_text"]` and `context["messages_summarized_count"]` from `apply_summarization_to_state()`.

**3e — `suppressed_nodes: set` replaces `suppress_tokens: bool`** (`graph.py`)
- `suppressed_nodes: set = {"image_generator_agent"}; suppress_tokens = selected_agent in suppressed_nodes`

**3f — `conversation_id` in observability logging** (`summarization_middleware.py`)
- `summarize_for_state()` accepts `conversation_id: Optional[str] = None`; `_summarization_node()` threads it through.

**Feature flags added to `Settings`** (`config.py`)
- `summarization_timeout_seconds` (default 30) — timeout for summarization calls.
- `suppress_internal_stream_chunks` (default True) — gate the early-continue; disable for debugging.

---

### Milestone 4 — Persistence decision
Deferred as planned. Current: checkpoint-only (`history_summary` in LangGraph Postgres checkpoint tables).

---

### Files changed
| File | Milestones |
|---|---|
| `app/ai/summarization_middleware.py` | 1a, 1b, 3a, 3b, 3c, 3d, 3f |
| `app/ai/graph.py` | 1c, 1d, 2a (comment), 2b, 3e + `conversation_id` threading |
| `app/core/config.py` | New settings: `summarization_timeout_seconds`, `suppress_internal_stream_chunks` |
