# Context Window Indicator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show each completed assistant response's current request context usage as a small circular indicator, backed by provider-aware context-window metadata that works through the server and sidecar backend.

**Architecture:** The canonical server owns model context metadata. Provider catalog sync enriches models with known input/output token limits, runtime resolution carries the selected model's context window, agent invocation combines that limit with existing token breakdown metadata, and `demo.py` renders a compact circle beside the assistant provider/model label. The sidecar remains a transparent proxy for model-config, provider catalog, message stream, and AI SDK message metadata.

**Tech Stack:** FastAPI, Pydantic v2, SQLAlchemy-backed provider metadata, LangChain model responses, Streamlit demo UI, pytest, httpx sidecar SSE proxy.

---

## Approved Scope

- Display a small circle in the Streamlit demo next to the assistant message's provider/model label.
- Circle represents current request usage as a percentage of the resolved model context window.
- Display after message completion, using persisted assistant message metadata.
- Use a backend-owned contract so a future non-Streamlit frontend can render the same indicator.
- Use a static registry plus provider API metadata. Unknown/custom model IDs get a neutral unknown state.
- Preserve sidecar behavior by passing enriched metadata through existing proxy and SSE paths.

## Current Code Findings

- Provider catalog normalization is in `app/services/provider_service.py`.
- The public provider/model API shapes are duplicated in `app/api/providers.py` and `app/api/model_config.py` as `ProviderModelOption`.
- Runtime model selection is resolved by `app/services/model_config_service.py` into `ResolvedRuntimeModelConfig` in `app/core/runtime_modeling.py`.
- Existing assistant metadata already includes `provider`, `model`, `provider_fallback`, and `token_breakdown` from `app/ai/agents/base_agent.py`.
- Token estimates and provider-reported usage extraction live in `app/ai/token_instrumentation.py`.
- Persisted messages include arbitrary `message_metadata`, so this feature should not require a DB migration.
- `client_backend/api/proxy.py` already proxies `/providers`, `/providers/{provider_type}/models`, `/model-config`, and `/model-config/options`.
- `client_backend/services/server_api.py` parses SSE JSON and drops only heartbeat events, so enriched `complete.message.metadata` should pass through unchanged.
- `demo.py` renders assistant provider/model captions in `render_message_bubble()` and model catalog rows in `render_models_view()`.

## Provider Compatibility Review

- Gemini: the Models API exposes `inputTokenLimit` and `outputTokenLimit`, and token docs say model token limits can be determined via the models endpoint. Implementation should read these fields during sync when present.
- OpenAI: the Models API object includes `id`, `created`, `object`, and `owned_by`; context-window limits are documented in model comparison/details pages, not returned by the model object. Implementation needs a local registry/pattern fallback for OpenAI.
- Anthropic: this project stores Anthropic provider credentials, but `ModelConfigService.SUPPORTED_PROVIDERS` currently includes only Gemini and OpenAI. Anthropic docs state model context windows are available in the model comparison table and programmatically through the Models API. Implementation should design the metadata model to support Anthropic later, but not claim runtime support until the existing model-config path includes Anthropic.

Sources checked:
- OpenAI Models API object: https://developers.openai.com/api/reference/resources/models
- OpenAI model comparison context-window table: https://developers.openai.com/api/docs/models/compare
- Gemini Models API token-limit fields: https://ai.google.dev/api/models#v1beta.models.list
- Gemini token counting and model token-limit docs: https://ai.google.dev/gemini-api/docs/tokens
- Anthropic model context-window metadata: https://platform.claude.com/docs/en/about-claude/models/overview

## Data Contract

Add this optional model metadata shape to normalized provider catalog entries and API responses:

```json
{
  "contextWindowTokens": 128000,
  "maxInputTokens": 128000,
  "maxOutputTokens": 16384,
  "contextWindowSource": "provider_api",
  "contextWindowKnown": true
}
```

Accepted `contextWindowSource` values:

- `provider_api`: returned by provider model metadata, currently expected for Gemini and future Anthropic.
- `registry`: matched from the local static registry.
- `heuristic`: matched from a conservative family-level pattern.
- `unknown`: custom or unrecognized model.

Add this optional assistant message metadata shape:

```json
{
  "context_window": {
    "provider": "openai",
    "model": "gpt-4o",
    "context_window_tokens": 128000,
    "max_input_tokens": 128000,
    "max_output_tokens": 16384,
    "source": "registry",
    "known": true,
    "used_tokens": 12000,
    "used_token_source": "actual_input",
    "usage_ratio": 0.09375,
    "display_state": "ok"
  }
}
```

Display states:

- `unknown`: no reliable context window.
- `ok`: below 70 percent.
- `warn`: 70 percent through 89 percent.
- `danger`: 90 percent or above.

Usage source priority:

1. `token_breakdown.actual.input_tokens`
2. `token_breakdown.estimated.total_tokens`
3. unknown

## File Structure

- Create `app/ai/model_context.py`: pure helpers for context-window registry lookup, provider metadata normalization, usage calculation, and display state.
- Modify `app/services/provider_service.py`: enrich normalized OpenAI/Gemini catalog entries with context-window fields.
- Modify `app/api/providers.py`: add context-window fields to `ProviderModelOption`.
- Modify `app/api/model_config.py`: add the same fields to `ProviderModelOption`.
- Modify `app/services/model_config_service.py`: expose model context metadata from catalog or fallback lookup when resolving runtime config.
- Modify `app/core/runtime_modeling.py`: add optional `context_window` metadata to `ResolvedRuntimeModelConfig`.
- Modify `app/ai/agents/base_agent.py`: attach final `context_window` metadata after actual usage extraction and runtime fallback resolution.
- Review `app/ai/agents/chat_agent.py`, `app/ai/agents/rag_agent.py`, and `app/ai/agents/planning_agent.py`: these call `_apply_runtime_metadata()` in additional paths; ensure context metadata is present for image/RAG/planning responses that bypass `invoke_model_with_history()`.
- Modify `demo.py`: render a small context usage circle next to the assistant provider/model label and optionally add context-window columns in the Models tab.
- Test with new or updated pytest coverage under `tests/` and `tests/client_backend/`.

## Tasks

### Task 1: Add Pure Context Metadata Helpers — DONE

**Files:**
- Create: `app/ai/model_context.py`
- Test: `tests/test_model_context_metadata.py`

- [x] Write tests for exact registry lookup, case-insensitive model IDs, unknown custom IDs, and provider API metadata taking precedence over registry metadata.
- [x] Implement a `ModelContextWindow` dataclass or typed dict with `provider`, `model`, `context_window_tokens`, `max_input_tokens`, `max_output_tokens`, `source`, and `known`.
- [x] Add a conservative registry for known OpenAI model families already used by the app, including `gpt-4o`, `gpt-4.1`, `gpt-5`, `o1`, `o3`, and `o4` families.
- [x] Add registry structure for Gemini and Anthropic families without depending on those providers being configured.
- [x] Implement `normalize_context_window_metadata(provider, model_id, raw_metadata)` that accepts provider API fields using snake_case or camelCase.
- [x] Implement `resolve_model_context_window(provider, model_id, catalog_metadata=None)` that returns known metadata or an unknown result.
- [x] Implement `build_context_window_usage(context_window, token_breakdown)` that chooses actual input tokens over estimated total tokens and returns `usage_ratio` and `display_state`.
- [x] Run `pytest tests/test_model_context_metadata.py` and confirm failures before implementation, then passing after implementation. **37/37 passing.**

**Design decisions:**
- Registry lookup uses exact match first, then longest-matching family prefix (so `gpt-4o-mini-2024-07-18` → `gpt-4o-mini` not `gpt-4o`).
- Catalog precedence requires a real signal (one of the recognized window keys with a positive int, or `context_window_known=True`). Empty/unrelated catalog dicts fall through to the registry.
- `normalize_context_window_metadata` picks the largest of multiple context candidates (defensive against under-reporting); `max_input_tokens` set equal to `context_window_tokens` when only one provided.
- `build_context_window_usage` returns the full "unknown" payload (used_tokens=None, display_state="unknown") when window is None/unknown, matching the spec.
- Denominator in usage ratio prefers `max_input_tokens` then falls back to `context_window_tokens`.
- `o4` family currently covers `o4-mini` only — bare `o4` was not publicly released as of the cutoff.

### Task 2: Enrich Provider Catalog Sync — DONE

**Files:**
- Modify: `app/services/provider_service.py`
- Test: `tests/test_provider_model_context_metadata.py`

- [x] Write tests for `_normalize_openai_model()` adding registry-derived context fields.
- [x] Write tests for `_normalize_gemini_model()` adding `input_token_limit` and `output_token_limit` from fake Gemini model metadata.
- [x] Update `_normalize_openai_model()` to call the new helper with registry fallback.
- [x] Update `_normalize_gemini_model()` to accept `input_token_limit` and `output_token_limit` parameters.
- [x] Update `_fetch_gemini_models_sync()` to pass `getattr(model, "input_token_limit", None)` and `getattr(model, "output_token_limit", None)`.
- [x] Keep catalog entries backward-compatible when older cached metadata lacks the new fields.
- [x] Run `pytest tests/test_provider_model_context_metadata.py`. **15/15 passing.**

**Design decisions:**
- Added two private helpers `_context_window_fields` and `_resolve_catalog_context_window` on `ProviderService` so OpenAI and Gemini normalizers share one path.
- Helpers always emit the 5 new keys (None / False / "unknown" when unknown), making downstream consumers' fallback paths trivial.
- Gemini provider-API metadata (`input_token_limit` / `output_token_limit`) takes precedence; falls back to registry when absent or unknown.
- `_fetch_gemini_models_sync` uses `getattr(model, ...)` so the catalog still builds if the SDK ever drops or renames those attributes.
- Test isolation uses `monkeypatch.setitem(sys.modules, "google.genai", fake_module)` to avoid real API hits.

### Task 3: Extend API Schemas — DONE

**Files:**
- Modify: `app/api/providers.py`
- Modify: `app/api/model_config.py`
- Test: `tests/test_provider_model_context_metadata.py`

- [x] Add optional fields to both `ProviderModelOption` classes: `context_window_tokens`, `max_input_tokens`, `max_output_tokens`, `context_window_source`, and `context_window_known`.
- [x] Confirm camelCase output aliases are produced through the existing `CamelModel` alias generator.
- [x] Add tests that validate a provider model payload serializes `contextWindowTokens`, `maxInputTokens`, `maxOutputTokens`, `contextWindowSource`, and `contextWindowKnown`.
- [x] Run the schema tests. **25/25 passing in this file; 62/62 across Tasks 1-3.**

**Design decisions:**
- Field types: `int | None`, `str | None`, `bool | None`, all defaulting to `None`. The tri-state on `context_window_known` (`None` legacy / `False` we looked and don't know / `True` known) distinguishes old cached payloads from explicit unknowns.
- `context_window_source` typed as `str | None` rather than `Literal[...]` for now — server-controlled, but consider tightening later.
- Two parallel `ProviderModelOption` classes exist (pre-existing tech debt) — kept in sync manually.
- Also fixed Task 2 mock setup: `_install_fake_genai` now also patches `google.genai` attribute via `monkeypatch.setattr` because `from google import genai` does `getattr` first before falling back to `sys.modules`.

### Task 4: Thread Context Metadata Through Runtime Resolution — DONE

**Files:**
- Modify: `app/core/runtime_modeling.py`
- Modify: `app/services/model_config_service.py`
- Test: `tests/test_runtime_model_overrides.py`

- [x] Add `context_window: dict[str, Any] | None = None` to `ResolvedRuntimeModelConfig`.
- [x] Add a private helper in `ModelConfigService` to resolve context metadata from the provider snapshot and selected model.
- [x] In `resolve_runtime_config()`, compute context metadata after final provider/model/fallback selection.
- [x] Include context metadata for `user_id is None` default runtime config.
- [x] Include context metadata in runtime fallback configs created by `_build_runtime_fallback_candidate()` or immediately after fallback selection.
- [x] Add tests that runtime overrides, fallback provider selection, default Gemini resolution, and custom models expose expected known/unknown context metadata.
- [x] Run `pytest tests/test_runtime_model_overrides.py`. **17/17 passing (12 pre-existing + 5 new); 79/79 across Tasks 1-4.**

**Design decisions:**
- Single helper `_resolve_context_window_metadata(provider, model_id, provider_snapshot)` that prefers catalog metadata then falls back to the registry.
- `context_window` computed once, AFTER the api_key-driven fallback swap, so it reflects the actual final provider/model — not the originally-requested one.
- `RuntimeFallbackConfig` left untouched — only `ResolvedRuntimeModelConfig` carries context_window. The post-swap helper call captures the final state regardless.
- `user_id is None` branch passes an empty snapshot `{}` forcing registry-only resolution (no provider catalog without a user).
- `base_agent.py` untouched — agent-side fallback metadata propagation is Task 5.

### Task 5: Persist Per-Message Usage Metadata — DONE

**Files:**
- Modify: `app/ai/agents/base_agent.py`
- Review/modify if needed: `app/ai/agents/chat_agent.py` — NOT modified (static fields only via `_apply_runtime_metadata`).
- Review/modify if needed: `app/ai/agents/rag_agent.py` — NOT modified.
- Review/modify if needed: `app/ai/agents/planning_agent.py` — NOT modified.
- Test: `tests/test_context_window_message_metadata.py`

- [x] Write tests for `_apply_runtime_metadata()` including model context metadata when present.
- [x] Write tests for final assistant metadata adding `context_window.used_tokens`, `used_token_source`, `usage_ratio`, and `display_state` from token breakdown.
- [x] Ensure fallback runtime changes recalculate the context metadata for the actual provider/model that answered.
- [x] Ensure unknown/custom model metadata is persisted as `known: false` with no misleading usage ratio.
- [x] Update `_apply_runtime_metadata()` to include the static context metadata.
- [x] After `extract_actual_usage(response)` updates `token_breakdown`, compute and attach usage metadata to `metadata["context_window"]`.
- [x] Apply the same helper in agent paths that build metadata outside `invoke_model_with_history()`, or document why those paths inherit `_apply_runtime_metadata()` plus no token usage. **Documented in docstring of `_merge_context_window_usage`.**
- [x] Run `pytest tests/test_context_window_message_metadata.py tests/test_context_overflow_retry.py`. **10 + existing tests passing; 91 total across Tasks 1-5.**

**Design decisions:**
- New `_merge_context_window_usage(metadata, token_breakdown)` helper on `BaseAgent` keeps merge logic unit-testable.
- `_apply_runtime_metadata` adds static context fields via `dict(...)` copy — runtime_config not mutated.
- `_create_fallback_runtime_config` uses the static registry only (no provider catalog access at that layer); custom/unknown fallback models correctly emit `known=false`.
- Other agent paths (chat vision, agentic RAG, planning emission) intentionally skip usage merge because they don't assemble a `token_breakdown`. Result: those messages carry static fields (provider, model, window limits, source, known) but lack used_tokens/usage_ratio/display_state. Decision documented in `_merge_context_window_usage` docstring.
- `build_context_window_usage` always returns a payload (unknown payload when known=False), so merging is unconditional once `metadata["context_window"]` exists.

### Task 6: Verify Sidecar Pass-Through — DONE

**Files:**
- Modify: `tests/client_backend/test_server_api.py`
- Modify or add: `tests/client_backend/test_messages.py`
- No production sidecar code expected unless tests reveal filtering.

- [x] Add a `stream_sse()` test proving non-heartbeat events with nested `message.metadata.context_window` pass through unchanged.
- [x] Add a `/messages/stream` sidecar route test proving `complete.message.metadata.context_window` is not stripped when proxied.
- [x] Add a model-config proxy test (existing proxy tests stub `proxy_server_request` itself, so the helper's body pass-through was uncovered).
- [x] Run `pytest tests/client_backend/test_server_api.py tests/client_backend/test_messages.py`. **7/7 passing.**

**Design decisions:**
- No production sidecar code changes — pass-through already works.
- `stream_sse` only filters heartbeats; `_build_sse_response` serializes whole events via `json.dumps`; `proxy_server_request` returns upstream body as `JSONResponse(content=response.json())`. All three layers preserve arbitrary nested fields.
- Tests stub boundaries (httpx `client.stream`, upstream server client, `request_response`) and exercise the real pass-through layer above. No "stub-the-thing-you-test" anti-pattern.
- All three new tests use deep dict equality (not substring/key presence) for the 11-field `context_window` payload.

### Task 7: Render The Streamlit Circle — DONE

**Files:**
- Modify: `demo.py`

- [x] Add CSS classes for a compact inline context circle: fixed 12-16 px size, no layout shift, neutral unknown state, ok/warn/danger colors, accessible title text. CSS appended to `APP_STYLE` (one-time injection at module load).
- [x] Add helper functions near existing message rendering helpers (`_get_context_window_metadata`, `_format_context_window_label`, `_render_context_window_indicator`, plus bonus `_format_tokens`).
- [x] Replace the assistant provider/model `st.caption()` call with a small HTML row that contains the provider/model text and the circle.
- [x] Keep `st.caption()` fallback if metadata is absent or malformed.
- [x] Use `unsafe_allow_html=True` only with escaped provider/model/title strings — both label and tooltip use `html.escape`; display_state allowlisted to {ok, warn, danger, unknown}.
- [x] Added `Context Window` and `Max Output` columns to the Models tab dataframe.
- [ ] Manual Streamlit verification — deferred (cannot run interactively in this session); test plan documented for the user.

**Design decisions:**
- CSS lives in the existing `APP_STYLE` global, injected once at module load.
- Display state always rendered (defaults to "unknown" with transparent fill and neutral gray border).
- HTML escaping covers all interpolated strings; `display_state` is allowlisted before being concatenated into the class attribute.
- Dataframe columns are unconditional — empty string for missing values keeps layout stable across providers.
- Both `title` and `aria-label` set for accessibility.

**Manual verification plan (run from a fresh shell):**
```
streamlit run demo.py
```
Then verify:
1. Low-usage known metadata → solid green circle, tooltip like `12k / 128k tokens (9%) - actual input`.
2. High-usage → amber (warn) or red (danger) circle with corresponding tooltip.
3. Unknown metadata → transparent circle with gray border; tooltip `Window unknown`.
4. Settings → Models tab shows `Context Window` and `Max Output` columns populated for synced providers.

### Task 8: AI SDK And Future Frontend Contract — DONE

**Files:**
- Review: `app/api/ai_sdk.py` — no production changes needed.
- Test: created `tests/test_ai_sdk_context_window.py`.

- [x] Confirm `get_conversation_messages_ai_sdk()` mirrors `message_metadata` to both `messageMetadata` and `metadata`. Lines 1060-1062 already mirror — no code change needed.
- [x] Add a regression test that persisted `context_window` appears in AI SDK `messageMetadata` for assistant messages. **2 tests passing.**
- [x] Do not add live streaming context events; documented as future extension in test docstring.

**Design decisions:**
- Direct async invocation with mocked `IMessageService` — avoids spinning up full DI container.
- `SimpleNamespace` for the fake message object (Pydantic plays nicer with duck-typed attrs than `MagicMock`).
- Asserts `model_dump(by_alias=True)` to inspect actual camelCase wire shape.
- Two tests: one pins the `context_window` sub-payload on both keys, the second pins `messageMetadata == metadata == original` so future selective-mirror refactors fail loudly.

### Mid-implementation bug fix: stale catalog backfill

Added `_backfill_context_window` to `ProviderService` so cached catalog entries pre-dating Task 2 transparently fill in `context_window_*` fields from the registry on read. Without this, the Streamlit demo's new Context Window / Max Output columns stayed blank until users manually re-synced providers. Wired into `get_cached_provider_models` and `get_cached_provider_status`. 4 new tests added.

Registry values verified against current public docs (no values changed):
- OpenAI gpt-5: 400k / 128k. gpt-4.1: 1,047,576 / 32,768. gpt-4o: 128k / 16,384. o1/o3/o4: 200k / 100k.
- Gemini 2.5 Pro: 1,048,576 / 65,536. Gemini 2.5 Flash: same. Gemini 3 preview: same conservative default.

### Task 9: End-To-End Verification — DONE

**Files:**
- No new files expected.

- [x] Run targeted backend tests:
  ```powershell
  pytest tests/test_model_context_metadata.py tests/test_provider_model_context_metadata.py tests/test_runtime_model_overrides.py tests/test_context_window_message_metadata.py tests/test_ai_sdk_context_window.py
  ```
  **Result: 95 passed.**

- [x] Run targeted sidecar tests:
  ```powershell
  pytest tests/client_backend/test_server_api.py tests/client_backend/test_messages.py
  ```
  **Result: 7 passed.**

- [x] Run existing context-overflow tests:
  ```powershell
  pytest tests/test_context_overflow_retry.py
  ```
  **Result: 2 passed.**

- [x] Run lint/format checks. Ruff is clean on all files touched by this work. The 2 E501 errors in `app/ai/agents/base_agent.py` (lines 186, 597) and 2 in `app/services/provider_service.py` (lines 74, 89) are PRE-EXISTING (verified via `git blame` — commits from Dec 2025 and March 2026, before this feature).

- [x] Combined feature suite passes: **104 tests passing** across `test_model_context_metadata`, `test_provider_model_context_metadata`, `test_runtime_model_overrides`, `test_context_window_message_metadata`, `test_ai_sdk_context_window`, `test_context_overflow_retry`, `tests/client_backend/test_server_api`, `tests/client_backend/test_messages`.

- [x] `demo.py` syntax check passes via `ast.parse`.

- [ ] Launch the Streamlit demo and verify no visual overlap in assistant messages. **Deferred — cannot run Streamlit interactively in this session.** Manual verification plan documented in Task 7.

## Summary

All 9 tasks complete. The feature surface introduces:
- Pure helpers in `app/ai/model_context.py` with registry + provider-API normalization, usage calculation, and display-state mapping.
- Provider catalog enrichment (`provider_service.py`) — every model entry carries the 5 context-window fields, with backfill on read so stale caches transparently fill from registry.
- API schemas extended (`providers.py`, `model_config.py`) with optional camelCase-aliased fields.
- Runtime resolver (`model_config_service.py`, `runtime_modeling.py`) threads `context_window` through `ResolvedRuntimeModelConfig` after the final fallback swap.
- Agent metadata (`base_agent.py`) attaches static + usage fields after `extract_actual_usage` updates `token_breakdown`; fallback paths also carry context_window via the static registry.
- Sidecar verified pass-through (no code changes — `stream_sse`, `_build_sse_response`, `proxy_server_request` all preserve nested fields).
- Streamlit demo (`demo.py`) renders an inline 12 px ok/warn/danger/unknown circle next to the provider/model caption, with HTML-escaped tooltip. Models tab gains Context Window and Max Output columns.
- AI SDK contract test guards `messageMetadata == metadata == message_metadata` so the wire shape stays mirrored.

Total new tests: 73 across 7 test files. Total feature-suite pass count: **104/104**.

## Post-implementation bug fix: `extract_actual_usage` shape mismatch

User reported the displayed context usage looked "miscalculated" and "not accumulated through the conversation". Root cause was in `app/ai/token_instrumentation.py:extract_actual_usage`:

- LangChain ≥ 0.2 makes `AIMessage.usage_metadata` a `UsageMetadata` TypedDict (a plain `dict` at runtime).
- The old code probed it via `hasattr(usage, "input_tokens")` — which is always `False` for a dict.
- Result: `token_breakdown.actual_input_tokens` stayed `None`, so `_merge_context_window_usage` fell back to the rough `estimated.total_tokens` heuristic (chars/4). The displayed value was both inaccurate AND grew at the wrong rate.

Fix: rewrote `extract_actual_usage` to try dict-subscript access first (handling the standard LangChain `UsageMetadata` shape), then attribute access for older providers, then fall back to `response_metadata.usage`/`token_usage` envelopes. Added 8 regression tests in `tests/test_context_window_message_metadata.py` covering every input shape.

Why this fixes "accumulation": the provider's `input_tokens` on each call IS the cumulative prompt size for that turn (system + full history + current + tools). It naturally grows as history grows — until trimmed by `chat_history_max_messages` (default 24) or `chat_history_max_tokens` (default 9000), at which point the displayed value plateaus near the budget. That plateau is by design and reflects what the model actually sees.

Final feature-suite pass count: **112/112**.

## Acceptance Criteria

- Synced model catalog responses include context-window metadata when known and safe unknown metadata when not known.
- Runtime assistant message metadata records the actual provider/model context window after fallback resolution.
- Message metadata includes context usage based on provider-reported input tokens when available, otherwise the existing estimate.
- Streamlit demo shows a small circle next to assistant provider/model labels for completed assistant messages.
- Unknown/custom models render a neutral circle and never display fabricated token limits.
- Sidecar backend passes provider/model-config metadata and completed message context metadata unchanged.
- AI SDK message history exposes `context_window` through existing `messageMetadata` and `metadata` fields.
- Existing provider fallback, custom model override, context-overflow retry, and sidecar streaming behavior continue to pass tests.

## Risks And Mitigations

- Provider context windows change over time. Keep the registry centralized in `app/ai/model_context.py`, prefer provider API metadata when available, and mark registry-derived data with `contextWindowSource: "registry"`.
- OpenAI custom or newly released models may not match the registry. Use unknown state unless an explicit registry or conservative heuristic match exists.
- Anthropic is partially present in provider storage but not in model runtime config. Keep helper support provider-agnostic, but do not add Anthropic runtime UI selection in this feature.
- Token estimates are approximate. Label estimated usage in tooltips and prefer provider-reported actual input tokens after completion.
- Streamlit HTML rendering can introduce unsafe text if not escaped. Escape all provider/model/title text before passing `unsafe_allow_html=True`.

## Deferred Work

- Live streaming context-window updates before completion.
- Provider-specific preflight token counting calls before generation.
- Anthropic runtime/provider catalog integration if the app expands `ModelConfigService.SUPPORTED_PROVIDERS`.
- Full frontend implementation outside `demo.py`.
