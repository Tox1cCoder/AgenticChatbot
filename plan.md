# Implementation Plan: Persistent Per-Agent Model Configuration (Per-User)

## Clarifying questions (please confirm)

1. **Scope:** Is this config **global per user across all conversations** (assumed ✅), or should it be per-conversation?
2. **Agents:** Should config apply only to **chat/rag/search/planning** (assumed ✅), or include image generation / router too?
3. **Overrides:** Remove **per-message overrides** from the Streamlit UI so model selection is unified via the Models tab (keep backend `modelConfig` support for API compatibility).
4. **Tuning:** Persist **temperature** per agent (assumed ✅). Any other knobs (max tokens, top_p)?
5. **Reasoning UI:** Is it acceptable to render **OpenAI reasoning summaries** (not full chain-of-thought) in the UI (assumed ✅)?

---

## 1) Target behavior (what you want)

### Models tab

- Shows a row per agent: **Chat**, **RAG**, **Search**, **Planning**.
- For each agent, user can pick **provider + model** (and temperature).
- Config is **persisted** per user and reused automatically for all messages.

### Provider keys

- OpenAI API key is stored **in DB per user** (encrypted).
- Gemini remains the **default + fallback** and is usable even if the user never stores any key.

### Runtime selection rules

- If a user configured an agent to use OpenAI but has no OpenAI key: **fallback to Gemini**.
- If OpenAI fails after retries: **fallback to Gemini** and surface fallback metadata in the UI.

---

## 2) Current codebase state (what already exists)

Already implemented and reusable:

- Provider key storage in DB (encrypted): `app/models/model_provider.py`, `app/services/provider_service.py`
- Provider APIs: `app/api/providers.py`
  - `POST /providers` (upsert key)
  - `DELETE /providers/{provider_type}`
  - `GET /providers/{provider_type}/models` (OpenAI model listing)
- Runtime can already consume a `model_request` mapping (currently fed via per-message `modelConfig`):
  - Schema/plumbing: `app/schemas/message.py`, `app/ai/schemas.py`, `app/services/message_service.py`, `app/ai/graph.py`
  - Agent selection: `app/ai/agents/base_agent.py`, `app/ai/agents/rag_agent.py`
- Demo uses the Models tab for per-agent configuration (no per-message selector in `demo.py`).

---

## 3) Proposed architecture (persistent per-agent config)

### 3.1 Data model

Keep `model_providers` for encrypted provider keys (already exists).

Add a **new DB table** for persistent per-agent selection:

- Table name: `agent_model_configs` (or `user_agent_model_configs`)
- Columns:
  - `id` (UUID PK)
  - `user_id` (UUID FK)
  - `agent_key` (string enum: `chat|rag|search|planning`)
  - `provider_type` (string enum: `gemini|openai` for now)
  - `model` (string, no validation by default)
  - `temperature` (float, nullable)
  - `created_at`, `updated_at`
- Constraint:
  - `UNIQUE(user_id, agent_key)` (so “save” is an upsert per agent)

Implementation files:

- `app/models/agent_model_config.py`
- `app/repositories/agent_model_config.py`
- `app/services/model_config_service.py`
- Alembic migration: `app/alembic/versions/*_add_agent_model_configs_table.py`

### 3.2 API design

Add a small “model config” API that the demo can call:

- `GET /model-config`
  - Returns a mapping for `chat/rag/search/planning`.
  - Should include “effective” values (DB override if present, else defaults from `AGENT_CONFIG`).
- `PATCH /model-config`
  - Upserts one or more agent configs.
  - Accepts partial updates (only modified agents).
- `POST /model-config/reset`
  - Deletes user configs (revert to defaults).

Schemas:

- `AgentModelConfig` (provider, model, temperature)
- `ModelConfigReadResponse` (mapping)
- `ModelConfigUpdateRequest` (partial mapping)

### 3.3 Runtime integration

Goal: apply **persistent config automatically** without requiring `modelConfig` on every message.

Proposed precedence:

1. **Per-message** `modelConfig` (if present) — optional/advanced
2. **Persistent per-agent config** (DB)
3. **Defaults** (Gemini + `AGENT_CONFIG`)

Implementation approach (minimal churn):

- In `MessageService` (both non-stream and stream paths):
  - If request includes `modelConfig`, keep using it.
  - Else load persistent mapping once via `ModelConfigService.get_effective_model_request(user_id)` and pass `model_request` into `AIService`.
- Do **not** fetch config inside each agent to avoid extra DB calls per tool loop.

### 3.4 Streamlit demo redesign

Models tab should become the single place to configure agent models:

1. **Provider keys**
   - OpenAI key: save/update/delete
   - Button to refresh OpenAI models
2. **Agent model mapping**
   - A grid/table with 4 rows (chat/rag/search/planning)
   - For each row:
     - Provider dropdown: `Gemini (default/fallback)` vs `OpenAI`
     - Model selector:
       - If OpenAI: dropdown from `/providers/openai/models` with manual override text input
       - If Gemini: a short dropdown of known working Gemini models + manual override text input
     - Temperature slider
   - Save + Reset buttons

Chat composer:

- Do not expose per-message model overrides in the Streamlit UI (model config is unified via the Models tab).

---

## 4) OpenAI “reasoning” rendering (docs + implementation plan)

### 4.1 What’s available (verified via SDK/runtime inspection)

OpenAI Responses API supports a request param `reasoning`:

- `effort`: `none|minimal|low|medium|high|xhigh`
- `summary`: `auto|concise|detailed`

The response can include:

- `ResponseReasoningItem` in `response.output` with `summary[].text`
- `response.usage.output_tokens_details.reasoning_tokens` (reasoning token count)

LangChain `ChatOpenAI` also supports a `reasoning` field and will switch to the Responses API when `reasoning` is set (see `ChatOpenAI._use_responses_api()`).

### 4.2 Plan to surface reasoning in UI

Backend:

- Decide a default behavior for OpenAI:
  - Either always request summaries when OpenAI is selected, or add a per-agent toggle later.
- When provider is OpenAI:
  - Pass `reasoning={"summary": "auto"}` (or user-selected `concise/detailed`) into ChatOpenAI creation.
  - Parse LangChain `AIMessage.content` blocks for items where `type == "reasoning"` and extract `summary[].text`.
  - Attach to assistant message metadata:
    - `reasoning_summary` (string)
    - `reasoning_tokens` (int, if available)
- Stream SSE:
  - Either include `reasoning_summary` in the final assistant message metadata, or emit an SSE event `type="reasoning_summary"` (similar to existing Gemini thinking UI).

Frontend (Streamlit):

- If assistant message metadata includes `reasoning_summary`, render it under an expander (“Reasoning (summary)”).
- Optionally show `reasoning_tokens` as a small caption (no chain-of-thought).

Important note:

- This plan aims to display **reasoning summaries only** (what the API explicitly returns), not internal chain-of-thought.

---

## 5) Implementation steps (spec-kit checklist)

### Phase A — Persisted config (DB)

- [x] Add `AgentModelConfig` SQLAlchemy model
- [x] Add Alembic migration for `agent_model_configs`
- [x] Add repository (upsert/get/reset)
- [x] Add service:
  - [x] `get_effective_model_request(user_id)` -> mapping compatible with existing `model_request`

### Phase B — API endpoints

- [x] Add router `app/api/model_config.py` (or extend `app/api/providers.py`)
- [x] Add `GET /model-config`, `PATCH /model-config`, `POST /model-config/reset`
- [x] Wire into `app/main.py` + DI container + `AppAutoInjector` wiring map

### Phase C — Runtime wiring

- [x] In `MessageService`, if no per-message `modelConfig`, load persistent config and pass to `AIService`
- [x] Ensure retries + fallback metadata remains consistent (`provider_fallback`)
- [x] Ensure Gemini works even when user has no stored key

### Phase D — Demo UI redesign

- [x] Replace Models tab UI with per-agent mapping editor
- [x] Preload existing mapping from `GET /model-config`
- [x] Save mapping via `PATCH /model-config`
- [x] Reset mapping via `POST /model-config/reset`
- [x] Remove per-message override UI from chat composer

### Phase E — OpenAI reasoning summaries

- [x] Add OpenAI reasoning config plumbing (likely via ModelFactory + agent invocations)
- [x] Extract `reasoning_summary` + `reasoning_tokens` into assistant metadata
- [x] Render summaries in Streamlit message bubbles

### Phase F — Verification

- [x] Update `verify_implementation.py` with new expected files and schema checks
- [ ] Manual test checklist:
  - [ ] No OpenAI key: Gemini works, OpenAI selection falls back
  - [ ] OpenAI key + model selected for chat only: chat uses OpenAI, rag/search/planning use defaults
  - [ ] Tool calling with OpenAI still works
  - [ ] Reasoning summary appears (when model supports it)

---

## 6) Acceptance criteria

- Models tab shows **chat/rag/search/planning** and allows setting **provider+model per agent**.
- Settings persist in DB and automatically apply to new messages.
- Gemini remains usable with no user-provided key.
- When OpenAI fails or is not configured, system falls back to Gemini and UI shows a fallback note.
- When OpenAI provides reasoning summaries, UI renders them (summary only) and optionally shows reasoning token count.

---

## Progress log / design decisions

### Phase A (completed)

- Implemented `agent_model_configs` table + SQLAlchemy model + repository + service.
- Updated `app/database/base.py` to import `Document` + `DocumentImage` to ensure SQLAlchemy relationship resolution during metadata/migration/verification.
- Updated `app/services/__init__.py` to use lazy imports to avoid importing heavy services (and their dependencies) when importing a single service module.
- Design decision: `ModelConfigService.get_effective_model_request(user_id)` returns **override-only** `model_request` (not defaults) to avoid forcing per-request model re-creation when no user overrides exist. UI reads defaults + overrides via `get_effective_model_config(user_id)`.

### Phase B (completed)

- Added `app/api/model_config.py` with `GET /model-config`, `PATCH /model-config`, and `POST /model-config/reset`.
- Wired the router into `app/main.py` and registered `ModelConfigService` in DI (`app/core/container.py`, `app/core/dependency_injection.py`).
- Design decision: API uses `provider` (not `provider_type`) in request/response payloads to match existing per-message `modelConfig` shape.
- Design decision: avoided `from __future__ import annotations` in `app/api/model_config.py` because auto-injection relies on concrete (non-string) type annotations.

### Phase C (completed)

- Updated `app/services/message_service.py` to apply persisted model config automatically when `modelConfig` is not provided per-message.
- Wired `ModelConfigService` into `MessageService` via DI (`app/core/container.py`).

### Phase D (completed)

- Updated `demo.py` Models tab to edit persistent per-agent settings (chat/rag/search/planning) via `GET /model-config`, `PATCH /model-config`, and `POST /model-config/reset`.
- Removed per-message override UI from the chat composer so model selection is unified via the Models tab.

### Phase E (completed)

- Enabled OpenAI reasoning summaries when using OpenAI models and surfaced `reasoning_summary` (and best-effort `reasoning_tokens`) in assistant message metadata.
- Rendered `reasoning_summary` in Streamlit message bubbles under a collapsible expander.
- Design decision: treat `reasoning.summary` as best-effort — if OpenAI rejects it (e.g., org not verified), retry the OpenAI call without reasoning summaries instead of falling back to Gemini.

### Phase F (completed)

- Updated `verify_implementation.py` to cover persisted per-agent model config (imports, file structure, migration checks, and basic wiring assertions).

### Phase G (follow-up UI cleanup)

- Replaced emoji UI icons in `demo.py` with Material icons (`:material/...:` / widget `icon=`) and added Material Symbols CSS support for HTML-rendered icons.
- Updated `demo_requirements.txt` to require Streamlit `>=1.34.0` to ensure Material icon support.
- Design decision: keep backend support for per-message `modelConfig` for compatibility, but the Streamlit demo no longer sends it.
