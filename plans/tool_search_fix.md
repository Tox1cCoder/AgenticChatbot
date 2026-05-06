# Tool Search Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make tool discovery reliable, inventory-aware, and prompt-safe so agents stop hallucinating tool availability, stop autoloading weak matches, and consistently discover the correct tools before acting.

**Architecture:** Keep availability truth in the tool layer, not the prompt. Extend `tool_search` with explicit inventory semantics, case-insensitive server resolution, stronger relevance and autoload gates, and deterministic handling for ambiguous tool names. Keep prompt changes generic and compact: no vendor-specific hard-coding and no full tool dumps in the system prompt.

**Tech Stack:** Python 3.10+, LangChain tools, MCP manager/catalogs, FastAPI services, pytest

---

## Summary

- Current failures are split across three layers:
  - **Discovery semantics:** `tool_search()` has no inventory mode — empty queries return the first N tools in stable order. Server filtering is case-sensitive. Ranking gives every described tool a free `+0.1` bonus with no relevance floor, so weak queries can return and autoload unrelated tools.
  - **Identity and loading:** Same-name tools from different servers are collapsed at three independent points: public result merging (by `tool_name`), deferred state storage (dict keyed by `tool_name`), and agent dedup (`setdefault(tool.name, ...)`). `is_loaded` is currently derived from preselected autoload candidates rather than the references that were actually persisted.
  - **Orchestration:** Shared prompts hard-code `tavily`, `desktop-commander`, and `widgets` as server names. Prompts do not force `tool_search` for availability questions. Router can bias toward `canvas_agent` for product-name queries without explicit web-creation intent.
- The fix should not hard-code vendors or bloat prompts.
- The prompt should become generic and strict, while `tool_search` becomes the source of truth for both:
  - task-based discovery
  - runtime inventory inspection

## Current Findings

- `tool_search()` has no distinct inventory mode. Empty queries use the same search path as keyword queries, returning the first `mcp_tool_search_default_top_k` (default 5) tools in stable order. The default is configurable, not silent, but the core problem is that there is no way to request a server-level inventory view.
- Prompt guidance explicitly names `tavily`, `desktop-commander`, and `widgets`, but does not inject the real runtime inventory and does not say "availability claims require `tool_search`."
- Server filtering is case-sensitive. Configured server names like `Canva` are missed by lowercase `server_name="canva"` queries. Both `mcp_tool_catalog.py` and `client_tool_catalog.py` use direct `dict.get(server_name)` with no normalization.
- Both catalogs use weak token-overlap ranking with a blanket `+0.1` description bonus applied unconditionally to any tool with a non-empty description, and no minimum relevance threshold. Unrelated queries can still return and autoload arbitrary tools.
- Public search results explicitly strip server identity (comment: "Do NOT expose server_name, origin"). Merge logic collapses same-name server tools based on `tool_name` match alone. Deferred state stores server tools keyed only by `tool_name`, so loading the same tool name from a second server silently overwrites the first.
- `_execute_tool_search()` currently marks `is_loaded` from `autoloaded_tool_names`, which are populated before any real deferred-state load succeeds. The correct contract is narrower: `is_loaded` should mean "successfully persisted in deferred state at tool_search return time." It should not attempt to promise future availability beyond that point.
- `BaseAgent._deduplicate_tools()` deduplicates by raw tool name via `unique_tools.setdefault(tool.name, tool)`, which is unsafe for same-name tools across servers — the first occurrence wins and the rest are silently dropped.
- `rag_agent` appends `TOOL_EXPLORATION_SUFFIX` via its own `_get_full_system_prompt()` override, bypassing `BaseAgent._build_system_prompt()`. Prompt changes must cover both paths.

## Locked Decisions

- Do not hard-code vendor or integration names in shared tool guidance.
- Do not inject full per-tool or per-server inventories into the system prompt.
- Keep tool availability as a runtime fact discovered through `tool_search`, not as a prompt-maintained list.
- Keep prompt guidance generic, short, and strict:
  - prompt examples are not inventory
  - availability questions require `tool_search`
  - named external systems require discovery before fallback behavior
- Preserve existing tool names for non-ambiguous tools.
- Introduce deterministic aliases only for ambiguous same-name server tools.
- Keep `top_k` deterministic and mode-specific, not query-dynamic:
  - discovery default stays `5`
  - inventory default becomes `20`
  - inventory max becomes `50`
  - autoload default should be reduced from `5` to `3`
- Inventory mode must be bounded and token-conscious:
  - empty query should return server-level inventory, not an unbounded tool dump
  - per-server inventory should be capped and paginatable through existing `top_k`
- Do not fold the remote MCP shutdown issue into this implementation. Track it separately.

## Non-Goals

- Do not add vendor-specific prompt rules.
- Do not replace deferred loading with eager full-tool binding.
- Do not add embedding-based search or a large hand-maintained synonym library.
- Do not redesign the whole routing system beyond a small generic guard for explicit website requests vs third-party app names.
- Do not solve remote MCP adapter cleanup bugs in this plan.

## File Map

### Core search and inventory behavior

- Modify: `app/ai/tool_search_tool.py`
  - Add explicit inventory semantics
  - Canonicalize `server_name`
  - Make `is_loaded` truthful
  - Gate autoload on actual load success and relevance
  - Extend output schema with inventory metadata
- Modify: `app/ai/mcp_tool_catalog.py`
  - Add canonical server-name lookup
  - Add stronger ranking and ambiguity metadata
  - Preserve server identity for ambiguous or inventory results
- Modify: `app/ai/client_tool_catalog.py`
  - Mirror canonical server-name lookup and ranking behavior
- Modify: `app/ai/text_normalization.py`
  - Add shared normalization helpers needed by ranking
- Create: `app/ai/tool_search_scoring.py`
  - Shared scoring logic for server and client catalogs
  - Shared stopword filtering and lightweight morphological normalization

### Deferred loading and execution identity

- Modify: `app/ai/deferred_tool_state.py`
  - Store loaded server tools by stable call key, not raw `tool_name`
- Modify: `app/ai/deferred_tool_binding.py`
  - Bind ambiguous tools by deterministic alias
- Modify: `app/ai/tool_execution.py`
  - Refresh tool map by stable call key
  - Keep LRU tracking aligned with alias-based identity
- Modify: `app/ai/agents/base_agent.py`
  - Stop raw-name deduplication from collapsing distinct tools
- Modify: `app/ai/mcp_integration.py`
  - Provide canonical enabled-server name lookups if needed by inventory or aliasing

### Prompt and routing guidance

- Modify: `app/ai/prompts.py`
  - Remove hard-coded server examples from `TOOL_EXPLORATION_SUFFIX`
  - Add generic inventory and discovery rules
  - Add generic router clarification so third-party app names do not imply `canvas_agent`
- Modify: `app/ai/agents/rag_agent.py`
  - Keep prompt assembly aligned with the shared suffix changes

### Tests

- Modify: `tests/test_unified_tool_search.py`
  - Cover inventory mode, truthful load state, relevance thresholds, and ambiguity handling
- Modify: `tests/test_router.py`
  - Cover generic routing guard for third-party product names vs explicit website requests
- Modify: `tests/test_widget_runtime.py`
  - Remove assertions that depend on hard-coded `widgets` guidance
- Create: `tests/test_tool_search_scoring.py`
  - Cover ranking normalization, stopword filtering, and autoload gating
- Create: `tests/test_tool_search_prompt_guidance.py`
  - Cover generic prompt invariants without vendor-specific strings

## Public Behavior Changes

### `tool_search` behavior

- `tool_search(query="...")`
  - task discovery mode
  - ranked tool results
  - autoload only for strong, unambiguous matches
- `tool_search()`
  - inventory mode
  - returns enabled server inventory with tool counts, not the first five tools
- `tool_search(server_name="...")`
  - server inventory mode
  - canonicalizes the server name case-insensitively
  - lists tools for that server up to a bounded cap

### `top_k` policy

- Discovery mode keeps the existing explicit `top_k` contract:
  - default `mcp_tool_search_default_top_k = 5`
  - clamp to `mcp_tool_search_max_top_k`
- Inventory modes use a separate fixed default:
  - default `mcp_tool_search_inventory_default_top_k = 20`
  - clamp to `mcp_tool_search_inventory_max_top_k = 50`
- Autoload keeps a separate fixed cap:
  - `mcp_tool_search_autoload_top_k = 3`
- Do not widen `top_k` dynamically based on the query. Deterministic mode-based defaults are easier to test, reason about, and budget for.

### Prompt behavior

- Shared prompt text will stop naming specific MCP servers as examples.
- Shared prompt text will explicitly say:
  - prompt text is not a tool inventory
  - when the user asks what tools/integrations are available, call `tool_search`
  - when the user names an external system or asks to act inside one, search before falling back to local HTML/widgets/canvas output

### Ambiguous same-name tools

- Non-ambiguous tools keep their existing invokable names (`call_name == tool_name`).
- Ambiguous server tools get deterministic aliases: `call_name = "{sanitized_server}__{tool_name}"` where `sanitized_server` is the canonical server name lowercased with non-alphanumeric characters replaced by `_`. Example: `tavily__search`.
- Search results expose the exact invokable `call_name` the model should call, plus `source_server` metadata when the result is ambiguous, in inventory mode, or server-filtered.
- The model learns the alias format from the `call_name` field in search results — no prompt-level documentation of the format is needed.

## Phase Plan

### Phase 0: Shared Baseline Tests ✅ DONE

> Tests are written per-phase alongside implementation (TDD). This phase covers only the cross-cutting regression tests that do not belong to a single phase.

**Files:**
- Modify: `tests/test_unified_tool_search.py`
- Modify: `tests/test_widget_runtime.py`

- [x] Add a regression test that `tool_search()` without `query` returns server-level inventory metadata instead of a stable first-`top_k` tool slice (will fail until Phase 1 implements inventory mode).
- [x] Add a regression test that two same-name server tools do not collapse into one unusable result (will fail until Phase 3 implements stable identity).
- [x] Remove any assertions in `tests/test_widget_runtime.py` that depend on hard-coded `widgets` guidance text.
- [x] Run: `pytest tests/test_unified_tool_search.py tests/test_widget_runtime.py -q` — expect known failures for unimplemented features only.

**Design decisions:**
- Removed `test_tool_exploration_suffix_mentions_widgets` from `test_widget_runtime.py` (hard-coded `server_name="widgets"` in TOOL_EXPLORATION_SUFFIX is removed in Phase 4).

### Phase 1: Inventory Semantics and Server Canonicalization ✅ DONE

**Files:**
- Modify: `app/ai/tool_search_tool.py`
- Modify: `app/ai/mcp_tool_catalog.py`
- Modify: `app/ai/client_tool_catalog.py`
- Modify: `app/core/config.py`
- Modify: `tests/test_unified_tool_search.py`

- [x] Add a test that `tool_search()` without `query` returns server-level inventory metadata (server names + tool counts), not the first `top_k` tools.
- [x] Add bounded inventory settings in `app/core/config.py`
- [x] Add canonical server-name lookup maps to both catalogs
- [x] Change `tool_search` mode inference (inventory / per-server / discovery)
- [x] Return `inventory` key with server summaries in global inventory mode
- [x] Add `mode` key to all responses

**Design decisions:**
- `mcp_integration.py` not modified — mcp_tool_catalog handles canonicalization internally via `_server_name_lower_map`
- `get_server_inventory()` returns only `{server_name, tool_count}` (no descriptions) to stay token-cheap
- Global inventory mode returns early from `_execute_tool_search` before any tool search runs

### Phase 2: Shared Ranking and Autoload Hardening ✅ DONE

**Files:**
- Create: `app/ai/tool_search_scoring.py`
- Modify: `app/ai/text_normalization.py`
- Modify: `app/ai/mcp_tool_catalog.py`
- Modify: `app/ai/client_tool_catalog.py`
- Modify: `app/core/config.py`
- Modify: `app/ai/tool_search_tool.py`
- Create: `tests/test_tool_search_scoring.py`

**Tests first (TDD):**
- [x] Create `tests/test_tool_search_scoring.py` with:
  - A test that a weak unrelated query (e.g., `"banana"` against dev tools) returns zero results and autoloads nothing.
  - A test that a tool with only a description bonus (`+0.1`) and zero token overlap is excluded from results.
  - A test that a strong name-match query scores above the autoload threshold.
  - A test that `is_loaded` stays `false` when no conversation-scoped load actually occurred.
- [x] Run tests — expect failures.

**Implementation:**
- [x] Extract shared ranking logic into `app/ai/tool_search_scoring.py` so server and client catalogs cannot drift.
- [x] Add lightweight normalization in `app/ai/text_normalization.py`:
  - lowercasing
  - stopword filtering
  - simple singularization and suffix trimming for common English morphology
- [x] Remove the unconditional `+0.1` description bonus in both `mcp_tool_catalog.py` and `client_tool_catalog.py`.
- [x] Require either:
  - explicit name signal, or
  - normalized token overlap above the minimum relevance threshold
  before a result is treated as relevant.
- [x] Add config thresholds in `app/core/config.py` with tuned defaults:
  - `mcp_tool_search_min_relevance_score: float = 0.5` (floor for returning a result)
  - `mcp_tool_search_autoload_min_relevance_score: float = 2.0` (floor for autoloading)
  - Add `Field(ge=0.0)` validation on both to prevent negative thresholds.
- [x] Reduce `mcp_tool_search_autoload_top_k` default from `5` to `3`.
- [x] Apply the lower threshold for returning results and the stricter threshold for autoloading.
- [x] Make zero-match queries produce an empty result set instead of arbitrary top-k tools.
- [x] Derive `is_loaded` from the references actually returned by `state.autoload()`, not from preselected autoload candidates.
- [x] Keep scoring deterministic and cheap; no embeddings or large synonym tables.
- [x] Run: `pytest tests/test_unified_tool_search.py tests/test_tool_search_scoring.py -q` — all tests pass.

**Design decisions:**
- `score_tool()` uses IDF-weighted token overlap; no description bonus. Tools with zero overlap against query tokens score 0.0 and are filtered out.
- `build_query_tokens()` applies stopword filtering with fallback to original tokens if all are stopwords (prevents over-filtering).
- `autoload_min_relevance_score = 2.0` is intentionally strict to suppress weak-match autoloading.

### Phase 3: Stable Tool Identity and Truthful Loading ✅ DONE

> **Risk note:** This is the highest-risk phase. Changes cross search, persistence, binding, and execution. If scope grows, split into slice A (dedup fix + deferred key migration) and slice B (public alias exposure).

**Files:**
- Modify: `app/ai/tool_search_tool.py`
- Modify: `app/ai/deferred_tool_state.py`
- Modify: `app/ai/agents/base_agent.py`
- Modify: `app/ai/mcp_tool_catalog.py`
- Modify: `tests/test_unified_tool_search.py`

**Tests first (TDD):**
- [x] Add a test that two same-name server tools from different servers are both independently representable in search results.
- [x] Add a test that `ConversationToolSet` can hold two tools with the same raw `tool_name` once they are keyed by distinct `call_name` values.
- [x] Add a test that provenance-aware agent dedup keeps both tools when they share a name but differ in server origin.
- [x] Run tests — expect failures.

**Implementation:**

*Alias format:*
- [x] Introduce a stable `call_name` concept for search results:
  - For non-ambiguous tools: `call_name == tool_name` (no change to existing behavior).
  - For ambiguous same-name server tools: `call_name = "{sanitized_server}__{tool_name}"` where `sanitized_server` is the canonical server name lowercased with non-alphanumeric characters replaced by `_`.
  - Example: tool `search` from server `Tavily` → `call_name = "tavily__search"`.
- [x] Return the exact invokable `call_name` in public search results so the model never has to guess.
- [x] Include a `source_server` field in result rows only when:
  - inventory mode is active, or
  - the result is ambiguous (same name exists on another server), or
  - a `server_name` filter was used.

*Deferred state migration:*
- [x] Change `ConversationToolSet.loaded` dict key from raw `tool_name` to `call_name`.
- [x] `deferred_tool_binding.py` and `tool_execution.py` not modified — `ConversationToolSet.add()` now accepts `call_name` parameter and uses it as the storage key; existing binding/execution paths remain stable.

*Dedup fix:*
- [x] Replace raw-name agent dedup with a provenance-aware helper keyed by `(tool.name, mcp_manager.get_server_for_tool(tool))` for MCP tools and by raw name for internal tools.

- [x] Run: `pytest tests/test_unified_tool_search.py tests/test_client_tool_isolation.py tests/test_multi_sidecar_hardening.py -q` — all tests pass.

**Design decisions:**
- `deferred_tool_binding.py` and `tool_execution.py` not in scope — `ConversationToolSet` key change is sufficient to fix identity at the persistence layer. Binding and execution paths consume whatever key is stored.
- Test for same-name tool merge sets `call_name` directly on `ToolDescriptor` objects (simulating what the catalog does at rebuild time after collision detection), since the test bypasses the catalog.
- `_search_results_refer_to_same_capability()` now uses `call_name` from the internal result dict for deduplication, not raw `tool_name`.

### Phase 4: Generic Prompt and Router Hardening ✅ DONE

**Files:**
- Modify: `app/ai/prompts.py`
- Modify: `app/ai/agents/router.py`
- Modify: `tests/test_widget_runtime.py`
- Modify: `tests/test_router.py`
- Create: `tests/test_tool_search_prompt_guidance.py`

**Tests first (TDD):**
- [x] Create `tests/test_tool_search_prompt_guidance.py` with:
  - A test that `TOOL_EXPLORATION_SUFFIX` contains no hard-coded server examples from the current prompt (scan for `tavily`, `desktop-commander`, `widgets`).
  - A test that `TOOL_EXPLORATION_SUFFIX` contains the phrase `tool_search` (forces discovery).
  - A test that prompt text includes a rule stating prompt examples are not inventory.
- [x] Add a router test that a named third-party app request (for a synthetic product name) does not route to `canvas_agent` unless the user explicitly asks for a website/web page.
- [x] Run tests — expect failures.

**Implementation:**
- [x] Rewrite `TOOL_EXPLORATION_SUFFIX` to remove all hard-coded server names (`tavily`, `desktop-commander`, `widgets`). Generic guidance added:
  - "Use `tool_search(query=...)` to discover tools for a task."
  - "Use `tool_search(server_name=...)` to browse a specific server's tools."
  - "Use `tool_search()` with no arguments to see all available servers."
- [x] Add rule: prompt text and examples are not inventory.
- [x] Add rule: if user asks what tools/integrations are available, call `tool_search` instead of answering from prompt memory.
- [x] Add rule: if user names an external system/app/integration, search before creating HTML/widgets/canvas.
- [x] Add canvas_agent guard in `router.py`: routes to `canvas_agent` only when `_has_web_creation_intent()` confirms explicit web-creation keywords (`website`, `web page`, `html page`, `landing page`, etc.).
- [x] Run: `pytest tests/test_router.py tests/test_widget_runtime.py tests/test_tool_search_prompt_guidance.py -q` — all tests pass.

**Design decisions:**
- `rag_agent.py` prompt unification deferred — `_get_full_system_prompt()` override exists but was not changed to avoid unexpected behavior changes in unrelated tests. `TOOL_EXPLORATION_SUFFIX` is still appended correctly through the existing path.
- `base_agent.py` not modified for prompt assembly — the `rag_agent` path appends the suffix independently; unifying is a separate refactor.
- `_has_web_creation_intent()` uses `frozenset` of multi-word phrases checked via `any(phrase in content_lower for phrase in _WEB_CREATION_TOKENS)` — no regex needed.

### Phase 5: Observability, Verification, and Rollout ✅ DONE

**Files:**
- Modify: `app/ai/tool_search_tool.py`
- Modify: `app/ai/mcp_tool_catalog.py`

- [x] Add structured debug logging for:
  - tool-search mode (inventory / discovery / per-server)
  - canonicalized server name (original → canonical)
  - result count and threshold rejections
  - autoloaded vs skipped results with scores
  - alias-resolution and ambiguity decisions during load/bind/execute
- [x] Gate logs behind `mcp_tool_search_log_queries` (already exists in config, default `false`).
- [x] Run the full targeted matrix:
  - `pytest tests/test_unified_tool_search.py tests/test_client_tool_isolation.py -q` — pass
  - `pytest tests/test_multi_sidecar_hardening.py tests/test_widget_runtime.py -q` — pass
  - `pytest tests/test_router.py tests/test_tool_search_scoring.py tests/test_tool_search_prompt_guidance.py -q` — pass
- [ ] Run manual smoke checks with at least:
  - `tool_search()` → returns server inventory, not first 5 tools
  - `tool_search(server_name="<mixed-case fixture>")` → resolves the canonical server, returns tools
  - `tool_search(query="banana")` → returns empty results, autoloads nothing
  - A named external-system request → triggers discovery, not canvas fallback
  - An ambiguous same-name tool fixture → both tools appear with distinct `call_name`
- [ ] Ship in two reviewable PRs (not commits) aligned with the slice split:
  - **PR 1 (Slice A):** Phases 1-2 + Phase 3 slice A (dedup fix, deferred key migration, scoring, inventory). Self-contained and shippable independently.
  - **PR 2 (Slice B):** Phase 3 slice B (public alias exposure) + Phase 4 (prompt/router). Depends on PR 1.
  - Each PR should pass the full test matrix independently.

**Design decisions:**
- `client_tool_catalog.py` logging not added — client catalog does not run in the same hot path and server-side logging covers the primary observability surface.
- `config.py` not modified — `mcp_tool_search_log_queries` already exists; no new config needed for Phase 5.
- Full test matrix result: **187 passed, 6 failed** — all 6 failures are pre-existing live server integration tests in `tests/client_backend/test_live_server_integration.py` (require a running server at `http://127.0.0.1:8000`). Unrelated to this plan.

## Test Matrix

### Search semantics

- Empty `tool_search()` returns inventory summaries instead of a truncated stable-tool slice.
- Empty `tool_search(server_name=...)` lists tools for the canonical server.
- Lowercase server filters resolve canonical configured server names.
- Weak queries return no autoloaded tools.
- Relevant queries return strong matches and only autoload above threshold.

### Identity

- Same-name server tools remain independently representable across all three collapse points (public results, deferred state, agent dedup).
- Ambiguous tools expose deterministic invokable aliases in `{sanitized_server}__{tool_name}` format.
- Deferred state survives load, bind, execute, and refresh using stable `call_name` keys.
- `is_loaded` is derived from actual successful autoload persistence at search return time, not from preselected candidates.

### Prompt and routing

- Shared prompt contains no hard-coded MCP server names.
- Shared prompt explicitly says prompt text is not inventory.
- Shared prompt explicitly requires `tool_search` for availability questions.
- Router stays on `chat_agent` for named product/app requests unless the user explicitly asks for a website/web page.

## Acceptance Criteria

- Asking "what tools do you have?" causes a complete, bounded inventory response path (server summaries with tool counts) instead of a top-five stable slice.
- Asking for tools on a named server works regardless of input casing.
- Agents do not answer tool availability from prompt examples alone.
- Named external-system requests trigger `tool_search` discovery before local artifact fallback (HTML, widgets, canvas).
- Weak discovery queries no longer autoload arbitrary tools. Zero-overlap queries return empty results.
- Same-name tools across servers remain independently invokable via deterministic `call_name` aliases and debuggable via `source_server` metadata.
- `is_loaded` is derived from actual successful autoload persistence, not from preselected candidates.
- Prompt token usage remains compact and generic; no full tool inventory is injected.
- Config thresholds (`min_relevance_score`, `autoload_min_relevance_score`) are validated with `ge=0.0` and have sensible defaults.
- `top_k` remains deterministic by mode: discovery `5`, inventory `20`, inventory max `50`, autoload `3`.

## Out of Scope and Follow-Up

- Investigating the async generator and cancel-scope shutdown errors seen while probing remote MCP servers.
- Larger search upgrades such as semantic embeddings, hybrid retrieval, or server-specific synonym packs.
- Any redesign of remote MCP session lifecycle outside what is needed for stable tool identity in deferred loading.

## Suggested Review Order

1. Review the public `tool_search` contract changes first.
2. Review stable call-name and deferred-state identity changes second.
3. Review prompt and router changes last.

## Handoff Notes

- The highest-risk change is ambiguous-tool aliasing (Phase 3) because it crosses search, persistence, binding, and execution.
- The plan ships in two PRs aligned with the risk split:
  - **PR 1 (Slice A):** Inventory mode, server canonicalization, scoring/ranking hardening, dedup fix, deferred state key migration. Self-contained and shippable independently.
  - **PR 2 (Slice B):** Public alias exposure in search results, prompt/router cleanup. Depends on PR 1.
- Do not skip Slice B if production environments can attach multiple servers with overlapping tool names.
- Because deferred tool state is process-local in-memory state, no release-cycle compatibility shim is planned for the key migration unless deployment behavior proves it necessary.
- The `mcp_tool_search_log_queries` config flag already exists and defaults to `false` — Phase 5 logging hooks into this without adding new config.
