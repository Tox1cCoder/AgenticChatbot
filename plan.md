# Deferred MCP Tool Discovery Overhaul Plan

Date: 2026-03-13
Status: Reviewed and adjusted — confirmed against codebase

## Summary

This plan overhauls the deferred MCP tool loading and `tool_search` flow so it is production-ready for large MCP inventories, stable under tool-name collisions, cheaper in tokens, and reliable at producing a final answer instead of looping on discovery.

The redesign covers all relevant code paths:

- MCP manager and registry
- MCP API/service schemas
- tool catalog and search
- deferred tool state and binding
- tool execution
- agent prompts and binding
- LangGraph orchestration and terminal recovery
- test coverage and telemetry

## Goals

1. Make tool identity unambiguous across servers, even when many servers expose the same tool name.
2. Make `tool_search` useful for large MCP servers with dozens of tools.
3. Reduce model-facing token cost of tool discovery responses.
4. Prevent repeated `tool_search` loops from exhausting the budget without a user-facing answer.
5. Remove legacy and redundant code paths that no longer match the main runtime flow.
6. Keep the resulting design deterministic, debuggable, and safe to roll out.

## Non-Goals

- Introducing embedding-based semantic search for tools in the first pass.
- Redesigning unrelated agent behaviors outside MCP discovery/execution integration.
- UI polish beyond changes required to support the new MCP identity and discovery model.

## Current-State Problems

### 1. Tool identity is name-only in the runtime

Current runtime state, binding, and execution collapse tools to bare `tool_name`. This causes collisions, replacements, and ambiguity across servers.

`ConversationToolSet.loaded` in `deferred_tool_state.py` is a `dict[str, LoadedTool]` keyed by bare `tool_name` with explicit replacement semantics: adding a tool silently overwrites any prior binding for that name. `mcp_integration.py` strips `server:` prefixes at load time (via `tool.name.split(":", 1)[-1]`), so the server identity is permanently discarded from the tool object name.

Collision detection already partially exists: `mcp_tool_catalog.py` tracks `_colliding_names` and `catalog.is_ambiguous(tool_name)` is guarded in `tool_search_tool.py` to skip autoloading ambiguous tools. The gap is at the **execution layer** — `tool_execution.py` builds `{t.name: t for t in tools}` with no ambiguity check; it silently picks whichever tool landed in the map last.

Impacted areas:

- `app/ai/deferred_tool_state.py`
- `app/ai/deferred_tool_binding.py`
- `app/ai/tool_execution.py`
- `app/ai/agents/base_agent.py`
- `app/ai/mcp_integration.py`
- `app/services/mcp_service.py`
- `app/api/mcp.py`

### 2. `tool_search` is too verbose and partly redundant

The current output includes fields that are either derivable or debug-oriented for the model:

- `display_name` is derivable from `server_name` + `tool_name`
- `call_as` duplicates `tool_name`
- `generation`, `latency_ms`, and `unavailable_servers` are runtime/debug metadata
- pretty-printed JSON wastes tokens: both `tool_search_tool.py:259` (module-level tool) and `:302` (inside `create_tool_search_tool`) call `json.dumps(result, indent=2)` — both must be changed to compact separators

### 3. Large MCP servers are not browseable

The current interface supports only:

- `query`
- `top_k`
- `server_name`

It does not support:

- pagination
- server exploration mode
- server summaries or capability grouping
- stable cursors for large result sets

### 4. Search ranking is too shallow

Catalog ranking is mostly exact match, substring match, and token overlap. It does not account for:

- server-level capabilities
- tool aliasing
- fuzzy near-matches
- ambiguity-aware expansion
- large inventory navigation

### 5. Autoloading is too aggressive

With defaults of `top_k=5`, `autoload_top_k=5`, and `max_loaded_tools_per_conversation=8`, a few exploratory searches can churn the loaded set before the model even executes a tool.

### 6. Orchestration can end without a final answer

The graph can hit iteration limits after many `tool_search` calls and terminate with a tool-call-only state or an empty assistant message rather than forcing one last synthesis pass.

### 7. There are divergent and likely legacy search-agent paths

The graph’s main runtime uses `BaseAgent.invoke_model_with_history`, while `SearchAgent` still has its own older prompt-building and streaming paths that are not aligned with the primary execution path.

## Target Architecture

## 1. Canonical Tool Identity

Introduce a canonical MCP tool reference used everywhere in runtime logic.

Proposed model:

```text
QualifiedToolRef
- server_name: str
- tool_name: str
- qualified_id: str            # canonical storage key, e.g. "github::search_issues"
- callable_name: str           # bound tool name exposed to the model
- schema_fingerprint: str
```

Note: `ToolReference(tool_name, server_name)` already exists in `mcp_tool_catalog.py` and is imported by `deferred_tool_state.py` and `tool_search_tool.py`. `QualifiedToolRef` replaces it — it is not a new parallel type. All import sites (`deferred_tool_state.py`, `tool_search_tool.py`, `deferred_tool_binding.py`, `tool_execution.py`) must be updated together.

Note: `schema_fingerprint` is already computed in `ToolDescriptor` (sha256[:16] of description + schema JSON). Lift it from `ToolDescriptor` into `QualifiedToolRef` — do not introduce a second computation.

### Decisions

- Storage, loading, and execution should key by `qualified_id`, not bare `tool_name`.
- Bound tools should expose a unique `callable_name` to the model.
- Bare-name lookup should only be allowed when the name is globally unique.
- Ambiguous bare-name lookup should fail fast rather than silently picking the first match.
- Phase 1 collision work is scoped to the **execution layer**: the catalog already detects collisions via `_colliding_names` and `is_ambiguous()`. The remaining gap is `tool_execution.py` silently building `{t.name: t}` maps with no ambiguity guard.

### Recommended callable-name strategy

Use deterministic, model-safe qualified names for bound MCP tools:

```text
mcp__{server_name}__{tool_name}
```

This removes runtime ambiguity and lets the model call the discovered tool directly.

The `"::"` separator is already in use as the **config-layer qualifier** for pinned tool specs (e.g. `"github::search_issues"` in `mcp_tool_search_pinned_tools`). These two formats serve distinct purposes and must coexist:

- `qualified_id` = internal storage key: `"server::tool_name"` (aligns with existing config convention)
- `callable_name` = model-visible bound name: `"mcp__server__tool_name"` (double underscore, safe for all model providers)

Do not conflate the two. Do not change the config format.

### Required code changes

- Define `QualifiedToolRef` as the shared identity type. It should live in `app/ai/mcp_tool_catalog.py` (same module as the existing `ToolReference` it replaces) or a dedicated `app/ai/mcp_types.py` if cross-module cleanliness is preferred. Choose one location and update all importers.
- Replace `ToolReference` with `QualifiedToolRef` at all import sites.
- Replace name-only maps and sets in deferred loading state and binding.
- Add a lightweight bound-tool adapter that wraps the real MCP tool while exposing `callable_name`.
- Update MCP manager and service methods to accept qualified references.

## 2. Catalog and Discovery Redesign

Keep a single discovery tool, but make it support both search and exploration. This minimizes schema count while making large inventories navigable.

### New `tool_search` modes

Recommended input model:

- `mode="search"`: ranked tool retrieval for a natural-language task
- `mode="browse"`: paginated listing of tools, optionally scoped to a server
- `mode="servers"`: list server summaries and capability hints

Additional inputs:

- `query: str | None`
- `server_name: str | None`
- `cursor: str | None`
- `limit: int | None`

### Catalog improvements

Augment the catalog with:

- server description and transport metadata
- tool args keywords
- normalized aliases and token variants
- collision metadata
- optional per-server capability summaries

Improve ranking with:

- exact callable-name match
- exact tool-name match
- server-name and server-description boosts
- token overlap over name, description, args, and server summary
- lightweight fuzzy match for near spellings
- ambiguity-aware expansion when top results are close

### Browse behavior

`browse` mode should support:

- stable cursor-based pagination
- browsing all tools within a server
- browsing all servers with counts
- query-within-server behavior

This is the minimum needed to make a 50-tool MCP server usable.

## 3. Compact Model-Facing Search Response

Replace the current verbose pretty JSON with a compact schema designed for the model.

### Proposed response shape

```json
{
  "mode": "search",
  "query": "github issues",
  "server": null,
  "results": [
    {
      "name": "mcp__github__search_issues",
      "server": "github",
      "origin": "search_issues",
      "summary": "Search issues in a repository",
      "args": "repo*, state, labels",
      "loaded": true,
      "confidence": "high"
    }
  ],
  "more": true,
  "next_cursor": "..."
}
```

### Response rules

- `name` is the exact callable tool name.
- Drop `display_name`.
- Drop `call_as`.
- Keep `origin` only if the callable name is qualified and differs from the source tool name.
- Replace `autoloaded` with per-result `loaded`.
- Remove debug-only fields from model-facing output.
- Serialize with compact JSON separators instead of pretty indentation.

### Token budget targets

- 5-result response should stay under roughly 250 model tokens in a representative case.
- Server browsing responses should return only what is needed to choose the next action.

## 4. Smarter `top_k` and Autoload Policy

Do not treat `top_k=5` as a fixed system behavior. Use it as a default display size, not as a hard-coded strategy.

### Recommended policy

- Default result count: 5
- High-confidence exact match: allow returning 1 to 3 results
- Ambiguous or flat score distribution: return up to 8 to 10 results
- Browse mode: page size default 10

### Autoload policy

- Autoload at most 1 to 2 tools by default
- Autoload only when confidence is above a threshold
- Do not autoload broad browse responses
- Never autoload all top 5 by default

### Deferred state policy

- Increase loaded-tool capacity only if needed after alias-based identity is in place
- Eviction should operate on qualified IDs
- Prefer preserving actually executed tools over merely autoloaded tools

## 5. Deferred Binding and Execution Redesign

### Binding changes

- Bind internal tools as today
- Bind `tool_search`
- Bind pinned MCP tools through qualified aliases
- Bind loaded deferred tools through qualified aliases

### Execution changes

- Tool maps must use `callable_name`, not bare `tool.name`
- Deferred loaded-tool state must store `qualified_id`
- MCP execution should resolve by qualified reference
- Ambiguous legacy lookups should return explicit errors

### API changes

The MCP API surface is currently ambiguous for tool details and execution.

Replace or extend current endpoints so they support qualified lookup:

- `GET /mcp/tools/{qualified_id}` or
- `GET /mcp/servers/{server_name}/tools/{tool_name}`
- `POST /mcp/tools/{qualified_id}/execute` or
- `POST /mcp/servers/{server_name}/tools/{tool_name}/execute`

The service layer must mirror the same identity model.

## 6. Orchestration Safeguards

The graph must stop discovery loops from ending in silence.

### New controls

- track consecutive `tool_search` calls per turn
- track duplicate `tool_search` query fingerprints per turn
- detect repeated search with no new loaded tools and no better results
- cap consecutive empty or duplicate searches

### Forced finalization path

Add a final-answer path for budget exhaustion and discovery failure:

- route to a dedicated finalization node instead of raw `end`
- invoke the selected agent one last time with tools disabled
- require it to either:
  - answer from gathered evidence, or
  - explicitly say no suitable tool was found / more detail is needed

### Budget tuning

Add dedicated search/discovery controls instead of relying only on the global ReAct limit:

- `search_agent_max_iterations`
- `max_tool_search_calls_per_turn`
- `max_duplicate_tool_search_calls`
- `max_empty_tool_search_results`

Keep global `react_agent_max_iterations` as a final guardrail.

## 7. Cleanup of Legacy and Redundant Code

### Remove or consolidate

- legacy `SearchAgent` prompt/build methods if graph no longer uses them
- `build_search_prompt` if it becomes unused after unification
- redundant `tool_search` output fields
- name-based dedupe helpers that break qualified identity
- compatibility branches that silently pick the first matching tool

### Cleanup criteria

- one discovery path
- one tool identity model
- one binding model
- one execution resolution path
- no silent ambiguity resolution

## Implementation Phases

## Phase 1: Foundation and Identity

Deliverables:

- add `QualifiedToolRef` and deterministic callable aliasing
- replace existing `ToolReference` with `QualifiedToolRef` and update all import sites (`deferred_tool_state.py`, `tool_search_tool.py`, `deferred_tool_binding.py`, `tool_execution.py`)
- update MCP manager lookup and execution to support qualified refs
- add execution-layer ambiguity guard in `tool_execution.py` (bare-name lookup fails loudly when multiple servers expose the same tool name)
- update service and API schemas for qualified lookup
- add bound-tool adapter abstraction

Files expected to change:

- `app/ai/mcp_integration.py`
- `app/ai/mcp_tool_catalog.py` (or new `app/ai/mcp_types.py` if `QualifiedToolRef` is extracted to a shared types module)
- `app/ai/deferred_tool_state.py`
- `app/ai/deferred_tool_binding.py`
- `app/ai/tool_execution.py`
- `app/ai/tool_search_tool.py`
- `app/services/mcp_service.py`
- `app/schemas/mcp.py`
- `app/api/mcp.py`

## Phase 2: Catalog and Search Tool V2

Deliverables:

- redesign `tool_search` input modes and output schema
- add server summaries and browse pagination
- improve ranking and ambiguity handling
- compact JSON serialization — change `json.dumps(result, indent=2)` at `tool_search_tool.py:259` (module-level `tool_search`) **and** `:302` (inside `create_tool_search_tool`) to `json.dumps(result, separators=(",", ":"))`
- dynamic result count and conservative autoloading

Files expected to change:

- `app/ai/tool_search_tool.py`
- `app/ai/mcp_tool_catalog.py`
- `app/core/config.py`

## Phase 3: Deferred Runtime and Orchestration Hardening

Deliverables:

- qualified-ID deferred state
- duplicate search suppression
- consecutive search loop guardrails
- dedicated finalization node / forced synthesis path — this is a **new LangGraph node** added to the `StateGraph` in `graph.py`, not an extension of the existing `auto_continue` Python loop. The `auto_continue` loop re-invokes the whole graph on `GraphRecursionError` (up to 5 rounds); the finalization node is a within-graph terminal path that forces a synthesis pass when discovery budget is exhausted mid-turn, before the recursion limit is hit.
- terminal recovery fixes for empty content + unresolved tool-call states

Files expected to change:

- `app/ai/deferred_tool_state.py`
- `app/ai/deferred_tool_binding.py`
- `app/ai/tool_execution.py`
- `app/ai/graph.py`
- `app/ai/prompts.py`
- `app/ai/agents/base_agent.py`

## Phase 4: Cleanup and Compatibility Removal

Deliverables:

- remove obsolete search-agent methods if unused
- remove deprecated bare-name-only code paths
- remove redundant tool-search response fields and helpers
- update `demo.py` MCP API calls that break on qualified endpoint changes — `demo.py` is a pure Streamlit REST client (no direct backend imports) calling endpoints including `GET /mcp/tools/{tool_name}` and `POST /mcp/tools/{tool_name}/execute`. If Phase 1 introduces qualified path segments (`GET /mcp/servers/{server}/tools/{tool_name}` and `POST /mcp/servers/{server}/tools/{tool_name}/execute`), these call sites in `demo.py` must be updated. Audit all `make_api_request` calls in `demo.py` that include a tool name in the path.

Files expected to change:

- `app/ai/agents/search_agent.py`
- `app/ai/prompts.py`
- `demo.py`
- any now-unused compatibility helpers found during implementation

## Acceptance Criteria

1. Two tools with the same original name from different servers can be discovered, loaded, and executed in the same conversation without collision.
2. A server with 50 tools can be explored through paginated browsing and/or server overview without relying on lucky query wording.
3. `tool_search` no longer emits redundant response fields for the model.
4. Repeated identical `tool_search` calls are suppressed or explicitly surfaced as duplicates.
5. The workflow produces a final user-facing answer when discovery exhausts the budget.
6. Discovery responses remain compact enough to avoid becoming the new token bottleneck.
7. Legacy ambiguous bare-name execution paths are removed or fail loudly.

## Test Plan

No tests were run in this planning pass.

### Unit tests

- catalog ranking and pagination
- qualified-ID parsing and normalization
- collision detection and ambiguous-lookup failure
- deferred state keyed by qualified ID
- autoload threshold and churn behavior
- compact response serialization

### Integration tests

- `tool_search` search mode with autoload
- `tool_search` browse mode with cursor paging
- execution of two same-name tools from different servers
- MCP API details/execute by qualified identity
- graph forced finalization after repeated discovery loops
- terminal recovery when the last assistant message contains tool calls but no text

### Regression tests

- large single-server inventory with 50+ fake tools
- repeated duplicate `tool_search` loop
- no-result search loop
- pinned tool behavior under qualified aliases
- non-deferred mode behavior if still supported

## Telemetry and Observability

Add counters and structured logs for:

- `tool_search_calls_per_turn`
- `duplicate_tool_search_calls`
- `tool_search_zero_result_count`
- `tool_search_autoload_count`
- `loaded_tool_evictions`
- `finalization_forced`
- `ambiguous_bare_name_lookup_attempts`
- average response token size for `tool_search`

These metrics are needed to verify the redesign actually reduces loops and token cost.

## Rollout Strategy

Recommended production rollout:

1. Land the qualified identity layer behind a feature flag.
2. Land the new `tool_search` contract and orchestration safeguards behind the same flag.
3. Run compatibility logging for ambiguous bare-name lookups and old endpoint usage.
4. Switch the new flow to default once the tests and telemetry are clean.
5. Remove legacy code and old response handling in a follow-up cleanup pass.

Suggested temporary flags:

- `mcp_tool_search_v2_enabled`
- `mcp_qualified_tool_identity_enabled`
- `mcp_tool_search_duplicate_guard_enabled`

## Risks and Mitigations

### Risk: alias renaming breaks assumptions in downstream code

Mitigation:

- isolate aliasing in a wrapper adapter
- keep source tool metadata available on the wrapper
- migrate all runtime maps together in one phase

### Risk: response schema change breaks consumers

Mitigation:

- version the response contract during rollout
- update demo/admin clients in the same branch

### Risk: browse mode increases tool complexity

Mitigation:

- keep one discovery tool with explicit modes
- keep output compact and cursor-driven

### Risk: orchestration guardrails become too restrictive

Mitigation:

- make thresholds configurable
- emit telemetry for suppressed searches and forced finalization

## Recommended First Implementation Order

1. Introduce qualified tool identity and alias-wrapped binding.
2. Fix ambiguous MCP API/service/manager lookup paths.
3. Redesign `tool_search` output to remove redundant fields and minify payloads.
4. Add server overview and browse pagination.
5. Reduce autoload aggressiveness and make result count dynamic.
6. Add duplicate search guards and forced finalization.
7. Remove old search-agent and bare-name compatibility code.

## Deliverable Definition of Done

The overhaul is complete when the runtime can reliably discover and call tools from large multi-server MCP installations, the model-facing discovery payload is compact and unambiguous, repeated search loops are controlled, and the system always returns a meaningful final assistant response even when tool discovery fails.
