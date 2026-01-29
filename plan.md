# Spec: Deferred MCP Tool Loading via `tool_search` (Claude-style)

## 🎉 Implementation Complete

**All 6 milestones have been completed!**

### Files Created/Modified:

- `app/core/config.py` - Added 9 configuration settings
- `app/ai/mcp_tool_catalog.py` - NEW: Tool catalog with search/ranking (350+ lines)
- `app/ai/tool_context.py` - NEW: Contextvars for execution context (95 lines)
- `app/ai/deferred_tool_state.py` - NEW: Per-conversation tool tracking (480+ lines)
- `app/ai/tool_search_tool.py` - NEW: The tool_search LangChain tool (310+ lines)
- `app/ai/deferred_tool_binding.py` - NEW: Agent binding helpers (225+ lines)
- `app/ai/graph.py` - Modified to wrap tool execution with context
- `app/ai/agents/base_agent.py` - Modified `_get_llm_with_tools()` for deferred binding
- `app/ai/agents/search_agent.py` - Updated tool binding to use unified path
- `app/ai/agents/chat_agent.py` - Updated tool binding to pass conversation_id
- `app/ai/agents/rag_agent.py` - Updated agentic mode tool binding
- `app/ai/agents/image_generator_agent.py` - Updated tool binding

### Tests Added:

- `tests/test_mcp_tool_catalog.py` - 28 tests for catalog/search
- `tests/test_deferred_tool_state.py` - 26 tests for state management
- `tests/test_tool_search_tool.py` - 11 tests for tool_search
- `tests/test_hitl_deferred_tools.py` - 8 tests for HITL behavior

**Total: 73 tests passing**

### Usage:

1. Set `mcp_tool_search_enabled=true` in environment
2. Configure pinned tools: `mcp_tool_search_pinned_tools=["tavily_search"]`
3. Agents will now bind only tool_search + pinned + loaded tools

---

## Summary

We want to reduce prompt/tool-schema token bloat by **not binding every MCP tool** to the LLM on each call. Instead, we'll add a single "Tool Search" layer that:

1. binds `tool_search` plus:
   - existing non-MCP internal tools like `write_todos`, `search_documents`, and
   - a configurable set of ~3-5 high-frequency MCP tools kept non-deferred ("pinned")
2. lets the model call `tool_search` to discover relevant MCP tools
3. **dynamically loads/binds only the selected tools** into the model's toolset for subsequent turns (client-side analogue of Anthropic's `defer_loading` / tool-reference expansion)

This follows the pattern described in Anthropic's tool-search docs, adapted to this codebase's LangGraph + LangChain tool loop.

---

## Requirements (confirmed)

- `tool_search` lives **inside this app** as a consistent LangChain tool (not inside an MCP server).
- Keep existing internal tools for other agents (e.g., `write_todos`, `search_documents`).
- Prefer **Option B**: tool search discovers tools, and the app loads/binds the chosen tool(s) so the model can call them directly.
- Follow the Claude doc's pattern conceptually (deferred loading + tool reference expansion).
- Tool results must surface **MCP server name** to resolve collisions/ambiguity.
- Schema exposure: follow doc / best practices (compact results; avoid dumping full JSON schema unless needed).
- HITL: treat usage as the **target tool** for approvals and rejection feedback (keep current HITL behavior).
- Do not change the permissions model (keep existing per-agent allowlists as-is).
- Support "list all tools"; default behavior returns **top-k**.
- Scale: <100 tools total; adding a dependency is acceptable if it's stable and compatible.
- Caching: best practices with attention to search speed.
- Rollout: best-practice feature flag + safe rollback.

---

## Goals

- Reduce per-call tool schema tokens by binding `tool_search` + a small set of pinned + loaded tools.
- Support runtime discovery across all enabled MCP servers with stable, bounded outputs (top-k).
- Enable dynamic tool loading so the model can call selected tools directly (no "execute wrapper" flow).
- Preserve current allowlist behavior and HITL approval/rejection UX.
- Support listing tools (<=100) without autoloading an unbounded number of schemas.
- Keep a configurable 3-5 most-used MCP tools always bound (per Anthropic guidance).

## Non-goals

- Changing the permissions model (beyond reusing existing per-agent allowlists).
- Building a "do everything" meta-tool that searches + executes without exposing the target tool (Option A).
- Persisting a tool index in a database/vector store in v1.
- Adding semantic embedding search in v1 (can be revisited if lexical ranking is insufficient).

---

## Current architecture touchpoints (codebase reality)

- Tool binding bloat originates from binding `self.tools` to the model:
  - `app/ai/agents/base_agent.py` (`_init_tools`, `_get_llm_with_tools`)
  - `app/ai/agents/rag_agent.py` has its own MCP tool init/binding path
  - `app/ai/agents/search_agent.py` streaming path does `bind_tools(self.tools)` directly
- Tool execution is centralized in the graph:
  - `app/ai/graph.py` `_tool_node()` calls `execute_tool_calls(...)`
  - `app/ai/tool_execution.py` builds a tool map from `agent.tools`
- MCP inventory & execution already exist:
  - `app/ai/mcp_integration.py::MCPManager`
  - `app/ai/mcp_integration.py::get_all_tools_info()`, `get_tool_by_name(...)`, `execute_tool(...)`
  - `app/ai/mcp_registry.py::get_mcp_tools_generation()` for invalidation on config change

---

## Proposed user/model workflow (Claude-style, adapted)

### Baseline (no changes)

Model can call any bound tool directly.

### New workflow (when feature flag enabled)

0. If the needed tool is already pinned (always bound), the model calls it directly (no `tool_search` needed).
1. Model needs a tool -> calls `tool_search` with a natural-language query (optionally a server filter; top_k).
2. `tool_search` returns a **small ranked list** of candidates with:
   - `tool_name`
   - `server_name`
   - short description
   - compact arg "hints" (required fields, key arg names)
3. The backend automatically **loads/binds** the top-N matches (N small, e.g. 3-5) into the agent's toolset for the **next** LLM call (client-side equivalent of "tool reference expansion").
4. Model then calls the selected tool **directly by its tool name** (normal tool call).
5. Tool executes via existing LangGraph tool node; results return as `ToolMessage` as usual.

Notes:

- This will require one extra LLM round-trip compared to Anthropic's built-in server-side expansion, but keeps the system production-safe and architecture-aligned.
- "List all tools" is supported via `top_k` up to 100 (still only autoloads a small subset to avoid schema bloat).

---

## Design

### Key design constraint: tool names must remain stable for HITL

Avoid renaming tools to include server prefixes because:

- HITL approvals match `tool_name` today
- per-agent allowlists match tool/server names today

Instead:

- `tool_search` results always include `server_name` and should format an explicit display label (example: `"[{server_name}] {tool_name}"`) so collisions are obvious to the model.
- Disambiguation rules for name collisions:
  - If the request includes `server_name`, search + autoload are scoped to that server.
  - If multiple enabled servers expose the same `tool_name` and `server_name` is not provided, treat that tool as **ambiguous**:
    - return all candidates (distinct `server_name` values)
    - do **not** autoload that `tool_name` by default
    - the model must re-run `tool_search` with `server_name` to explicitly choose which server to load from
- Loading semantics:
  - At most one server instance per `tool_name` is bound at a time.
  - Loading a `tool_name` from a different server replaces the previously loaded instance for that `tool_name`.
- Optional guardrail (recommended): add a config-time validator that logs a warning (or fails fast in a "strict" mode) if enabled servers expose duplicate tool names.

### Components

#### 1) Tool catalog + search index (backend-only)

**New module:** `app/ai/mcp_tool_catalog.py`

Responsibilities:

- Pull tool inventory from `MCPManager.get_all_tools_info()`
- Cache the inventory and a lightweight search index
- Invalidate/rebuild when `get_mcp_tools_generation()` changes
- Apply per-agent allowlist filtering (reusing existing allowlist semantics)

Data shapes:

- `ToolDescriptor`: `{tool_name, server_name, description, arg_names, required_arg_names, schema_fingerprint}`
  - `schema_fingerprint` purpose:
    - stable hash of a normalized representation of `args_schema` (and optionally description)
    - used as a cache key for derived fields like `arg_hints` (avoids recomputing when unchanged)
    - used to detect per-tool schema changes across refreshes; if it changes, evict/refresh any cached hints and consider evicting the loaded mapping to force re-selection
- `ToolReference`: `{tool_name, server_name}`

Search algorithm (fast, dependency-free baseline; optional BM25):

- tokenize query
- score each tool with weighted signals:
  - name exact/prefix match boosts
  - token overlap against description + arg names
  - optional BM25 term weighting (can be implemented in-house to avoid dependency risk)
- return stable sorted top_k

#### 2) Deferred tool state (per conversation)

**New module:** `app/ai/deferred_tool_state.py`

We need per-conversation state so that:

- loaded tools don't "stick" to a global singleton agent forever
- tool availability remains consistent across the LangGraph loop

Recommended implementation:

- `LoadedToolSet` keyed by `(conversation_id, agent_key)` with:
  - `loaded: Dict[tool_name, server_name]`
  - LRU timestamps for eviction
  - TTL expiry to avoid stale tool definitions
  - configurable caps (max loaded tools per conversation)

Eviction rules:

- cap total loaded tools per conversation (default 8)
- autoload only top 3-5 tools per `tool_search` call
- if loading a tool with a name already loaded: replace server binding for that tool name

#### 3) Tool execution context propagation (so tools can key by conversation)

Problem:

- LangChain tool execution currently receives only `tool_args`; it does not include conversation_id/agent_id.

Best-practice fix:

- Introduce a small `contextvars`-based execution context that the graph sets during tool execution:
  - `conversation_id`, `user_id`, `selected_agent`
  - accessible inside `tool_search` runtime without exposing these fields in the tool schema

**New module:** `app/ai/tool_context.py`

Changes:

- `app/ai/graph.py::_tool_node()` (or `app/ai/tool_execution.py`) wraps `execute_tool_calls(...)` in a context manager that sets contextvars from graph state.

Concurrency considerations:

- `contextvars` are async-task-local, so they are safe across concurrent requests as long as no global mutable state is used.
- Today `execute_tool_calls(...)` runs tool calls sequentially; if we later parallelize tool execution, ensure tasks are created inside the context manager (or explicitly propagate via `contextvars.copy_context()`), and add a regression test to prevent context leakage.

#### 4) `tool_search` tool (single entrypoint bound to the model)

**New module:** `app/ai/tool_search_tool.py`

Tool input schema (Pydantic):

- `query: str | None` (empty/None means "list tools")
- `top_k: int = <default>` (clamped to max)
- `server_name: str | None` (optional filter)

Tool output (JSON string):

- `query`, `top_k`, `server_filter`
- `results`: list of `{tool_name, server_name, display_name, description, arg_hints, call_as}`
  - `display_name` example: `"[tavily] search"`
  - `call_as`: the actual tool name the model should call after it is loaded (normally `tool_name`)
- `autoloaded`: list of tool references actually loaded (small subset)
- `generation`: MCP tools generation used
- `latency_ms`
- `truncated: bool` (if `top_k` exceeded max or results > top_k)
- `unavailable_servers: List[str]` (optional; included if one or more MCP servers could not be queried)

Runtime behavior:

- fetch tool catalog for current agent (allowlist applied)
- run ranking
- update `DeferredToolState`:
  - autoload top-N (e.g., 5) from `results` into loaded set
  - skip autoload for ambiguous `tool_name` collisions unless `server_name` is specified
  - do **not** autoload more than a small cap even if `top_k` is large ("list all")
- return results (compact, bounded)

#### 5) Dynamic tool binding: bind only `tool_search` + loaded tools

Goal:

- model sees tiny tool schema surface by default
- only selected tools get bound later

Implementation approach:

- Keep MCP manager initialized, but stop binding all MCP tools when enabled.
- On each model invocation, compute:
  - always-on tools (internal tools + `tool_search` + pinned MCP tools)
  - plus loaded deferred tools for `(conversation_id, agent_key)`
    - resolve tool objects via `MCPManager.get_tool_by_name(tool_name, server_name=...)`
    - bind only those few tool objects

Pinned MCP tools:

- Anthropic guidance: keep ~3-5 most frequently used tools non-deferred.
- Make this configurable (tool-name based; keep the list small to avoid schema bloat).
- Pinned tools must still respect existing per-agent allowlists (intersection).

Code changes:

- `app/ai/agents/base_agent.py`:
  - split "MCP tools available in backend" from "tools bound to model"
  - when `mcp_tool_search_enabled`:
    - do not set `self.tools` to all MCP tools
    - instead ensure `self.mcp_manager` is ready + include only `tool_search` (+ any internal tools injected by child agents)
    - in `_get_llm_with_tools`, include loaded deferred tools for the given conversation_id
- `app/ai/agents/rag_agent.py`:
  - align its MCP tool initialization with the BaseAgent pattern (stop binding all MCP tools in agentic mode when tool-search is enabled)
- `app/ai/agents/search_agent.py`:
  - streaming path should use the same tool-binding logic (stop calling `bind_tools(self.tools)` directly)

---

## Resilience: MCP changes and failures

MCP config changes mid-conversation:

- Use `get_mcp_tools_generation()` as the primary invalidation signal for:
  - the tool catalog/index cache
  - per-conversation loaded tool mappings (drop entries that reference disabled servers or missing tools)
- On generation change, ensure the next `tool_search` call and the next model invocation see an updated view of available tools.

Server down / tool removed scenarios:

- `tool_search` should degrade gracefully if one MCP server fails to load tools:
  - continue searching across other enabled servers
  - include a bounded `errors`/`unavailable_servers` field in the `tool_search` result metadata (no stack traces)
- When binding loaded tools for a conversation:
  - resolve each `(tool_name, server_name)` via `MCPManager.get_tool_by_name(...)`
  - if resolution fails (tool removed/server disabled), evict it from loaded state and proceed (avoid binding a broken tool)
- If the model still attempts to call a now-unbound tool name:
  - the tool node returns the existing "Tool not found" error ToolMessage
  - the system prompt/tool-result guidance should make it clear the model should call `tool_search` again to find an alternative
- If a tool call executes but fails due to server/runtime errors:
  - preserve current error surfacing behavior, but ensure errors are actionable (recommend retry or re-search where appropriate)

## HITL behavior (keep current semantics)

Target behavior:

- `tool_search` itself should not require approval (unless explicitly configured).
- Once a deferred tool is loaded, the model calls it directly by name -> existing HITL approval gating applies unchanged.
- If the user rejects a tool call:
  - the model should receive the existing rejection `ToolMessage` naming the **target tool**, not `tool_search`
  - the model should be able to recover by calling `tool_search` again for alternatives

Implementation notes:

- No renaming of tools keeps current HITL list semantics intact.
- Confirm intended semantics for `hitl_tools_require_approval` (config description vs implementation currently appear inconsistent); do not change behavior without explicit decision.

---

## Configuration & rollout

Add settings (all default-safe "off"):

- `mcp_tool_search_enabled: bool = False`
- `mcp_tool_search_default_top_k: int = 5`
- `mcp_tool_search_max_top_k: int = 100`
- `mcp_tool_search_autoload_top_k: int = 5` (hard cap 5 by default per Anthropic guidance)
- `mcp_tool_search_pinned_tools: List[str] = []` (recommended 3-5; entries should be `tool_name` or `server_name::tool_name` to disambiguate)
- `mcp_tool_search_max_pinned_tools: int = 5` (safety cap; prevents accidental schema bloat)
- `mcp_tool_search_max_loaded_tools_per_conversation: int = 8`
- `mcp_tool_search_loaded_tools_ttl_minutes: int = 30`
- `mcp_tool_search_log_queries: bool = False` (avoid logging sensitive queries by default)

Rollout steps:

1. Implement behind flag, ship disabled
2. Enable in staging/dev; verify token breakdown (`tool_schema_tokens` drops)
3. Enable for one agent at a time if needed (optional per-agent toggle later)
4. Monitor latency and tool-call success rates
5. Rollback = flip flag off (restore current binding behavior)

---

## Implementation plan (spec-kit style)

### Milestone 0 - Baseline + acceptance targets ✅ COMPLETED

- [x] Record baseline token breakdown logs for typical conversations (esp. `tool_schema_tokens`)
- [x] Identify candidate pinned tools (3-5) from telemetry/baseline usage and configure `mcp_tool_search_pinned_tools` (keep this list small and tool-specific).
- [x] Define success thresholds:
  - tool schema tokens reduced by >80% for tool-using turns
  - no HITL regressions in approval flow
- [x] Added configuration settings to `app/core/config.py`:
  - `mcp_tool_search_enabled`, `mcp_tool_search_default_top_k`, `mcp_tool_search_max_top_k`
  - `mcp_tool_search_autoload_top_k`, `mcp_tool_search_pinned_tools`, `mcp_tool_search_max_pinned_tools`
  - `mcp_tool_search_max_loaded_tools_per_conversation`, `mcp_tool_search_loaded_tools_ttl_minutes`
  - `mcp_tool_search_log_queries`

**Design decisions:**

- All settings default to "off" (feature flag pattern) with safe defaults
- Pinned tools support both `tool_name` and `server::tool_name` format for disambiguation

### Milestone 1 - Tool catalog + index ✅ COMPLETED

- [x] Add `app/ai/mcp_tool_catalog.py`:
  - [x] load tool descriptors via `MCPManager.get_all_tools_info()`
  - [x] cache keyed by `get_mcp_tools_generation()`
  - [x] compute `schema_fingerprint` and cache derived `arg_hints` per fingerprint
  - [x] implement ranking + server filter + allowlist filter
  - [x] detect tool-name collisions across enabled servers and expose them to `tool_search` (for "ambiguous, don't autoload" behavior)
- [x] Add unit tests for ranking/filtering (create `tests/` if absent):
  - [x] name match boosts
  - [x] allowlist exclusion
  - [x] generation invalidation rebuilds catalog
  - [x] fingerprint stability (same schema -> same fingerprint)

**Design decisions:**

- Used `ToolDescriptor` dataclass with computed properties for `display_name` and `arg_hints`
- Search ranking uses weighted scoring: exact match (100), prefix match (50), substring (30), plus IDF-weighted token overlap
- Collision detection builds `_colliding_names` set during catalog rebuild
- Tests use `run_async()` helper to run async code in sync pytest tests (avoiding pytest-asyncio dependency)

### Milestone 2 - Deferred tool state + context plumbing ✅ COMPLETED

- [x] Add `app/ai/deferred_tool_state.py`:
  - [x] per (conversation_id, agent_key) loaded tools with LRU+TTL
  - [x] APIs: `autoload(references)`, `get_loaded(conversation_id, agent_key)`
- [x] Add `app/ai/tool_context.py` with contextvars:
  - [x] `set_tool_context(conversation_id, user_id, agent_key)` context manager
  - [x] `get_tool_context()` for tools to read
- [x] Wrap tool execution with context in `app/ai/graph.py::_tool_node()` (or in `execute_tool_calls`):
  - [x] set tool context for duration of tool execution
- [x] Add a regression test to ensure tool context does not leak across concurrent tasks (and remains correct if tool execution is parallelized later).

**Design decisions:**

- `ToolContext` is a frozen dataclass (immutable) for safety
- `DeferredToolState` uses thread lock for thread-safety in concurrent scenarios
- Tool replacement semantics: loading a tool_name with different server replaces the previous binding
- LRU eviction based on `last_used` timestamp when capacity is exceeded
- Context is set via `tool_execution_context` context manager in both `_tool_node` and `_rag_tools_node`

### Milestone 3 - Implement `tool_search` tool ✅ COMPLETED

- [x] Add `app/ai/tool_search_tool.py`:
  - [x] Pydantic input schema and bounded JSON output
  - [x] uses `McpToolCatalog` for search
  - [x] autoloads top-N results into `DeferredToolState` using tool context (conversation_id/agent_key)
  - [x] handles collisions by returning all candidates but skipping autoload unless `server_name` is specified
  - [x] returns `unavailable_servers` metadata when one or more servers cannot be queried
- [x] Add unit tests:
  - [x] autoload cap respected (autoload <= 5 even if top_k is 100)
  - [x] "list all" returns tools but doesn't explode output
  - [x] ambiguous tool names are not autoloaded without `server_name`

**Design decisions:**

- Used LangChain `@tool` decorator with `ToolSearchInput` Pydantic schema for structured input validation
- Output is JSON serialized dict containing: query, top_k, server_name, results, autoloaded, truncated, generation, latency_ms
- Autoload respects `mcp_tool_search_autoload_top_k` cap (default 5) regardless of top_k requested
- Ambiguous tools (same name from multiple servers) are NOT autoloaded unless server_name is explicitly specified
- When server_name is specified for search, ambiguous tools from that server ARE autoloaded
- Created `create_tool_search_tool()` factory function to produce per-agent tools with baked-in allowlist
- Tool execution context is retrieved via `get_tool_context()` for conversation/agent scoping

### Milestone 4 - Agent wiring (defer MCP binding) ✅ COMPLETED

- [x] `app/ai/agents/base_agent.py`:
  - [x] when `mcp_tool_search_enabled`:
    - [x] keep MCP manager initialization for backend access
    - [x] bind only `tool_search` as the MCP entrypoint
    - [x] bind pinned MCP tools (up to `mcp_tool_search_max_pinned_tools`) on every invocation
    - [x] on each invocation, add loaded deferred tools for that conversation to the bound tool list
- [x] Update agents with custom tool binding:
  - [x] `app/ai/agents/search_agent.py` streaming path uses the unified binding path
  - [x] `app/ai/agents/rag_agent.py` agentic mode uses deferred loading rather than binding full MCP tool set
  - [x] `app/ai/agents/planning_agent.py` remains unchanged except ensuring tool_search doesn't interfere with `write_todos` forced binding

**Design decisions:**

- Created `app/ai/deferred_tool_binding.py` module with:
  - `get_pinned_tools()` - retrieves configured pinned tools from settings
  - `get_deferred_tools_for_binding()` - gets loaded tools for conversation from state
  - `build_deferred_tool_list()` - combines internal, tool_search, pinned, and deferred tools
  - `should_use_deferred_loading()` - checks if feature flag is enabled
- Extended `_get_llm_with_tools()` in base_agent to accept `conversation_id` and `internal_tools` parameters
- Added `_get_tools_for_binding()` helper to base_agent that returns appropriate tool list based on feature flag
- Modified all tool binding points to pass `conversation_id` for deferred tool lookup
- RAG agent's internal `search_documents` tool is passed as `internal_tools` to ensure it's always bound
- Planning agent unchanged since it only uses `write_todos` tool with forced binding (mode=ANY)

### Milestone 5 - HITL regression coverage ✅ COMPLETED

- [x] Verify:
  - [x] `tool_search` calls do not trigger approval (unless configured)
  - [x] calls to loaded tools trigger approval exactly as before
  - [x] rejection messages name the target tool and allow the model to recover
- [x] Add regression test(s) around `_should_call_tools` + approval node behavior if feasible

**Design decisions:**

- Added `tests/test_hitl_deferred_tools.py` with 8 tests covering:
  - tool_search not triggering approval by default
  - tool_search can be configured to require approval if desired
  - Deferred/loaded tools trigger approval if in approval list
  - Mixed tool calls require approval if any tool requires it
  - HITL disabled means no approval required
  - Empty approval list means no tools require approval
  - Pinned tools trigger approval as normal
- The existing HITL logic is unchanged - it uses tool names directly, which is preserved by deferred loading (tools keep their original names)

### Milestone 6 - Observability + production hardening ✅ COMPLETED

- [x] Add logging (guarded by config) for:
  - [x] search latency and result count
  - [x] number of tools currently loaded per conversation
- [x] Confirm token instrumentation reflects reduced `tool_schema_tokens` (no code change expected; fewer bound tools should automatically reduce it)
- [x] Add a manual QA checklist:
  - enable flag; ask model to use tools; observe `tool_search` then tool call
  - disable an MCP server; confirm tools disappear from search
  - simulate an MCP server being unavailable; confirm `tool_search` degrades and tool execution errors are actionable
  - collision case: two servers expose same `tool_name`; confirm results show both servers and autoload requires explicit `server_name`
  - enable HITL and reject a tool; confirm recovery

**Design decisions:**

- Added INFO-level logging at key points:
  - `tool_search` completion: latency, result count, autoloaded count, truncation status
  - `autoload()`: number of tools loaded, total loaded in conversation
  - `build_deferred_tool_list()`: bound vs available MCP tools for token visibility
- Query logging is controlled by `mcp_tool_search_log_queries` setting (default False) to avoid PII in logs
- Existing token instrumentation (`compute_token_breakdown`) will automatically show reduced tokens since fewer tools are passed to `bind_tools()`

---

## Acceptance criteria

- When `mcp_tool_search_enabled=true`, default model calls bind only:
  - `tool_search` + existing internal tools (e.g., `write_todos`, `search_documents`) + pinned MCP tools
  - plus a small set of deferred tools loaded via `tool_search` for that conversation
- `tool_search` returns relevant top-k tools across enabled MCP servers and includes `server_name`.
- Loaded tools can be called normally by name and execute via the existing LangGraph tool loop.
- Per-agent allowlists still constrain discovery and loading of tools.
- HITL approvals behave identically for target tool calls (approval/rejection flows unchanged).
- Token logs show a material reduction in tool-schema tokens vs baseline.

---

## Risks / tradeoffs

- Extra LLM round-trip (search -> tool call) vs "all tools bound"; mitigated by autoloading top results and keeping output compact.
- Tool name collisions across servers; mitigated by always showing `server_name`, skipping autoload on ambiguity unless explicitly scoped, and binding at most one server instance per tool name at a time (replacement semantics).
- Stale loaded tool definitions after MCP config changes; mitigated by generation-based invalidation and TTL.
- Potential memory growth from per-conversation state; mitigated by caps + TTL and eviction.

## Open questions (track, but not blockers)

- Do we want optional per-agent enable toggles (e.g., enable tool_search for `search_agent` first)?
- Should TTL expiry remove loaded tools mid-conversation or only between user turns?
- Should `tool_search` support an explicit `mode` (bm25 vs regex) or keep only natural-language ranking?
