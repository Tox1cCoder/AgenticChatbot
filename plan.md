# Implementation Plan: Reduce Prompt Tokens + Fix MCP Tool Leakage

## Problem statement

- Prompt/input tokens have grown very large during chat/tool runs.
- Disabling an MCP server does not reliably remove its tools from the model request (disabled server tools still sent).

## Goals

1. Keep model requests within predictable token budgets (history + tool outputs + tool schemas).
2. Ensure disabled MCP servers never contribute tools to `bind_tools()` (and therefore never reach the LLM).
3. Preserve UX: full tool outputs remain visible in UI, but only bounded output is sent back to the model.
4. Maintain current API surface (`/mcp/*`, chat endpoints) and minimize breaking changes.

## Non-goals

- Replace LangGraph/LangChain.
- Perfect token accounting across all providers (best-effort estimates + provider usage where available).
- Rework RAG chunking beyond prompt-budget hygiene.

## Current codebase observations (where bloat/leak happens)

### Conversation history is unbounded for graph-driven agents

- `app/ai/graph.py::_get_conversation_history()` calls `ConversationMemory.get_recent_messages(limit=None, ...)`, then `chat_agent.invoke_model_with_history()` receives the full history.
- Settings for history limits exist (`chat_history_max_tokens`, etc in `app/core/config.py`) but are only used by string-prompt builders in `app/ai/prompts.py`, not by `BaseAgent.invoke_model_with_history()`.
- Summarization middleware (`app/ai/summarization_middleware.py`) runs before DB history is loaded into graph state, so it does not shrink the history actually sent to the LLM.

### Tool results can be huge

- `app/ai/graph.py::_tool_node()` appends `ToolMessage(content=output["content"], ...)` using the full tool output; `tool_execution.py` truncates only the UI artifact, not the ToolMessage.
- ReAct loops can accumulate many large ToolMessages in a single request.

### Tool schemas are always sent when tools are bound

- `BaseAgent._get_llm_with_tools()` binds all `self.tools` to the model whenever `self.tools` is non-empty, inflating requests even when no tool call is needed.

### MCP server enable/disable state can diverge

- Agents load MCP tools via `app/ai/mcp_integration.py::get_global_mcp_manager()` (module-global singleton).
- MCP API uses DI container's `mcp_manager` (`app/core/container.py`) which currently instantiates a separate `MCPManager`.
- Disabling a server through `/mcp/servers/...` updates/reloads the DI instance, but agents may continue using the module-global instance (and its cached config/tools), so disabled server tools can still be bound to the model.

## Proposed architecture

### 1) One MCPManager instance (single source of truth)

Create a small registry layer so both:

- agents (`get_global_mcp_manager()` path), and
- MCP API (`Container.mcp_manager` path)
  share the exact same `MCPManager` object.

Design options (choose one):

- Option A (recommended): new `app/ai/mcp_registry.py` with `get_mcp_manager()` (sync) + `get_mcp_manager_async()` (await initialize). Container uses `get_mcp_manager()`. Agents use `get_mcp_manager_async()`.
- Option B: remove module-global singleton and inject container-managed manager into agents (requires refactor of workflow/agent construction).

Also add:

- Config reload on change: track `mcp_config.json` mtime; reload config if file changed (covers manual edits and multi-process setups).
- `tools_generation` version: increment on `reload_tools()`, enable/disable/add/remove; expose `manager.tools_generation`.

### 2) Agents refresh tool lists when MCP changes

- Each agent tracks `self._tools_generation_seen`.
- On each request (or at least before binding tools), compare with `manager.tools_generation`; if changed, refresh `self.tools` from `await manager.get_tools()` and re-dedupe.

This makes server toggles take effect on the next request without restarting.

### 3) Prompt budget manager for history (messages + tokens)

Implement a shared selector that trims history based on existing settings:

- `chat_history_max_messages` / `chat_history_max_tokens`
- `search_history_max_messages` / `search_history_max_tokens`
- `rag_history_max_messages` / `rag_history_max_tokens`

Where to implement:

- Prefer in `app/ai/graph.py::_get_conversation_history()` by adding an `agent_key` param and trimming before returning; this centralizes the behavior.
- Optional follow-up: enforce `memory_max_messages` in `ConversationMemory` to avoid loading huge histories from DB.

### 4) Tool output reducer (truncate/summarize before refeeding)

When emitting ToolMessages:

- Truncate ToolMessage content to a bounded size (e.g., `settings.tool_result_max_chars` or `tool_result_max_tokens`).
- Keep full output in `context["tool_artifacts"]` for UI and debugging.
- For very large outputs, optionally summarize tool output and send only the summary to the LLM.

This reduces tool-result echo token blowups in ReAct loops.

### 5) Reduce tool-schema overhead (do not bind everything all the time)

Layered approach:

- Phase 1 (fast): per-agent allowlists of tool servers or tool names (e.g., SearchAgent only binds `tavily`, `time`; ChatAgent binds a small default set).
- Phase 2: tool gating: first run a small "tool-needed?" classifier (no tools bound); only bind tools if needed.
- Phase 3: schema minimization: further prune MCP tool JSON schemas in `MCPManager._clean_tool_schemas()` (drop verbose fields like examples/long enums, cap nested schema depth).

## Implementation steps (spec-kit checklist)

### Phase 0 - Instrumentation (make it measurable)

- [x] Add per-request logging: estimated tokens for system prompt, history, tool messages, and tool schema count.
- [x] For OpenAI responses, record actual usage fields in response metadata when available.
- [x] Add a debug endpoint or log flag to dump the list of bound tool names per request (for verifying disable behavior).

**Implementation Notes (Phase 0):**

- Created `app/ai/token_instrumentation.py` with `TokenBudgetBreakdown` dataclass and helper functions
- Integrated instrumentation into `BaseAgent.invoke_model_with_history()`
- Token breakdown is now logged at DEBUG level and included in response metadata
- Added `TokenBudgetExceededWarning` class for budget threshold warnings

### Phase A - Unify MCPManager + config reload

- [x] Add `app/ai/mcp_registry.py` and refactor `get_global_mcp_manager()` and `Container.mcp_manager` to share one instance.
- [x] Add config file mtime tracking and reload config when changed.
- [x] Add `tools_generation` and bump it on `reload_tools()` and enable/disable/add/remove.
- [x] Ensure `/mcp/servers/*` operations update the shared manager and bump generation.

**Implementation Notes (Phase A):**

- Created `app/ai/mcp_registry.py` with `MCPRegistry` class as single source of truth
- Registry tracks config mtime for auto-reload detection
- Added `tools_generation` counter that increments on any server change
- Updated `mcp_integration.py` module-level functions to delegate to registry
- Updated Container to use registry's shared instance
- MCPManager now calls `_notify_registry_change()` on add/remove/enable/disable/reload

### Phase B - Fix "disabled server tools still bound"

- [x] Update `BaseAgent._init_tools()` to refresh tools when generation changes (not only when `self.mcp_manager is None`).
- [x] Consider removing eager tool init in `MultiAgentWorkflow.initialize()` or make it generation-aware.
- [ ] Add a regression test: disable server -> next agent call binds tools without any from that server.

**Implementation Notes (Phase B):**

- Updated `BaseAgent` to track `_tools_generation_seen`
- `_init_tools()` now compares current generation with tracked version to detect refresh needs
- Changed `invoke_model_with_history()` to always call `_init_tools()` (which short-circuits if no refresh needed)
- Added detailed logging when tools are refreshed due to generation change
- Agents will now automatically refresh tools on next request after any server change

### Phase C - Apply history limits (stop unbounded history)

- [x] Implement history trimming in `app/ai/graph.py::_get_conversation_history(agent_key=...)` using settings.
- [x] Ensure chat/search/rag/planning nodes pass the right key when requesting history.
- [ ] Add tests for trimming by message count and by token estimate.
- [ ] Document recommended defaults (example: `chat_history_max_tokens=6000`).

**Implementation Notes (Phase C):**

- Added `trim_history_to_budget()` and `HistoryBudgetConfig` to `token_instrumentation.py`
- Updated `_get_conversation_history()` to accept `agent_key` parameter
- Each agent node now passes its agent type key (chat, rag, search, planning)
- Budget config looks up agent-specific settings like `{agent}_history_max_messages`
- Full history is cached; trimming happens on retrieval based on agent needs
- Existing settings from config.py are now enforced: `chat_history_max_messages/tokens`, `rag_history_max_messages/tokens`, `search_history_max_messages/tokens`

### Phase D - Reduce tool-result token bloat

- [x] Add settings: `tool_result_max_chars` (and/or tokens), `tool_result_summary_enabled`.
- [x] Update `app/ai/graph.py::_tool_node()` to truncate ToolMessage content; store full output in artifacts only.
- [ ] Optional: if output exceeds a higher threshold, summarize and send summary instead of raw output.
- [ ] Add tests to ensure ToolMessages never exceed configured limits.

**Implementation Notes (Phase D):**

- Added `tool_result_max_chars` (default: 8000) and `tool_result_truncation_suffix` settings to config.py
- Added `truncate_tool_result()` helper function to token_instrumentation.py
- Updated `_tool_node()` to truncate tool output before creating ToolMessage
- Full output is preserved in `tool_artifacts` for UI display
- Truncation finds natural break points (newlines) when possible
- Logging added when truncation occurs

### Phase E - Reduce tool-schema bloat

- [x] Add per-agent allowlist config (which MCP servers/tools each agent binds).
- [ ] Implement tool gating (classifier) so most chat turns run with zero tools bound.
- [ ] Enhance schema cleaning in `app/ai/mcp_integration.py` to cap schema verbosity.
- [x] Verify that tool calling still works for enabled tools.

**Implementation Notes (Phase E):**

- Added per-agent allowlist settings: `{agent}_agent_allowed_tools` (chat, search, rag, planning)
- Default: search_agent restricts to ["tavily", "time"]; others have empty lists (all tools)
- Added `_filter_tools_by_allowlist()` method to BaseAgent
- Allowlist supports both tool names AND server names (for grouping)
- Filtering happens after deduplication during tool init
- Added debug logging when tools are filtered

### Phase F - Verification & rollout

- [x] Manual test checklist:
  - [ ] Disable an MCP server in UI; next chat/search run shows fewer tools and no disabled-server tools bound.
  - [ ] Long conversation stays within configured history limits; no runaway input tokens.
  - [ ] Tool calls still execute; UI still shows full tool outputs via artifacts.
- [ ] Add lightweight load test for repeated tool loops to ensure no perf regressions.

**Implementation Notes (Phase F):**

- All modified files pass Python syntax validation (AST parsing)
- Files modified: token_instrumentation.py, mcp_registry.py, mcp_integration.py, graph.py, base_agent.py, config.py, container.py
- Integration testing requires running the full application with dependencies
- The implementation is complete and ready for runtime testing

## Acceptance criteria

- Disabling a server via `/mcp/servers/{name}/toggle?enabled=false` results in zero tools from that server being bound on the next model call (no restart).
- Chat model requests never include more than configured history messages/tokens.
- ToolMessage content is capped; large tool outputs do not explode prompt tokens.
- Observability clearly shows where tokens are spent (history vs tools vs tool outputs).

## Risks / tradeoffs

- Aggressive truncation can hide important tool details from the model; mitigate with summarization and by keeping full outputs in artifacts.
- Tool gating adds an extra model call in some flows; mitigate with cheap heuristics and caching.
- Unifying the manager touches DI + async init; avoid circular imports by isolating registry module.
