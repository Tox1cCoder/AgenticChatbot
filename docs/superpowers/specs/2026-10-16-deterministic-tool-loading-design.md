# Deterministic tool loading — design

Date: 2026-10-16. Decided in conversation with Thai after the `take100` rename broke the skill (fix `ccc1af12`).

## Why
Deferred tools are callable only after `tool_search` autoloads them. Autoload fires for one result only, and only when that result is "high" confidence and leads the runner-up by 8 points. Both thresholds come out of hand-tuned bonuses (`tool_search_scoring.py`: +40/+14/+12/+35/+65, capability word lists). So when the ranking is wrong, the tool cannot be called, and nothing reports it: the model just gives up. `take100` hit exactly this. A one-word skill name put its command tool 6 points ahead of Desktop Commander, not 8, so the tool stayed unloaded, even though `activate_skill` had already given the model its exact name.

A mature harness (Claude Code, observed in its own session) keeps ranking and binding separate:
- Ranking may be fuzzy.
- Selecting tools by exact name always loads them.
- Search results are loaded and the model chooses among them.
- Deferred tool names are listed in context.
- Policy (allowlists, exclusions, approval gates) is separate from ranking.

## Goals
1. Loading by exact name is deterministic: `tool_search(select=[...])` loads every named tool that exists and that the agent may use.
2. `activate_skill` loads its own command tool. A skill never depends on search.
3. Keyword search loads its top results (default 3). A confidence band or margin never withholds a tool.
4. The system prompt lists the deferred tool names the agent may load, so the model can select without searching.
5. A tool the model is using stays loaded: expiry and LRU run from last use for client tools, as they already do for server tools.

## Non-goals (Plan B, separate)
- Replacing `_score_candidate` / `_capability_specific_adjustment` with BM25, and rewriting the tests that pin hand-tuned confidence values. Ranking keeps its current scorer until then. The `ccc1af12` +100 exact-name boost stays.

## Anti-goals
- Selection must never reveal or load a tool the agent's allowlist forbids. Excluded raw web providers (`tavily_search`, `tavily_extract`, `brave_image_search`) stay unloadable.
- No new runtime dependency.
- Do not load tools when the model browses: inventory and per-server inventory load nothing.

## Decisions
- Selection is a new `select: list[str] | None` field on the `tool_search` input. A `select:` prefix in `query` is not supported: one way to do it.
- A raw server tool name that several servers share is ambiguous. Selecting it returns its aliases (`<server>__<tool>`) and loads nothing.
- The external-open safety rule stays: when a client action tool exists, server openers are not loaded. It becomes an explicit load-block flag instead of riding on `_autoload_eligible`.
- The output drops `recommended_tool` and `requires_refinement`. `next_action` becomes `call_a_loaded_tool` | `refine_search` | `inspect_inventory`.
- Claude's call: default caps are `mcp_tool_search_autoload_top_k` 1 → 3 and `mcp_tool_search_max_loaded_tools_per_conversation` 8 → 16, because each search now loads up to 3 tools.
- Claude's call: the prompt inventory is capped by a new setting, `mcp_tool_search_prompt_inventory_max_tools`, default 60. Past the cap it lists that many names and tells the model to use `tool_search` for the rest.
