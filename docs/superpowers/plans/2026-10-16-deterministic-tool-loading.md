# Deterministic Tool Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a deferred tool callable whenever the model asks for it by name, when a skill is activated, or when search returns it. A confidence score or a ranking margin must never withhold it.

**Architecture:**
- Separate binding from ranking. Add exact selection (`tool_search(select=[...])`) in a new `app/ai/tool_selection.py`, which is the one place that resolves names to load references.
- `activate_skill` reuses that module to load its own command tool.
- Keyword search loads its top 3 results, ungated. Inventory modes load nothing.
- The system prompt lists the deferred tool names the agent may load.
- Client tool names and client-tool expiry are fixed first, because exact selection depends on both.

**Tech Stack:** Python 3.13 (`.venv`), LangChain/LangGraph tools, pydantic, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-16-deterministic-tool-loading-design.md`

## Global Constraints
- Line length 100 (ruff E501 is enforced). Functions ≤ 100 lines, cyclomatic complexity ≤ 8, ≤ 5 positional parameters.
- Run tests and lint with the app interpreter: `.venv/Scripts/python.exe -m pytest ...` and `.venv/Scripts/python.exe -m ruff check ...`.
- Never load a tool the agent's allowlist forbids or that `RAW_WEB_TOOL_NAMES` excludes (`app/ai/deferred_tool_binding.py:88-94`).
- Inventory mode (empty `query`, with or without `server_name`) loads nothing.
- No new runtime dependency.
- Commit after each task with only that task's files staged. Other sessions share this working tree, so never use `git add -A` and never stash.

## Review Focus
1. **Selecting a name a custom agent may not use.** It must come back as `not_found` and must not load, so that existence does not leak. Test in Task 3.
2. **Selecting an excluded raw web tool** (`tavily_search`). It must be refused and not loaded, with the existing web guidance. Test in Task 3.
3. **The device reconnects between activation and the call.** The new session means the load returns nothing. Activation must not claim the tool is callable; it falls back to telling the model to `select` it. Test in Task 4.
4. **A long conversation with many searches.** A client tool used in this turn must not be evicted by LRU, and a tool idle past the TTL must expire. Test in Task 2.
5. **Inventory at the limits** (0 tools, more than the cap, a 500-character description). The prompt block must stay bounded and well formed. Test in Task 6.

---

### Task 1: Client tool names agree between catalog and runtime

`ClientToolCatalog._rebuild_from_catalog` (`app/ai/client_tool_catalog.py:227-232`) builds names without de-duplication. The runtime (`client_runtime_tools._build_exposed_name`, L94-117) appends `_2`, `_3` on collision. So selecting by exact name, and binding by name (`base_agent.py:768`), miss a tool whose name collided.

**Files:**
- Modify: `app/ai/client_runtime_tools.py` (add a public helper next to `get_exposed_client_tool_name`, L169)
- Modify: `app/ai/client_tool_catalog.py:227-232`
- Test: `tests/test_client_tool_isolation.py`

**Interfaces:**
- Produces: `exposed_names_by_qualified_id(catalog: dict[str, Any]) -> dict[str, str]` in `client_runtime_tools.py`, built from `_parse_tool_specs(catalog)` (`spec.qualified_tool_id -> spec.exposed_name`). `ClientToolCatalog` descriptors use these names.

- [ ] **Step 1: Write the failing test.** Add `test_catalog_names_match_runtime_names_on_collision`. Build a session catalog with an MCP entry and a skill entry that collide, using the same fixture shape as `tests/test_skills_tool.py:108-151`. Assert that `{d.tool_name for d in catalog.list_all()} == {t.name for t in <runtime tools for that catalog>}`. Assert that `catalog.get_tool("client__skill_client_skill__run_skill_command_2")` is not None.
- [ ] **Step 2: Run it and confirm it fails.** Run `.venv/Scripts/python.exe -m pytest tests/test_client_tool_isolation.py -k collision -v`. Expected: FAIL, because `get_tool` returns None for the `_2` name.
- [ ] **Step 3: Implement.** Add `exposed_names_by_qualified_id`. In `_rebuild_from_catalog`, take each entry's exposed name from that map, keyed by the entry's qualified id. Fall back to the current sanitize expression only for an entry the map lacks.
- [ ] **Step 4: Run the suites.** Run `.venv/Scripts/python.exe -m pytest tests/test_client_tool_isolation.py tests/test_skills_tool.py tests/test_custom_agent_client_tool_resync.py tests/test_client_tool_scope.py -q`. Expected: all pass.
- [ ] **Step 5: Commit** with the message `fix(client-tools): name catalog tools exactly as the runtime binds them`.

### Task 2: Client tools stay loaded while they are used

`_mark_tool_used_if_deferred` (`app/ai/tool_execution.py:761-782`) calls `DeferredToolState.mark_tool_used` (`deferred_tool_state.py:788`). That method only looks at server tools, so a client tool's TTL and LRU run from its last `add`.

**Files:**
- Modify: `app/ai/deferred_tool_state.py` (new method beside `mark_tool_used`)
- Modify: `app/ai/tool_execution.py:761-782`
- Modify: `app/core/config.py:1792`: change the default of `mcp_tool_search_max_loaded_tools_per_conversation` from 8 to 16
- Test: `tests/test_deferred_tool_state_lifecycle.py`

**Interfaces:**
- Produces: `DeferredToolState.mark_client_tool_used(conversation_id: str | None, agent_key: str | None, tool_name: str, device_id: str | None, session_id: str | None = None, user_id: str | None = None) -> None`. It updates `LoadedClientTool.last_used` and the scope's `last_active`. It does nothing when the tool is not loaded.

- [ ] **Step 1: Write the failing tests.**
  - `test_client_tool_ttl_runs_from_last_use`: load a client tool. Advance the monotonic clock (patch it as the existing lifecycle tests do) to TTL − 1 min, then call `mark_client_tool_used`, then advance 2 min more. Assert the tool is still in `get_loaded_client_tools`. Advance past the TTL again with no use, and assert it is gone.
  - `test_used_client_tool_survives_lru`: with `max_tools=2`, load A and B, mark A used, load C. Assert A and C are loaded and B is gone.
- [ ] **Step 2: Run them and confirm they fail.** Run `.venv/Scripts/python.exe -m pytest tests/test_deferred_tool_state_lifecycle.py -k "last_use or survives_lru" -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Add `mark_client_tool_used`, resolving the session the same way `get_loaded_client_tools` does (L710-732). In `_mark_tool_used_if_deferred`, call it for names starting with `client__`, passing the tool context's `device_id` and `user_id`. Change the config default to 16.
- [ ] **Step 4: Run the suites.** Run `.venv/Scripts/python.exe -m pytest tests/test_deferred_tool_state_lifecycle.py tests/test_client_tool_isolation.py tests/test_tool_execution_recovery.py -q`. Expected: all pass.
- [ ] **Step 5: Commit** with the message `fix(deferred-tools): keep client tools loaded while they are used`.

### Task 3: Exact selection — `tool_search(select=[...])`

**Files:**
- Create: `app/ai/tool_selection.py`
- Modify: `app/ai/mcp_tool_catalog.py` (two lookups on `McpToolCatalog`)
- Modify: `app/ai/tool_search_tool.py`: add a `select` field to `ToolSearchInput` (L115-143), and branch to selection at the top of `_execute_tool_search` (L213), before inventory and search
- Modify: `app/core/config.py`: add `mcp_tool_search_max_select: int = 8` next to L1764, and register it in the `_non_negative_int` validator list (L2130-2139)
- Test: create `tests/test_tool_selection.py`

**Interfaces:**
- Consumes: `exposed_names_by_qualified_id` (Task 1). Also `ClientToolCatalog.get_tool(name, server_name=None)` (`client_tool_catalog.py:407`), the catalogs' `_descriptor_matches_allowlist` statics, `ToolReference` and `ClientToolReference`, and `DeferredToolState.autoload` / `autoload_client_tools`.
- Produces:
  - `McpToolCatalog.get_by_call_name(name: str) -> ToolDescriptor | None`. It matches `get_call_name()`; a raw name matches only when it is not ambiguous.
  - `McpToolCatalog.aliases_for(tool_name: str) -> list[str]`. It returns the call names of every descriptor with that raw name, sorted.
  - In `app/ai/tool_selection.py`:
    ```python
    @dataclass(frozen=True)
    class Selection:
        server_refs: list[ToolReference]
        client_refs: list[ClientToolReference]
        found: list[dict[str, Any]]      # public dicts: tool_name, description, arg_hints
        not_found: list[str]
        ambiguous: dict[str, list[str]]  # raw name -> aliases
        excluded: list[str]              # names refused by the runtime exclusions

    def resolve_selection(names: list[str], *, server_catalog: McpToolCatalog | None,
                          client_catalog: ClientToolCatalog | None,
                          server_allowlist: list[str] | None, client_allowlist: list[str] | None,
                          excluded_tool_names: frozenset[str]) -> Selection
    def load_selection(selection: Selection, *, ctx: ToolContext,
                       client_session_id: str | None) -> set[str]   # names actually loaded
    ```
  - The `tool_search` output for a selection is `{"mode": "select", "results": [found... with is_loaded], "not_found", "ambiguous", "excluded", "loaded_count", "next_action"}`. `next_action` is `"call_a_loaded_tool"` when anything loaded, else `"refine_search"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_tool_selection.py`. Use fake catalogs in the style of `tests/test_unified_tool_search.py` and the real `DeferredToolState`. Each test asserts loaded names via `get_loaded` / `get_loaded_client_tools`.
  - `test_select_loads_named_server_and_client_tools`: selecting `["read_file", "client__skill_take100__run_skill_command"]` loads both. The output has `mode == "select"`, `loaded_count == 2`, and every result has `is_loaded is True`.
  - `test_select_loads_regardless_of_query_scoring`: a tool that no keyword query ranks "high" still loads by name.
  - `test_select_unknown_name_is_not_found`
  - `test_select_name_outside_custom_agent_allowlist_is_not_found_and_not_loaded` (Review Focus 1)
  - `test_select_excluded_raw_web_tool_is_refused` (Review Focus 2): `"tavily_search"` appears in `excluded`, nothing loads, and `guidance` is the existing web-provider text (`tool_search_tool.py:595-601`).
  - `test_select_ambiguous_raw_name_returns_aliases`: two servers expose `search`, so `ambiguous == {"search": ["a__search", "b__search"]}` and nothing loads. Selecting `"a__search"` loads it.
  - `test_select_more_than_cap_is_refused_whole`: 9 names with cap 8 produce an `error` key and no loads.
  - `test_select_duplicates_load_once`
- [ ] **Step 2: Run them and confirm they fail.** Run `.venv/Scripts/python.exe -m pytest tests/test_tool_selection.py -v`. Expected: FAIL (import error, `tool_selection` does not exist).
- [ ] **Step 3: Implement** the two catalog lookups, `tool_selection.py`, the `select` field and the branch in `_execute_tool_search`.
  - Field description: "Exact tool names to load, as listed under Deferred tools or returned by an earlier search. Loads every named tool you may use; query and top_k are ignored when this is set."
  - Order of resolution per name: client catalog first (names start with `client__`), then server `get_by_call_name`, then `aliases_for`.
  - Allowlists and exclusions are applied before a name counts as found.
- [ ] **Step 4: Run the suites.** Run `.venv/Scripts/python.exe -m pytest tests/test_tool_selection.py tests/test_unified_tool_search.py tests/test_tool_search_runtime_exclusions.py tests/test_custom_agents_tools.py tests/test_specialist_middleware.py -q`. Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat(tool-search): load tools by exact name with select`.

### Task 4: `activate_skill` loads its own command tool

**Files:**
- Modify: `app/ai/skills_tool.py:111-155`
- Modify: `app/ai/tool_execution.py:56`: change `TOOL_LOADING_TOOLS` to `{"tool_search", "activate_skill"}` so the same-batch map refresh (L2173-2184) runs after activation too
- Modify (only if Step 2's custom-agent test fails on the allowlist): `app/ai/custom_agent_runtime.py:102-120`. `client_tool_search_allowlist()` also returns `skill::<name>::run_skill_command` for each name in `allowed_skill_refs`.
- Test: `tests/test_skills_tool.py`

**Interfaces:**
- Consumes: `resolve_selection` and `load_selection` (Task 3), with `server_catalog=None` and the agent's client allowlist. Also `get_tool_context()` for `conversation_id` and `agent_key`, and `get_client_tool_catalog(device_id, user_id)`.
- Produces: activation output. On a successful load the binding paragraph reads "The command tool `<name>` is loaded and callable now. Call it with an argv array; do not call the internal qualified id `<qid>` directly." When nothing loaded, it reads "Load it with tool_search(select=["<name>"]), then call it with an argv array."

- [ ] **Step 1: Write the failing tests.**
  - `test_activation_loads_the_command_tool`: after `activate_skill("take100")` in a tool context with a conversation id, `get_loaded_client_tools` contains `client__skill_take100__run_skill_command`, and the output contains "loaded and callable now".
  - `test_activation_after_session_change_does_not_claim_callable` (Review Focus 3): the catalog session differs from the bound session, so nothing is loaded and the output contains `tool_search(select=`.
  - `test_custom_agent_activation_binds_the_allowed_skills_command`: a custom agent whose `allowed_skill_refs` includes the skill gets the command tool from `_get_tools_for_binding` after activation.
  - Update the existing `_2`-name test at L108-151 to the new wording.
- [ ] **Step 2: Run them and confirm they fail.** Run `.venv/Scripts/python.exe -m pytest tests/test_skills_tool.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.** After a successful dispatch, resolve `[exposed_name]` and load it, then write the wording above. Apply the `custom_agent_runtime` change only if the custom-agent test fails on the allowlist.
- [ ] **Step 4: Run the suites.** Run `.venv/Scripts/python.exe -m pytest tests/test_skills_tool.py tests/test_skill_device_isolation.py tests/test_skill_installation_chat_integration.py tests/test_custom_agents_tools.py tests/test_hitl_client_and_deferred.py tests/test_tool_execution_recovery.py -q`. Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat(skills): activation loads the skill's command tool`.

### Task 5: Search loads its top results; ranking no longer gates binding

**Files:**
- Modify: `app/ai/tool_search_tool.py`
  - autoload loop (L458-516): drop the `_autoload_eligible` and confidence gates; keep the score floor, the ambiguity skip and the exclusions; honor a new internal `_load_blocked` flag
  - `_prioritize_external_action_results` (L731-759): set `_load_blocked = True` where it now sets `_autoload_eligible = False`
  - `_build_recommendation` (L690-728): replace it with `_next_action(public_results: list[dict]) -> str`
  - inventory paths (L331-351, L385-400, and per-server inventory): load nothing
  - both docstrings (L841-881, L921-947)
- Modify: `app/core/config.py:1764`: change the default of `mcp_tool_search_autoload_top_k` from 1 to 3
- Modify: `app/ai/prompts.py:337-351` (`TOOL_EXPLORATION_SUFFIX`), `app/ai/prompts.py:173-178` and `app/ai/prompts.py:311-314`
- Modify: `README.md:735`
- Test: `tests/test_unified_tool_search.py`, `tests/test_tool_search_runtime_exclusions.py`, `tests/test_tool_search_prompt_guidance.py`, `tests/test_tool_search_scoring.py:148`

**Interfaces:**
- Produces: the discovery output `{"query", "mode", "results", "next_action", "loaded_count", "more_available", "resolved_server_name"?, "guidance"?, "debug_scores"?}`. It no longer has `recommended_tool` or `requires_refinement`. `next_action` is `"call_a_loaded_tool" | "refine_search" | "inspect_inventory"`.
- Prompt wording (exact): "Results with `is_loaded: true` are callable now: pick the one that fits and call it. To load a specific tool by name, use `tool_search(select=[...])`. Refine the query only when no loaded result fits."

- [ ] **Step 1: Write the failing tests and update the pinned ones.**
  - New: `test_search_loads_top_three_without_confidence_gate`. Three medium-confidence matches are all loaded and `next_action == "call_a_loaded_tool"`.
  - New: `test_per_server_inventory_loads_nothing`. The current behavior autoloads the first listed tool.
  - New: `test_external_open_server_opener_stays_blocked`. This replaces the `_autoload_eligible` assertion at `test_unified_tool_search.py:973`.
  - Rewrite these to the new shape:
    - the `recommended_tool` / `requires_refinement` / `next_action` assertions at `test_unified_tool_search.py` L730, L842, L1002
    - `test_tool_search_runtime_exclusions.py:109` (keep: raw web tools never load)
    - `test_tool_search_prompt_guidance.py:97` (assert the new wording, and that "recommended_tool" and "requires_refinement" are absent)
    - `test_tool_search_scoring.py:148` (keep: below the score floor, nothing loads)
  - Leave the `rank_tool_candidates` confidence/eligibility tests alone: they test ranking, which Plan B replaces.
- [ ] **Step 2: Run them and confirm they fail.** Run `.venv/Scripts/python.exe -m pytest tests/test_unified_tool_search.py tests/test_tool_search_runtime_exclusions.py tests/test_tool_search_prompt_guidance.py -q`. Expected: the new and rewritten tests FAIL.
- [ ] **Step 3: Implement** the loop, the flag, `_next_action`, the no-load inventory paths, the docstrings, the prompt text and the README line.
- [ ] **Step 4: Run every tool-search, deferred and specialist suite.** Run `.venv/Scripts/python.exe -m pytest tests/test_tool_search_accuracy.py tests/test_tool_search_action_intent.py tests/test_tool_search_scoring.py tests/test_unified_tool_search.py tests/test_tool_search_runtime_exclusions.py tests/test_tool_search_prompt_guidance.py tests/test_tool_selection.py tests/test_deferred_tool_binding.py tests/test_specialist_middleware.py tests/test_planning_binds_only_what_it_executes.py tests/test_hitl_client_and_deferred.py -q`. Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat(tool-search): load top results; ranking no longer gates binding`.

### Task 6: Deferred tool names in the system prompt

**Files:**
- Create: `app/ai/deferred_tool_inventory.py`
- Modify: `app/ai/agents/base_agent.py:2228`: append the inventory right after `TOOL_EXPLORATION_SUFFIX`
- Modify: `app/ai/agents/rag_agent.py:225` and `:1304`: same placement
- Modify: `app/ai/agents/custom_agent.py`: override the new suffix method to pass the spec's allowlists, following the `_build_skills_suffix` override at L94-104
- Modify: `app/ai/agents/planning_agent.py`: override the new suffix method to return "". Planning strips `TOOL_EXPLORATION_SUFFIX` with `str.replace` (L189), which would leave an inventory appended after it in place, and Planning binds no `tool_search`.
- Modify: `app/core/config.py`: add `mcp_tool_search_prompt_inventory_max_tools: int = 60` and register it in `_non_negative_int`
- Test: create `tests/test_deferred_tool_inventory.py`

**Interfaces:**
- Consumes: the catalogs' `list_all(allowlist=...)`, `ordinary_excluded_tool_names` (`deferred_tool_binding.py:97-112`), and `tool_search_profiles.infer_tool_profile(...).purpose`.
- Produces:
  - `build_deferred_tool_inventory(*, server_tools: list[ToolDescriptor], client_tools: list[ClientToolDescriptor], excluded_tool_names: frozenset[str], max_tools: int) -> str`. It returns "" when no tool remains.
  - `BaseAgent._build_deferred_tool_inventory_suffix(self, *, user_id: str | None, device_id: str | None) -> str`.
- Format (exact):
  ```
  ── Deferred tools ──
  Load any of these with tool_search(select=["name", ...]) before calling it.
  • <call name> – <purpose, ≤ 80 chars>
  ...
  (+N more — use tool_search(query=...) to find them)
  ── End Deferred tools ──
  ```
  The last `(+N more …)` line appears only past the cap. Lines are sorted by call name.

- [ ] **Step 1: Write the failing tests.**
  - `test_inventory_lists_call_names_with_short_purpose`
  - `test_inventory_omits_excluded_raw_web_tools`
  - `test_inventory_is_empty_without_tools` (returns "")
  - `test_inventory_caps_and_counts_the_rest`: 70 tools with cap 60 give 60 lines plus "(+10 more".
  - `test_inventory_truncates_long_purpose` (Review Focus 5)
  - `test_custom_agent_inventory_respects_its_allowlist`
  - `test_planning_prompt_has_no_inventory`: Planning replaces `TOOL_EXPLORATION_SUFFIX`, so the inventory must not leak in.
- [ ] **Step 2: Run them and confirm they fail.** Run `.venv/Scripts/python.exe -m pytest tests/test_deferred_tool_inventory.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement.** Write the module, the BaseAgent method (it reads the catalogs the way `_execute_tool_search` does, with the same client-only-scope rule), the custom-agent override, the three injection sites and the setting.
- [ ] **Step 4: Run the suites.** Run `.venv/Scripts/python.exe -m pytest tests/test_deferred_tool_inventory.py tests/test_tool_search_prompt_guidance.py tests/test_planning_binds_only_what_it_executes.py tests/test_custom_agents_tools.py tests/test_search_agent_time_context.py tests/test_widget_runtime.py -q`. Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat(prompt): list deferred tool names the agent can load`.

### Task 7: Live verification and memory

- [ ] **Step 1: Run the full suite.** Run `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`. Expected: the same pass/skip counts as before this plan, plus the new tests. Investigate any failure; per `test-and-lint-baseline` memory, the suite was fully green.
- [ ] **Step 2: Live chat check (needs Thai).** With the sidecar connected:
  - Ask "tháng này tôi OT bao nhiêu giờ".
  - Expected trace: `activate_skill("take100")` is followed directly by `client__skill_take100__run_skill_command`, with no `tool_search` in between.
  - Then ask for something that needs an MCP tool by name. Expected: `tool_search(select=[...])` once, then the tool.
- [ ] **Step 3: Update memory.** Update `take100-merged-skill-and-applications-api.md`: loading is deterministic now, and the +100 boost is ranking-only. Add a line to MEMORY.md under "MCP and tools".
