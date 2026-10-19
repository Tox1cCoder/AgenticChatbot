# Deterministic Tool Loading Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan
> task by task. Use `superpowers:subagent-driven-development` only if the user chooses
> delegated execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make exact selection and skill activation independent of ranking, with
truthful loading results and one shared authorization/loading path.

**Architecture:** `tool_selection.py` owns request policy, authorized catalog
snapshots, exact resolution and shared loading. Search keeps ranking but delegates
loading to that module; skill activation does the same. Acquire prompt inventory
asynchronously and format it in a pure `deferred_tool_inventory.py` function.

**Tech Stack:** Existing Python `.venv`, LangChain/LangGraph, Pydantic and pytest.
No new runtime dependency or database migration.

**Spec:** `docs/superpowers/specs/2026-10-16-deterministic-tool-loading-design.md`.
Read it before implementation; it defines error precedence, identity and budgets.

## Global constraints

- Line length 100. New/extracted functions at most 100 lines, complexity at most 8,
  and at most 5 positional parameters; use keyword-only policy/context arguments.
- Use `.venv/Scripts/python.exe -m pytest` and `-m ruff check`.
- `None` allowlist means unrestricted; `[]` means none. Filter before aliases,
  counts, ranking, public output or prompt rendering.
- Ordinary loading excludes raw web providers by raw and public identity. Preserve
  only the explicit existing server-side diagnostic opt-in.
- One shared discovery/activation loader; retain existing approval and dispatch checks.
- Inventory never loads or refreshes tools. A last-use refresh does not pin a turn.
- Keep loaded capacity at 8 per origin scope. Autoload defaults to 3 eligible results.
- Select limit defaults to 8. Prompt inventory defaults to 20 entries, with a fixed
  4,000-character total budget and 80-character normalized purposes.
- No unrelated refactor, ranking replacement, provider framework or dependency.
- Record Task 9's baseline before starting Task 1. Use the execution worktree.
  Other sessions may have edits: never stash or use `git add -A`; stage only files
  changed for the task. Commit passing tasks according to session instructions.

## File responsibilities and dependency order

| File | Responsibility |
| --- | --- |
| `app/ai/client_runtime_tools.py`, `client_tool_catalog.py` | Identical validated client names and raw identity |
| `app/ai/mcp_tool_catalog.py` | Unique server call names |
| `app/ai/deferred_tool_state.py` | Accurate batches, capacity and last-use lifecycle |
| `app/services/custom_agent_capability_resolver.py` | Selected skills become exact live command refs |
| `app/ai/tool_selection.py` (new) | Policy, snapshots, resolution and shared loading |
| `app/ai/tool_search_tool.py` | Thin entry points and mode orchestration |
| `app/ai/skills_tool.py` | Activate instructions, then request command loading |
| `app/ai/deferred_tool_inventory.py` (new) | Pure bounded prompt rendering |
| Agent factories and workflow middleware | Prepare request data, bind and refresh consistently |

Tasks 1–3 establish identity, lifecycle and authorization. Task 4 supplies the loader.
Tasks 5–7 integrate selection, activation and discovery. Task 8 adds bounded prompt
inventory. Task 9 verifies behavior and actual simplification.

### Task 1: Make catalog identities agree with callable runtime names

**Files:** Modify `app/ai/client_runtime_tools.py`, `app/ai/client_tool_catalog.py`,
`app/ai/mcp_tool_catalog.py`. Test `tests/test_client_tool_isolation.py`,
`tests/test_mcp_tool_catalog_refresh.py`, `tests/test_deferred_tool_binding.py`.

**Interfaces:** Add
`exposed_names_by_qualified_id(catalog: dict[str, Any]) -> dict[str, str]` from
`_parse_tool_specs`. Add `source_tool_name: str = ""` to `ClientToolDescriptor`,
populated from the validated runtime entry's raw name. Add server lookup
`get_by_call_name(name: str) -> ToolDescriptor | None`; it resolves exact public
names only. Task 4 resolves raw shorthand and visible aliases over filtered snapshots,
never an unfiltered `aliases_for` response.

- [ ] **Step 1: Add regression tests.** Reuse the MCP/skill collision fixture in
  `tests/test_skills_tool.py::test_activation_appends_exact_model_callable_skill_tool_name`.
  Assert catalog/runtime name sets agree, the skill is accessible by the `_2` name,
  rejected parser entries are absent from both, and the raw source name survives.
  Add server fixtures for `a-b/search` versus `a_b/search`, alias-versus-raw-name
  collision, input-order reversal, and duplicate qualified IDs. Valid public names
  must be unique and deterministic; duplicate identities produce a catalog error.

  ```python
  def test_client_name_map_uses_runtime_collision_suffix():
      from app.ai.client_runtime_tools import exposed_names_by_qualified_id

      catalog = {"tools": [
          {"name": "run_skill_command", "origin": "mcp",
           "server_name": "skill_take100", "qualified_id": "mcp::collision"},
          {"name": "run_skill_command", "origin": "skill",
           "server_name": "skill_take100",
           "qualified_id": "skill::take100::run_skill_command"},
      ]}
      names = exposed_names_by_qualified_id(catalog)
      assert names["skill::take100::run_skill_command"] == (
          "client__skill_take100__run_skill_command_2"
      )
      assert len(set(names.values())) == 2
  ```

- [ ] **Step 2: Run the regression tests and record their failures.**
  `.venv/Scripts/python.exe -m pytest tests/test_client_tool_isolation.py tests/test_mcp_tool_catalog_refresh.py -q`
- [ ] **Step 3: Implement shared client naming and unique server aliases.** Skip
  catalog entries missing from the validated name map; never guess a fallback name.
  Reuse the map in `get_exposed_client_tool_name`. Reserve all server raw names before
  allocating aliases; process qualified IDs in sorted order. Keep noncolliding names.
  On an alias conflict append `__` plus a SHA-256 qualified-ID prefix, growing the
  prefix until unique. Newly generated aliases must fit 64 characters. Truncate only
  their readable prefix, never the identity suffix; use a digest form when needed.
  Check uniqueness rather than assuming no collision.
  Reject duplicate qualified IDs. Never use randomized `hash()` or input order.
- [ ] **Step 4: Verify each alias binds the correct runtime tool.** Run the suites
  above plus `tests/test_skills_tool.py`, `tests/test_custom_agent_client_tool_resync.py`,
  `tests/test_client_tool_scope.py`, `tests/test_deferred_tool_binding.py`; Ruff changed
  files. Suggested commit: `fix(tools): align catalog and runtime identities`.

### Task 2: Make state results and client last-use tracking truthful

**Files:** Modify `app/ai/deferred_tool_state.py`, `app/ai/tool_execution.py`.
Test `tests/test_deferred_tool_state_lifecycle.py`, `tests/test_tool_execution_recovery.py`.

**Interfaces:** Add `mark_client_tool_used` with keyword-only fields
`conversation_id`, `agent_key`, `tool_name`, `device_id`, `session_id`, `user_id`,
`catalog_version`, `tool_instance_id`; return `bool`. Extend
`_mark_tool_used_if_deferred(tool_name, *, tool_metadata=None)` to pass the actual
executed tool's metadata. Add `DeferredToolState.load_batch` taking keyword-only
`conversation_id`, `agent_key`, `server_refs`, `client_refs`, `device_id`, `session_id`,
`user_id`, `max_tools`; return the retained server/client reference lists.

- [ ] **Step 1: Add lifecycle and batch regression tests.** Use at minute 29 must
  survive minute 31 with TTL 30, then expire after 30 idle minutes. With capacity two,
  touching A before loading C evicts B. Wrong user/device/session/instance cannot
  touch a same-named entry. Existing autoload methods loading A/B/C at capacity two
  return only B/C. `load_batch` with three refs at capacity two raises `ValueError`
  before either pool changes; a mixed batch fitting each origin succeeds.

  ```python
  def test_autoload_reports_only_retained_references():
      from app.ai.deferred_tool_state import DeferredToolState
      from app.ai.mcp_tool_catalog import ToolReference

      state = DeferredToolState()
      refs = [ToolReference(name, "files") for name in ("a", "b", "c")]
      returned = state.autoload("conv", "chat", refs, max_tools=2)
      retained = state.get_loaded("conv", "chat")
      assert {r.tool_name for r in returned} == {r.tool_name for r in retained}
      assert len(returned) == 2
  ```

- [ ] **Step 2: Run new tests and confirm intended failures.**
  `.venv/Scripts/python.exe -m pytest tests/test_deferred_tool_state_lifecycle.py -q`
- [ ] **Step 3: Implement under the existing state lock.** Deduplicate refs, validate
  both origin capacities before mutation, and reuse scope add/cleanup operations.
  Do not acquire the non-reentrant lock twice. Existing autoload methods filter their
  returns against final retained state. Client use refreshes entry and scope only when
  exact bound identity agrees. Pass `getattr(tool, "metadata", None)` after successful
  execution; failed/denied calls and old sessions do not refresh a new instance.
- [ ] **Step 4: Verify lifecycle, isolation, recovery and approval suites.** Include
  `tests/test_client_tool_isolation.py` and `tests/test_hitl_client_and_deferred.py`;
  Ruff changed files. Suggested commit:
  `fix(deferred-tools): report retained loads and track client use`.

### Task 3: Authorize selected skill commands through exact live references

**Files:** Modify `app/services/custom_agent_capability_resolver.py`,
`app/ai/agents/custom_agent.py`, and `app/ai/custom_agent_runtime.py` only where needed
for resolved refs. Test `tests/test_custom_agent_capability_resolver.py`,
`tests/test_custom_agent_client_tool_resync.py`, `tests/test_custom_agents_tools.py`.

**Interfaces:** Keep `resolve_custom_agent_capabilities` and `CapabilityResolution`.
Extend `resolved_client_tool_refs` to include commands for selected, enabled,
device-local live skills. `CustomAgent._request_spec` remains the request policy
source; make its resolved spec available to subsequent factories without mutating
`self._spec` or caching per-user policy on the shared agent.

- [ ] **Step 1: Add tests for a selected skill with no selected client tools.** The
  resolved refs include its current command instance. Final `_get_tools_for_binding`
  admits that command once loaded and rejects unrelated skill commands. Repeat after
  session/catalog/instance rotation, with a missing command, disabled skill, name
  collision and second device. Repeated selections do not duplicate refs. Existing
  management capability-status tests must still pass.
- [ ] **Step 2: Run the three suites and confirm the new assertions fail.**
- [ ] **Step 3: Resolve canonical names before deriving command IDs.** Match each
  selected ref to a live skill using `skill_refs_match`. Derive
  `skill::<live.name>::run_skill_command`; require the matching live tool's client-skill
  origin and requesting device. Copy exact live identity into a ref with `type="client"`
  and deduplicate by full identity. Missing commands are unavailable capabilities.
  Keep `client_ref_matches_metadata` strict. Do not merely add wildcard or qualified
  strings to a search allowlist. Use a small resolver helper for this mapping.
- [ ] **Step 4: Verify all three suites plus `tests/test_skill_device_isolation.py`;
  Ruff changed files.** Suggested commit:
  `fix(custom-agents): resolve selected skill command instances`.

### Task 4: Introduce one authorized snapshot and loading implementation

**Files:** Create `app/ai/tool_selection.py`. Modify `app/ai/deferred_tool_binding.py`
to re-export moved exclusions. Create `tests/test_tool_selection.py`.

**Interfaces:** Define these types in `tool_selection.py`:

```python
@dataclass(frozen=True)
class ToolLoadingPolicy:
    server_allowlist: tuple[str, ...] | None = None
    client_allowlist: tuple[str, ...] | None = None
    excluded_tool_names: frozenset[str] = frozenset()
    allow_raw_web_tools: bool = False

@dataclass(frozen=True)
class ToolCatalogSnapshot:
    server_tools: tuple[ToolDescriptor, ...]
    client_tools: tuple[ClientToolDescriptor, ...]
    server_generation: int
    client_session_id: str | None
    client_catalog_version: int | None

@dataclass
class Selection:
    server_refs: list[ToolReference]
    client_refs: list[ClientToolReference]
    results: list[dict[str, Any]]
    not_found: list[str]
    ambiguous: dict[str, list[str]]
    excluded: list[str]
```

`async acquire_tool_catalog_snapshot(*, ctx: ToolContext,
policy: ToolLoadingPolicy) -> ToolCatalogSnapshot` acquires and copies current data.
`resolve_selection(names: list[str], *, snapshot: ToolCatalogSnapshot,
policy: ToolLoadingPolicy) -> Selection` is pure.
`async load_selection(selection: Selection, *, snapshot: ToolCatalogSnapshot,
ctx: ToolContext, policy: ToolLoadingPolicy) -> set[str]` performs the shared load,
updates result `is_loaded`/`unavailable_reason`, and returns retained public names.
Callers await it; runtime counterpart verification may need asynchronous MCP access.

- [ ] **Step 1: Add resolution/loading tests with real descriptors and state.**
  Cover `None` versus empty allowlists on both origins; forbidden aliases/descriptions
  absent; two visible raw matches, one, and none; raw/public/client-source exclusions;
  diagnostic opt-in; low-ranked exact selection; duplicates; missing conversation/device;
  mixed-origin capacity preflight; stale session, catalog republish, wrong instance,
  and a session change during loading.

  ```python
  def test_empty_server_allowlist_cannot_resolve_a_tool():
      from app.ai.mcp_tool_catalog import ToolDescriptor
      from app.ai.tool_selection import (
          ToolCatalogSnapshot, ToolLoadingPolicy, resolve_selection,
      )

      tool = ToolDescriptor("read_file", "files", "Read a file", [], [], "fp")
      snapshot = ToolCatalogSnapshot((tool,), (), 0, None, None)
      selection = resolve_selection(
          ["read_file"], snapshot=snapshot,
          policy=ToolLoadingPolicy(server_allowlist=()),
      )
      assert selection.not_found == ["read_file"]
      assert selection.results == []
      assert selection.server_refs == []
  ```

- [ ] **Step 2: Run the new suite and record import/behavior failures.**
  `.venv/Scripts/python.exe -m pytest tests/test_tool_selection.py -q`
- [ ] **Step 3: Implement policy and resolution.** Move `RAW_WEB_TOOL_NAMES` and
  `ordinary_excluded_tool_names` here. This module must not import
  `deferred_tool_binding` or `tool_search_tool`. Obtain catalog lists without passing
  empty allowlists; explicitly filter `None`/empty/nonempty policy and copy descriptors
  into a snapshot. Keep authorized excluded descriptors internally for select's
  `excluded` response; discovery/inventory remove them before ranking/counting. Apply
  policy again in the pure resolver so manually supplied snapshots cannot bypass it.
  Resolve public names first, then visible raw server shorthand. Respect client-only
  scope even without a device. Catalog failure yields empty affected-origin data and
  internal diagnostics; never fall back to another device or a broader policy.
- [ ] **Step 4: Implement shared loading.** Check context and both capacities before
  mutation. Validate current server generation and ownership-checked client session,
  catalog and instance against the snapshot; mark stale refs unavailable without
  substitution. Call `load_batch` once, inspect final retained state and recheck identity.
  A change after loading makes affected results unloaded. Confirm refs have live runtime
  counterparts through existing binding helpers; a state entry alone is not proof its
  schema can be offered. Keep existing execution approval checks.
- [ ] **Step 5: Verify selection, state, deferred-binding and runtime-exclusion suites;
  Ruff changed files.** Suggested commit:
  `refactor(tools): centralize authorized deferred loading`.

### Task 5: Wire exact selection through every public entry point

**Files:** Modify `app/ai/tool_search_tool.py`, `app/core/config.py`, `.env.example`.
Extend `tests/test_tool_selection.py`, `tests/test_config_validation.py`,
`tests/test_custom_agents_tools.py`, `tests/test_tool_search_runtime_exclusions.py`.

**Interfaces:** Add `select: list[str] | None = None` to `ToolSearchInput`, exported
`tool_search`, factory `tool_search_impl`, and `_execute_tool_search`, forwarding it
through every call. Add `mcp_tool_search_max_select=8` with nonnegative validation;
zero disables explicit selection. Add the environment example entry.

- [ ] **Step 1: Test actual `.ainvoke` calls.** Exercise exported, ordinary factory,
  and custom-agent factory tools. Assert select mode reaches the loader; unknown names
  return `not_found`; denied names and aliases remain hidden; ordinary custom-agent
  raw-provider selection is refused without callers supplying exclusions. Validate
  `[]`, blanks, duplicates, limit after deduplication, select with server_name,
  query/top-k precedence, zero limit, and missing context.

  ```python
  import pytest

  @pytest.mark.asyncio
  async def test_public_tool_rejects_empty_select_instead_of_listing_inventory():
      import json
      from app.ai.tool_search_tool import tool_search

      result = json.loads(await tool_search.ainvoke({"select": []}))
      assert result["mode"] == "select"
      assert result["error"] == "select must contain at least one tool name"
      assert result["loaded_count"] == 0
  ```

- [ ] **Step 2: Run the four suites and confirm new public-entry-point tests fail.**
- [ ] **Step 3: Route selection before query/inventory logic.** Validate names before
  catalog I/O; preserve input order when deduplicating. Empty select and blank names
  are errors; reject nonempty server_name with select. Ignore query/top-k for valid
  selection. Acquire the snapshot, resolve names and await the shared loader. Ordinary
  factories always supply ordinary exclusions; custom factories use the current
  resolved request spec. Keep diagnostic opt-in a server-side keyword, never a model
  input. Do not append another loading loop.
- [ ] **Step 4: Match response fields and error precedence in the spec.** Forbidden
  names are `not_found` before exclusion/ambiguity classification. Request/capacity
  errors change neither pool. Partial unknown names can coexist with loaded authorized
  results. `loaded_count` counts distinct currently loaded returned results.
- [ ] **Step 5: Verify public schemas, final binding, all four suites and Ruff.**
  Suggested commit: `feat(tool-search): select deferred tools by exact name`.

### Task 6: Load activated commands and refresh the actual execution map

**Files:** Modify `app/ai/skills_tool.py`, `app/ai/agents/base_agent.py`,
`app/ai/agents/custom_agent.py`, `app/ai/tool_execution.py` and affected execution
integration in `app/ai/workflow/middleware.py`.
Test `tests/test_skills_tool.py`, `tests/test_tool_execution_recovery.py`,
`tests/test_specialist_middleware.py`, `tests/test_hitl_client_and_deferred.py`.

**Interfaces:** Add optional keyword-only `loading_policy: ToolLoadingPolicy | None`
to `create_activate_skill_tool`. Base callers construct ordinary policy; custom callers
pass policy from the same resolved request spec used by search and binding.
Add `activate_skill` to `TOOL_LOADING_TOOLS`.

- [ ] **Step 1: Add activation-to-call tests.** Exercise a base agent and a skill-only
  custom agent, including the collision-suffixed name. After activation, assert the
  command appears in final binding and executes next without search. Cover same-batch
  activation/command, reconnect during dispatch, republish during loading, missing
  conversation, missing command, failed activation and an unrelated forbidden command.
  A mutating command must still encounter approval.
- [ ] **Step 2: Run all four suites and confirm new behavior fails.**
- [ ] **Step 3: Resolve/load the canonical command after successful dispatch.** Capture
  the activation session/version before awaiting dispatch; do not derive an old command
  name from a new session afterward. Use the shared snapshot/resolver/loader and supplied
  policy, awaiting the shared loader. Failed dispatch appends no success paragraph.
  If loaded, say
  `The command tool <name> is loaded for this session.` and give argv guidance. For a
  changed session/version, explain tools need refreshing; do not suggest selecting an
  obsolete name. Suggest `tool_search(select=[...])` only for a current available command
  when selection can succeed. Missing/forbidden commands get unavailable guidance.
- [ ] **Step 4: Refresh through current agent binding.** Preserve invocation exclusions,
  scope and strict custom policy. Replace stale external map entries instead of only
  appending unseen names; retain required invocation-local internal tools. Middleware
  must offer live schemas on the next model call. Do not create an activation-specific
  refresh implementation alongside the existing one.
- [ ] **Step 5: Run the four suites plus `tests/test_skill_device_isolation.py`,
  `tests/test_skill_installation_chat_integration.py`,
  `tests/test_custom_agent_client_tool_resync.py`,
  `tests/test_client_runtime_result_binding.py`; Ruff changed files.** Suggested commit:
  `feat(skills): load authorized commands during activation`.

### Task 7: Simplify discovery and delete obsolete binding gates

**Files:** Modify `app/ai/tool_search_tool.py`, `app/ai/tool_search_scoring.py`,
`app/ai/mcp_tool_catalog.py`, `app/ai/client_tool_catalog.py`, `app/ai/prompts.py`,
`app/core/config.py`, `.env.example`, `README.md`.
Test `tests/test_unified_tool_search.py`, `tests/test_tool_search_scoring.py`,
`tests/test_tool_search_action_intent.py`, `tests/test_tool_search_accuracy.py`,
`tests/test_tool_search_runtime_exclusions.py`, `tests/test_tool_search_prompt_guidance.py`.

**Interfaces:** Make `_execute_tool_search` keyword-only orchestration of normalization,
snapshot acquisition and mode handlers. Extract inventory, discovery and response
helpers within global size limits. Selection delegates to Task 4. Delete
`RecommendedTool`, `autoload_eligible`, `high_confidence_margin`, `_build_recommendation`,
`recommended_tool`, `requires_refinement`, and `call_recommended_tool` from active
contracts. Retain confidence calculations only where ranking diagnostics use them.

- [ ] **Step 1: Add discovery behavior tests; update obsolete contract tests.** Three
  medium-confidence eligible results load. Blocked server openers do not consume load
  slots; retain blocking with and without client actions for external-open discovery,
  while deliberate authorized selection succeeds. Global, per-server and inferred
  named-integration inventory make zero mutations. No-match searches produce no
  fabricated scores or fallback loads. Previously loaded status is truthful. Capacity
  below autoload count is respected. Ranking tests assert ordering/relevance/confidence
  instead of deleted eligibility fields; keep exact skill-name ranking coverage.
- [ ] **Step 2: Run the six suites and confirm the new assertions fail.**
- [ ] **Step 3: Unify discovery loading.** Rank visible nonexcluded candidates. Pass
  `settings.mcp_tool_search_min_relevance_score` explicitly to `rank_tool_candidates`;
  replace its `value or 1.0` floor expression with an explicit `None` check so zero works.
  Remove `mcp_tool_search_autoload_min_relevance_score` from config, examples and active
  docs; existing deployment keys become obsolete, ignored keys rather than another
  threshold. Change autoload default/example to 3; keep loaded capacity/example at 8.
  Collect the first eligible returned candidates, scanning past discovery-blocked
  results, capped by the autoload setting, returned count and each origin's capacity.
  Resolve their public names through Task 4 and await `load_selection` once.
- [ ] **Step 4: Delete dead plumbing and test-only compatibility.** Remove the old
  state-loading loops and gate-only eligibility/margin calculations. Preserve numeric
  ranking bonuses. Replace legacy plain-descriptor/tuple merge fallbacks with the
  actual `ToolSearchScore` contract and migrate fake catalogs in the same task; never
  fabricate `1000 - idx` scores. Preserve inventory's separate unscored contract.
  Update output models for select mode, error/guidance and next actions; remove
  recommendation-specific logging. Do not leave unused response models.
- [ ] **Step 5: Update prompts and docs.** Use this guidance:
  `Results with is_loaded: true are available in this session. Pick the one that fits
  and call it. Load a known tool with tool_search(select=[...]). Refine the query when
  no loaded result fits.` Explain browsing without loading, the removed environment
  key and per-origin capacity. Discovery-blocked results explain server-host execution;
  they do not claim a tool is forbidden by security policy.
- [ ] **Step 6: Verify all six suites plus selection, binding, specialist and approval
  suites; Ruff changed files.** Suggested commit:
  `refactor(tool-search): share loading and remove recommendation gates`.

### Task 8: Acquire prompt inventory asynchronously and render within budget

**Files:** Create `app/ai/deferred_tool_inventory.py`, `tests/test_deferred_tool_inventory.py`.
Modify `app/ai/workflow/specialists.py`, `app/ai/workflow/middleware.py`,
`app/ai/agents/base_agent.py`, `app/ai/agents/custom_agent.py`,
`app/ai/agents/rag_agent.py`, `app/ai/agents/chat_agent.py`,
`app/ai/agents/search_agent.py`, `app/ai/agents/canvas_agent.py`,
`app/ai/agents/image_generator_agent.py`, `app/ai/agents/planning_agent.py`,
`app/core/config.py`, `.env.example`, `README.md` at affected preparation/call sites.

**Interfaces:** Pure
`build_deferred_tool_inventory(*, snapshot: ToolCatalogSnapshot,
policy: ToolLoadingPolicy, offered_names: frozenset[str], max_tools: int,
max_chars: int = 4000) -> str`.
Add `mcp_tool_search_prompt_inventory_max_tools=20`, nonnegative validation and example.
Use `deferred_tool_inventory: str = ""` as an explicit prompt-factory input or
invocation-local request extra. Do not acquire catalogs in a synchronous suffix method.

- [ ] **Step 1: Add formatting tests.** Cover empty input, zero cap, 70 visible tools,
  long purposes/newlines, a name too long for the remaining budget, stable sorting,
  no truncated names, accurate overflow, already offered tools, and total length at
  most 4,000. Both limits include header/footer/overflow text. Add integration tests
  for empty custom allowlists, reconnect/rebased refs, raw exclusions, concurrent users
  sharing one agent, client-only scope without a device, worker restrictions, deferred
  mode off, forced-final mode, Planning, direct SDK and both RAG paths. Eligible prompts
  contain the block exactly once.

  ```python
  def test_inventory_is_empty_when_disabled():
      from app.ai.deferred_tool_inventory import build_deferred_tool_inventory
      from app.ai.tool_selection import ToolCatalogSnapshot, ToolLoadingPolicy

      result = build_deferred_tool_inventory(
          snapshot=ToolCatalogSnapshot((), (), 0, None, None),
          policy=ToolLoadingPolicy(), offered_names=frozenset(), max_tools=0,
      )
      assert result == ""
  ```

- [ ] **Step 2: Run the inventory and prompt integration tests; confirm failures.**
- [ ] **Step 3: Implement pure formatting.** Reapply policy; remove excluded and offered
  names, sort, normalize purpose whitespace and cap purposes at 80 characters. Reserve
  complete header/footer/overflow space before adding entries. Omit entries instead of
  truncating names. Return empty when no entry fits. Format `Deferred tools`, a select
  instruction, complete `name — purpose` rows, optional
  `(+N more; use tool_search(query=...))`, and `End deferred tools`. Counts never include
  forbidden entries.
- [ ] **Step 4: Prepare data before synchronous formatting.** Initialize tools and
  resolve the custom request spec first. Acquire the snapshot with explicit
  conversation/user/device/scope/policy. In specialist construction, finalize offered
  tools, disabled-tool handling and worker restrictions before inventory preparation
  and system-prompt construction. Pass the rendered string through request-local data
  to prompt factories; never cache per-user inventory on the agent. Catalog failure
  yields an empty block with internal diagnostics rather than a failed answer.
  The live binding middleware must suppress/remove inventory when current offered
  tools no longer contain `tool_search`; do not retain a stale initial block after
  restrictions change. Direct SDK and RAG preparation await acquisition before their
  synchronous formatters. Never use `asyncio.run` inside a running event loop.
- [ ] **Step 5: Verify inventory, prompt guidance, specialist, custom-agent, planning,
  search time-context and widget suites; Ruff changed files.** Suggested commit:
  `feat(prompts): add bounded request-scoped deferred tool inventory`.

### Task 9: Verify behavior, measure complexity and document limits

**Files:** Verification only, plus README corrections for inaccurate claims. Use a
local report or temporary output for measurements; no new benchmark framework or
unrequested external agent-memory edits.

- [ ] **Step 1: Record fresh baselines before Task 1.** Record branch, dirty files,
  interpreter version and full-suite outcome. The preceding review's 394 targeted
  passes and clean Ruff are useful evidence, not a full-suite baseline. Record schema
  tokens, inventory tokens and search rounds for the scenarios below with fixed
  catalogs/prompts and the same existing tokenizer. Do not invent before measurements
  after implementation.
- [ ] **Step 2: Run final automated checks.**

  ```powershell
  .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
  .venv/Scripts/python.exe -m ruff check app/ai app/services/custom_agent_capability_resolver.py app/core/config.py tests
  rg -n 'autoload_eligible|high_confidence_margin|recommended_tool|requires_refinement|call_recommended_tool|mcp_tool_search_autoload_min_relevance_score' app tests .env.example README.md
  rg -n '\.autoload\(|\.autoload_client_tools\(|\.load_batch\(' app/ai
  ```

  Expected: no new test/lint failures relative to baseline. No obsolete active
  runtime/test consumers; literal assertions that obsolete response keys are absent
  and a README migration note for the removed environment key are allowed. Loader
  search shows one discovery/activation caller of state loading;
  account for checkpoint restore/state internals separately. Fix extra callers rather
  than renaming them to evade the check.
- [ ] **Step 3: Verify representative traces.** With the sidecar connected, request
  `tháng này tôi OT bao nhiêu giờ`: activation must proceed directly to the canonical
  command. Repeat with a skill-only custom agent, collision-suffixed command, exact
  server selection, inventory browsing, reconnect between activation/call, and a
  mutating command requiring approval. If the live environment is unavailable, record
  unrun checks explicitly; automated tests are not live evidence.
- [ ] **Step 4: Compare actual costs.** Record changed runtime LOC, executor size,
  loader call sites, removed symbols, offered schemas/tokens, inventory tokens and
  search rounds. Require one shared loader, no obsolete binding gates, mode helpers
  within size limits, capacity still 8, and inventory within budget. State measured
  context increases as costs. Do not claim general codebase debloating or reduced
  total tokens without evidence.
- [ ] **Step 5: Review the final diff and report.** Check authorization, approvals,
  migration guidance and unrelated edits. Report behavior, tests, measured tradeoffs
  and unrun live checks. Update external agent memory only if explicitly requested.

## Review coverage

| Verified review gap | Required task/test coverage |
| --- | --- |
| Public wrappers omit select | Task 5 real `.ainvoke` tests for every factory |
| Empty allowlists and alias leaks | Tasks 4, 5 and 8 filter before output/counts |
| Skill-only custom binding fails | Task 3 exact refs; Task 6 binding/execution |
| Old-session load reports success | Tasks 2, 4 and 6 identity/reconnect tests |
| Ordinary custom raw-provider bypass | Tasks 4 and 5 factory defaults/exclusions |
| Sanitized and raw/alias collisions | Task 1 unique deterministic identities |
| Batch reports evicted tools; caps conflict | Tasks 2 and 4 preflight/final-state tests |
| Synchronous or stale inventory | Task 8 async request-local data and restrictions |
| Conflicting discovery guarantees | Task 7 one return floor and explicit routing preference |
| Stale models/config/logging | Tasks 5 and 7 complete contract/environment migration |
| Complexity added without deletion | Tasks 7 and 9 removed gates, one loader, measured costs |
