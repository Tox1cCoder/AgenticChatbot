# Deterministic tool loading — design

Date: 2026-10-16. Revised after checking the implementation and reproducing the
catalog, authorization, session, and capacity edge cases.

## Problem and outcome

Search currently decides both which tools rank well and which tools can be called.
Its confidence and margin gates can withhold a known skill command after activation
has returned its exact name. Client catalog names can also disagree with runtime
names after collisions, and executing a client tool does not refresh its idle timer.

Make exact selection and skill activation independent of ranking. Search, selection,
and activation must use the same authorization and loading implementation. Report
only tools that remain loaded and can be bound in the current execution scope.

## Scope and simplification

- Add `tool_search(select=[...])` and load a skill's command after activation.
- Keep the current lexical scorer and its ranking bonuses, including the exact-name
  boost. Replacing ranking with BM25 is a separate project.
- Remove `autoload_eligible`, its gate-specific margin parameter/calculation,
  recommendation logic, obsolete response fields, and duplicate loading loops.
- Keep confidence and match reasons as ranking diagnostics; neither controls loading.
- Extract the existing 400-line search executor into mode-specific helpers. Do not
  introduce a plugin framework, provider abstraction, or second authorization engine.
- Add only two runtime modules: `tool_selection.py` for policy, snapshots, resolution
  and loading; `deferred_tool_inventory.py` for pure prompt formatting.
- No new runtime dependency or database migration.

## Authorization and identity

`None` allowlists mean unrestricted; empty allowlists mean no tools. New code filters
descriptors explicitly rather than passing an empty list to legacy catalog methods
that interpret it as unrestricted. Filter before resolution, ambiguity responses,
inventory counts, ranking, and prompt rendering. Forbidden names return `not_found`;
their descriptions and aliases must not appear in output or debug scores.

Ordinary factories merge `RAW_WEB_TOOL_NAMES` with invocation exclusions. Move the
existing constants/helper into `tool_selection.py` and re-export them from
`deferred_tool_binding.py` to preserve imports without a circular dependency.
Exclusions compare raw source names and public names, including aliased server tools
and prefixed client tools. Only the existing explicit server-side
`allow_raw_web_tools=True` diagnostic opt-in bypasses raw-provider exclusions.

Client names come from the same validated parser used to construct runtime tools;
do not invent fallback names for entries that parser rejects. Server call names must
be unique across the complete catalog, including sanitized-server collisions and
aliases that collide with another raw tool name. Preserve existing names unless
they collide. Resolve conflicts in sorted qualified-ID order, reserving all raw names
before allocating aliases and adding a checked digest suffix when necessary. Reject
duplicate qualified identities instead of selecting an arbitrary descriptor.
Newly generated aliases fit 64 characters; only the readable prefix may be truncated.

A raw name with several visible server matches returns only their visible public
aliases and loads nothing. With one visible match, resolve it; with none, return
`not_found`. Exact public names take precedence over raw-name shorthand.

Custom agents derive command authorization from selected skills resolved on the
requesting device. Extend the existing capability resolver to include each selected,
enabled live skill's `skill::<canonical-name>::run_skill_command` as an exact current
client reference. Preserve device, session, catalog version and tool-instance checks.
Use the resolved request spec for search, activation, inventory, binding and execution.
Do not authorize by a guessed skill name or broaden the strict matcher.

## Loading and lifecycle

One asynchronous `load_selection` function is the only discovery/activation caller of deferred
state loading. It receives policy, a catalog snapshot, and references selected from
that snapshot. It checks scope, availability and identity before mutation, and checks
the retained current state afterward. Pinned binding and checkpoint restore retain
their existing lifecycle roles and do not become discovery paths.

For clients, validate the ownership-checked active session against the captured
session, catalog version, device and tool instance. Repeat validation after loading.
On a reconnect or republish, report affected tools as unavailable; do not silently
substitute a new tool under an old name. Runtime dispatch still performs its final
identity check, since a disconnect can happen after any response.

Exact selection preflights distinct resolved references against each origin's
capacity. If either origin cannot hold the requested set, reject before either pool
changes. Existing older tools may be evicted by LRU. Batch state methods return only
references retained at the end of the batch. A concurrent later operation can still
evict a tool; this feature introduces no turn-wide pinning.

Refresh client last-use timestamps after successful execution using the actual bound
tool's identity. Never refresh another session's same-named tool. Idle expiry remains
measured from last use. Keep capacity at **8 per server/client scope**, rather than
doubling it without evidence; this setting is not a combined server/client cap.

## Public tool contract

- `select: list[str] | None = None` is accepted and forwarded by the exported tool,
  ordinary factory wrapper and custom-agent factory.
- Non-`None` `select` takes precedence over `query` and `top_k`. A nonempty
  `server_name` with `select` is an error; exact names define scope.
- Strip surrounding whitespace, reject blank names, and deduplicate in input order.
  Apply the select limit after deduplication. `select=[]` is an error, never inventory.
- `mcp_tool_search_max_select` defaults to **8**; zero disables explicit selection.
  Oversized or invalid requests perform no state mutation.
- Unknown/forbidden names do not prevent other valid names loading. Request and
  capacity errors reject the entire load. Missing conversation context reports an
  error and loads nothing. Missing client sessions leave client tools unavailable
  without suppressing independently available server results.

Selection output contains `mode`, `results`, `not_found`, `ambiguous`, `excluded`,
`loaded_count`, and `next_action`, with optional `error` and `guidance`. Only authorized
but excluded names appear in `excluded`. Results contain `tool_name`, `description`,
`arg_hints`, and `is_loaded`; unavailable results also contain `unavailable_reason`.
`loaded_count` counts distinct returned results that are currently loaded, including
already-loaded tools; it does not count attempted additions.

Discovery returns up to `top_k` ranked visible results. Load up to **3 eligible
results** by default, scanning past blocked results. The single discovery floor,
`mcp_tool_search_min_relevance_score`, controls which matches are returned. Remove
the second autoload floor and its configuration. A returned eligible result is not
withheld because of confidence, margin, or another score threshold.

The external-open rule is a discovery routing preference: for an `external_open`
query, do not auto-load server tools classified as `external_open` or `shell_exec`,
whether or not a client action exists. Keep those results unloaded and explain that
the action would run on the server. Deliberate exact selection may load an authorized
server tool; execution authorization and approval still apply. This is not a new
security boundary inferred from query wording.

Global and per-server inventory never mutate loaded state. They may accurately mark
previously loaded tools, but their `next_action` remains `inspect_inventory`.
Discovery and selection use `call_a_loaded_tool` when any result has `is_loaded=true`,
otherwise `refine_search`. Errors include explicit guidance rather than suggesting
synonym retries for forbidden tools.

Delete `recommended_tool`, `RecommendedTool`, `requires_refinement` and
`call_recommended_tool` from runtime schemas, responses, logging and prompts.
Activation failures append no command-loading instructions. After successful
activation, provide the canonical exposed command name and claim it is loaded only
when the common loader confirms it. Use refresh guidance for changed sessions;
suggest exact selection only when it can succeed in the current scope.

## Prompt inventory lifecycle and budget

Acquire catalog data asynchronously before building prompts. Keep the renderer pure
and synchronous. Prepare invocation-local snapshots after tool initialization and
custom-agent reference resolution; never cache a user's snapshot on a shared agent
or use persisted stale instance IDs. Pass scope and exclusions explicitly rather
than assuming a tool-execution context exists while constructing a prompt.

Render inventory only when deferred loading is enabled and `tool_search` is among
the final tools offered to the model. Planning and forced-final responses receive
none. Apply worker restrictions first. Cover BaseAgent prompts, direct SDK paths,
and both RAG prompt paths without duplicate injection.

`mcp_tool_search_prompt_inventory_max_tools` defaults to **20**; zero disables it.
The complete block is capped at **4,000 characters**, including header, footer and
overflow guidance. Sort by public call name, normalize purpose whitespace and cap
each purpose at 80 characters. Omit entries rather than truncate callable names.
Overflow counts only authorized, nonexcluded deferred entries omitted by either cap.
Do not repeat already offered tools. Empty inventory returns an empty string.

## Verification and acceptance

Use `.venv/Scripts/python.exe` for tests and Ruff. Test actual LangChain `.ainvoke`
entry points and final binding/execution, rather than relying only on fake catalogs
or state assertions. Cover empty allowlists, filtered aliases, ordinary custom-agent
raw exclusions, collisions, reconnects, republish, cross-device use, capacity mismatch,
same-batch activation/call, idle expiry, prompt scope/budgets, and approval gates.

Record a fresh baseline before implementation; do not infer a green repository from
memory. The preceding review ran 394 targeted tests successfully and Ruff on 14
related files; that is not a full-suite or live-sidecar baseline.

Record before/after loader call sites, search-executor size, removed eligibility and
recommendation symbols, representative offered-schema tokens, inventory tokens, and
search rounds to a successful call. Use the same existing tokenizer for both runs.
Added inventory has an explicit context cost; do not claim overall token savings
without measurement. Structural acceptance: one shared discovery/activation loader,
no obsolete binding gates, and no increase to loaded-tool capacity.
