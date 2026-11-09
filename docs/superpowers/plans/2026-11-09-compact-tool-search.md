# Compact tool-search responses

**Goal:** Remove redundant model-context metadata and distinguish unavailable catalogs from genuine search misses.

**Design:** Keep existing ranking, allowlists, scope, deduplication, and autoload eligibility. Emit `results` and one `next_action` for discovery; keep inventory mode and canonical server names where needed. Omit echoed queries, recommendation copies, refinement booleans, load counts, false pagination flags, and normal-output scoring diagnostics. Use the existing debug setting for scores, confidence, and match reasons. Catalog failures report fixed public error codes and recovery guidance without exception details; healthy sources remain usable.

**Scope:** Apply the same contract to the exported tool, factory-created tools, schema, prompts, and documentation. No dependencies, database changes, new search algorithm, or changes to long-term conversation memory.

## Implementation

- [x] Add regression tests for the 44-character empty-result response, loaded results, inventory, true-only pagination, debug output, and unavailable/partially unavailable catalogs. Run them before implementation to confirm expected failures.
- [x] Update the response builder and schemas, preserve diagnostics in logs/debug output, remove obsolete recommendation logic, and distinguish catalog failures. Preserve server/client discovery isolation and healthy-source fallback.
- [x] Update prompt guidance, existing contract tests, and README. Run tool-search, binding, execution, and agent integration tests plus Ruff; review the final diff.

Expected empty discovery output:

```json
{"results":[],"next_action":"refine_search"}
```

Discovery uses `call_a_loaded_tool` if a returned result was successfully loaded. Otherwise use `refine_search`, or `report_unavailable` when a catalog failed. Inventory uses `inspect_inventory` when available. Failure outputs contain `error: "catalog_unavailable"`, `unavailable_sources`, and fixed recovery guidance; partial successes retain their actionable results and report the unavailable sources.

Review regressions also cover manager-level swallowed connection failures, refresh failure preserving the old snapshot, recovery without a generation bump, disconnected devices versus connected empty catalogs, disallowed-origin lookup suppression, and exact offline integrations retaining their identity instead of fuzzy-resolving to a healthy neighbor.

Completed verification: 358 tool-search, catalog/manager, binding, execution, and agent integration tests passed. Scoped Ruff checks and `git diff --check` passed. Independent code review found no remaining important issues after the fixes.

Verification command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_tool_search_output_contract.py tests/test_unified_tool_search.py tests/test_tool_search_prompt_guidance.py tests/test_tool_search_runtime_exclusions.py tests/test_tool_search_accuracy.py tests/test_tool_search_scoring.py tests/test_tool_search_action_intent.py -q
```
