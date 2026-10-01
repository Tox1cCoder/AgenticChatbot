# Refactor remediation implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement the independent workstreams with regression tests and review.

**Goal:** Repair the confirmed generation defects and remaining code/configuration/test issues from the 1 October review.

**Architecture:** Keep the existing generation control, turn coordinator, HITL repository, and graph budget mechanisms. Share their established behavior across first turns, approval resumes, and continued epochs; add no parallel lifecycle authority.

**Tech stack:** Python 3.11+, FastAPI, SQLAlchemy/PostgreSQL, LangGraph, pytest, Ruff.

## Constraints

- Worktree: `.worktrees/refactor-remediation`, branch `fix/refactor-remediation`.
- Interpreter: absolute original `.venv/Scripts/python.exe`.
- Never run database-backed tests against the app database. Use `TEST_DATABASE_URL` for `chatbot_test`.
- Preserve existing preview changes copied into the worktree. Do not commit or stage unrelated existing work.
- Never print plaintext skill passwords or write them into tracked fixtures.
- File ownership below separates parallel edits. Notify the coordinator before touching another workstream's files.

## Task 1 â€” Complete package deletion, templates, and checkout reproducibility (coordinator)

Files: response/exception `__init__.py`, `.env.example`, `.env.client.example`, `.gitattributes`, evaluation fixture files, `ai_sdk_v6.py` (single TimeoutError edit).

- [x] Confirm the existing import failure with the continuation test; remove only the matching deleted imports and exports.
- [x] Update templates: optional database-persisted server keys, loopback API bind, remove fields absent from Settings except REDIS_PASSWORD/LANGSMITH_PROJECT, add client trust/upload limits.
- [x] Pin evaluation bytes to LF using Git attributes; verify fixture hashes against a checkout using Windows conversion.
- [x] Replace asyncio.TimeoutError with TimeoutError without changing preview behavior.
- [x] Run package/bundle/preview/fixture tests and Ruff.

## Task 2 â€” Generation/approval lifecycle (generation worker)

Files: `message_service.py`, `generation_control_service.py`, HITL/generation repositories and schemas when needed, container lifecycle wiring, generation/HITL tests.

- [x] Promote and improve the review probes into behavior regression tests. Observe failures for first-yield disconnect, malformed research accounting, continued-epoch approval, concurrent approval resume, stopped approval resume, missed Stop signal, old producer attribution, and discarded partials.
- [x] Put all post-lease work under cleanup, settle accounting errors before publishing, and stamp current producer in the continuation transition.
- [x] Share approval persistence and durable Stop checking with continued epochs; persist partials before Stop/disconnect settlement.
- [x] Acquire `_hold_turn` before interrupt claim; return retriable conflict without consuming the approval. Fail closed on actual lifecycle transition errors, while allowing genuinely legacy records with no generation.
- [x] Register/watch approval resumes and deterministically close nested generators; invalidate stopped pending approvals with user/conversation/thread scope.
- [x] Check Stop/approval races, error and cancellation paths, second-turn acceptance, and persisted-message continuity. Keep newly extracted helpers at complexity <=8.
- [x] Run focused generation/HITL tests; coordinate PostgreSQL verification with the coordinator.

## Task 3 â€” Planning metadata and worker context (planning worker)

Files: planning workflow/agent/specialist modules, graph metadata helpers, `config.py` (obsolete consecutive-error setting only), related tests. Do not edit message_service or contract prose.

- [x] Add failing tests demonstrating that actual model calls/budget exits reach persisted/live planning metadata, and that supplied parent context reaches worker model inputs.
- [x] Derive planning count/reached state from existing accountant, consistently across resume/Continue. Put parent context in bounded untrusted task input.
- [x] Remove unused `tool_execution_consecutive_errors_limit` and validator references; use existing execution limits.
- [x] Run planning/worker/context/metadata and configuration tests. Tell the coordinator exact contract changes.

## Task 4 â€” Upload staging, reconnect test, and local skill/test independence (support worker)

Files: document upload/staging routes/services (excluding message_service), `test_tool_execution_recovery.py`, sanitized Take100 module and its tests, local ignored skill files and secret-binding support when needed. Do not edit `.env*`, `.gitattributes`, or `config.py`.

- [x] Reproduce event-loop disk work and add a test proving staging runs off the loop while keeping size/error cleanup behavior.
- [x] Reproduce the reconnect test alone. Keep its functional/deadline contract and remove cold import cost from the measured operation; avoid simply increasing a bound.
- [x] Make tracked Take100 tests import a sanitized tracked client module rather than requiring an ignored personal skill file.
- [x] Migrate plaintext skill passwords to existing skill-secret bindings without displaying them. Remove credential values from skill prose; document service rotation separately if remote access is unavailable.
- [x] Run upload, reconnect-in-isolation, Take100, and skill-binding tests.

## Task 5 â€” Integration and final review (coordinator)

- [x] Review each workstream diff and its red/green evidence. Resolve important findings before integration.
- [x] Run the full configured suite against the dedicated database and Ruff; verify preserved preview and bundle behavior.
- [x] Compare original workspace file hashes against the captured baseline before applying any edits; merge concurrent changes rather than overwriting them.
- [x] Integrate explicit reviewed files into the original checkout, preserving existing staged deletions and unrelated staging.
- [x] Re-run relevant checks in the original workspace and update the remediation status. Report exact test results and any genuinely external unfinished action.


## Verified result — 2026-10-01

- Full suite in the original checkout: **6502 passed, 0 failed, 15 skipped, 1 deselected**; 2 non-failing experimental LangGraph streaming warnings.
- Dedicated PostgreSQL lifecycle/approval gate: 194 passed; 30 new regressions rechecked after the final helper extraction.
- Planning/context/configuration gate: 358 passed. Upload/client/reconnect gate: 133 passed. Fresh integration contract checks: 78 passed.
- Final multi-approval cleanup/validation gate: 100 passed; independent scoped reviewer checks: 32 passed. Completing, pausing, failing, or disconnecting one resumed turn preserves other pending turns' custom-agent guards.
- Final routing-vocabulary/lifecycle gate after naming cleanup: 101 passed.
- Ruff and Git whitespace checks pass. Existing preview changes and staged deletions are preserved. The matching package-export repairs are staged with those deletions; the two new shared integration files are staged for the bundle's tracked-file inventory. Other fixes remain in the working tree. No commits or pushes were made.
- Tested using the existing Python 3.13.7 environment; Python 3.11 itself was not executed. Local Qdrant client now matches the existing 1.18.0 repository pin.
- Both ignored local skill files declare encrypted secret bindings; four values were persisted and verified without printing them. Direct CLI binding lookup and TLS verification pass. No credential migration network requests were made.

The renewed request supersedes the earlier Task 6 deferral. The subsequent
[final security remediation](2026-10-01-final-security-remediation.md) completes the
MCP restrictions and rotates the Take100 password through its supported account control.
The attendance rotation still requires its current site URL; its saved tunnel is offline.
