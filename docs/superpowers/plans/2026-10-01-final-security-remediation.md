# Final security remediation implementation plan

> **For agentic workers:** Use the dispatching-parallel-agents and requesting-code-review skills for independent MCP security and account-control work.

**Goal:** Finish the previously deferred MCP restrictions and the exposed-password account actions.

**Architecture:** Extend existing FastAPI authentication with an explicit administrator
allowlist. Guard the common API server-add path so URL parsing cannot bypass the stdio
opt-in. Rotate through supported account controls with encrypted replacement custody.

**Tech stack:** FastAPI, Pydantic settings, pytest, requests, existing local SkillSecretStore.

## Constraints

- Reuse `.worktrees/refactor-remediation`; preserve current original edits and staging.
- No live application database writes or migrations. Dedicated PostgreSQL test database only.
- Never log credential values or authenticated response bodies. No guessed remote write URLs.
- No commits or pushes of unrelated preview work.

## Task 1 — Server MCP access and stdio enforcement

- [x] Add failing HTTP tests for non-admin mutation/direct execute, admin HTTP operations, default stdio refusal through both add routes, explicit opt-in, and redacted reads.
- [x] Implement `ADMIN_USER_IDS`, `MCP_ALLOW_API_STDIO`, administrator dependency, and transport enforcement using existing exception/DI patterns.
- [x] Verify bundled manager initialization remains available, supported settings parsing, and URL-command refusal before startup/persistence. Bundled manager code is unchanged; tests replace external transport and do not launch a real stdio process.
- [x] Update example environment and relevant MCP documentation. Run focused tests and Ruff.

## Task 2 — Account rotation

- [x] Discover Take100's supported self-account password control and verify the account identity without exposing credentials.
- [x] Persist encrypted recovery custody before the Take100 rotation, verify a fresh replacement-password login, update the actual encrypted binding, and remove the obsolete session. Subsequent verified CLI reads create a fresh valid cache.
- [x] Restore the already-pinned local parser dependencies and configure the ignored Windows launcher to use native certificate chain validation while retaining TLS verification. Actual `check-day` and `list` reads succeed.
- [x] Report the unreachable attendance site and request its current URL while continuing Task 1.
- [ ] Rotate the attendance password after receiving the current site URL. The saved tunnel returns HTTP 404 with `ERR_NGROK_3200`; its password binding remains unchanged.

## Task 3 — Integration and verification

- [x] Independently review MCP guards and bypass cases, including direct tool testing and enabling existing stdio servers.
- [x] Verify local admin identity and set its allowlist in the ignored installation environment. Existing signed tokens, an unexpired current refresh token, and the matching active canonical user were checked in a read-only DB transaction; the API was not started.
- [x] Merge explicit reviewed files against captured hashes, preserving all unrelated edits.
- [x] Run the full configured suite on the original checkout, Ruff, and relevant account verification. Record exact outcomes and the external attendance URL blocker.

## Verified result — 2026-10-01

- Original-checkout full suite on the dedicated `chatbot_test` database: **6548 passed, 0 failed, 15 skipped, 1 deselected** in 341.10 seconds. Two non-failing experimental LangGraph streaming warnings remain.
- MCP regression cycle: 17 expected failures before implementation; four additional expected failures for normalized existing-stdio enable cases. Focused gate: 95 passed. Independent final review gate: 72 passed, with both actual container flag values checked separately.
- Ruff and Git whitespace checks pass. All 1312 earlier files outside the eight-file MCP integration scope were preserved byte-for-byte before updating these status records. Existing index entries and preview work were preserved during integration. The user subsequently authorized committing all changes and pushing them through the repository's configured dual-remote workflow.
- `ADMIN_USER_IDS` is configured for this installation's verified existing profile. `MCP_ALLOW_API_STDIO=false` is explicit. Authentication and administration changes take effect when the application next loads its configuration; no live startup or migrations were performed.
- Take100's native password change returned HTTP 200; an independent replacement login verified the same account. Root separately checked encrypted recovery custody and the active binding. The previous Take100 password was retained in encrypted recovery; obsolete local session removal preceded creation of the new verified cache. No passwords, tokens, or authenticated bodies were emitted.
- Remaining external action: attendance password rotation. The current attendance URL was requested; neither open browser tabs nor the saved offline tunnel supplied a replacement address.

## Publication verification — 2026-10-01

- Fresh full suite before committing: **6548 passed, 0 failed, 15 skipped, 1 deselected** in 323.78 seconds; the same two non-failing LangGraph warnings.
- Commits separate cleanup/configuration, document staging, client/runtime integration, generation lifecycle, planning context/accounting, answer previews, MCP security, and status documentation.
- Ignored local environment, skill credentials, sessions, and encrypted recovery artifacts stay outside the commits. Attendance rotation remains blocked on the current site URL.
