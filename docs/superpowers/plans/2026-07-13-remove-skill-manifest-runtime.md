# Standards-Compatible Skill Command Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the private manifest-driven skill runtime with a `SKILL.md`-only, device-local command runtime that installs and invokes bundled commands without global `PATH` dependencies.

**Architecture:** The registry discovers one bundle root and source hash per `SKILL.md`. A runtime manager publishes one fixed `run_skill_command` tool for ready bundles, an environment manager prepares optional Python projects in profile-local virtual environments, and the execution engine resolves argv only inside the selected bundle/runtime. All calls retain device/session/catalog validation, approval, secret isolation, output limits, redaction, and auditing.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, asyncio/threaded subprocess execution, `venv`, `tomllib`, pytest, Ruff.

---

### Task 1: Define the standard bundle model and discovery behavior

**Files:**
- Modify: `client_backend/services/local_skills_registry.py`
- Test: `tests/client_backend/test_skills_registry.py`
- Delete: `shared/skills/manifest.py`
- Delete: the obsolete private execution-metadata parser tests

- [x] Add failing registry tests proving direct and nested bundles record a confined `bundle_root`, deterministic `source_hash`, executable assets, and no private manifest fields.
- [x] Run `pytest -q tests/client_backend/test_skills_registry.py` and confirm the new assertions fail against the existing metadata model.
- [x] Add bundle-root resolution and safe asset discovery to `SkillMetadata`; remove all manifest parsing, fields, imports, and serialized summaries.
- [x] Delete the private manifest model and its validation-only test module.
- [x] Re-run the registry tests and confirm they pass.

### Task 2: Build readiness and fixed command catalog entries

**Files:**
- Modify: `client_backend/services/skill_runtime/manager.py`
- Test: `tests/client_backend/test_skill_runtime_manager.py`
- Replace: `tests/client_backend/test_skill_capability_catalog.py`

- [x] Write failing tests for `instruction_only`, bundled-command `ready`, Python-project `setup_required`, stale runtime, and the single fixed command-tool schema.
- [x] Run the two focused test modules and confirm failures are caused by the old manifest API.
- [x] Replace manifest readiness with bundle/runtime inspection and publish only `skill::<name>::run_skill_command`, marked `mutation: true` and carrying the source hash.
- [x] Re-run the focused tests and confirm they pass.

### Task 3: Add isolated Python runtime preparation

**Files:**
- Create: `client_backend/services/skill_runtime/environment.py`
- Modify: `client_backend/core/paths.py`
- Test: `tests/client_backend/test_skill_environment.py`

- [x] Write failing tests for setup preview, approval requirement, staged virtual-environment preparation, runtime metadata, stale detection, failure rollback, and confined cleanup.
- [x] Run `pytest -q tests/client_backend/test_skill_environment.py` and confirm import/behavior failures.
- [x] Implement deterministic preview and `SkillEnvironmentManager` with profile-local staging, bounded non-interactive setup, redacted logs, atomic promotion, and runtime inspection.
- [x] Re-run the environment tests and confirm they pass.

### Task 4: Make installation accept complete one-skill bundles

**Files:**
- Modify: `client_backend/services/skill_runtime/install.py`
- Modify: `client_backend/schemas/skills.py`
- Modify: `client_backend/api/skills.py`
- Test: `tests/client_backend/test_skill_installation.py`
- Test: `tests/client_backend/test_skills_api.py`

- [x] Add failing tests for preview/hash confirmation, direct and nested one-skill sources, zero/multiple-skill rejection, automatic portable-asset readiness, approved Python setup, rollback, and uninstall cleanup.
- [x] Run both focused modules and confirm the new contract fails against the direct-only installer.
- [x] Implement preview/install/setup endpoints and atomic complete-bundle installation without validating or returning private manifest status.
- [x] Re-run both focused modules and confirm they pass.

### Task 5: Replace typed capability execution with confined argv execution

**Files:**
- Modify: `client_backend/services/skill_runtime/execution.py`
- Delete: `client_backend/services/skill_runtime/permissions.py`
- Replace: `tests/client_backend/test_skill_execution_engine.py`
- Delete: `tests/client_backend/test_skill_permissions.py`

- [x] Write failing execution tests for the reserved command id, fixed argv validation, mandatory approval, bundled `bin/` and `scripts/` resolution, runtime command resolution, scoped `PATH`, workspace/skill cwd, command rejection, timeout, output cap, and no shell interpretation.
- [x] Run the execution module and confirm failures exercise the old manifest path.
- [x] Implement the single command executor, resolve only skill-owned commands, inject sidecar-generated roots, and keep normalized envelopes/audit behavior.
- [x] Delete the manifest-specific permission evaluator and tests.
- [x] Re-run the execution tests and confirm they pass.

### Task 6: Namespace encrypted secrets per skill

**Files:**
- Modify: `client_backend/services/skill_runtime/secrets.py`
- Modify: `client_backend/api/skills.py`
- Modify: `client_backend/schemas/skills.py`
- Replace: `tests/client_backend/test_skill_secrets.py`

- [x] Write failing tests proving bindings are keyed by skill, only names are listed, values remain encrypted, execution receives only the selected skill's bindings, and two profile roots do not share data.
- [x] Run the secrets tests and confirm failures against the flat store.
- [x] Implement per-skill encrypted storage and `GET/POST/DELETE /skills/{name}/secrets` operations; remove the flat secret endpoint.
- [x] Re-run the secrets and API tests and confirm they pass.

### Task 7: Publish, activate, approve, and dispatch the fixed command tool

**Files:**
- Modify: `client_backend/services/runtime_bridge.py`
- Modify: `app/ai/client_runtime_tools.py`
- Test: `tests/client_backend/test_runtime_bridge.py`
- Replace: `tests/client_backend/test_skill_dispatch.py`
- Replace: `tests/client_backend/test_skill_hitl.py`

- [x] Write failing tests proving the catalog publishes only ready command tools, activation adds ready/setup guidance, every command is mutation-gated, and stale session/catalog/tool-instance requests are rejected before spawn.
- [x] Run the focused bridge/dispatch/HITL tests and confirm the old capability collector fails them.
- [x] Wire readiness collection, generated activation footers, approval context, and optional source-hash tool-instance derivation into the bridge.
- [x] Re-run focused tests and confirm they pass.

### Task 8: Replace fixtures and prove the original failure is fixed

**Files:**
- Replace: `tests/fixtures/skills/echo_python/`
- Delete: `tests/fixtures/skills/binary_probe/`
- Replace: `tests/client_backend/test_skill_example_fixtures.py`
- Modify: `skills/google_calendar/SKILL.md`
- Modify: `skills/google_calendar/google_calendar_cli.py`
- Modify: `skills/google_calendar/__main__.py`
- Create: `skills/google_calendar/bin/cli-anything-google-calendar.py`

- [x] Create a failing end-to-end fixture test where the named CLI is absent from global `PATH` but available from the bundle and invoked through `run_skill_command`.
- [x] Run the fixture test and confirm it fails before the runtime changes are complete.
- [x] Convert fixtures to `SKILL.md` plus bundled scripts only, add a portable Calendar launcher, and make Calendar imports bundle-local.
- [x] Re-run the fixture and Calendar credential-free tests and confirm both pass.

### Task 9: Remove obsolete APIs, tests, and documentation

**Files:**
- Delete: `plans/skill_runtime_refactor.md`
- Modify: `docs/skill-runtime.md`
- Modify: `README.md`
- Modify: `client_backend/services/skill_runtime/__init__.py`
- Modify: `shared/skills/errors.py`
- Update: `tests/client_backend/test_skill_audit.py`
- Update: any remaining tests reported by repository search

- [x] Search the repository for every obsolete private execution-manifest symbol and record any remaining references.
- [x] Delete or rewrite every obsolete runtime, test, fixture, and documentation path; retain only fixed command-runtime concepts.
- [x] Update audit tests for `run_skill_command` and mandatory approval.
- [x] Re-run the complete `tests/client_backend/test_skill*.py` group.

### Task 10: Full verification and branch completion

**Files:**
- Verify all modified files

- [x] Run `ruff check` on all modified Python files and fix reported issues.
- [x] Run focused skill tests plus device-isolation, unified-tool-search, and multi-sidecar hardening tests.
- [x] Run the full pytest suite and record exact pass/fail counts.
- [x] Run the obsolete-reference search again and require zero matches.
- [x] Inspect `git diff --check`, `git status --short`, and the complete diff for unintended changes.
- [ ] Use `superpowers:finishing-a-development-branch` for the final integration handoff.
