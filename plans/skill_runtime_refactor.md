# Production Skill Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Markdown-only skill activation plus ad hoc shell execution with a production-ready, manifest-driven skill runtime that can install, discover, validate, expose, and execute arbitrary user-added skills through typed, permissioned, auditable capabilities.

**Architecture:** Skills remain client-owned and sidecar-executed. User-added bundles can be scanned from configured roots or installed into a profile-local skill root, and each executable bundle can optionally include a machine-readable `skill.json` manifest. The sidecar validates bundles, runs readiness checks, syncs skill capabilities as typed client tools, and executes them through a generic `SkillExecutionEngine` instead of asking the model to construct raw shell commands. The canonical server orchestrates and dispatches to the active device session; it does not execute arbitrary user skill code.

**Tech Stack:** FastAPI, client sidecar runtime, WebSocket device bridge, Pydantic models, JSON Schema, LangChain tools, existing client MCP/tool catalog path, pytest.

---

## Problem Statement

The current skill system treats skills primarily as Markdown instructions. `activate_skill` loads the skill text and the model decides what command to run. For executable skills, this creates a fragile boundary: the model may call a command that is not installed, call it through the wrong shell, miss required environment variables, or leak secrets into prompt-visible text.

Google Calendar exposed this failure mode, but it is only an example fixture. The issue is general: a user can add any skill into `CLIENT_SKILLS_ROOTS`, or install a bundle through the sidecar, and the system needs a mature way to install, discover, validate, expose, execute, and audit those skills without hardcoding individual integrations, package names, or providers.

## Current Code Findings

- `client_backend/services/local_skills_registry.py` scans `CLIENT_SKILLS_ROOTS`, parses `SKILL.md`, and stores Markdown content. It does not parse an execution manifest.
- `app/ai/skills_tool.py` creates `activate_skill`, which dispatches `client_skill::activate` to the active sidecar and returns the full skill instructions as text.
- `client_backend/services/runtime_bridge.py` handles `client_skill::activate` by returning the skill Markdown content. It does not expose executable skill capabilities.
- `client_backend/services/local_mcp_manager.py` already has the strongest local execution pattern: it loads a local config, discovers capabilities, syncs tool metadata, and dispatches calls by qualified id.
- `app/ai/client_runtime_tools.py` already converts sidecar-published client tool catalog entries into namespaced model-facing tools such as `client__server__tool`.
- `app/ai/client_tool_catalog.py` separately indexes sidecar-published client tools for `tool_search`, but currently filters out anything except `origin == "mcp"`.
- Desktop Commander is a local MCP server, not a skill runtime. It can run shell commands, but it does not know whether a skill is installed, ready, permissioned, or safe to execute.
- The repo currently has no `skill.json` files. The Google Calendar skill in `skills/google_calendar` expects a command named `cli-anything-google-calendar`, but the repo does not install that command and does not package `cli_anything.google_calendar`; this should be treated as one failing fixture, not as the architecture driver.

## Design Principles

- Skills are local capabilities owned by the connected device.
- Markdown explains how to reason; manifests define how to execute.
- The model calls typed capabilities, not raw shell commands, for normal skill execution.
- The runtime must not hardcode any specific skill name, provider, package, or command.
- The installer and runtime must not branch on Google Calendar, or on any other specific example skill. Example skills prove the generic contract only.
- Installing a skill means safely registering or copying a validated bundle into a sidecar-controlled skill root; dependency installation is explicit and user-approved, never an implicit side effect of scanning.
- Secrets are injected only at execution time and never placed in prompts, chat history, or normal logs.
- Readiness failures should be detected before tool execution when possible.
- Existing Markdown-only skills continue to work as instruction-only skills.
- The sidecar enforces session binding, permissions, timeouts, output limits, and audit records.

## Skill Bundle Contract

A skill directory can contain:

```text
skills/<skill-name>/
  SKILL.md
  skill.json
  requirements.txt
  package.json
  runners/
  tests/
  README.md
```

`SKILL.md` remains the human/model instruction file. `skill.json` is optional. If missing, the skill is instruction-only and activation behaves as it does today.

### Universal Installation Contract

Skills can become available in two generic ways:

1. **Scanned roots:** the sidecar scans configured `CLIENT_SKILLS_ROOTS`, exactly as it does today.
2. **Profile-installed bundles:** the sidecar accepts a local skill bundle path, validates it, copies or links it into a user profile skill root, writes install metadata, and refreshes catalogs.

The install path must be generic:

- Accept any directory bundle that contains `SKILL.md`; optionally accept zip archives after path-traversal validation.
- Parse `SKILL.md` and optional `skill.json` during installation to fail early on malformed bundles.
- Install into a sidecar-controlled profile directory such as `<profile>/skills/installed/<safe-name>-<hash>/`.
- Write local install metadata such as source path, source hash, installed timestamp, bundle name, manifest status, and enabled state.
- Never store secrets in install metadata.
- Never auto-install Python, Node, system, or shell dependencies during scan. The first implementation reports missing dependencies and repair hints; dependency installation can be added later behind explicit user approval and the same audit path.
- Refresh the local skill registry and runtime catalogs after install, uninstall, enable, disable, or reload.
- Reject duplicate active skill names unless the user disables or uninstalls the existing skill first.

This contract is deliberately provider-neutral. A calendar skill, a CAD skill, a local file utility, and a domain-specific API wrapper all use the same install, readiness, catalog, execution, permission, secret, and audit machinery.

### Manifest Shape

```json
{
  "schema_version": "1.0",
  "name": "example-calendar",
  "display_name": "Example Calendar",
  "description": "Inspect and manage calendar events.",
  "runtime": {
    "type": "python_module",
    "module": "skills.example_calendar.cli",
    "entrypoint": "cli"
  },
  "dependencies": {
    "python": ["click>=8", "requests>=2"],
    "node": [],
    "system": []
  },
  "secrets": [
    {
      "name": "EXAMPLE_CALENDAR_ACCESS_TOKEN",
      "required": true,
      "description": "OAuth access token with calendar scopes."
    }
  ],
  "permissions": [
    "network:api.example.com",
    "calendar:read",
    "calendar:write"
  ],
  "capabilities": [
    {
      "name": "event_list",
      "description": "List events from a calendar.",
      "input_schema": {
        "type": "object",
        "properties": {
          "calendar_id": { "type": "string", "default": "primary" },
          "time_min": { "type": "string" },
          "time_max": { "type": "string" }
        },
        "required": ["time_min", "time_max"]
      },
      "execution": {
        "argv": [
          "--json",
          "event",
          "list",
          "--calendar-id",
          "{calendar_id}",
          "--time-min",
          "{time_min}",
          "--time-max",
          "{time_max}"
        ]
      },
      "permissions": ["calendar:read"],
      "secrets": ["EXAMPLE_CALENDAR_ACCESS_TOKEN"],
      "mutation": false
    }
  ]
}
```

The manifest is declarative. It identifies runtime type, dependencies, secrets, permissions, and typed capabilities. It does not contain arbitrary prompt instructions.

## Runtime Types

The first production slice should support these generic runtime types:

- `python_module`: import a Python module and call a known Click/Typer/function entrypoint.
- `python_script`: execute a Python script path under the skill directory.
- `binary`: execute an installed command after `PATH` resolution and preflight.

Reserved future runtime types:

- `node_package`: execute a Node package through `npx` or a local package command.
- `mcp_server`: expose tools from a skill-owned MCP server through the existing local MCP machinery.
- `shell`: restricted fallback that requires explicit permission and never interpolates untrusted values into a single shell string.

The manifest validator should reject unsupported runtime types in the first slice rather than accepting metadata that cannot execute safely. The runtime should prefer structured argument arrays over shell strings. Shell execution should be an explicit escape hatch, not the default integration path, and should only be added after explicit permission and HITL enforcement are already in place.

## Execution Flow

```text
Skill install or reload
 -> SkillBundleInstaller validates SKILL.md and optional skill.json
 -> SkillBundleInstaller copies or links bundle into profile skill root
 -> LocalSkillsRegistry refreshes installed and configured skill roots
 -> RuntimeBridge refreshes skill and tool catalogs if connected

Sidecar startup
 -> LocalSkillsRegistry scans SKILL.md and skill.json from configured and profile-installed roots
 -> SkillRuntimeManager validates manifests
 -> SkillRuntimeManager runs preflight checks
 -> RuntimeBridge syncs instruction skills and executable capabilities

Chat turn
 -> Model activates relevant skill instructions when useful
 -> Model sees executable skill capabilities as client tools
 -> Model calls a typed skill capability
 -> Server dispatches tool call to active sidecar session
 -> Sidecar validates session, capability, permissions, secrets, and arguments
 -> SkillExecutionEngine runs capability with scoped env and timeout
 -> Sidecar returns structured result and writes audit record
```

## Data Model

### Skill Catalog Entry

Extend the existing sidecar skill catalog entries with execution metadata:

```json
{
  "name": "example-calendar",
  "description": "Inspect and manage calendar events.",
  "enabled": true,
  "content_length": 1234,
  "install": {
    "source": "profile",
    "installed": true,
    "source_hash": "sha256:..."
  },
  "execution": {
    "manifest_present": true,
    "status": "ready",
    "capability_count": 2,
    "missing_dependencies": [],
    "missing_secrets": [],
    "permissions": ["network:api.example.com", "calendar:read"]
  }
}
```

### Skill Capability Tool Catalog Entry

Executable skill capabilities should sync through the existing client tool catalog path:

```json
{
  "name": "event_list",
  "description": "List events from a calendar.",
  "origin": "skill",
  "server_name": "skill_example_calendar",
  "qualified_id": "skill::example-calendar::event_list",
  "input_schema": {
    "type": "object",
    "properties": {
      "calendar_id": { "type": "string" },
      "time_min": { "type": "string" },
      "time_max": { "type": "string" }
    },
    "required": ["time_min", "time_max"]
  },
  "readiness": {
    "status": "ready"
  }
}
```

The model-facing tool name can be derived by the existing client tool naming pattern, for example:

```text
client__skill_example_calendar__event_list
```

Server-side tool metadata should preserve the catalog `origin` value (`skill`), while model-facing tool metadata should expose a normalized tool origin such as `client_skill`. This keeps skill tools device-scoped like client MCP tools while allowing search, approval, audit, and analytics code to distinguish skill tools from MCP server tools.

### Execution Result

All skill executions should return a normalized envelope:

```json
{
  "ok": true,
  "skill": "example-calendar",
  "capability": "event_list",
  "result": {
    "items": []
  },
  "stdout": "",
  "stderr": "",
  "duration_ms": 231,
  "audit_id": "skill-exec-20260708-0001"
}
```

Failure shape:

```json
{
  "ok": false,
  "skill": "example-calendar",
  "capability": "event_list",
  "error": {
    "code": "MISSING_SECRET",
    "message": "Required secret EXAMPLE_CALENDAR_ACCESS_TOKEN is not configured.",
    "repair": {
      "type": "configure_secret",
      "secret": "EXAMPLE_CALENDAR_ACCESS_TOKEN"
    }
  },
  "duration_ms": 12,
  "audit_id": "skill-exec-20260708-0002"
}
```

## Standard Error Codes

- `SKILL_INSTALL_INVALID`
- `SKILL_INSTALL_CONFLICT`
- `UNSAFE_BUNDLE_PATH`
- `SKILL_MANIFEST_INVALID`
- `SKILL_NOT_READY`
- `CAPABILITY_NOT_FOUND`
- `UNSUPPORTED_RUNTIME`
- `MISSING_DEPENDENCY`
- `MISSING_SECRET`
- `PERMISSION_REQUIRED`
- `PERMISSION_DENIED`
- `COMMAND_NOT_FOUND`
- `INVALID_ARGUMENTS`
- `EXECUTION_TIMEOUT`
- `OUTPUT_TOO_LARGE`
- `NON_JSON_OUTPUT`
- `REMOTE_API_ERROR`
- `RUNTIME_ERROR`

## Secrets

Add a sidecar `SkillSecretStore` abstraction with providers:

- process environment
- encrypted local profile storage
- user-approved OAuth or credential setup flow

Secrets are resolved by name at execution time and injected into the child process or runtime call environment. Logs and returned errors must redact secret values.

The first implementation can read from environment variables and local profile storage. OAuth refresh-token flows can be added later behind the same interface.

## Permissions

Each capability declares required permissions. The sidecar enforces them before execution.

Initial permission families:

- `network:<host>`
- `filesystem:read:<path>`
- `filesystem:write:<path>`
- `process:spawn`
- `mutation`
- `domain:<domain>:<action>` or exact domain labels such as `calendar:read`, `cad:write`, or `crm:read`
- `desktop:automation`
- `shell:execute` (reserved until shell runtime is explicitly implemented)

Generic runtime enforcement should cover host allowlists, workspace path allowlists, mutation flags, and shell restrictions. Domain-specific permission names can exist as labels, but enforcement must not depend on hardcoded skill names.

## Audit Records

Each execution should write a local audit record under the sidecar profile:

```json
{
  "timestamp": "2026-07-08T06:30:00Z",
  "user_id": "server-user-id",
  "device_id": "device-id",
  "session_id": "sidecar-session-id",
  "skill": "example-calendar",
  "capability": "event_list",
  "qualified_id": "skill::example-calendar::event_list",
  "arguments_redacted": {
    "calendar_id": "primary",
    "time_min": "2026-07-08T00:00:00+07:00",
    "time_max": "2026-07-09T00:00:00+07:00"
  },
  "status": "ok",
  "duration_ms": 231
}
```

Audit records must never store raw secret values.

## File Structure

Create:

- `shared/skills/manifest.py`: Pydantic models and validation for `skill.json`.
- `shared/skills/errors.py`: normalized skill runtime error codes and error envelope helpers.
- `client_backend/services/skill_runtime/__init__.py`: package marker.
- `client_backend/services/skill_runtime/manager.py`: manifest registry, readiness checks, and capability catalog generation.
- `client_backend/services/skill_runtime/install.py`: safe, generic skill bundle installation, uninstall, and install metadata.
- `client_backend/services/skill_runtime/execution.py`: generic execution engine and runtime-type dispatch.
- `client_backend/services/skill_runtime/secrets.py`: secret lookup and redaction.
- `client_backend/services/skill_runtime/permissions.py`: permission evaluation.
- `client_backend/services/skill_runtime/audit.py`: local execution audit writer.
- `tests/client_backend/test_skill_manifest.py`: manifest validation tests.
- `tests/client_backend/test_skill_installation.py`: generic install/uninstall tests for local skill bundles.
- `tests/client_backend/test_skill_runtime_manager.py`: discovery and readiness tests.
- `tests/client_backend/test_skill_execution_engine.py`: runner and error handling tests.
- `tests/client_backend/test_skill_capability_catalog.py`: catalog sync contract tests.

Modify:

- `client_backend/services/local_skills_registry.py`: attach optional manifest metadata to scanned skills.
- `client_backend/services/runtime_bridge.py`: route `skill::<skill>::<capability>` dispatch requests to `SkillExecutionEngine`.
- `client_backend/services/local_mcp_manager.py`: no major behavior change; use it as the reference pattern for local capability discovery.
- `app/ai/client_runtime_tools.py`: accept client tool catalog entries with `origin == "skill"` in addition to `origin == "mcp"`.
- `app/ai/client_tool_catalog.py`: index `origin == "skill"` entries for `tool_search`, deferred loading, and custom-agent client tool allowlists.
- `app/ai/skills_tool.py`: keep Markdown activation unchanged; optionally include readiness summary in activated skill text.
- `app/services/client_runtime_store.py`: preserve skill capability catalog metadata if the existing tool catalog shape needs extra fields.
- `client_backend/api/skills.py`: expose install, uninstall, readiness, and capability status in local skill APIs.
- `README.md`: document user-added executable skill bundles.

## Implementation Tasks

### Task 1: Add Manifest Models

**Files:**
- Create: `shared/skills/manifest.py`
- Create: `tests/client_backend/test_skill_manifest.py`

- [x] Define Pydantic models for `SkillManifest`, `SkillRuntimeSpec`, `SkillDependencySpec`, `SkillSecretSpec`, `SkillCapabilitySpec`, and `SkillCapabilityExecutionSpec`.
- [x] Validate that each capability has a non-empty `name`, `description`, `input_schema`, and `execution`.
- [x] Validate runtime `type` against the first-slice supported runtime type set: `python_module`, `python_script`, and `binary`.
- [x] Reject capability names that cannot be converted into safe tool identifiers.
- [x] Add tests for valid manifests, missing required fields, invalid or reserved runtime types, duplicate capability names, malformed schemas, and provider-neutral manifests with no Google-specific fields.
- [x] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_manifest.py -q` → 48 passed.

### Task 2: Load Manifests During Skill Scan

**Files:**
- Modify: `client_backend/services/local_skills_registry.py`
- Test: `tests/client_backend/test_skills_registry.py` or create `tests/client_backend/test_skill_runtime_manager.py`

- [x] Extend `SkillMetadata` with optional `manifest_path`, `manifest`, and `manifest_error` fields.
- [x] During `_load_skill`, look for `skill.json` in the same directory as `SKILL.md`.
- [x] Parse the manifest with `shared.skills.manifest`.
- [x] Preserve Markdown-only behavior when `skill.json` is missing.
- [x] Include install and execution status summaries in `to_dict()` and `to_sync_dict()` without exposing absolute local paths to the canonical server.
- [x] Add tests proving Markdown-only skills still load, manifest-backed skills include execution metadata, and installed skill metadata does not leak absolute profile paths.
- [x] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skills_registry.py -q` → 11 passed.

### Task 3: Add Readiness Checks

**Files:**
- Create: `client_backend/services/skill_runtime/manager.py`
- Create: `client_backend/services/skill_runtime/secrets.py`
- Test: `tests/client_backend/test_skill_runtime_manager.py`

- [x] Implement readiness states: `instruction_only`, `ready`, `not_ready`, and `invalid`.
- [x] Check Python imports for Python dependencies when possible.
- [x] Check executable availability for `binary` runtime using `shutil.which`.
- [x] Check required secrets through `SkillSecretStore`.
- [x] Return repair hints for missing dependency, missing executable, missing secret, unsupported runtime, and permission-required states. Do not auto-install dependencies during readiness checks.
- [x] Add tests for ready skill, missing secret, missing binary, invalid manifest, and Markdown-only skill.
- [x] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_runtime_manager.py -q` → 16 passed.

### Task 4: Add Generic Skill Bundle Installation

**Files:**
- Create: `client_backend/services/skill_runtime/install.py`
- Modify: `client_backend/services/local_skills_registry.py`
- Modify: `client_backend/api/skills.py`
- Test: `tests/client_backend/test_skill_installation.py`

- [x] Implement a `SkillBundleInstaller` that accepts a local directory path containing `SKILL.md`, validates the optional `skill.json`, computes a source hash, and installs the bundle under the current user's profile skill directory.
- [x] Add a registry-managed profile install root, for example `get_profile_subdir(user_id, "skills") / "installed"`, to `LocalSkillsRegistry._resolve_skill_roots()` so installed bundles are scanned without requiring users to edit `CLIENT_SKILLS_ROOTS`. (Extracted to `get_installed_skills_root(user_id)` in `core/paths.py` as the single source of truth.)
- [x] Reject unsafe bundles: missing `SKILL.md`, malformed front matter, malformed manifest, duplicate active skill name, path traversal (target escaping profile root), and symlinks in the bundle. (Zip archives deferred — plan marks them optional.)
- [x] Store install metadata locally with bundle name, source hash, source path, installed timestamp, manifest status, and enabled state. Do not store secrets.
- [x] Add local API operations for install, uninstall, and install-status. Generic request bodies (`source_path` / `name`); no provider-specific mentions.
- [x] Refresh the skill registry and runtime catalogs after install or uninstall if the runtime bridge is connected.
- [x] Add tests for installing a Markdown-only skill, installing a manifest-backed skill, rejecting duplicates, rejecting malformed manifests, uninstalling an installed skill, and proving the installed skill appears in the normal skill catalog.
- [x] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_installation.py -q` → 13 passed, 1 skipped (Windows symlink-privilege; logic covered cross-platform).

### Task 5: Sync Skill Capabilities As Client Tools

**Files:**
- Modify: `client_backend/services/runtime_bridge.py`
- Modify: `app/ai/client_runtime_tools.py`
- Modify: `app/ai/client_tool_catalog.py`
- Test: `tests/client_backend/test_skill_capability_catalog.py`
- Test: `tests/test_unified_tool_search.py`

- [x] Build skill capability catalog entries from ready manifest capabilities.
- [x] Add those entries to the existing sidecar tool catalog in `_build_tool_catalog`.
- [x] Use qualified ids shaped as `skill::<skill-name>::<capability-name>`.
- [x] Allow `app/ai/client_runtime_tools.py` to parse entries with `origin == "skill"`.
- [x] Allow `app/ai/client_tool_catalog.py` to index entries with `origin == "skill"` so `tool_search`, deferred loading, and custom-agent allowlists can discover skill capability tools.
- [x] Add a client tool origin constant such as `TOOL_ORIGIN_CLIENT_SKILL = "client_skill"` and set skill tool metadata to that value while preserving the catalog entry's raw `origin == "skill"` (as `metadata["catalog_origin"]`).
- [x] Generate exposed names with the existing `client__...` prefix pattern.
- [x] Add tests proving MCP tools and skill capability tools can coexist without name collisions and can both be found through client-side `tool_search`.
- [x] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_capability_catalog.py -q` → 8 passed.
- [x] Run: `.venv\Scripts\python.exe -m pytest tests/test_unified_tool_search.py -q` → 16 passed.

### Task 6: Add Permission Evaluator Before Execution

**Files:**
- Create: `client_backend/services/skill_runtime/permissions.py`
- Test: `tests/client_backend/test_skill_permissions.py`

- [x] Implement permission evaluation for `network`, `filesystem`, `process`, and mutation families before any child process or runtime entrypoint is invoked.
- [x] Treat `mutation: true` capabilities as blocked unless the current permission policy explicitly grants the mutation or a later HITL approval grants it.
- [x] Block `shell` runtime entirely in the first slice because `shell` is a reserved future runtime type.
- [x] Return normalized `PERMISSION_REQUIRED` or `PERMISSION_DENIED` errors with redacted arguments and no secret values.
- [x] Add tests for allowed read capability, blocked write capability, blocked process spawn without permission, blocked mutation capability, and redacted permission error payloads.
- [x] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_permissions.py -q` → 27 passed.

### Task 7: Implement Execution Engine

**Files:**
- Create: `client_backend/services/skill_runtime/execution.py`
- Create: `client_backend/services/skill_runtime/errors.py` or `shared/skills/errors.py`
- Test: `tests/client_backend/test_skill_execution_engine.py`

- [ ] Implement `SkillExecutionEngine.execute(qualified_tool_id, arguments, context)`.
- [ ] Validate capability exists and current readiness is `ready`.
- [ ] Validate arguments against capability `input_schema`.
- [ ] Call the permission evaluator before resolving secrets, rendering arguments, or spawning a process.
- [ ] Render argument placeholders only into argv lists, never into a shell string.
- [ ] Implement `python_module`, `python_script`, and `binary` runtimes first.
- [ ] Apply timeout and output size limits.
- [ ] Parse JSON stdout when the capability declares JSON output.
- [ ] Return normalized success and error envelopes.
- [ ] Add tests for success, invalid arguments, command not found, timeout, output too large, and non-JSON output.
- [ ] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_execution_engine.py -q`

### Task 8: Route Skill Tool Dispatch

**Files:**
- Modify: `client_backend/services/runtime_bridge.py`
- Test: `tests/client_backend/test_runtime_bridge.py` or create `tests/client_backend/test_skill_dispatch.py`

- [ ] Keep `client_skill::activate` behavior unchanged for Markdown activation.
- [ ] Route `qualified_tool_id` values that start with `skill::` to `SkillExecutionEngine`.
- [ ] Keep existing session, catalog version, and tool instance validation.
- [ ] Return runtime errors through the existing `ToolDispatchResult` error path.
- [ ] Add tests for valid skill dispatch, stale session rejection, catalog mismatch rejection, and missing capability rejection.
- [ ] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_dispatch.py -q`

### Task 9: Add HITL Approval Flow

**Files:**
- Modify: `client_backend/services/runtime_bridge.py`
- Modify: `app/services/message_service.py` if skill permissions need the existing HITL interrupt path
- Test: `tests/client_backend/test_skill_hitl.py`

- [ ] Treat `mutation: true` capabilities as approval-required unless the permission policy already grants them.
- [ ] Redact sensitive arguments from permission prompts and logs.
- [ ] Integrate with the existing HITL interrupt path so approved skill mutations resume through the same session, catalog version, and tool instance validation.
- [ ] Add tests for pending mutation approval, denied mutation approval, approved mutation execution, stale session rejection after approval, and catalog mismatch rejection after approval.
- [ ] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_hitl.py -q`

### Task 10: Add Secret Store

**Files:**
- Modify: `client_backend/services/skill_runtime/secrets.py`
- Modify: `client_backend/api/skills.py`
- Test: `tests/client_backend/test_skill_secrets.py`

- [ ] Implement environment-backed secret lookup.
- [ ] Implement encrypted local profile-backed secret storage using the existing local profile root.
- [ ] Add redaction helpers for logs and returned errors.
- [ ] Add local API endpoints for listing required secret names and setting a secret value.
- [ ] Add tests for env lookup, stored lookup, missing required secret, and redaction.
- [ ] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_secrets.py -q`

### Task 11: Add Audit Trail

**Files:**
- Create: `client_backend/services/skill_runtime/audit.py`
- Modify: `client_backend/services/skill_runtime/execution.py`
- Test: `tests/client_backend/test_skill_audit.py`

- [ ] Write one audit JSONL record per execution under the current profile.
- [ ] Include user id, device id, session id, skill name, capability name, qualified id, redacted arguments, status, duration, and error code.
- [ ] Keep raw stdout/stderr out of audit records by default.
- [ ] Add tests proving secret values are not stored.
- [ ] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_audit.py -q`

### Task 12: Add Generic Example Fixtures

**Files:**
- Create: `tests/fixtures/skills/echo_python/SKILL.md`
- Create: `tests/fixtures/skills/echo_python/skill.json`
- Create: `tests/fixtures/skills/binary_probe/SKILL.md`
- Create: `tests/fixtures/skills/binary_probe/skill.json`
- Modify: `skills/google_calendar/skill.json` if keeping Google Calendar as an optional real-world fixture
- Modify: `skills/google_calendar/google_calendar_cli.py` or package imports if keeping Google Calendar in the automated fixture set
- Modify: `skills/google_calendar/SKILL.md` if keeping Google Calendar in the automated fixture set
- Test: `tests/client_backend/test_skill_example_fixtures.py`
- Test: `skills/google_calendar/tests/test_full_e2e.py` only if Google Calendar remains part of the automated fixture set

- [ ] Add at least two provider-neutral executable skill fixtures that prove the runtime is universal: one `python_module` or `python_script` fixture, and one `binary` fixture that uses an installed command available in CI.
- [ ] Add manifests for those fixtures without mentioning Google Calendar or any provider-specific field.
- [ ] Add dry-run fixture tests proving the skill capability path works without live credentials.
- [ ] If Google Calendar remains in the repo, add its manifest as an example of the same generic contract and fix its package/import path so it is runnable from the repo. Do not add any Google-specific runtime manager, dispatcher, permission evaluator, secret store, or catalog code.
- [ ] Add a regression test that scans all fixture manifests and fails if core runtime code contains example-specific skill name checks.
- [ ] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_example_fixtures.py -q`
- [ ] Optional if Google Calendar is kept in the automated fixture set: `.venv\Scripts\python.exe -m pytest skills/google_calendar/tests -q`

### Task 13: Update Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/skill-runtime.md`

- [ ] Document the difference between instruction-only skills and executable skill bundles.
- [ ] Document the two generic installation paths: configured scan roots and profile-installed bundles.
- [ ] Document `skill.json` schema with examples.
- [ ] Document runtime types and security constraints.
- [ ] Document dependency readiness and repair hints, including the rule that dependencies are not silently installed during scan.
- [ ] Document secret setup and readiness states.
- [ ] Document how sidecar users reload skills and inspect readiness.
- [ ] Add a short troubleshooting section for command not found, missing secret, invalid manifest, and permission denied.

### Task 14: Full Regression

**Files:**
- Existing test suite

- [ ] Run client-side tests: `.venv\Scripts\python.exe -m pytest tests/client_backend -q`
- [ ] Run skill-related server tests: `.venv\Scripts\python.exe -m pytest tests/test_skills_tool.py tests/test_skills_architecture.py tests/test_client_tool_scope.py -q`
- [ ] Run MCP/client runtime tests: `.venv\Scripts\python.exe -m pytest tests/test_multi_sidecar_hardening.py tests/test_client_invocation_isolation.py tests/test_client_tool_isolation.py tests/test_unified_tool_search.py -q`
- [ ] Run a manual sidecar readiness check through the local `/skills` API.
- [ ] Run a manual sidecar install check by installing a provider-neutral local skill bundle through the local `/skills` API and verifying it appears in readiness and tool catalogs.
- [ ] Run one dry-run executable skill capability through a chat turn and verify the model calls the typed tool rather than constructing a raw shell command.

## Acceptance Criteria

- A user can add a new manifest-backed skill under `CLIENT_SKILLS_ROOTS` without changing backend code.
- A user can install a new provider-neutral skill bundle into the sidecar profile skill root without changing backend code or editing environment variables.
- The sidecar reports whether each executable skill is ready before the model attempts execution.
- Missing executable, missing dependency, missing secret, invalid manifest, and permission denial all produce structured, actionable errors.
- Executable skill capabilities appear as typed client tools scoped to the active device session.
- Executable skill capabilities are discoverable both through direct client runtime binding and through client-side `tool_search`.
- The model does not need to construct raw shell commands for normal skill execution.
- Desktop Commander remains available as a generic local MCP tool, but skill execution does not depend on it.
- Secrets are never included in prompts, chat history, normal logs, or audit records.
- Existing Markdown-only skills still activate and behave as instruction-only skills.
- At least two non-Google example skills work as generic manifest-backed fixtures. Google Calendar may also work as an additional fixture, but no core runtime code special-cases it.

## Migration Strategy

1. Ship manifest parsing, generic installation, and readiness with no execution changes.
2. Sync ready capabilities as opt-in client tools while keeping `activate_skill` unchanged.
3. Add permission evaluation before enabling any executable dispatch path.
4. Enable execution for safe dry-run/read-only capabilities first.
5. Enable mutation capabilities after permission and HITL enforcement is in place.
6. Add provider-neutral fixtures first; optionally migrate one real skill, such as Google Calendar, as an extra reference fixture.
7. Document the manifest and installation contracts so users can add future skills without code changes.

## Risk Notes

- Arbitrary shell skills are the highest-risk path. Keep them disabled unless explicitly permissioned.
- Dependency installation can become unsafe if automatic installs run without review. The first production version should detect missing dependencies and provide repair hints instead of auto-installing packages.
- Skill bundle installation can become unsafe if archives are extracted naively. Reject path traversal, symlinks that escape the profile skill root, and duplicate active names.
- Long-running local tools need cancellation and timeout handling through the existing runtime bridge.
- Local profile secrets must be encrypted or protected with OS facilities before storing long-lived credentials.
- The server should treat sidecar-published capabilities as untrusted metadata and continue validating session, catalog version, and tool instance ids.

---

## Implementation Progress

Executed via subagent-driven development (controller = Opus, implementers/reviewers = sonnet). Controller reviews + commits (subagents never commit, per project delegation rules). Each task: implement → controller verifies pytest → commit → independent task review → fix loop → record.

| Task | Status | Commit(s) | Notes |
|------|--------|-----------|-------|
| 1. Manifest models | ✅ done | `05b6a28` (base `22360f9`) | 48 tests. Review found 2 Important (gitignore over-reach, unenforced non-empty input_schema); both fixed. |
| 2. Load manifests during scan | ✅ done | `166fa4b` (base `05b6a28`) | 11 registry tests. Review Approved; 1 Important (embedded-path leak in redactor) + 2 Minors fixed proactively before Task 4 relies on it. |
| 3. Readiness checks | ✅ done | `7f5289b` (base `166fa4b`) | New skill_runtime/ package (manager + minimal secret store); 16 tests. Review found 1 Important (dep-name parser skipped whitespace-padded reqs, masking missing deps); fixed by switching to packaging.Requirement. |
| 4. Bundle installation | ✅ done | `af416e1` (base `23b4c00`) | install.py + shared/skills/errors.py + /skills install/uninstall/installed API; 13 tests (+1 platform-skip). Implementer hit a transient 529 mid-task (resumed). Review Needs-fixes→fixed: 3 Important (symlink rejection [plan-required], disabled-reinstall replace semantics, UNSAFE_BUNDLE_PATH test) + DRY helper. NOTE: user committed `beb1fdb`/`23b4c00` to this branch concurrently — no file overlap. |
| 5. Capability tools | ✅ done | `4e28e75` (base `af416e1`) | Ready skills' capabilities sync as client tools (manager.capability_catalog_entries + runtime_bridge merge + client_runtime_tools/client_tool_catalog origin widening); 8 catalog tests + coexistence search test. Review Approved (3 Minors; applied logger.exception). runtime_bridge has 8 pre-existing baseline ruff errors (deferred to final lint cleanup). |
| 6. Permission evaluator | ✅ done | `b06d312` (base `4e28e75`) | Pure pre-exec permission evaluation (permissions.py); 27 tests. Review found 2 CRITICAL over-grants (empty/root fs path-prefix opened whole FS; mutation gate bypassable via granted token/`*`) — both fixed + regression-tested; re-review confirmed Resolved. |

## Design Decisions Log

### Cross-cutting
- **`errors.py` location:** `shared/skills/errors.py` (per File Structure section), not the `client_backend/services/skill_runtime/errors.py` alternative in Task 7 — so both server-side (`app/ai`) and client_backend can import normalized codes.
- **Google Calendar migration (Task 12) is optional.** The two provider-neutral fixtures (python + binary) are the required deliverable; gcal only if cheap.

### Task 1 — Manifest models
- **`.gitignore` bug fixed (in scope, enabling).** A bare `skills` pattern was ignoring the *importable* `shared/skills/` package (its `front_matter.py` is imported by both server and client but was untracked — a latent fresh-clone break) and would ignore future `tests/fixtures/skills/`. Fixed by anchoring the pattern to `/skills/` (top-level personal bundles only). Previously-untracked `shared/skills/front_matter.py` + `__init__.py` are now committed. First attempt used broad `!/shared/skills/**` negations — review caught that this also re-tracked `.env`/`*.db`/`*.log` inside the subtree; replaced with the anchored form.
- **`json_output: bool` on `SkillCapabilityExecutionSpec`.** Part of the execution contract now (not speculative) because Task 7 parses JSON stdout only when the capability declares JSON output. Keeps the manifest schema stable across tasks.
- **`input_schema` must be a non-empty dict; `argv` MAY be empty.** The brief's "non-empty input_schema/execution" is enforced for `input_schema` (validator) and for `execution` (its required `argv` field means `{}` fails). `argv: []` is intentionally allowed — a `python_module`/`binary` capability may invoke its entrypoint with no positional args.
- **Reserved vs unknown runtime types** produce distinct error messages (`shell`/`node_package`/`mcp_server` → "reserved for a future slice"; anything else → "unknown"). Validation surfaces only as `pydantic.ValidationError`; no custom error-code system yet (that is Task 7 / `shared/skills/errors.py`).
- **Minor review items deferred to final triage:** two near-identical non-empty-string description validators (2 call sites, not extracted).

### Task 2 — Load manifests during scan
- **Strict scope:** Task 2 only *attaches* the manifest and emits a coarse summary from its presence/validity. `execution.status` is exactly `instruction_only` / `invalid` / `manifest_present`. NO readiness (import/`which`/dep/secret) checks — those are Task 3; NO install logic — Task 4.
- **Invalid `skill.json` is never fatal.** A missing manifest → instruction-only (unchanged). A malformed/invalid one → skill still loads instruction-only with `manifest_error` set. The `.exists()` probe + read + parse all sit inside one try/except that only sets `manifest_error`, so even an `OSError` from a broken symlink can't drop an otherwise-valid skill.
- **Sync privacy boundary hardened.** `to_sync_dict()` (server-synced) never emits `manifest_path` or any absolute path. `install_metadata` (reserved, populated by Task 4) is surfaced only through a redacting `_install_summary`: an allow-list of safe keys (`installed`, `source_hash`, `bundle_name`, `source`) AND `_looks_like_absolute_path` which detects POSIX/Windows-drive/UNC absolute paths — including a path *embedded* in a larger string (checks whitespace-split tokens), cross-OS. `to_dict()` (device-local) may include `manifest_path`.
- **`install_metadata` declared but not populated here** — a stable, redacted hook so Task 4 inherits a safe-by-default surface.

### Task 3 — Readiness checks
- **New package `client_backend/services/skill_runtime/`.** `secrets.py` = env-only `SkillSecretStore` (`get`/`has`), explicitly deferring encrypted profile storage + redaction + API to Task 10 (which extends, not replaces, it). `manager.py` = `SkillRuntimeManager.evaluate_readiness(manifest, manifest_error)` → `SkillReadiness`.
- **Readiness states:** `invalid` (manifest_error set), `instruction_only` (no manifest, no error), else aggregate `ready`/`not_ready` over five signals: unsupported runtime (defensive only — load_manifest already rejects), binary command present + on PATH (`shutil.which`), Python deps present, binary-missing-command, required secrets present. Repair hints are structured dicts; `REPAIR_HINT_TYPES` includes `permission_required` (shape reserved for Task 6, never emitted here).
- **Dependency check is presence-only, not version-match** (avoids false negatives from workable-but-mismatched versions). Uses `packaging.requirements.Requirement(req).name` (already a pinned dep) — robustly handles extras, markers, and whitespace, and returns None (→ skip, don't false-fail) only for genuinely unparseable strings. (Initial hand-rolled regex skipped whitespace-padded requirements → could mask a missing dep; review caught it.)
- **`node`/`system` deps are NOT availability-checked in this slice** (python only), per scope.

### Task 4 — Generic bundle installation
- **`shared/skills/errors.py` pulled forward** (File Structure lists it; Task 7 will reuse). Full normalized error-code set + `SkillRuntimeError(code, message, repair)` + `error_payload()`. Task 4 is the first task that needs normalized codes (install failures).
- **Directory install only; zip archives deferred** (plan marks them optional). Reduces the extraction attack surface; directory install satisfies every Task-4 test.
- **Security controls:** rejects missing SKILL.md, malformed front matter (present but no name), invalid manifest, duplicate ACTIVE skill name (disabled does not block), install/uninstall targets escaping the profile root (`is_under_root` before every copy/rmtree), and **symlinks in the bundle** (`os.walk(followlinks=False)` + reject; the plan's Risk Notes require this — copytree also uses `symlinks=True` as TOCTOU defense). Install metadata never stores secrets.
- **Disabled-reinstall = replace:** reinstalling over a DISABLED same-named bundle removes the stale install dir first (guarded by `is_under_root`), so the reinstall is discoverable and scan-dedup never arbitrates between two same-named bundles. Known limitation: a disabled duplicate coming from a CONFIGURED root (not profile-installed) can still shadow a profile install via first-match-wins dedup — pre-existing behavior, noted for final triage.
- **`get_installed_skills_root(user_id)` in `core/paths.py`** is the single source of truth for the install location (installer writes, registry scans) — prevents drift.
- **`pyproject.toml` ruff:** added `flake8-bugbear.extend-immutable-calls` for FastAPI DI markers (Depends/Query/…), clearing pre-existing B008 false positives repo-wide (no behavior change). The API's `Depends`-in-defaults is idiomatic FastAPI.
- **Concurrent user commits:** while Task 4 was implemented, the user committed `beb1fdb` (tool-execution-policy design doc) and `23b4c00` (attachments/ai_sdk feature) to this same branch. No overlap with skill-runtime files. Per-task review bases now use each task commit's actual parent, not the previous task commit.

### Task 5 — Sync skill capabilities as client tools
- **`SkillRuntimeManager.capability_catalog_entries(skill, manifest, readiness)`** emits one client-tool catalog entry per capability, but ONLY when `readiness.status == "ready"` — the model is never offered a capability it can't execute. server_name = `skill_<name with - → _>`; qualified_id = `skill::<skill>::<capability>` (original name preserved).
- **`runtime_bridge._build_tool_catalog`** appends ready-skill entries (via `_collect_skill_capability_tools`, a static method) into the `tools` list BEFORE the tool_instance_id stamping loop, so skill tools get the same session/catalog-version/tool_instance_id validation as MCP tools. The collection is wrapped in try/except (logged via `logger.exception`) so a skill-runtime hiccup can never break MCP tool sync.
- **Origin model:** added `TOOL_ORIGIN_CLIENT_SKILL = "client_skill"`. Model-facing `tool_origin` = client_skill for skills; the RAW catalog origin (`"skill"`) is preserved as `metadata["catalog_origin"]` so approval/audit/search can distinguish skill tools from MCP tools. `client_runtime_tools` + `client_tool_catalog` widened their origin gate from `{"mcp"}` to `{"mcp","skill"}`.
- **Execution NOT wired yet:** a `skill::...` tool call will fail at dispatch until Task 8 routes it — expected between-task state, committed in sequence.
- **runtime_bridge pre-existing lint (8 errors: 6 E402 from imports split around `_make_tool_instance_id`, 2 E501)** left for a final lint-cleanup commit; the new SkillRuntimeManager import went in the clean top block (0 new errors).

### Task 6 — Permission evaluator
- **Pure static evaluation** (`permissions.py`): no IO/execution/HITL/secrets. `SkillPermissionPolicy(granted, allow_mutation)` with `deny_all()`/`allow_all()`; `SkillPermissionEvaluator.evaluate(capability, runtime) -> PermissionDecision`.
- **Code semantics:** `PERMISSION_DENIED` = hard/ungrantable (reserved `shell` runtime, checked first, unconditional even under allow_all). `PERMISSION_REQUIRED` = grant/approval could unblock (ungranted network/fs/process/domain token, or unapproved mutation with `requires_approval=True`).
- **Token matching:** blanket `*`; family wildcards `network:*`/`filesystem:read:*`/`filesystem:write:*`; filesystem path-prefix with a `/` boundary. `process:spawn` and domain labels are exact-only.
- **Two Critical over-grants caught in review + fixed:** (1) an empty or root-only fs path grant (`filesystem:read:` / `filesystem:read:/`) used to match the entire filesystem — now skipped (`if not prefix: continue`); use `filesystem:read:*` to grant everything. (2) mutation was satisfiable via the resource-grant set (a literal `"mutation"` token OR blanket `"*"`) — now gated SOLELY on `allow_mutation`. Both regression-tested; re-review confirmed Resolved.
- **Redaction:** `redact_arguments` keeps keys, replaces every value with `"<redacted>"`; `PermissionDecision.to_error_payload` routes args through it so no argument value (possible secret) ever reaches an error/prompt/log.
- Not yet wired into execution — Task 7 must call the evaluator before invoking any runtime and refuse on any non-allowed decision.
