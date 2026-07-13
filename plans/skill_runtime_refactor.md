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

- [x] Implement `SkillExecutionEngine.execute(qualified_tool_id, arguments, context)`.
- [x] Validate capability exists and current readiness is `ready`.
- [x] Validate arguments against capability `input_schema` (jsonschema).
- [x] Call the permission evaluator before resolving secrets, rendering arguments, or spawning a process.
- [x] Render argument placeholders only into argv lists, never into a shell string.
- [x] Implement `python_module`, `python_script`, and `binary` runtimes first.
- [x] Apply timeout (clamped) and output size limits.
- [x] Parse JSON stdout when the capability declares JSON output.
- [x] Return normalized success and error envelopes.
- [x] Add tests for success, invalid arguments, command not found, timeout, output too large, and non-JSON output (+ permission block, missing secret, redaction, scoped env, path escape).
- [x] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_execution_engine.py -q` → 16 passed.

### Task 8: Route Skill Tool Dispatch

**Files:**
- Modify: `client_backend/services/runtime_bridge.py`
- Test: `tests/client_backend/test_runtime_bridge.py` or create `tests/client_backend/test_skill_dispatch.py`

- [x] Keep `client_skill::activate` behavior unchanged for Markdown activation.
- [x] Route `qualified_tool_id` values that start with `skill::` to `SkillExecutionEngine`.
- [x] Keep existing session, catalog version, and tool instance validation.
- [x] Return runtime errors through the existing `ToolDispatchResult` error path (SkillRuntimeError code + repair preserved).
- [x] Add tests for valid skill dispatch, stale session rejection, catalog mismatch rejection, and missing capability rejection.
- [x] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_dispatch.py -q` → 6 passed (+7 runtime_bridge regression).

### Task 9: Add HITL Approval Flow

**Files:**
- Modify: `client_backend/services/runtime_bridge.py`
- Modify: `app/services/message_service.py` if skill permissions need the existing HITL interrupt path
- Test: `tests/client_backend/test_skill_hitl.py`

- [x] Treat `mutation: true` capabilities as approval-required unless the permission policy already grants them.
- [x] Redact sensitive arguments from permission prompts and logs.
- [x] Integrate with the existing HITL interrupt path so approved skill mutations resume through the same session, catalog version, and tool instance validation.
- [x] Add tests for pending mutation approval, denied mutation approval, approved mutation execution, stale session rejection after approval, and catalog mismatch rejection after approval.
- [x] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_hitl.py -q` → 23 passed (+72-test HITL/tool/skill regression).

### Task 10: Add Secret Store

**Files:**
- Modify: `client_backend/services/skill_runtime/secrets.py`
- Modify: `client_backend/api/skills.py`
- Test: `tests/client_backend/test_skill_secrets.py`

- [x] Implement environment-backed secret lookup.
- [x] Implement encrypted local profile-backed secret storage using the existing local profile root.
- [x] Add redaction helpers for logs and returned errors.
- [x] Add local API endpoints for listing required secret names and setting a secret value.
- [x] Add tests for env lookup, stored lookup, missing required secret, and redaction.
- [x] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_secrets.py -q` → 13 passed.

### Task 11: Add Audit Trail

**Files:**
- Create: `client_backend/services/skill_runtime/audit.py`
- Modify: `client_backend/services/skill_runtime/execution.py`
- Test: `tests/client_backend/test_skill_audit.py`

- [x] Write one audit JSONL record per execution under the current profile.
- [x] Include user id, device id, session id, skill name, capability name, qualified id, redacted arguments, status, duration, and error code.
- [x] Keep raw stdout/stderr out of audit records by default.
- [x] Add tests proving secret values are not stored (incl. special-character secrets).
- [x] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_audit.py -q` → 8 passed.

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

- [x] Add at least two provider-neutral executable skill fixtures that prove the runtime is universal: one `python_module` or `python_script` fixture, and one `binary` fixture that uses an installed command available in CI. (echo-python python_script + binary-probe binary using `python`.)
- [x] Add manifests for those fixtures without mentioning Google Calendar or any provider-specific field.
- [x] Add dry-run fixture tests proving the skill capability path works without live credentials.
- [~] If Google Calendar remains in the repo, add its manifest... — SKIPPED (optional per plan; kept scope tight; core runtime is provider-neutral without it).
- [x] Add a regression test that scans all fixture manifests and fails if core runtime code contains example-specific skill name checks. (Scans skill_runtime/*.py + shared/skills; implementer proved it fails when a fixture name is hardcoded.)
- [x] Run: `.venv\Scripts\python.exe -m pytest tests/client_backend/test_skill_example_fixtures.py -q` → 5 passed.
- [~] Optional Google Calendar suite — N/A (gcal migration skipped).

### Task 13: Update Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/skill-runtime.md`

- [x] Document the difference between instruction-only skills and executable skill bundles.
- [x] Document the two generic installation paths: configured scan roots and profile-installed bundles.
- [x] Document `skill.json` schema with examples.
- [x] Document runtime types and security constraints.
- [x] Document dependency readiness and repair hints, including the rule that dependencies are not silently installed during scan.
- [x] Document secret setup and readiness states.
- [x] Document how sidecar users reload skills and inspect readiness.
- [x] Add a short troubleshooting section for command not found, missing secret, invalid manifest, and permission denied.

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
| 6. Permission evaluator | ✅ done | `812decb` (base `4e28e75`) | Pure pre-exec permission evaluation (permissions.py); 27 tests. Review found 2 CRITICAL over-grants (empty/root fs path-prefix opened whole FS; mutation gate bypassable via granted token/`*`) — both fixed + regression-tested; re-review confirmed Resolved. |
| 13. Documentation | ✅ done | `f51a6ac` (base `def0a30`) | New docs/skill-runtime.md (full guide) + README Skills System 'Executable skills' subsection. Docs-only → controller-written/reviewed. |
| 12. Example fixtures | ✅ done | `fe2e69e` (base `3060750`) | Two provider-neutral executable fixtures (echo-python python_script, binary-probe binary) + test proving they load/execute without creds + genericness regression (core has no hardcoded skill names). 5 tests. Test-only/additive → controller-reviewed (no per-task reviewer dispatch); final review covers it. Google Calendar migration SKIPPED (optional). |
| 11. Audit trail | ✅ done | `0335a8b` (base `476831f`) | audit.py SkillAuditWriter → profile audit.jsonl per execution; wired into execute() for ALL outcomes incl. permission-denied. 8 tests. Review found 1 CRITICAL secret leak: _redact_arguments redacted the json.dumps()'d text, so a secret with a quote/backslash/non-ASCII char round-tripped back unredacted; fixed by walking the RAW structure (redact string leaves before serializing) + special-char test; re-review confirmed Resolved. |
| 10. Secret store | ✅ done | `f5bbd99` (base `f4d71c3`) | SkillSecretStore extended with encrypted per-profile storage (profile-first, env-fallback) + set/delete/list + redact_secret_values; /skills secrets API. 13 tests. Review Approved; Important (weak sibling key file) fixed by delegating at-rest to core.security DPAPI/Fernet primitive (no key file); added corruption-tolerance tests. |
| 9. HITL approval for mutations | ✅ done | `1857958` (base `1b8789c`) | A: propagate `mutation` (manifest→catalog→spec→tool metadata→CallIdentity) + identity_requires_approval auto-gates as last resort. B: redact_sensitive_args in approval prompt. C: ToolDispatchRequest.mutation_approved → sidecar runs approved mutation (still re-validates session/catalog/instance). 23 tests + 72 regression. Review Approved; 2 Important fixed (unified mutation def via shared is_mutation(); denied-mutation test). |
| 8. Route skill dispatch | ✅ done | `e26b84c` (base `aea00bb`) | runtime_bridge routes `skill::` → SkillExecutionEngine; ok=false → error path (code+repair preserved); validation untouched (skill tools are normal catalog entries). 6 dispatch tests. Review Approved on production code; 1 Important test-only (vacuous missing-capability assertion) fixed by driving _handle_tool_request end-to-end. |
| 7. Execution engine | ✅ done | `498b165` (base `975bbe2`) | SkillExecutionEngine (execution.py) — subprocess.run via to_thread (Selector-loop-safe), no shell, permission-before-secrets/spawn, scoped env, secret redaction, timeout+output caps; 16 tests. Review found 2 CRITICAL secret leaks (full os.environ inheritance; `_redact` substring collision) + 2 Important (manager/store divergence, unbounded timeout) — all fixed + regression-tested; re-review Approved (fixed NaN-clamp Minor too). User committed 4 image-attachment commits concurrently (no overlap). |

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

### Task 7 — Execution engine
- **Selector-loop-safe subprocess:** the sidecar runs a Selector event loop on Windows (for psycopg), where `asyncio.create_subprocess_exec` raises NotImplementedError. So execution uses `subprocess.run(argv_list, ...)` wrapped in `asyncio.to_thread` — never `shell=True`, never a joined shell string. Argv placeholders render each into a discrete list element.
- **Ordered flow:** parse → lookup → readiness gate → jsonschema validation (+ schema defaults) → **permission check** → resolve secrets → render argv → build cmd by runtime → run → parse. Permission is strictly before secrets/render/spawn.
- **Runtimes:** `binary` (shutil.which), `python_script` (path confined to skill dir via is_under_root), `python_module` (`python -m module`, PYTHONPATH prepends skill dir).
- **Security fixes from review (2 CRITICAL + 2 Important):** (1) child env is built from an ALLOW-LIST (`_ENV_PASSTHROUGH_NAMES`) + this capability's injected secrets — NOT full `os.environ` (which would leak every other skill's env-backed secret); (2) `_redact` replaces secret values LONGEST-FIRST (a shorter secret that is a substring of a longer one previously leaked a fragment, non-deterministically via hash seed); (3) the readiness manager defaults to the engine's own secret store (no divergence); (4) `_clamp_timeout` bounds a caller timeout to (0, 300s], rejecting NaN/inf/non-numeric.
- **Secret redaction** applied to stdout, stderr, and any error derived from them (RUNTIME_ERROR/NON_JSON_OUTPUT). INVALID_ARGUMENTS messages use only the schema path/keyword, never the offending value.
- **First-slice policy default** = `SkillPermissionPolicy(granted={"*"}, allow_mutation=False)` (trust declared resource grants for user-installed skills; block mutations → PERMISSION_REQUIRED → Task 9 HITL; block shell). Revisit strictness later (final-triage item).
- **audit_id stays None** — Task 11 wires the audit writer.

### Task 8 — Route skill dispatch
- `runtime_bridge._execute_tool_request` routes `qualified_tool_id.startswith("skill::")` to a new `_execute_skill_capability` (before the MCP fallback, after the `client_skill::activate` branch — the two never collide since `client_skill::activate` doesn't start with `skill::`).
- **ok=false → error path (design choice):** the engine never raises, so `_execute_skill_capability` raises `SkillRuntimeError(code, message, repair)` on `ok=false`, matching how MCP dispatch surfaces errors (return-on-success / raise-on-error). `_build_runtime_error_context` was enhanced to preserve the normalized `code` (PERMISSION_REQUIRED/MISSING_SECRET/…) and put `repair` in the error detail, instead of the generic exception class name. On success it returns the `result` payload.
- **Validation untouched:** skill tools are ordinary catalog entries (Task 5), so `_validate_tool_request` (session / catalog-version / tool_instance_id / unknown-tool) already covers them — a stale session, catalog mismatch, or unknown capability is rejected before the engine is ever constructed. No special-casing added.
- HITL/mutation approval and audit are still deferred (Tasks 9/11); a blocked mutation surfaces as PERMISSION_REQUIRED for now.

### Task 9 — HITL approval for skill mutations (A+C)
- **Trigger (A):** `mutation` now flows manifest → `capability_catalog_entries` → `ClientRuntimeToolSpec` → `_build_tool` metadata → `resolve_call_identity` → `CallIdentity.mutation`. `identity_requires_approval` auto-gates a mutation as a LAST RESORT (after the per-tool/server/global precedence ladder, so an explicit policy entry can pre-approve it; the `master_enabled` HITL toggle still wins). Reuses 100% of the existing LangGraph interrupt/approve/deny/resume machinery — resume re-validation already covered `client_skill` provenance.
- **Mutation definition unified:** `SkillCapabilitySpec.is_mutation()` (`self.mutation or "mutation" in self.permissions`) is the single source of truth, used by both the permission evaluator (Task 6) and the catalog signal — so a capability declaring mutation via the permissions token still gates (fails closed → approvable, not permanently unexecutable).
- **Redaction (B):** `redact_sensitive_args` masks values under secret/token/password/api_key/credential/authorization-style keys in the approval prompt (`_build_tool_interrupt_request`), applied generically. Skill args are non-secret by design (secrets injected at exec), so this is a conservative defensive backstop that leaves MCP prompts unaffected. Shallow-only (documented tradeoff; skill args are flat).
- **Dispatch signal (C):** `ToolDispatchRequest.mutation_approved` (server-derived, not client-forgeable). `_dispatch_client_tool` sets it = `spec.mutation` (reaching dispatch means the gate approved/pre-granted). The sidecar `_execute_skill_capability` runs the mutation with an `allow_mutation=True` policy ONLY when the flag is set, and STILL re-validates session/catalog/tool_instance first — an approved-but-stale mutation is rejected. The sidecar remains the final enforcement authority (default policy blocks mutation without the flag).

### Task 10 — Encrypted secret store + API
- `SkillSecretStore` now checks encrypted per-profile storage first, then env — so a user can set a secret via the local API without an env var + restart, while env-only construction (readiness manager, execution engine, their tests) is unchanged (no active user id → env-only reads, raising writes).
- **At-rest protection delegated to `core.security.encrypt_local_secret`/`decrypt_local_secret`** (the review caught that a self-managed sibling key file adds ~no protection vs a same-user attacker). That primitive is OS-user-bound DPAPI on Windows (no key file to guard), managed-Fernet fallback elsewhere — the same one `upstream_auth` uses. One JSON envelope per profile (`<profile>/skills/secrets.json`). Reads tolerate missing/malformed/undecryptable/non-dict state → empty, never crash.
- `redact_secret_values(text, values)` — shared longest-first redaction for logs/errors.
- API: `POST /skills/secrets` (never echoes value; no-profile→400), `GET /skills/{name}/secrets` (names + required + configured booleans only, never values; ordered before `/{name}`).
- **Perf note (final triage):** `get()`/`has()` re-read+decrypt the profile file per lookup; a readiness check / execution with N secrets does N decrypt cycles. Consider per-call caching if it bites.

### Task 11 — Audit trail
- `SkillAuditWriter.write()` appends one JSON line per execution to `<profile>/skills/audit.jsonl`: timestamp, audit_id, user/device/session, skill, capability, qualified_id, `arguments_redacted`, status, duration_ms, error_code. Never includes stdout/stderr or a secret value. Wired into `SkillExecutionEngine.execute` for EVERY outcome (success, SkillRuntimeError, broad Exception, and the permission-denied early return — a blocked mutation/shell is a security event worth recording). `audit_id` (`skill-exec-<utc>-<hex>`) is generated once and matches the returned envelope.
- Best-effort: the whole write is swallowed on failure (disk/serialize/no-profile) so audit can never break execution; no active user → skip (no shared/global file).
- **CRITICAL secret leak caught in review + fixed:** `_redact_arguments` originally redacted the `json.dumps()`'d text — but `json.dumps` escapes quotes/backslashes/non-ASCII, so a secret containing any of those never matched and `json.loads` round-tripped the raw secret back into the record. Fixed to walk the RAW argument structure and redact string leaves (incl. nested) BEFORE serializing; regression test uses a `p@ss"w\ord/café`-style secret.
- **Minor (final triage):** the 4 near-identical `_audit.write` call sites in execute() could be one helper (drift risk); audit append is sync disk I/O on the loop (unlike the subprocess offload); a non-string arg leaf numerically equal to a string secret isn't redacted (unreachable today — secret_values are always string env secrets).
