# Portable Device-Local Skill Runtime Design

**Date:** 2026-07-13  
**Status:** Approved architecture; implementation pending written-spec review  
**Supersedes:** The assumption in `plans/skill_runtime_refactor.md` that every executable skill must provide `skill.json` and every binary must already exist on the sidecar's global `PATH`

## Problem

The current runtime discovers a normal Agent Skill from `SKILL.md`, but it only exposes executable tools when a colocated `skill.json` is present. A plain skill therefore activates as Markdown instructions. If those instructions name a CLI, the model searches for a generic process tool such as Desktop Commander and launches the command in that MCP server's inherited environment.

That behavior is not portable. A command that was installed in the skill author's development environment may not exist on the user's machine, in the sidecar interpreter, or on the Desktop Commander process's `PATH`. The Google Calendar example demonstrates this failure: the installed skill teaches the model the command name, but the bundle does not transport an installed console entry point or sufficient package metadata to create one.

The runtime needs to support ordinary `SKILL.md` skills with bundled scripts or commands, while retaining the stronger typed `skill.json` path. Execution must remain local to the exact client device selected for the chat turn.

## Goals

- Make a portable Agent Skill with `SKILL.md` and bundled executable assets callable without requiring `skill.json`.
- Invoke skill commands through the skill runtime, not through Desktop Commander or another general-purpose MCP shell tool.
- Give each installed skill a local execution environment whose command resolution does not depend on global `PATH` mutation.
- Support explicit local installation of a standard Python project into an isolated per-skill virtual environment.
- Preserve typed manifest capabilities as the preferred path for deterministic schemas, fine-grained permissions, secret declarations, and mutation classification.
- Bind discovery, tool publication, approval, dispatch, execution, secrets, and audit records to one user, device, runtime session, catalog version, and tool instance.
- Fail before model invocation when a bundle is not portable or its runtime cannot be prepared.
- Keep installation and execution provider-neutral; no calendar-specific or package-name-specific branches are allowed.

## Non-goals

- Inferring a missing executable, package source, secret, or dependency solely from prose in `SKILL.md`.
- Silently executing arbitrary setup hooks while scanning configured skill roots.
- Mutating the user's global Python installation or global `PATH`.
- Adding unrestricted shell-string execution. The first implementation accepts an argument vector only.
- Automatically bootstrapping Node, Rust, native package managers, containers, or multiple interdependent plugins in this slice.
- Full Claude Code or Codex plugin marketplace compatibility. This slice adopts their portable bundle conventions where they fit the existing one-skill installer.

## Compatibility Contract

### Supported bundle shapes

The registry and installer distinguish the directory containing `SKILL.md` from the bundle root that owns executable resources.

Direct skill:

```text
my-skill/
  SKILL.md
  skill.json              # optional typed capabilities
  bin/                    # optional bundled commands
  scripts/                # optional scripts
  pyproject.toml          # optional Python project
  requirements.lock      # optional locked Python dependencies
```

Nested single-skill bundle:

```text
my-skill/
  skills/
    SKILL.md              # accepted legacy single-skill layout
  bin/
  scripts/
  pyproject.toml
```

For a scanned root, the first directory below that root containing the discovered `SKILL.md` tree is the bundle root. For a profile installation, the copied install directory is authoritative. `SkillMetadata` stores both `skill_path` and `bundle_root`; runtime confinement uses `bundle_root`.

The profile installer accepts a source directory when it contains exactly one discoverable `SKILL.md`, either directly or below `skills/`. A source containing zero or multiple skills is rejected in this slice with a structured portability error. This avoids guessing how several skills share installation state.

### Portable executable assets

An instruction skill is command-capable when at least one of these is true:

- The bundle contains a platform-resolvable command under `bin/`.
- The bundle contains a supported Python script under `scripts/`.
- An explicit installation has successfully built a Python environment from `pyproject.toml` and that environment contains console entry points.

A ready `skill.json` remains executable through its typed capabilities, but it does not automatically receive the synthetic command capability unless the bundle also satisfies one of the portable-command conditions above.

A bare command mentioned only in Markdown is not considered installed. Activation reports the skill as `setup_required` instead of encouraging a generic shell fallback.

## Architecture

### 1. Bundle discovery

`LocalSkillsRegistry` passes the configured scan root into `_load_skill` and records a confined `bundle_root`. Installed bundle metadata supplies the authoritative bundle root and source hash. Registry serialization never sends the absolute root to the canonical server.

The execution summary gains:

```json
{
  "status": "ready",
  "mode": "portable_command",
  "manifest_present": false,
  "command_tool": true,
  "setup": {
    "status": "ready",
    "runtime_id": "sha256-prefix",
    "repair_hints": []
  }
}
```

Top-level readiness remains compatible with the existing values: `instruction_only`, `ready`, `not_ready`, or `invalid`. The nested setup status is one of `not_applicable`, `setup_required`, `setting_up`, `ready`, `failed`, or `stale`. A skill with no executable assets remains a valid instruction-only skill.

### 2. Local runtime preparation

A new `SkillEnvironmentManager` owns runtime preparation and inspection. Runtime files live under the active user's local sidecar profile:

```text
<profile>/<user>/skills/runtimes/<install-id>/<source-hash>/
  runtime.json
  venv/                    # when Python setup is needed
  setup.log                # redacted, size-capped
```

Runtime preparation occurs only as part of an explicit `/skills/install` or `/skills/setup` action. Merely scanning `CLIENT_SKILLS_ROOTS` never installs dependencies or executes build code.

For a Python bundle, preparation creates a virtual environment using the sidecar interpreter, installs the local bundle, performs a readiness probe, and atomically promotes the staged runtime only after all steps succeed. The installer uses non-interactive subprocess calls with bounded time and output. A failed update leaves the previous ready installation intact.

Locked dependency input is preferred. When `requirements.lock` is present, setup installs those dependencies first and installs the local project without separately resolving its dependencies. Unlocked or remote dependencies are surfaced in the install preview as `needs_review`; the explicit install/setup confirmation authorizes the disclosed downloads for that device only. Setup never writes to the global interpreter.

`runtime.json` records the source hash, interpreter identity, platform, resolved command names, setup timestamp, and environment format version. A source-hash or interpreter mismatch marks the environment stale and requires rebuild.

### 3. Skill-scoped command resolution

The child process receives a scoped `PATH` in this order:

1. `<bundle-root>/bin`
2. The per-skill virtual environment's `Scripts` directory on Windows or `bin` directory on POSIX
3. The sidecar's sanitized infrastructure `PATH`

The process environment also includes `SKILL_ROOT` and `SKILL_RUNTIME_ROOT`. These values are constructed by the sidecar and are not supplied by the model. The global process environment is never modified.

Command lookup resolves only an argv first element. A bare name resolves against the scoped `PATH`. A relative `scripts/<name>.py` value resolves inside `bundle_root` and is launched with the skill environment's Python interpreter. Native executable files under `bin/` are supported. Shell, batch, and PowerShell scripts are not auto-interpreted in this slice. No shell parses operators, redirects, substitutions, or pipelines. Bundle paths remain confined to `bundle_root`; workspace `cwd` values remain confined to configured workspace roots.

### 4. Installation API and confirmation

Runtime setup uses a hash-bound two-step API so the user can review downloads without a time-of-check/time-of-use gap:

- `POST /skills/install/preview` accepts `source_path` and returns the discovered skill, source hash, bundle shape, executable assets, dependency/setup plan, requested permissions, and whether confirmation is required.
- `POST /skills/install` accepts `source_path`, `expected_source_hash`, and `approve_setup`. It rejects a changed source hash and performs the disclosed setup only when approved.
- `POST /skills/{name}/setup` rebuilds an installed or configured-root skill after the same preview/hash confirmation contract.

The existing install request remains valid for instruction-only bundles and portable bundles that require no dependency setup. A request that would download or build without `approve_setup` returns `SKILL_SETUP_REQUIRED` and its preview rather than partially installing.

### 5. Synthetic command capability for plain skills

Every ready portable-command skill publishes one reserved client capability in addition to any manifest capabilities:

```text
qualified id: skill::<skill-name>::run_skill_command
model tool:   client__skill_<safe-name>__run_skill_command
```

Input schema:

```json
{
  "type": "object",
  "properties": {
    "argv": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 1
    },
    "cwd": {
      "type": "string",
      "enum": ["skill", "workspace"],
      "default": "workspace"
    }
  },
  "required": ["argv"],
  "additionalProperties": false
}
```

`run_skill_command` is reserved and rejected as a user manifest capability name to prevent collisions. It routes through `SkillExecutionEngine`, the same timeout/output/redaction/audit boundary used by typed capabilities.

Because a plain Markdown skill cannot reliably declare whether an arbitrary command mutates state, the synthetic command capability is treated as a mutation for approval purposes. Approval may later be cached for an exact skill, device, source hash, executable, and argv prefix. Typed manifest capabilities retain their existing per-capability mutation flag and avoid unnecessary approval for declared read-only operations.

### 6. Activation and model guidance

`activate_skill` returns the instructions plus a machine-generated runtime footer. For a ready plain skill, the footer names the exact skill command tool and tells the model to pass an argv array. It explicitly says not to use Desktop Commander or search for another shell executor for commands belonging to the skill.

For a non-ready skill, activation returns structured status and repair hints. The model must report the setup requirement instead of guessing `npx`, `pip`, or another package manager command.

The synthetic command tool is also indexed by `tool_search`, but its metadata identifies it as the execution tool for one specific skill, not as a general shell capability.

### 7. Secrets

Typed `skill.json` secrets continue to be resolved exactly as today.

For a plain skill, secrets are not inferred from Markdown. A user may create per-skill secret bindings through `GET/POST /skills/{name}/secrets`. Only secret names explicitly bound to that installed skill are injected into its command environment. Bindings and encrypted values remain in that machine's profile and are never synchronized to another device. The existing flat secret endpoint remains available for typed-manifest compatibility during migration.

If a command needs an undeclared or unbound environment variable, it fails as not configured. The runtime must not fall back to inheriting the sidecar's complete environment.

### 8. Device and session isolation

Installation state, environments, secret bindings, audit logs, and readiness checks live only in the local sidecar profile. The canonical server receives sanitized catalogs, not executable files or local paths.

Every skill tool binding carries:

- `user_id`
- `device_id`
- `session_id`
- `catalog_version`
- `tool_instance_id`
- skill source hash through the tool instance derivation

The canonical server verifies that the requested device belongs to the user and that the binding still matches the active session and catalog. The request is queued by `device_id`. The receiving sidecar repeats session, catalog, capability, source-hash, permission, and readiness checks before spawning anything.

With the same account connected from two machines, each machine publishes its own catalogs. A skill installed on Machine A is absent from Machine B. If both machines install the same named skill, their device/session/tool-instance bindings remain distinct. A new turn from Machine B cannot reuse Machine A's deferred binding or approval.

Mutation approval displays the target device name and the skill source hash prefix so the user can see where the action will run.

## Installation and Execution Flow

```text
Explicit install on one sidecar
  -> locate exactly one SKILL.md and its bundle root
  -> reject symlinks/path traversal and validate metadata
  -> compute source hash and inspect executable/dependency inputs
  -> return install preview when dependency setup needs confirmation
  -> copy bundle into a staged profile directory
  -> build staged per-skill environment if required
  -> run readiness probe
  -> atomically promote bundle + runtime metadata
  -> refresh that sidecar's skill and tool catalogs

Chat turn bound to that device
  -> model activates relevant skill
  -> activation returns instructions + runtime footer
  -> model calls the skill's run_skill_command tool with argv[]
  -> server validates user/device/session/catalog/tool instance
  -> approval gate runs for the legacy command capability
  -> sidecar repeats validation and resolves skill-scoped command/env
  -> child runs locally; output is bounded, redacted, normalized, audited
```

## Errors and Recovery

Add or standardize these repair outcomes:

- `SKILL_PORTABILITY_UNSUPPORTED`: the bundle has no executable assets or usable setup metadata for a command it expects.
- `SKILL_SETUP_REQUIRED`: executable assets exist but the local environment has not been prepared.
- `SKILL_SETUP_FAILED`: environment creation or readiness probing failed; include a redacted repair summary.
- `SKILL_RUNTIME_STALE`: source hash, interpreter, platform, or runtime format changed.
- `COMMAND_NOT_FOUND`: the requested argv executable is absent from the bundle, skill environment, and allowed infrastructure path.
- Existing permission, secret, timeout, output, JSON, and runtime codes remain unchanged.

Install/setup failures are atomic. Catalog publication excludes synthetic command tools until readiness is `ready`. A stale runtime never executes and provides a rebuild hint. Uninstall removes the installed bundle, its runtime environments, per-skill secret bindings, and cached approvals after confined-path validation.

## Testing Strategy

All production changes follow test-driven development. The first failing regression reproduces the reported behavior without using a provider-specific name:

1. Install a nested plain `SKILL.md` bundle with a bundled Python console command and no `skill.json`.
2. Ensure the global `PATH` does not contain that command.
3. Verify setup creates a per-skill environment and readiness becomes `ready`.
4. Verify the tool catalog exposes `skill::<name>::run_skill_command`.
5. Activate the skill and verify the runtime footer points to that tool rather than a generic process tool.
6. Dispatch argv through the runtime bridge and verify the bundled command executes successfully.

Additional required coverage:

- Direct and nested bundle-root discovery and confinement.
- Zero/multiple `SKILL.md` install rejection.
- Bundle `bin/` precedence without global `PATH` mutation.
- Python environment staging, atomic promotion, staleness, failure rollback, and uninstall cleanup.
- No dependency installation during configured-root scanning.
- Command argv validation, workspace confinement, timeout, output cap, redaction, and command-not-found repair.
- Plain-skill command approval and typed-capability mutation behavior.
- Per-skill secret injection with no cross-skill or cross-device leakage.
- Same user with Machine A and Machine B: catalog, deferred binding, approval, request queue, and execution remain on the selected device.
- Session rotation, catalog refresh, source update, and tool-instance mismatch reject before process spawn.
- Existing executable-manifest fixtures and instruction-only skills remain compatible.

## Documentation Changes

After implementation:

- Update `plans/skill_runtime_refactor.md` with a portable-command follow-up phase and completed verification commands.
- Update `docs/skill-runtime.md` with supported bundle layouts, explicit setup behavior, synthetic command tools, scoped `PATH`, per-device installation, and migration guidance.
- Update the main README installation examples to distinguish instruction-only, portable-command, and typed-manifest skills.
- Document that copying only `SKILL.md` or an inner package directory cannot transport an external CLI; the install source must contain the complete portable bundle.

## Acceptance Criteria

- A newly installed, properly packaged plain Agent Skill can run its bundled CLI on a machine where that CLI was never globally installed.
- The model calls the skill-specific command capability and does not route skill execution through Desktop Commander.
- No install or execution step mutates global `PATH` or the global Python environment.
- An incomplete bundle fails at install/readiness with actionable structured diagnostics rather than `npx`/package-manager guessing.
- Typed `skill.json` capabilities continue to work and remain the path for fine-grained schemas, secrets, and read/write classification.
- Two machines connected to the same account cannot reuse each other's skill catalog entries, approvals, sessions, secrets, or runtime environments.
- Focused skill-runtime tests and the full existing regression suite pass.
