# Portable Device-Local Skill Runtime Design

**Date:** 2026-07-13
**Status:** Approved; implementation authorized
**Contract:** Open Agent Skills-compatible `SKILL.md` bundles only

## Problem

The current sidecar can discover and activate a standard Agent Skill, but executable operations depend on a private machine-readable extension. A normal skill therefore teaches the model a command name without giving the sidecar a skill-owned way to resolve it. The model then searches for a generic process tool and runs the command in that tool's unrelated environment.

The Google Calendar failure demonstrates the result: activation succeeded, but Desktop Commander could not find the CLI because the executable existed only in the author's development environment. Trying `npx` could not help because the package was never published there.

## Goals

- Use `SKILL.md` as the only required skill definition and compatibility contract.
- Support the standard optional `bin/`, `scripts/`, and other bundled assets without private metadata.
- Give every command-capable skill one fixed, typed `run_skill_command` tool.
- Resolve commands only from the selected skill's bundle and device-local prepared runtime.
- Install Python projects into an isolated, per-user, per-device environment after explicit setup approval.
- Keep discovery side-effect free: scanning never installs packages or runs setup code.
- Bind catalog publication, approval, dispatch, execution, secrets, and audit records to the selected device and runtime session.
- Fail with structured setup or portability guidance instead of guessing a global package-manager command.

## Non-goals

- Inferring missing package source, dependencies, or secret names from prose.
- Running shell strings, redirects, pipelines, substitutions, or profile scripts.
- Mutating the sidecar's global interpreter or global `PATH`.
- Automatically installing dependencies while scanning configured roots.
- Supporting multi-skill plugin installation, Node, Rust, containers, or native package managers in this slice.
- Creating one model tool per CLI subcommand. The CLI remains responsible for its own command grammar.

## Skill Bundle Contract

A direct skill bundle has this shape:

```text
my-skill/
  SKILL.md               # required metadata and instructions
  bin/                   # optional bundled commands
  scripts/               # optional executable scripts
  pyproject.toml         # optional installable Python project
  requirements.lock      # optional locked dependency input
  references/            # optional documentation
  assets/                # optional resources
```

A distributable bundle may contain one nested skill:

```text
my-plugin/
  skills/
    my-skill/
      SKILL.md
  bin/
  scripts/
  pyproject.toml
```

The profile installer accepts exactly one discoverable `SKILL.md`. Zero or multiple skills are rejected because this installer creates one isolated runtime and one command namespace per bundle.

`SkillMetadata` records both the `SKILL.md` path and the bundle root. Installed bundles use the directory containing `install.json` as the authoritative root. Configured-root skills use their direct skill directory unless the configured root itself has plugin-level executable assets.

## Command Capability

A skill is command-capable when at least one condition is true:

- `bin/` contains a supported bundled executable;
- `scripts/` contains a supported Python script; or
- a prepared Python environment contains console entry points installed from `pyproject.toml`.

Every ready command-capable skill publishes exactly one tool:

```text
qualified id: skill::<skill-name>::run_skill_command
model tool:   client__skill_<safe-name>__run_skill_command
```

Its fixed schema is:

```json
{
  "type": "object",
  "properties": {
    "argv": {
      "type": "array",
      "items": {"type": "string"},
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

The capability is always classified as mutating because unstructured Markdown cannot safely classify arbitrary argv. The existing human-in-the-loop gate must approve each call before the sidecar resolves secrets or starts a process.

## Discovery and Readiness

Registry scanning computes a deterministic source hash, rejects unsafe bundle paths, discovers executable assets, and produces one of these states:

- `instruction_only`: valid skill with no bundled executable assets or Python project;
- `ready`: a bundled command/script is directly runnable, or a matching prepared runtime exists;
- `not_ready`: setup is required, failed, or stale;

Unsafe or structurally invalid bundles are rejected or omitted before catalog publication rather than represented as runnable entries.

The nested setup state is `not_applicable`, `setup_required`, `setting_up`, `ready`, `failed`, or `stale`. Catalog publication excludes the command tool unless readiness is `ready`.

Scanning never runs build code. A Python project found under a configured root remains `setup_required` until the user explicitly approves setup.

## Runtime Preparation

`SkillEnvironmentManager` stores runtimes under the active sidecar profile:

```text
<profile>/<server>/<user>/skills/runtimes/<skill>/<source-hash>/
  runtime.json
  venv/
  setup.log
```

Preparation creates a staged virtual environment with the sidecar interpreter, installs the local bundle non-interactively, probes installed console entry points, writes redacted and size-capped diagnostics, then atomically promotes the stage. Failure removes the stage and leaves any previous ready runtime untouched.

`runtime.json` records the source hash, interpreter, platform, environment format version, setup time, and discovered command names. A source, interpreter, platform, or format mismatch makes the runtime stale.

Bundles declaring Python dependencies or build requirements require a hash-bound preview and `approve_setup=true`. Setup never writes to the global interpreter.

## Installation API

- `POST /skills/install/preview` accepts `source_path` and returns the skill name, source hash, bundle shape, executable assets, Python setup inputs, and whether confirmation is required.
- `POST /skills/install` accepts `source_path`, optional `expected_source_hash`, and `approve_setup`. A changed hash is rejected before copying or building.
- `POST /skills/{name}/setup` accepts `expected_source_hash` and `approve_setup` and prepares an installed or configured-root skill.
- `POST /skills/uninstall` removes the installed bundle and its confined runtime and secret state.

Install copies the complete bundle to a staged profile directory, prepares its runtime when approved, writes install metadata at the bundle root, and atomically promotes it. A failed replacement does not delete the previous installed bundle.

## Skill-Scoped Resolution and Execution

The model supplies only argv and the logical cwd choice. It cannot supply environment paths.

For argv element zero, the sidecar accepts only:

- a bare command resolved from the skill's `bin/` directory or prepared runtime command directory; or
- a relative Python path under `scripts/`.

It never falls back to an arbitrary system executable. Python files are launched with the prepared environment's interpreter when present, otherwise the sidecar interpreter. Shell, batch, PowerShell, and command-string interpretation are rejected.

The child receives a scoped environment containing:

1. `<bundle-root>/bin`;
2. the prepared environment command directory;
3. a sanitized infrastructure `PATH` needed by the runtime;
4. sidecar-generated `SKILL_ROOT` and `SKILL_RUNTIME_ROOT` values; and
5. encrypted secrets explicitly bound to this skill.

The global environment is not modified. Output, timeout, redaction, normalized errors, and auditing remain enforced at the single subprocess boundary.

## Activation Guidance

`activate_skill` returns the original instructions followed by a generated runtime footer.

For a ready skill, the footer names its exact `run_skill_command` tool and instructs the model to pass argv without using Desktop Commander or searching for another shell executor.

For a non-ready skill, the footer reports structured setup or portability guidance and explicitly forbids guessing `npx`, `pip`, or another installer command.

## Secrets

Secrets are stored as encrypted, per-skill bindings in the active machine's user profile. `POST /skills/{name}/secrets` sets one binding; `GET /skills/{name}/secrets` returns configured names but never values. Execution injects only bindings belonging to the selected skill.

Secret state is not synchronized through the canonical server and is not inherited from another skill or device. The child never receives the sidecar's complete environment.

## Device and Session Isolation

Bundles, runtimes, secret bindings, and audit logs exist only in the local sidecar profile. The canonical server receives sanitized catalogs, not local paths or executable files.

Every advertised tool remains bound to `user_id`, `device_id`, `session_id`, `catalog_version`, and `tool_instance_id`. The catalog entry also carries the source hash. The server verifies ownership and binding before queueing by device; the receiving sidecar repeats catalog, session, source, readiness, approval, and command checks before spawning.

Two machines using one account publish separate catalogs. A skill installed on Machine A is absent from Machine B. If both install the same named skill, their runtime sessions, tool instances, secrets, approvals, and local environments remain distinct.

## Errors

- `SKILL_PORTABILITY_UNSUPPORTED`: instructions expect a command but the bundle transports no executable assets or setup source.
- `SKILL_SETUP_REQUIRED`: a Python project exists but setup has not been approved or completed.
- `SKILL_SETUP_FAILED`: preparation failed with redacted diagnostics.
- `SKILL_RUNTIME_STALE`: source, interpreter, platform, or runtime format changed.
- `COMMAND_NOT_FOUND`: argv zero is not a command owned by the selected skill.
- Existing permission, secret, timeout, output-limit, install, and runtime errors remain normalized.

## Acceptance Criteria

- A newly installed standard Agent Skill can run its bundled command on a device where that command was never globally installed.
- The model receives and calls the skill-specific command tool instead of Desktop Commander.
- No private per-skill execution manifest exists in the source tree, fixtures, runtime, API, or documentation.
- No install or execution mutates global `PATH` or the global Python environment.
- Incomplete bundles fail with actionable setup or portability diagnostics before process execution.
- All command calls pass through approval, confinement, timeout, redaction, and audit boundaries.
- Two devices on one account cannot reuse each other's catalog entries, approvals, sessions, secrets, runtimes, or executions.
- Focused runtime tests and the full regression suite pass.
