# Skill Runtime

The skill runtime lets a connected device (the client sidecar) expose **user-added skills** as typed, permissioned, auditable capabilities that the model can call — without the model constructing raw shell commands. Skills remain client-owned and are executed on the sidecar; the canonical server only orchestrates and dispatches to the active device session.

This document covers the executable skill runtime. For the plain Markdown/frontmatter loading that predates it, see the "Skills System" section of the main [README](../README.md).

---

## Instruction-only vs. executable skills

A skill is a directory containing at least a `SKILL.md`:

```text
skills/<skill-name>/
  SKILL.md          # human/model instructions (YAML front matter + markdown body)
  skill.json        # OPTIONAL machine-readable execution manifest
  runner.py         # runtime code (for python_script)
  requirements.txt  # optional
  README.md         # optional
```

- **Instruction-only skill** — `SKILL.md` with no `skill.json`. Behaves exactly as before: `activate_skill` loads the markdown so the model can reason with it. Nothing is executed.
- **Executable skill** — `SKILL.md` **plus** a valid `skill.json` manifest. Its declared capabilities become typed client tools the model can call directly (`skill::<skill>::<capability>`), executed by the sidecar's `SkillExecutionEngine`.

A missing or malformed `skill.json` never breaks loading — the skill degrades to instruction-only (with a `manifest_error` recorded).

---

## Two ways to add a skill

Both paths are generic and provider-neutral — no skill name, provider, package, or command is hardcoded anywhere in the runtime.

1. **Scanned roots.** The sidecar scans the directories in `CLIENT_SKILLS_ROOTS` for `SKILL.md` files, exactly as it always has. To develop against this repo's `skills/` folder, add its absolute path to the sidecar's `CLIENT_SKILLS_ROOTS`.
2. **Profile-installed bundles.** Install a local directory bundle through the sidecar's `/skills/install` API. It is validated, copied into a sidecar-controlled profile skill root (`<profile>/skills/installed/<safe-name>-<hash>/`), recorded with install metadata, and picked up by the normal scan — no environment-variable edits or restart required.

Dependencies are **never** installed as a side effect of scanning or installing. The runtime reports missing dependencies with repair hints; installing them is an explicit, separate, user-approved step.

---

## The `skill.json` manifest

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
  "dependencies": { "python": ["click>=8", "requests>=2"], "node": [], "system": [] },
  "secrets": [
    { "name": "EXAMPLE_CALENDAR_ACCESS_TOKEN", "required": true, "description": "OAuth access token." }
  ],
  "permissions": ["network:api.example.com", "calendar:read", "calendar:write"],
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
      "execution": { "argv": ["--json", "event", "list", "--time-min", "{time_min}", "--time-max", "{time_max}"] },
      "permissions": ["calendar:read"],
      "secrets": ["EXAMPLE_CALENDAR_ACCESS_TOKEN"],
      "mutation": false
    }
  ]
}
```

Validation (`shared/skills/manifest.py`) enforces:

- `schema_version` must be a supported version (`"1.0"`).
- `name` matches `^[a-zA-Z0-9][a-zA-Z0-9_-]*$`; each capability `name` matches `^[a-zA-Z][a-zA-Z0-9_]*$` (it becomes part of the `skill::<skill>::<capability>` id and a `client__…` tool name).
- Each capability has a non-empty `description`, a non-empty `input_schema` object, and an `execution` block. Capability names are unique within a manifest.
- `runtime.type` is one of the first-slice types below; reserved/unknown types are rejected at load time.

`execution.argv` is a list of template strings. `{placeholder}` tokens are substituted with the matching argument value, **each into its own argv element** — never concatenated into a shell string. Set `execution.json_output: true` when the capability prints a JSON document to stdout; the engine parses it into `result`.

---

## Runtime types

First-slice, supported:

| Type | How it runs |
|---|---|
| `python_module` | `python -m <module>` with the bundle dir on `PYTHONPATH`. |
| `python_script` | `python <bundle-dir>/<script>` (script path is confined to the bundle dir). |
| `binary` | an installed command resolved via `PATH` (`shutil.which(command)`). |

Reserved for future slices and **rejected today**: `node_package`, `mcp_server`, `shell`.

### Security constraints

- **Never a shell.** Every runtime runs an argv **list** via `subprocess.run` (wrapped in a worker thread for the sidecar's event loop). No value is ever interpolated into a shell string.
- **Permission check before anything happens.** The permission evaluator runs before secrets are resolved, arguments are rendered, or a process is spawned.
- **Scoped environment.** The child process receives only an allow-list of infrastructure environment variables plus the capability's own declared secrets — never the sidecar's full environment (which would leak other skills' secrets).
- **Path confinement.** A `python_script` path that escapes the bundle directory is refused; a bundle containing a symlink is rejected at install time.
- **Timeouts & output caps.** Each call has a timeout (default 30s, capped) and an output-size limit.
- **Secret redaction.** Resolved secret values are stripped from returned stdout/stderr, error messages, logs, and audit records.

---

## Readiness

Before the model can call a capability, the sidecar computes the skill's readiness (`SkillRuntimeManager.evaluate_readiness`):

| Status | Meaning |
|---|---|
| `instruction_only` | no `skill.json` — nothing to execute |
| `ready` | manifest valid, dependencies present, binary on PATH, required secrets configured |
| `not_ready` | one or more requirements unmet (see repair hints) |
| `invalid` | `skill.json` present but failed to parse/validate |

Only **ready** skills expose their capabilities as callable tools. Readiness checks are detection-only and provide structured **repair hints**:

- `install_dependency` — a declared Python dependency is not importable (dependencies are never auto-installed).
- `install_command` — a `binary` runtime's command is not on `PATH`.
- `configure_secret` — a required secret is not set.
- `unsupported_runtime` / `invalid_manifest` — manifest problems.

Python dependencies are **presence-checked** (via installed distribution metadata), not version-matched, to avoid false negatives.

---

## Secrets

Secrets are declared in the manifest (`secrets` / a capability's `secrets`) and resolved **only at execution time**, injected into the child process environment. They never appear in prompts, chat history, normal logs, or audit records.

`SkillSecretStore` resolves a secret from **encrypted per-profile storage first, then the process environment**. Profile storage is protected by the shared local-secret primitive (OS-user-bound DPAPI on Windows, managed Fernet elsewhere) — there is no separate key file to guard.

> **First-slice limitation.** Secrets live in a single flat namespace and the env fallback resolves any variable by name, so a manifest that declares a secret named after an existing process env var will receive it, and two skills declaring the same secret name share a value. This is a least-privilege gap, not a sandbox boundary — skills already run unsandboxed as the local user (they could read the environment or token files directly). Per-skill secret namespacing is a planned hardening.

Set and inspect secrets through the local API (values are never returned):

- `POST /skills/secrets` — body `{ "name": "...", "value": "..." }`.
- `GET /skills/{name}/secrets` — the skill's declared secret names, whether each is `required`, and a `configured` presence boolean (never the value).

---

## Permissions and mutation approval

Each capability declares `permissions` and a `mutation` flag. The sidecar enforces them before execution:

- Resource permission families: `network:<host>`, `filesystem:read:<path>`, `filesystem:write:<path>` (path-prefix grants with a `/` boundary), `process:spawn`, exact domain labels (e.g. `calendar:read`), and `mutation`.
- A denied permission returns a structured `PERMISSION_REQUIRED` (grantable) or `PERMISSION_DENIED` (hard, e.g. reserved `shell`) error with **redacted** arguments.
- `mutation: true` capabilities require **human approval**. They route through the existing HITL interrupt path: the turn pauses, the user approves or rejects (with sensitive argument values redacted from the prompt), and only an approved mutation dispatches — still re-validated for session, catalog version, and tool-instance id before it runs.

---

## Audit trail

Every capability execution writes one JSON line to `<profile>/skills/audit.jsonl`:

```json
{
  "timestamp": "2026-07-08T06:30:00+00:00",
  "audit_id": "skill-exec-20260708T063000-a1b2c3",
  "user_id": "...", "device_id": "...", "session_id": "...",
  "skill": "example-calendar", "capability": "event_list",
  "qualified_id": "skill::example-calendar::event_list",
  "arguments_redacted": { "time_min": "...", "time_max": "..." },
  "status": "ok", "duration_ms": 231, "error_code": null
}
```

Records never contain raw stdout/stderr or any secret value. Auditing is best-effort — a write failure never breaks execution.

---

## Managing skills (sidecar API)

All endpoints live on the sidecar (the server has none of its own):

| Endpoint | Purpose |
|---|---|
| `GET /skills` | list local skills |
| `GET /skills/{name}` | one skill's detail |
| `PATCH /skills/{name}/toggle?enabled=` | enable/disable a skill |
| `POST /skills/reload` | rescan configured + installed roots |
| `POST /skills/install` | install a local directory bundle (`{ "source_path": "..." }`) |
| `POST /skills/uninstall` | uninstall an installed bundle (`{ "name": "..." }`) |
| `GET /skills/installed` | list profile-installed bundles |
| `GET /skills/{name}/secrets` | declared secret names + `configured` booleans |
| `POST /skills/secrets` | set a secret value |

After an install, uninstall, or reload, the skill registry and the runtime tool catalog are refreshed, so newly-ready capabilities appear as tools without a restart.

---

## Standard error codes

`SKILL_INSTALL_INVALID`, `SKILL_INSTALL_CONFLICT`, `UNSAFE_BUNDLE_PATH`, `SKILL_MANIFEST_INVALID`, `SKILL_NOT_READY`, `CAPABILITY_NOT_FOUND`, `UNSUPPORTED_RUNTIME`, `MISSING_DEPENDENCY`, `MISSING_SECRET`, `PERMISSION_REQUIRED`, `PERMISSION_DENIED`, `COMMAND_NOT_FOUND`, `INVALID_ARGUMENTS`, `EXECUTION_TIMEOUT`, `OUTPUT_TOO_LARGE`, `NON_JSON_OUTPUT`, `REMOTE_API_ERROR`, `RUNTIME_ERROR` (defined in `shared/skills/errors.py`).

---

## Troubleshooting

| Symptom | Likely cause & fix |
|---|---|
| `COMMAND_NOT_FOUND` | A `binary` runtime's command is not on `PATH`. Install it (readiness reports it as `not_ready` with an `install_command` hint). |
| `MISSING_SECRET` | A required secret is not configured. Set it via `POST /skills/secrets`; check status via `GET /skills/{name}/secrets`. |
| `SKILL_MANIFEST_INVALID` / status `invalid` | `skill.json` failed validation (unknown runtime type, missing/empty required field, bad capability name). Fix the manifest; the skill loads instruction-only meanwhile. |
| `PERMISSION_DENIED` | The capability requires the reserved `shell` runtime or an explicitly denied permission — not grantable in this slice. |
| `PERMISSION_REQUIRED` | A `mutation` needs approval, or a resource permission is not granted under the current policy. Approve the mutation, or grant the permission. |
| `SKILL_NOT_READY` | Missing dependency/binary/secret at execution time. See the readiness repair hints via `GET /skills/{name}`. |
| Capability doesn't appear as a tool | The skill isn't `ready`, or the catalog hasn't refreshed — call `POST /skills/reload`. |

---

## Example fixtures

Two provider-neutral executable fixtures under `tests/fixtures/skills/` demonstrate the contract end-to-end:

- `echo_python/` — a `python_script` skill whose `echo` capability returns its `{message}` argument as JSON.
- `binary_probe/` — a `binary` skill that runs the `python` command with a `-c` script.

They install, validate, and execute with no live credentials — the same generic machinery a calendar, CAD, CRM, or file-utility skill would use.
