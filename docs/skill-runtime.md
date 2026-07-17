# Device-Local Skill Command Runtime

The sidecar implements the open Agent Skills directory contract: every skill has a `SKILL.md`, and executable skills transport their own commands or scripts inside the same bundle. Skills, runtimes, secrets, and audit logs remain local to the device that published them.

## Bundle format

```text
my-skill/
  SKILL.md               # required metadata and instructions
  bin/                   # optional bundled command launchers
  scripts/               # optional Python scripts
  pyproject.toml         # optional installable Python project
  requirements.lock      # optional locked dependencies
  references/            # optional documentation
  assets/                # optional resources
```

The installer also accepts a distribution containing exactly one nested `SKILL.md`, such as `skills/my-skill/SKILL.md`, while executable assets remain at the copied bundle root. A source with zero or multiple skills is rejected.

The front-matter `name` must be 1-64 lowercase letters, digits, or single hyphens, with no leading or trailing hyphen.

`CLIENT_SKILLS_ROOTS` scanning is read-only. It discovers assets and readiness but never executes setup code or installs dependencies.

## Readiness

- `instruction_only`: no bundled command, script, or Python project.
- `ready`: `bin/` or `scripts/` contains a directly runnable asset, or a matching prepared environment exists.
- `not_ready`: Python setup is required, failed, or stale.

Structurally unsafe bundles are rejected during installation or omitted during configured-root scanning; they are not published as runnable catalog entries.

Prepared Python environments live below the active user's sidecar profile under `skills/runtimes/<skill>/<source-hash>/`. Setup uses a staged virtual environment and atomically promotes it only after installation and console-command discovery succeed. It never modifies the global interpreter.

Use the hash-bound workflow for Python projects:

```http
POST /skills/install/preview
POST /skills/install
POST /skills/{name}/setup
```

`approve_setup=true` and the exact `expected_source_hash` returned by preview are required before executing project-controlled build code. Setup freshly rehashes the bundle, so a missing or changed hash is rejected.

For an already configured or installed skill, `GET /skills/{name}` returns its current `sourceHash` for the setup request.

## Model tool

Every ready executable skill publishes one tool:

```text
skill::<skill-name>::run_skill_command
```

Input:

```json
{
  "argv": ["bundled-command", "arg1", "arg2"],
  "cwd": "workspace"
}
```

`argv` must be a non-empty string array. `cwd` is either `workspace` or `skill`. No shell parses the arguments.

The sidecar resolves argv element zero only from:

1. the selected bundle's `bin/` directory;
2. the selected skill's prepared Python command directory; or
3. a relative `.py` path below that bundle's `scripts/` directory.

It does not fall back to an arbitrary executable on the machine. Bundle discovery publishes Python launchers, current-platform native executables, and executable extensionless POSIX commands; batch, command, PowerShell, shell, and non-executable files are omitted. Python files run with the prepared environment's interpreter when one exists, otherwise the sidecar interpreter.

The child receives a scoped `PATH`, `SKILL_ROOT`, `SKILL_RUNTIME_ROOT`, and only encrypted bindings belonging to the selected skill. The sidecar's complete environment is never inherited.

## Activation

`activate_skill` returns the skill instructions plus generated runtime guidance. The sidecar reports the internal command binding; the canonical activation layer resolves that binding against the current device catalog and appends the exact model-callable `client__...` tool name, including any collision suffix. It tells the model not to call the internal binding directly, use Desktop Commander, or search for another shell executor. A non-ready footer reports setup guidance and prohibits guessing `npx`, `pip`, or another installer.

The ready footer also reports the skill's configured secret binding names (never values) and explains the remediation path for missing credentials: the user adds the secret binding in the skill's secret settings, and the runtime injects it into the command environment on the next run. The model is told it cannot set environment variables itself and must never pass secret values as command arguments, set them through another tool, or repeat them in conversation. The skill command failure hint carries the same instruction, because bundled CLIs typically phrase credential errors as "set ENV_VAR" advice the model cannot follow.

## Approval, isolation, and audit

The fixed command tool is always marked mutating because free-form Markdown cannot safely classify arbitrary argv. It passes through the existing approval-policy gate before secret lookup or process creation; the default policy asks a human, while an explicit per-tool policy may preapprove it.

Catalog and dispatch remain bound to user, device, runtime session, catalog version, tool instance, and skill source hash. The canonical server verifies ownership and binding before queueing to the selected device; the sidecar repeats catalog and session validation before execution.

Two machines on the same account publish independent catalogs and use separate profile storage. Neither machine can reuse the other's runtime, secrets, approval, session, or tool instance.

Every attempted command writes one profile-local JSONL audit record containing the skill, fixed capability name, redacted arguments, result status, duration, device, and session. Raw command output and secret values are excluded.

Command confinement is not an operating-system sandbox. An approved skill command runs with the sidecar user's filesystem and network privileges, just as coding-agent commands do; install only bundles you trust. The runtime limits command selection, environment/secrets, device routing, time, and audit scope, but it does not virtualize the host. Command output is intentionally uncapped so the complete redacted terminal error can reach the agent.

## Secret API

```http
POST   /skills/{name}/secrets
GET    /skills/{name}/secrets
DELETE /skills/{name}/secrets/{secret_name}
```

The POST body is `{"name":"ACCESS_TOKEN","value":"..."}`. GET returns configured names only. Values are encrypted at rest and namespaced by skill.

## Errors

| Code | Meaning |
|---|---|
| `SKILL_INSTALL_INVALID` | The bundle source, structure, or preview hash is invalid. |
| `UNSAFE_BUNDLE_PATH` | A symlink or path escapes a confined root. |
| `SKILL_SETUP_REQUIRED` | Python setup needs explicit approval. |
| `SKILL_SETUP_FAILED` | Environment creation or installation failed. |
| `SKILL_RUNTIME_STALE` | Source, interpreter, platform, or runtime format changed. |
| `SKILL_NOT_READY` | The command tool cannot currently execute. |
| `PERMISSION_REQUIRED` | Human approval has not been supplied. |
| `COMMAND_NOT_FOUND` | argv zero is not owned by the selected skill. |
| `INVALID_ARGUMENTS` | argv or cwd failed the fixed schema. |
| `EXECUTION_TIMEOUT` | The command exceeded its bounded timeout. |
| `RUNTIME_ERROR` | The owned command returned non-zero or failed unexpectedly; the message carries the complete redacted terminal output (stderr, or stdout when stderr is empty). |

## Complete example

The tracked `tests/fixtures/skills/google_calendar/` bundle contains `SKILL.md` and `bin/cli-anything-google-calendar.py`. Its end-to-end test invokes `cli-anything-google-calendar --json ...` with an empty global `PATH`. Copying only the Markdown file or an inner Python package is insufficient; installation must receive the complete bundle.
