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

The installer also accepts a distribution containing exactly one nested `SKILL.md`, such as `skills/my-skill/SKILL.md`, while executable assets remain at the copied bundle root.

An uploaded archive may instead carry a whole library: every folder that directly contains a `SKILL.md` is one skill, and the set installs together. The single-skill case deliberately keeps the archive root as its bundle root, because that is where a nested distribution puts its `bin/`; only a library gives each skill its own folder as a root.

## Progressive disclosure

A skill keeps `SKILL.md` short and points at companion documents for the parts that only sometimes apply. Activation lists every readable file in the bundle by relative path, and `read_skill_resource` returns one on demand.

Reads are confined to the selected, enabled skill's bundle: the path is re-checked after the OS resolves it, links are refused rather than followed, only regular files are read, the size is capped, and content must decode as UTF-8. Binary assets ship with the skill and can be used by its commands, but never enter a prompt as text.

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

## Archive upload and installation lifecycle

A browser installs a skill by uploading a ZIP; the two-step shape is what keeps
uploaded code from running before anyone approves it.

**Staging** (`POST /skills/uploads`) validates the container, extracts it under
confinement, and reads its metadata. Nothing is executed. The archive must be a
real ZIP by central-directory parse -- never by extension or client-supplied MIME
type -- and every member is checked before a byte is written: traversal segments,
absolute POSIX/Windows/UNC names, reserved device names, trailing dot or space
components, paths beyond the configured depth and length, duplicates that collide
after casefolding and NFC normalization, encrypted members, unsupported
compression, device/socket/pipe entries, and every size, count, and ratio limit.
Symbolic links are skipped rather than rejected -- source archives commonly carry
one, the bundle hasher refuses links anyway, and extracting one is the only
dangerous option -- and the count is reported as ``archive.skippedLinkCount``.

Members stream into a temporary sibling that is promoted only after all of them
succeed, so a rejected archive leaves nothing behind.

One redundant wrapper directory is stripped: zipping a folder produces
`my-skill/SKILL.md`, and the bundle root is what publishes `bin/` and `scripts/`.
A root with several entries is left alone.

Staged uploads are scoped to one user, expire (30 minutes by default), and are
bounded per profile by outstanding count, total bytes, attempt rate, and
available disk space. An unknown, expired, or foreign upload id returns the same
404, so an id cannot be probed.

**Installation** (`POST /skills/uploads/{uploadId}/install`) requires the
`expectedSourceHash` the user previewed, `approveSetup` for a Python project, and
`replaceSourceHash` to overwrite an existing skill. Replacement is permitted only
for bundles under the profile's installed root; a skill from a configured root is
never rewritten. The work runs asynchronously against a persisted receipt:

`validating` -> `waitingForLock` -> `copying` -> `preparingRuntime` ->
`committing` -> `refreshingCatalog` -> `syncingCatalog`

Cancellation is honored until the atomic promotion begins and reports
`SKILL_OPERATION_COMMITTED` afterwards. A process killed mid-install is
reconciled on the next start by comparing the operation's source hash against
what is actually installed, so an interrupted operation resolves to the outcome
that really happened rather than to its last recorded state.

Publishing the catalog to the canonical server is best-effort and reported as
`catalogSyncStatus`; a committed local install with a failed sync is a success
with `pending`, never a failure.

## Errors

| Code | Meaning |
|---|---|
| `SKILL_INSTALL_INVALID` | The bundle source, structure, or preview hash is invalid. |
| `SKILL_SOURCE_CHANGED` | The uploaded or installed hash is stale; re-preview before retrying. |
| `SKILL_CONFIGURED_ROOT_CONFLICT` | The colliding skill lives in a configured root the sidecar does not manage. Published to clients as `SKILL_INSTALL_CONFLICT`. |
| `SKILL_ARCHIVE_INVALID` | The ZIP is malformed, encrypted, corrupt, or uses unsupported compression. |
| `SKILL_ARCHIVE_PATH_UNSAFE` | A member path escapes the bundle, collides on this filesystem, or is not portable. |
| `SKILL_ARCHIVE_TOO_LARGE` | An upload, expansion, per-file, or compression-ratio limit was exceeded. |
| `SKILL_ARCHIVE_TOO_MANY_FILES` | The archive holds more entries than allowed. |
| `SKILL_ARCHIVE_TYPE_UNSUPPORTED` | Only a single `.zip` archive is accepted. |
| `SKILL_UPLOAD_NOT_FOUND` | The upload is unknown, expired, or owned by another profile. |
| `SKILL_UPLOAD_STATE_INVALID` | The upload cannot make the requested transition. |
| `SKILL_UPLOAD_CONSUMED` | The upload already has a different installation request. |
| `SKILL_UPLOAD_QUOTA_EXCEEDED` | Outstanding uploads, stored bytes, or attempt rate exceeded. |
| `SKILL_OPERATION_NOT_FOUND` | The installation is unknown, expired, or owned by another profile. |
| `SKILL_OPERATION_COMMITTED` | Too late to cancel; poll for the result. |
| `SKILL_INSTALL_LOCKED` | Another mutation holds this skill's lock. Retryable. |
| `SKILL_STORAGE_INSUFFICIENT` | Not enough free local disk space. Retryable. |
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
