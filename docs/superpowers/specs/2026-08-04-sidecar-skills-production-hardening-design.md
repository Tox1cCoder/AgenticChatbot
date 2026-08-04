# Sidecar and Skills Production Hardening Design

**Date:** 2026-08-04

**Status:** Approved for implementation planning

## Goal

Restore reliable sidecar startup on frontend developers' Windows machines and
bring the shipped Phase 1 modern-skills implementation up to its documented
security, correctness, recovery, and maintainability guarantees.

## Scope

This work covers five related areas:

1. Windows source-bundle bootstrap and launcher behavior.
2. Skill-resource confinement and metadata privacy.
3. Correct discovery and installation of nested single-skill archives.
4. Transactional installation and recovery for skill collections.
5. Focused removal of dead, deprecated, and duplicated code in the launcher and
   skills subsystem.

The existing HTTP response shapes, device binding, per-skill enablement,
hash-bound approval, encrypted secrets, and upload-only installation policy remain
compatible.

## Non-goals

- Implementing Phase 2 namespaced skill identity.
- Fetching or updating skills from remote URLs.
- Implementing catalog search or changing the prompt-injection threshold.
- Treating `allowed-tools` as a restrictive tool allowlist.
- Removing a published compatibility route or response field without evidence
  that no supported consumer still uses it.
- Refactoring unrelated application code.

## Verified Current Problems

### Launcher regression

The launcher emitted by both bundle builders has four bootstrap defects:

- It calls `Get-FileHash`, which is unavailable in the reported frontend
  environment even though the rest of the script can run there.
- It invokes `py -3.10`, which requests exactly Python 3.10. On a machine with
  only a newer compatible Python, the launcher fails despite its own "3.10+"
  requirement.
- It considers the virtual environment healthy when
  `.venv/Scripts/python.exe` merely exists. A partial environment without `pip`
  is reused forever.
- Native command failures are not checked at every boundary. Virtual-environment
  creation and the first pip command can fail before a later command obscures the
  useful error. The final sidecar exit code is not deliberately propagated.

The existing tests inspect generated source strings, so they pass without
exercising these behaviors.

### Phase 1 implementation gaps

- `discover_collection()` deliberately gives a nested single skill the archive
  root as its bundle root, but `SkillUploadService.skill_roots()` later assumes
  each returned root directly contains `SKILL.md`. Preview succeeds and install
  can fail.
- Collection installation commits skills one at a time. On a later failure,
  rollback calls `uninstall()` for earlier entries. That removes an updated skill
  instead of restoring its previous bundle and also removes name-bound runtime
  state and secrets.
- Operation recovery records only the first preview's hash. A process crash after
  the first collection member commits can therefore be reported as success while
  the remaining members were never installed.
- The resource listing excludes `install.json`, but the read path does not.
  Directly requesting it can expose installation metadata, including an absolute
  `source_path` for path-based installs.
- Resource-link policy is applied to the resolved final target rather than every
  original path component, so the implementation does not fully match its
  "refuse links" contract.
- The proposal says collection installation records a shared `collection_id`, but
  no such field exists in the codebase.

### Proposal inaccuracies

The official [Agent Skills specification](https://agentskills.io/specification)
requires non-empty bounded `name` and `description` fields, requires the name to
match the parent directory, and defines optional `license`, `compatibility`,
`metadata`, and experimental `allowed-tools` fields. The current proposal and
parser do not describe or validate that whole contract.

`allowed-tools` represents tools pre-approved to run. It is not a declaration
that every other tool must become unavailable, so Phase 3 must not describe it as
a restrictive allowlist without a separate product policy.

The system already supports hash-guarded replacement through the upload/install
flow. Gap G3 is therefore specifically the absence of persisted collection
provenance, installed version, and update discovery—not the total absence of an
update path.

## Architecture

### 1. Canonical bundle templates

Create one source directory for the four generated handoff files:

- `requirements-client.txt`
- `start-client-backend.ps1`
- `start-client-backend.bat`
- `README.client_backend.md`

Both the Python and PowerShell bundle builders copy these canonical files. They
continue to build the same directory and ZIP outputs, but no builder embeds a
second launcher, requirements manifest, or README.

Builder tests assert artifact behavior and artifact equality, not equality
between two duplicated source strings.

### 2. Launcher bootstrap contract

The PowerShell launcher supports Windows PowerShell 5.1 and current PowerShell 7
without relying on optional/autoloaded modules.

It will:

1. Resolve the bundle and configuration paths as today.
2. Compute SHA-256 with `System.Security.Cryptography.SHA256` and
   `System.IO.File`, avoiding `Get-FileHash`.
3. Discover a compatible host interpreter in this order:
   - `py -3`
   - `python`
   - `python3`
4. Execute a version probe and accept Python 3.10 or newer. A command that exists
   but is a Microsoft Store placeholder or an older Python is skipped with a
   diagnostic.
5. Validate an existing venv by executing its Python. If it is missing or broken,
   recreate the bundle-owned `.venv` using the selected host interpreter.
6. Probe `python -m pip --version`. If pip is missing, run
   `python -m ensurepip --upgrade`, then probe again. If the interpreter does not
   provide `ensurepip`, stop with an actionable message rather than repeatedly
   attempting `pip install`.
7. Compare an environment fingerprint containing a bootstrap schema version,
   venv Python major/minor version, and requirements SHA-256. A legacy marker or
   changed fingerprint triggers dependency installation.
8. Install requirements with `python -m pip --disable-pip-version-check -r ...`.
   Do not upgrade pip from the network during every bootstrap.
9. Check the exit code immediately after every native process. The marker is
   written atomically only after pip succeeds.
10. Start `client_backend`, capture its exit code before cleanup, restore the
    caller's location, and exit with the sidecar's code.

The launcher owns `.venv`; repair or recreation never targets a caller-supplied
directory. Configuration files are not deleted or overwritten during repair.

### 3. Resource access policy

`list_skill_resources()` and `read_skill_resource()` will use one shared policy
for paths and file types.

The policy will:

- reject empty and absolute paths and all paths escaping the selected bundle;
- reject `install.json` and excluded runtime/cache directories on both listing
  and reading;
- inspect every existing component from the bundle root with link/reparse-point
  checks before opening the file;
- accept only regular files at or below the configured byte limit;
- accept only UTF-8 text;
- read bytes once, enforce the size limit on the bytes actually read, and then
  decode, reducing metadata/read race exposure;
- cache the resource manifest by the registry's live source hash so activation
  does not repeatedly decode every companion file.

The activation text will call the manifest a discoverability list. A truncated
manifest does not become an authorization boundary: an exact referenced path can
still be read if it passes the same policy. This keeps large legitimate skills
usable while making exclusions unconditional.

### 4. Explicit discovered-skill model

Introduce a discovered-skill value with separate fields for:

- `bundle_root`: the directory whose assets belong to the skill;
- `skill_file`: the exact `SKILL.md` document;
- parsed name and metadata needed by preview/install.

A direct skill uses the same directory for `bundle_root` and
`skill_file.parent`. A nested single-skill archive keeps the archive root as its
bundle root and retains the nested document path. Each member of a multi-skill
collection uses its own skill directory for both.

Preview, upload receipt creation, installation, and re-discovery consume this
same model. No layer reconstructs identity by assuming `bundle_root / SKILL.md`.
The uploaded receipt persists safe bundle-relative document paths, so install
uses exactly the members the user previewed rather than rediscovering a mutable
shape from scratch.

### 5. Collection transaction coordinator

Collection installation becomes a logical transaction with durable recovery.
Physical files still require multiple renames, so correctness comes from staging,
retained backups, a shared lock, and a journal—not from claiming that several
filesystem paths can be changed in one OS operation.

#### Prepare

- Revalidate every previewed source hash and replacement hash.
- Prepare every bundle and any new hash-scoped runtime without changing the
  installed catalog.
- Record, for every member, its staged path, target path, prior bundle path and
  hash, action, and runtime cleanup information.
- If preparation fails, remove only new staged bundles/runtimes. Existing bundles,
  runtimes, enablement, and secrets remain untouched.

#### Commit

- Acquire one profile-wide skills-mutation lock shared by install, uninstall, and
  catalog refresh paths.
- Write a transaction journal before the first promotion.
- For each member, move an existing bundle to a retained backup and promote the
  staged bundle, recording progress after each durable step.
- Do not refresh the registry or publish a catalog until every member is promoted.
- Mark the journal `committed` before deleting backups.
- Refresh and publish one catalog generation for the completed set.
- Clean old backups and obsolete hash-scoped runtimes after the committed state is
  durable. Secret bindings remain name-bound and are preserved.

#### Failure and recovery

- A failure before the durable `committed` state moves the journal to
  `rolling_back`, removes promoted new targets, and restores every prior bundle.
- Recovery resumes rollback for any pre-commit journal, regardless of how many
  members had been promoted.
- Recovery finishes cleanup for a committed journal and verifies the complete set
  of expected installed hashes before marking the operation successful.
- Operation receipts store the transaction identifier and the complete expected
  member set. They no longer infer collection success from the first skill hash.
- If rollback cannot complete, the operation remains failed and retryable with a
  durable cleanup receipt; it is never reported as a partial success.

The same coordinator handles one-skill installs, eliminating different commit
semantics between direct and collection workflows.

### 6. Compatibility and lifecycle

- Existing installed bundles without collection metadata continue to scan.
- Existing plain-hash launcher markers are treated as stale and replaced after a
  successful verified install.
- Existing API aliases and the single-skill `preview` compatibility projection
  remain until the cleanup audit proves they have no supported consumers.
- The transaction identifier is operational recovery metadata, not the persistent
  collection identity proposed for Phase 2.
- Per-skill enable/disable state remains keyed by the current bare skill name in
  this phase. Namespacing remains necessary before colliding libraries can
  coexist.

## Error Handling and Diagnostics

- Launcher errors name the failing executable and phase, preserve the meaningful
  native output, and state the required remediation.
- No failed dependency installation writes a success marker.
- Resource failures use client-safe messages and never echo excluded metadata or
  outside paths.
- Transaction failures keep the original error as primary and log rollback
  failures separately with the transaction identifier.
- Recovery decisions are based on the full journal and installed hashes, never a
  timestamp or a single collection member.

## Cleanup Phase

Cleanup runs only after the new behavior is protected by passing tests.

Required removals and consolidation:

- Remove embedded `REQUIREMENTS_CONTENT`, `START_PS1_CONTENT`,
  `START_BAT_CONTENT`, and `README_CONTENT` copies from both builders.
- Remove tests that grep or split duplicated launcher source when behavioral tests
  cover the same contract.
- Remove unused `LocalSkillsRegistry._split_front_matter()` and
  `_extract_yaml_value()` wrappers if the final reference scan still shows no
  callers.
- Remove stale task-number comments and inaccurate claims about atomic promotion,
  rollback, and collection identity.
- Consolidate duplicate SKILL.md discovery and parsing into the discovered-skill
  model.
- Audit direct-path install routes and legacy compatibility projections using
  repository references, frontend contracts, and route tests. Remove only items
  proven unsupported; otherwise mark the retention reason and intended removal
  gate explicitly.
- Run a final dead-symbol and deprecated-reference search over the touched
  subsystem and remove any additional evidence-backed dead paths.

Cleanup must not combine unrelated style changes or repository-wide refactors
with this fix.

## Documentation Corrections

Update `docs/superpowers/plans/2026-08-03-modern-skills-architecture-proposal.md`
to:

- describe Phase 1 as shipped and subsequently hardened, with the hardening
  commits/tests cited;
- remove the false shared-`collection_id` statement unless persistent collection
  identity is actually implemented;
- refine G3 to persisted provenance/version and update discovery;
- keep G4 and G6 as open issues;
- expand G5 to the full current Agent Skills frontmatter/validation gap;
- describe `allowed-tools` as experimental pre-approval metadata;
- distinguish the operational transaction identifier from future namespaced
  collection identity;
- replace the unsupported `~50` threshold with a measurement requirement or label
  it explicitly as a hypothesis;
- record the transactional limitations that were fixed and the tests proving the
  final guarantees.

## Testing Strategy

All production changes follow red-green-refactor.

### Launcher behavior

Windows integration tests execute the generated launcher in temporary bundles and
cover:

- startup when `Get-FileHash` is unavailable;
- a machine with compatible Python newer than 3.10 but no exact 3.10;
- repair of a venv created with `--without-pip`;
- actionable failure when neither pip nor ensurepip can be provided;
- requirements-content changes independent of timestamps;
- no marker after a failed install;
- successful marker reuse;
- sidecar exit-code propagation;
- identical output from both builders using canonical templates.

Tests use an empty/local requirements fixture and a minimal temporary
`client_backend` module so they do not access the network.

### Skills behavior

- Direct reads of `install.json` and excluded paths fail.
- Linked files and linked intermediate directories fail.
- Listed resources satisfy the same size/text policy as reads.
- A nested single-skill ZIP previews and installs end to end with top-level assets.
- A fresh multi-skill install rolls back every promoted member after a later
  failure.
- A collection update restores every previous bundle after a later failure and
  preserves secrets and previous runtimes.
- Recovery rolls back each possible interrupted pre-commit point.
- Recovery completes committed cleanup only when the full expected hash set is
  installed.
- Catalog readers cannot observe or publish an intermediate transaction.

### Final verification

- Run the complete launcher/bundle test module.
- Run all client-backend skills, upload, operation, runtime bridge, catalog,
  device-isolation, and chat-integration tests.
- Run Ruff checks and formatting checks for every touched Python file.
- Build the bundle with both builders and compare their generated trees.
- Inspect and smoke-test the rebuilt `dist/client-backend-bundle.zip` from a clean
  extraction.
- Confirm `git status` contains no unexpected generated or user-file changes.

## Acceptance Criteria

- The reported frontend-machine failure is covered by behavioral tests and the
  sidecar starts from a clean or pip-less bundle-owned venv when compatible Python
  is installed.
- No optional PowerShell module is required by the launcher.
- Failed bootstrap commands never produce a success marker or a zero exit code.
- `install.json` cannot be read through the model-facing resource tool.
- Nested single skills install with the exact bundle shape that was previewed.
- Collection failure or pre-commit crash converges to the complete previous state;
  committed recovery requires the complete new state.
- Existing secrets and old runtimes survive failed updates.
- Both builders consume one canonical set of handoff files.
- The architecture proposal accurately distinguishes shipped behavior, remaining
  gaps, and future work.
- Relevant tests, lint, bundle comparison, and clean-extraction smoke tests pass.
