# Production Sidecar Skill ZIP Installation Design

**Date:** 2026-07-31

**Status:** Approved for planning

**Owners:** Sidecar backend, Streamlit client, AI SDK frontend
**Frontend contract:** [`plans/SKILL_INSTALLATION_FE_CONTRACT.md`](../../../plans/SKILL_INSTALLATION_FE_CONTRACT.md)

## Problem

The device sidecar can already preview and install a complete Agent Skill bundle
from a directory path. That contract is suitable for trusted local automation,
but a browser cannot safely provide a usable local directory path. The frontend
needs a drag-and-drop ZIP workflow.

The existing read routes also use an initialized in-memory registry unless the
caller explicitly invokes `POST /skills/reload`. This makes list and detail
responses vulnerable to stale local state. Installation and catalog
synchronization are currently coupled closely enough that a synchronization
failure after a successful filesystem commit can make the API appear to report
an installation failure.

Adding executable archive upload increases the consequences of archive
traversal, decompression bombs, weak local authentication, concurrent
installation races, partial failure, and ambiguous frontend retries. The new
workflow must address those concerns without removing the existing path-based
API.

## Goals

- Let a user drag and drop one ZIP containing exactly one Agent Skill bundle.
- Preserve preview and explicit setup approval before project-controlled build
  code can execute.
- Reuse the existing validated, hash-bound, atomic directory installer.
- Keep existing path-based installation routes for trusted local tooling.
- Make new installs conflict-safe and updates explicit and hash-guarded.
- Protect the sidecar from hostile or malformed archives and storage
  exhaustion.
- Isolate staged uploads and operations by server profile and authenticated
  user.
- Make installation retry-safe across client timeouts and sidecar restarts.
- Separate a committed local mutation from downstream catalog-sync status.
- Return fresh, generation-tagged catalog data consistently.
- Provide one complete frontend contract for Streamlit and the AI SDK frontend.
- Verify that a newly installed skill reaches device-bound AI SDK chat without
  leaking to another device.

## Non-goals

- Installing TAR, GZIP, 7z, RAR, or multi-skill plugin archives.
- Downloading a bundle from a URL.
- Treating archive validation or malware scanning as an OS sandbox.
- Allowing the browser to submit an arbitrary local `sourcePath`.
- Automatically approving Python build or dependency installation.
- Replacing a skill discovered from a configured read-only root.
- Changing the canonical AI SDK chat stream protocol.
- Synchronizing skill bundle bytes, secrets, runtimes, or local paths to the
  canonical server.

## Existing Behavior to Preserve

- `POST /skills/install/preview` accepts a local directory path.
- `POST /skills/install` installs a local directory path.
- The exact source hash is required before Python setup may run.
- Installed bundles, runtimes, secrets, and audit data remain device-local.
- A bundle contains exactly one discoverable `SKILL.md`.
- Ready executable skills publish
  `skill::<skill-name>::run_skill_command`.
- `/skills/*` routes require an authenticated sidecar session.
- The sidecar exposes the same management router under `/skills` and
  compatibility alias `/api/skills`.

## Architecture

Archive ingestion, installation, and catalog projection remain separate units.

```text
multipart ZIP
    |
    v
SkillUploadService
  - stream and quota enforcement
  - archive preflight
  - confined extraction
  - persisted upload receipt
    |
    v
SkillBundleInstaller.preview(extracted_root)
    |
    v
user confirmation
    |
    v
SkillInstallationService
  - persisted operation state
  - per-user/per-skill lock
  - source/update hash checks
    |
    v
SkillBundleInstaller.install(extracted_root)
    |
    v
SkillCatalogService
  - refresh
  - generation
  - serialization
  - best-effort runtime sync
```

### SkillUploadService

`SkillUploadService` owns temporary upload bytes, archive validation,
extraction, expiry, quotas, cancellation, and upload receipts. It never prepares
a runtime or runs bundle code.

Staged data lives below the active profile:

```text
<profile>/<server>/<user>/skills/uploads/<upload-id>/
  upload.json
  bundle.zip
  extracted/
```

The upload ID is cryptographically random. Client filenames are display
metadata only and are never used as filesystem names.

### SkillInstallationService

`SkillInstallationService` owns the asynchronous installation-operation state
machine, request idempotency, conflict policy, locking, and crash recovery. It
delegates bundle validation, copying, and runtime preparation to the existing
`SkillBundleInstaller`.

Operation receipts live below:

```text
<profile>/<server>/<user>/skills/operations/<operation-id>.json
```

The operation receipt contains normalized metadata and the terminal result or
error. It never contains archive bytes, Markdown content, dependency output,
secrets, or staging paths.

### SkillCatalogService

`SkillCatalogService` becomes the single entry point for registry refresh,
frontend serialization, persistent catalog generation, and runtime-bridge
synchronization. API routes must not independently compose these steps.

The service distinguishes:

- local catalog refresh and committed catalog generation;
- runtime catalog synchronization to the canonical server.

A local mutation remains successful if the subsequent runtime synchronization
fails. The response exposes `catalogSyncStatus: "pending"` and synchronization
is retried.

## API

The detailed wire contract is in
[`plans/SKILL_INSTALLATION_FE_CONTRACT.md`](../../../plans/SKILL_INSTALLATION_FE_CONTRACT.md).

### Create and Preview an Upload

```http
POST /skills/uploads
Content-Type: multipart/form-data

file=<one .zip file>
```

The route streams and validates the archive, extracts it into a confined
staging directory, runs the existing side-effect-free installer preview, and
returns `201 Created` with:

- `uploadId`;
- `state: "staged"`;
- `createdAt` and `expiresAt`;
- archive and expanded byte counts;
- file count;
- source hash;
- bundle name, shape, executable assets, and setup preview.

No setup code runs during this request.

### Start Installation

```http
POST /skills/uploads/{uploadId}/install
Content-Type: application/json
```

Request fields:

- `expectedSourceHash`: must equal the staged preview hash;
- `approveSetup`: explicit Python setup approval;
- `replaceSourceHash`: omitted for a new install; required for an update and
  must equal the currently installed profile bundle hash.

The route returns `202 Accepted` with an operation resource. It does not keep
the browser request open while Python setup runs.

New installation returns `409 Conflict` if the name already resolves from any
root. Update is allowed only when the conflicting skill is a profile-installed
bundle and `replaceSourceHash` matches. A configured-root skill cannot be
overwritten even when disabled.

### Read or Cancel Installation

```http
GET    /skills/installations/{operationId}
DELETE /skills/installations/{operationId}
```

Operation states are:

- `pending`;
- `running`;
- `succeeded`;
- `failed`;
- `cancelled`.

Cancellation succeeds only before the atomic commit boundary. After commit,
the operation completes normally so local state is never reported ambiguously.

### Cancel a Staged Upload

```http
DELETE /skills/uploads/{uploadId}
```

Deletion is idempotent for the owning user. An upload cannot be removed while
its installation is running.

### Catalog Routes

- `GET /skills` performs a bounded freshness check and returns a complete
  catalog snapshot.
- `GET /skills/{name}` performs the same freshness check before lookup.
- `POST /skills/reload` forces a full rescan, attempts runtime synchronization,
  and returns the same complete catalog shape.
- Install, update, uninstall, setup, and toggle return the catalog snapshot
  produced by their committed mutation or operation result.

Every catalog response contains:

- `deviceId`;
- `catalogGeneration`;
- `catalogSyncStatus`;
- `skills`;
- `totalCount`;
- `enabledCount`.

Catalog generation is persisted per user and increments only when the
frontend-visible serialized catalog changes.

## Archive Security

The implementation must not call `ZipFile.extractall`.

### Upload and Expansion Limits

Defaults are configurable:

- 25 MiB uploaded bytes;
- 100 MiB total expanded bytes;
- 50 MiB expanded bytes per file;
- 2,000 entries;
- bounded path depth and portable relative path length;
- bounded outstanding uploads and total staged storage per user;
- bounded compression ratio.

`Content-Length` may reject a request early but is not authoritative. Limits
are enforced while streaming the upload and each expanded member.

### Preflight

All entries are checked before the first extracted file is written. Reject:

- a non-ZIP payload or unsupported ZIP feature;
- encrypted members;
- unsupported compression methods;
- malformed or truncated metadata;
- absolute, drive-letter, or UNC paths;
- empty, dot, `..`, NUL-containing, or traversal path components;
- backslash path ambiguity;
- excessive path depth or length;
- trailing dots or spaces;
- Windows reserved device names;
- duplicate or Unicode/case-normalized colliding paths;
- symlinks, devices, FIFOs, sockets, or other non-regular entry types;
- declared counts or sizes above configured limits.

The extractor writes one member at a time under a freshly created random root,
checks containment before opening the destination, uses exclusive file
creation, re-enforces expanded size limits while copying, and treats CRC or
short-read failures as invalid archives.

The extracted tree is passed through the existing bundle hash implementation,
which rejects links and non-regular files and detects source changes. The
installer rehashes during installation, preserving its existing time-of-check
and time-of-use defense.

### Trust Boundary

A valid skill remains executable code chosen by the user. Preview and install
screens must state that approved setup and later skill commands execute with
the local sidecar user's host privileges. Archive validation prevents archive
filesystem attacks and resource exhaustion; it does not establish publisher
trust or sandbox the installed code.

An optional malware-scanner integration may reject a staged archive before
preview. Its result supplements, but does not replace, structural validation
or explicit approval.

## Authentication and Isolation

All skill-management routes require the sidecar local session dependency.

The current upstream-token compatibility path must be hardened:

- an unverified JWT subject may be used only as a profile lookup hint;
- after restoration, the supplied bearer must exactly match the stored active
  upstream access token using constant-time comparison;
- a matching subject alone never authorizes a request.

The sidecar remains loopback-bound by default. Non-loopback binding requires an
explicit unsafe-network configuration. Production CORS uses configured
frontend origins rather than `*`.

Upload and operation lookup re-resolve the active authenticated user. Unknown,
expired, foreign-user, and foreign-server IDs return the same `404` response.
The active user is checked again when an operation starts.

Every server identifier, user identifier, device identifier, and profile
subdirectory component is validated before path construction.

## Concurrency and Idempotency

The workflow uses:

- an in-process async single-flight lock for catalog refresh;
- a cross-process profile lock for filesystem mutations;
- a per-user/per-skill mutation lock for install, update, setup, and uninstall;
- a per-upload lock for state transitions.

Locks have bounded acquisition time and return a normalized retryable error
rather than waiting forever.

Starting installation creates and persists the operation before background
work begins. The upload records the normalized install-request fingerprint.

- Repeating the same start request returns the existing operation.
- Reusing the upload with different parameters returns `409 Conflict`.
- A terminal success retains a small receipt until expiry so a client can
  recover after a response timeout.
- Archive contents are removed immediately after successful commit.

An update verifies `replaceSourceHash` while holding the skill mutation lock.
The old bundle and runtime remain authoritative until the new bundle and
runtime are fully prepared. Atomic promotion then switches the bundle. Cleanup
of the old bundle is retryable after commit.

## Persistence and Recovery

Upload, operation, and catalog metadata use versioned JSON written with a
temporary file and atomic replacement. Profile permissions are restricted
where supported by the OS.

Startup recovery:

- deletes expired staged uploads;
- removes abandoned extraction and installer stages;
- restores or cleans recoverable backups;
- marks an interrupted pre-commit operation failed and retryable;
- reconciles a post-commit operation from installed metadata;
- retries pending catalog synchronization;
- does not delete a known-good previous bundle because a replacement failed.

Cleanup also runs opportunistically during upload and installation operations.
Quotas count both live and expired-but-not-yet-cleaned data, preventing cleanup
lag from bypassing storage limits.

## Errors

New routes use the standard response envelope for success and failure:

```json
{
  "success": false,
  "code": "SKILL_ARCHIVE_TOO_LARGE",
  "message": "The expanded skill bundle exceeds 100 MiB.",
  "data": null,
  "error": {
    "uploadId": "optional-safe-id",
    "retryable": false
  }
}
```

Unknown exception text, local paths, subprocess commands, dependency logs,
archive contents, and secrets never appear in the wire response.

Status mapping:

- `400`: malformed archive, invalid bundle, or invalid state transition;
- `401`: missing or invalid sidecar session;
- `404`: unknown, expired, or foreign upload/operation;
- `409`: name conflict, stale source hash, consumed upload, or commit-boundary
  cancellation;
- `413`: uploaded or expanded resource limit;
- `415`: unsupported media/archive format;
- `422`: request-schema validation;
- `423`: bounded lock contention;
- `507`: insufficient local storage.

Existing path-install error compatibility remains documented. New schemas
accept both camelCase and existing snake_case names, reject unknown fields,
and serialize camelCase.

## Catalog Consistency

Catalog serialization is deterministic. The generation input includes all
frontend-visible fields that can affect behavior, including source hash,
enabled state, readiness, setup state, and installation origin.

`GET /skills` performs a freshness check with a short TTL. Concurrent callers
share one scan. `POST /skills/reload` bypasses the TTL.

When local state changes:

1. commit filesystem/state mutation;
2. refresh the local registry;
3. persist a new generation if the projection changed;
4. return/record the committed local result;
5. synchronize the runtime catalog;
6. record `synced` or `pending`.

A runtime-sync failure after step 3 does not roll back or misreport the local
operation. Reconnect, explicit reload, and a retry task may converge pending
state.

## Client Integration

### Streamlit

The existing JSON-only API helper gains a separate multipart upload helper or
an explicit `files`/`form` mode that never sends `json` simultaneously.

The Skills panel adds:

- ZIP selection and upload;
- preview summary;
- executable-code warning;
- setup approval control when required;
- explicit update confirmation with the current hash;
- operation polling with bounded backoff;
- cancel and retry;
- catalog refresh from the terminal operation result.

Session state stores only upload IDs, operation IDs, and safe preview metadata.
It does not retain archive bytes after upload. Cancel, logout, and selecting a
new archive attempt best-effort staged-upload cleanup.

### AI SDK Frontend

Skill management remains ordinary sidecar HTTP, not an AI SDK UI Message
Stream. The frontend uses the same management contract as Streamlit.

Query caches are scoped by `deviceId` and `catalogGeneration`. A successful
operation invalidates:

- skill list and affected detail;
- device tool catalog/capability views;
- HITL settings that group client-skill tools.

The frontend shows a committed local success with pending catalog sync as
"Installed locally; connecting skill to chat", not as failure.

AI SDK chat continues to send `deviceId`. Once the runtime catalog is synced,
the next turn can resolve the new skill. No chat-stream event schema changes.

## Observability and Audit

Local audit events cover upload accepted/rejected, preview, install/update
requested, operation transitions, cancellation, cleanup, setup, catalog sync,
and uninstall.

Events may record:

- upload and operation IDs;
- skill name and source hashes;
- compressed/expanded bytes and file count;
- normalized outcome/error code;
- duration and sync status.

They do not record archive contents, Markdown, local paths, secret values, or
raw dependency output.

Structured metrics/logging cover rejection categories, extraction time, setup
time, lock contention, recovery, cleanup, staged storage, and catalog-sync
backlog.

## Verification

### Archive Unit Tests

- valid direct and nested ZIPs;
- non-ZIP, truncated ZIP, invalid CRC, encrypted entry, unsupported
  compression;
- forward- and backslash traversal, absolute, drive, and UNC paths;
- duplicate and case/Unicode-colliding paths;
- symlink and non-regular Unix-mode entries;
- reserved Windows names, trailing dots/spaces, excessive depth/length;
- upload, entry, total expansion, count, ratio, quota, and disk-space limits;
- zero and multiple `SKILL.md` files;
- cleanup after every rejection point.

### API and Isolation Tests

- authentication and strict schemas;
- camelCase and snake_case request compatibility;
- user/server ownership and indistinguishable foreign-ID `404`;
- upload preview never invokes setup;
- start-operation idempotency and parameter mismatch;
- new-install conflict and explicit hash-guarded update;
- configured-root conflict even when disabled;
- polling, cancellation, expiry, and terminal receipts.

### Concurrency and Fault Injection

- two installs/updates of the same name;
- stale `replaceSourceHash`;
- uninstall or setup during replacement;
- upload deletion during install;
- concurrent list/reload/mutation calls;
- lock timeout;
- failure and restart before/after copy, runtime preparation, promotion,
  metadata commit, catalog refresh, and runtime synchronization.

### Client and Chat Integration

- Streamlit sends authenticated multipart without JSON and polls operations;
- Streamlit clears safe session state and renders normalized failures;
- `/skills` and `/api/skills` remain equivalent;
- AI SDK frontend examples validate against the wire contract;
- install a real fixture ZIP, synchronize its device catalog, and resolve it
  in a device-bound AI SDK chat turn;
- another device and a turn without `deviceId` cannot resolve the skill;
- existing Streamlit, skill runtime, runtime bridge, AI SDK HTTP/stream, lint,
  and full regression suites pass.

## Acceptance Criteria

- A browser uploads one ZIP once, previews it, approves setup if required, and
  observes installation through a retry-safe operation.
- No archive can write outside its staging root or exceed configured upload,
  expansion, entry, path, or quota limits.
- A new install never silently replaces an existing skill.
- An update requires the currently installed source hash and cannot replace a
  configured-root skill.
- A failed replacement leaves the previous skill and runtime usable.
- A committed mutation remains reported as committed when catalog sync fails.
- List, detail, reload, and mutation responses use one deterministic catalog
  projection with persisted generation.
- Foreign users/devices cannot discover or consume staged uploads,
  operations, bundles, runtimes, secrets, or receipts.
- Streamlit and the AI SDK frontend follow the same documented sidecar
  contract.
- A newly installed skill becomes available only to chat turns bound to the
  originating device after catalog synchronization.
