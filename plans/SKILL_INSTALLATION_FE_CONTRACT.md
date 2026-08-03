# Sidecar Skill Installation — Frontend Contract

This document is the frontend handoff for device-local skill discovery,
drag-and-drop ZIP installation, updates, setup, and catalog refresh. It
complements [`AI_SDK_FE_CONTRACT.md`](AI_SDK_FE_CONTRACT.md), which owns the
chat stream.

## Ownership and Base URL

Skills are owned by the **device sidecar**, not the canonical server.

Use the authenticated local sidecar base URL:

```text
http://127.0.0.1:8100
```

The preferred routes are `/skills/*`. The sidecar also exposes compatibility
aliases under `/api/skills/*`.

Every request requires:

```http
Authorization: Bearer <local-session-token>
```

Do not send skill ZIPs, upload IDs, local paths, or skill secrets to the
canonical server.

## Response Envelope

Success:

```json
{
  "success": true,
  "message": "Skill archive staged",
  "data": {},
  "error": null
}
```

Failure:

```json
{
  "success": false,
  "code": "SKILL_ARCHIVE_INVALID",
  "message": "The ZIP archive is malformed.",
  "data": null,
  "error": {
    "retryable": false
  }
}
```

Rules:

- Response keys and `data` fields are camelCase.
- New JSON request bodies use camelCase.
- The sidecar temporarily accepts documented snake_case request equivalents.
- Unknown request fields are rejected.
- Never display raw `error` content as trusted HTML.
- Upload and operation responses never expose staging/runtime filesystem paths,
  archive contents, setup logs, or secret values. Catalog responses retain the
  existing display-only `folderPath`.

## Endpoint Summary

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/skills` | Get the current device catalog. |
| `GET` | `/skills/{name}` | Get one skill and its instruction body. |
| `POST` | `/skills/reload` | Force rescan and runtime-catalog sync; returns the catalog. |
| `POST` | `/skills/uploads` | Upload, validate, extract, and preview one ZIP. |
| `DELETE` | `/skills/uploads/{uploadId}` | Cancel a staged upload. |
| `POST` | `/skills/uploads/{uploadId}/install` | Start a new install or guarded update. |
| `GET` | `/skills/installations/{operationId}` | Poll installation state. |
| `DELETE` | `/skills/installations/{operationId}` | Cancel before commit. |
| `PATCH` | `/skills/{name}/toggle?enabled=` | Enable or disable one skill. |
| `POST` | `/skills/{name}/setup` | Prepare/rebuild an installed Python runtime. |
| `POST` | `/skills/uninstall` | Remove a profile-installed skill. |
| `GET` | `/skills/installed` | List profile-installed bundle metadata. |
| `POST` | `/skills/{name}/secrets` | Set one write-only secret binding. |
| `GET` | `/skills/{name}/secrets` | List configured secret names. |
| `DELETE` | `/skills/{name}/secrets/{secretName}` | Remove one binding. |

The existing path-based `POST /skills/install/preview` and
`POST /skills/install` routes remain for trusted local tooling. Browser clients
must not use them.

## Catalog Shape

`GET /skills`, `POST /skills/reload`, and terminal mutation results share this
catalog shape:

```json
{
  "success": true,
  "message": "Skills retrieved",
  "data": {
    "deviceId": "device-uuid",
    "catalogGeneration": 42,
    "catalogSyncStatus": "synced",
    "skills": [
      {
        "name": "google-calendar",
        "description": "Manage Google Calendar.",
        "enabled": true,
        "folderPath": "C:\\Users\\…\\skills\\installed\\google-calendar-a1b2c3",
        "sourceHash": "64-hex-characters",
        "installSource": "profile",
        "commandCapable": true,
        "runtimeStatus": "ready",
        "setupStatus": "ready"
      }
    ],
    "totalCount": 1,
    "enabledCount": 1
  },
  "error": null
}
```

### Catalog fields

| Field | Meaning |
|---|---|
| `deviceId` | Device cache boundary. Never merge catalogs from different values. |
| `catalogGeneration` | Monotonic per-user value. A lower generation must not replace a higher cached generation for the same device. |
| `catalogSyncStatus` | `synced`, `pending`, or `disconnected`. |
| `installSource` | `profile` for installed bundles, `configured` for read-only configured roots. |
| `runtimeStatus` | `ready`, `not_ready`, or `instruction_only`. |
| `setupStatus` | `not_applicable`, `ready`, `setup_required`, `failed`, or `stale`. |
| `commandCapable` | `true` only when the runtime status is `ready`. |
| `sourceHash` | Current bundle hash; use it for setup and guarded update. |
| `folderPath` | Local display-only path. Never send it to the canonical server. |

`GET /skills/{name}` returns the same skill fields plus `content`.

## Upload a ZIP

```http
POST /skills/uploads
Authorization: Bearer <local-session-token>
Content-Type: multipart/form-data
```

Multipart fields:

| Field | Type | Rules |
|---|---|---|
| `file` | file | Exactly one `.zip`. Do not base64-encode it. |

Do not manually set the multipart boundary. Let `fetch`, Axios, or the HTTP
library set `Content-Type`.

Success is `201 Created`:

```json
{
  "success": true,
  "message": "Skill archive staged",
  "data": {
    "uploadId": "opaque-random-id",
    "state": "staged",
    "createdAt": "2026-07-31T10:00:00Z",
    "expiresAt": "2026-07-31T10:30:00Z",
    "archive": {
      "filename": "google-calendar.zip",
      "compressedBytes": 24576,
      "expandedBytes": 98304,
      "fileCount": 14,
      "skippedLinkCount": 0
    },
    "preview": {
      "name": "google-calendar",
      "sourceHash": "64-hex-characters",
      "bundleShape": "direct",
      "executableAssets": {
        "bin": ["google-calendar.py"],
        "scripts": [],
        "pythonProject": false
      },
      "setup": {
        "pythonProject": false,
        "dependencies": [],
        "buildRequirements": [],
        "declaredCommands": [],
        "dependencyLock": null,
        "confirmationRequired": false
      },
      "existingSkill": null
    },
    "operationId": null
  },
  "error": null
}
```

`archive.skippedLinkCount` counts symbolic links dropped during extraction rather
than materialized. Surface it when non-zero: a bundle that depended on a link is
otherwise silently incomplete.

`setup.dependencyLock` names the bundle's lock file when it ships one
(`requirements.lock`), otherwise `null`. `operationId` is `null` until an
installation claims this upload, then names it; a client that already tracks the
`202` response does not need to read it.

When a skill with the same name exists, `preview.existingSkill` is:

```json
{
  "name": "google-calendar",
  "sourceHash": "current-installed-hash",
  "installSource": "profile",
  "enabled": true,
  "replaceable": true
}
```

A configured-root collision reports `replaceable: false`. The UI must not show
an update action for it.

Uploading never executes setup code.

## Confirmation UI

Before starting installation, display:

- skill name;
- archive expanded size and file count;
- bundled command/script names;
- Python dependencies and build requirements when present;
- whether this is a new install or update;
- the executable-code trust warning below.

Required warning:

> Install only skills you trust. Approved setup and skill commands run locally
> with your user account's filesystem and network access. ZIP validation does
> not sandbox the installed code.

When `preview.setup.confirmationRequired` is `true`, require a deliberate setup
approval control. Do not pre-check it.

When `existingSkill.replaceable` is `true`, require a separate deliberate
“Update existing skill” confirmation. Do not infer update from the matching
name.

## Start a New Installation

```http
POST /skills/uploads/{uploadId}/install
Content-Type: application/json
```

```json
{
  "expectedSourceHash": "hash-from-preview",
  "approveSetup": false
}
```

Do not send `replaceSourceHash` for a new installation.

## Start a Guarded Update

```json
{
  "expectedSourceHash": "hash-from-preview",
  "approveSetup": true,
  "replaceSourceHash": "preview.existingSkill.sourceHash"
}
```

If the installed bundle changes after preview, the API returns `409`; upload or
reload again and present the new preview. Never silently retry with a new hash.

## Installation Start Response

Success is `202 Accepted`:

```json
{
  "success": true,
  "message": "Skill installation queued",
  "data": {
    "operationId": "opaque-operation-id",
    "uploadId": "opaque-upload-id",
    "state": "pending",
    "statusUrl": "/skills/installations/opaque-operation-id",
    "createdAt": "2026-07-31T10:01:00Z",
    "expiresAt": "2026-07-31T11:01:00Z"
  },
  "error": null
}
```

Starting the same upload again with the same normalized body returns the same
operation. Starting it with different fields returns `409`.

## Poll Installation

```http
GET /skills/installations/{operationId}
```

Pending/running:

```json
{
  "success": true,
  "message": "Skill installation running",
  "data": {
    "operationId": "opaque-operation-id",
    "uploadId": "opaque-upload-id",
    "state": "running",
    "phase": "preparingRuntime",
    "createdAt": "2026-07-31T10:01:00Z",
    "expiresAt": "2026-07-31T11:01:00Z",
    "startedAt": "2026-07-31T10:01:01Z",
    "finishedAt": null,
    "result": null,
    "failure": null
  },
  "error": null
}
```

`phase` values:

- `validating`;
- `waitingForLock`;
- `copying`;
- `preparingRuntime`;
- `committing`;
- `refreshingCatalog`;
- `syncingCatalog`.

Treat phase values as display hints. Ignore unknown future values.

Succeeded:

```json
{
  "success": true,
  "message": "Skill installed",
  "data": {
    "operationId": "opaque-operation-id",
    "uploadId": "opaque-upload-id",
    "state": "succeeded",
    "phase": "syncingCatalog",
    "createdAt": "2026-07-31T10:01:00Z",
    "expiresAt": "2026-07-31T11:01:00Z",
    "startedAt": "2026-07-31T10:01:01Z",
    "finishedAt": "2026-07-31T10:01:08Z",
    "result": {
      "action": "installed",
      "name": "google-calendar",
      "sourceHash": "64-hex-characters",
      "runtimeStatus": "ready",
      "catalog": {
        "deviceId": "device-uuid",
        "catalogGeneration": 43,
        "catalogSyncStatus": "synced",
        "skills": [],
        "totalCount": 1,
        "enabledCount": 1
      }
    },
    "failure": null
  },
  "error": null
}
```

For an update, `result.action` is `updated`.

Failed operation polling remains HTTP `200` because the operation resource was
retrieved successfully:

```json
{
  "success": true,
  "message": "Skill installation failed",
  "data": {
    "operationId": "opaque-operation-id",
    "state": "failed",
    "phase": "preparingRuntime",
    "expiresAt": "2026-07-31T11:01:00Z",
    "result": null,
    "failure": {
      "code": "SKILL_SETUP_FAILED",
      "message": "The skill runtime could not be prepared.",
      "retryable": true
    }
  },
  "error": null
}
```

Render `failure.message`. Branch on `failure.code`, not text.

### Polling policy

- Poll immediately once.
- Then use bounded backoff such as 500 ms, 1 s, 2 s, then 3 s.
- Stop on `succeeded`, `failed`, or `cancelled`.
- Stop when `expiresAt` passes.
- Pause or slow polling when the page is hidden.
- A network error does not mean installation failed; resume polling the same
  `statusUrl`.

## Cancellation

Cancel a staged upload:

```http
DELETE /skills/uploads/{uploadId}
```

Cancel an operation before commit:

```http
DELETE /skills/installations/{operationId}
```

Cancellation may return `409` once the operation crosses the commit boundary.
When that happens, keep polling to a terminal state.

Unknown, expired, and foreign IDs all return `404`.

## Catalog Synchronization UX

`catalogSyncStatus`:

| Value | UI behavior |
|---|---|
| `synced` | Skill is installed and available to device-bound chat. |
| `pending` | Show “Installed locally; connecting skill to chat.” Retry via reload or background reconciliation. |
| `disconnected` | Show “Installed locally; connect this device to use the skill in chat.” |

Do not label a committed installation as failed solely because synchronization
is pending or the runtime bridge is disconnected.

AI SDK chat requests must continue including the sidecar-provided `deviceId`.
A turn without `deviceId` has no local skills.

## Cache Rules

Scope the skill cache by:

```text
sidecarBaseUrl + deviceId
```

For the same device, do not replace a cached catalog with a response whose
`catalogGeneration` is lower.

After successful install/update:

1. install `result.catalog` immediately if its generation is current/newer;
2. invalidate the affected skill detail;
3. invalidate any device tool/capability catalog;
4. invalidate HITL settings because executable skill tools are grouped there.

After toggle, setup, uninstall, or reload, apply the returned catalog using the
same generation rule.

On logout or device change, discard uploads, operations, skill lists, details,
and polling state from the previous cache boundary.

## Error Codes

| Status | Code | Meaning / FE action |
|---|---|---|
| `400` | `SKILL_ARCHIVE_INVALID` | Malformed, encrypted, or corrupt ZIP. Select another file. |
| `400` | `SKILL_ARCHIVE_PATH_UNSAFE` | A path inside the ZIP escapes the bundle, collides on this filesystem, or is not portable. Repackage it. |
| `400` | `SKILL_BUNDLE_INVALID` | ZIP does not contain exactly one valid skill. A repository archive holding several is the usual cause; the message names them. |
| `400` | `SKILL_UPLOAD_STATE_INVALID` | Upload cannot perform the requested transition. Reload its state. |
| `400` | `SKILL_SETUP_REQUIRED` | The bundle declares a Python project; resend with `approveSetup: true`. |
| `400` | `SKILL_SETUP_FAILED` | The runtime could not be prepared. Retryable. |
| `401` | `UNAUTHENTICATED` | Clear local auth state and log in again. |
| `404` | `SKILL_UPLOAD_NOT_FOUND` | Upload is unknown, expired, or foreign. Re-upload. |
| `404` | `SKILL_OPERATION_NOT_FOUND` | Operation is unknown, expired, or foreign. Reload catalog before retrying. |
| `409` | `SKILL_INSTALL_CONFLICT` | Name exists; require explicit update where permitted. |
| `409` | `SKILL_SOURCE_CHANGED` | Uploaded or installed hash is stale. Re-preview. |
| `409` | `SKILL_UPLOAD_CONSUMED` | Upload already has a different operation request. |
| `409` | `SKILL_OPERATION_COMMITTED` | Too late to cancel; continue polling. |
| `413` | `SKILL_ARCHIVE_TOO_LARGE` | Compressed/expanded limit exceeded. |
| `413` | `SKILL_ARCHIVE_TOO_MANY_FILES` | Entry-count limit exceeded. |
| `413` | `SKILL_UPLOAD_QUOTA_EXCEEDED` | Cancel another staged upload or wait for cleanup. |
| `415` | `SKILL_ARCHIVE_TYPE_UNSUPPORTED` | Only ZIP is accepted. |
| `423` | `SKILL_INSTALL_LOCKED` | Another mutation is active. Retry using server guidance. |
| `507` | `SKILL_STORAGE_INSUFFICIENT` | Free local disk space, then retry. |

Setup/runtime codes already documented in
[`docs/skill-runtime.md`](../docs/skill-runtime.md) remain valid.

## TypeScript-Oriented Flow

```ts
async function stageSkillZip(file: File, token: string) {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${sidecarBase}/skills/uploads`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });
  return parseSidecarEnvelope(response);
}

async function startSkillInstall(
  uploadId: string,
  preview: SkillPreview,
  approveSetup: boolean,
  replaceSourceHash?: string,
) {
  return sidecarJson(`/skills/uploads/${encodeURIComponent(uploadId)}/install`, {
    method: "POST",
    body: {
      expectedSourceHash: preview.sourceHash,
      approveSetup,
      ...(replaceSourceHash ? { replaceSourceHash } : {}),
    },
  });
}

async function waitForSkillInstall(statusUrl: string, signal: AbortSignal) {
  for (const delay of [0, 500, 1000, 2000, 3000]) {
    if (delay) await abortableDelay(delay, signal);
    const operation = await sidecarJson(statusUrl, { method: "GET", signal });
    if (["succeeded", "failed", "cancelled"].includes(operation.data.state)) {
      return operation.data;
    }
  }

  // Continue at the capped interval until terminal, expiry, or cancellation.
}
```

The example is behavioral pseudocode. Use the frontend's normal query/mutation
library and central authenticated sidecar transport.

## Streamlit Reference Behavior

The repository's Streamlit client is a reference consumer of this contract.

- `st.file_uploader(..., type=["zip"])` selects one archive.
- The upload helper sends `files={"file": (...)}` and never also sends `json`.
- Preview metadata, `uploadId`, and `operationId` may be stored in
  `st.session_state`.
- Uploaded bytes are not retained after the staging request completes.
- Polling uses reruns or fragments rather than a blocking loop that freezes the
  page.
- Selecting a different ZIP or pressing Cancel best-effort deletes the old
  staged upload.
- Logout clears all skill-upload session keys.
- Terminal success applies the returned catalog and clears stale cached GET
  responses.

Streamlit and the AI SDK frontend must not implement different conflict,
approval, polling, or synchronization semantics.

## Existing Management Operations

### Force reload

```http
POST /skills/reload
```

Returns the complete catalog shape. Prefer its returned catalog over issuing an
immediate second `GET /skills`.

### Toggle

```http
PATCH /skills/{name}/toggle?enabled=false
```

Returns the refreshed catalog.

### Setup

```http
POST /skills/{name}/setup
Content-Type: application/json

{
  "expectedSourceHash": "current-catalog-hash",
  "approveSetup": true
}
```

Setup approval is explicit and hash-bound.

### Uninstall

```http
POST /skills/uninstall
Content-Type: application/json

{ "name": "google-calendar" }
```

Only profile-installed skills can be uninstalled. Configured-root skills remain
read-only.

### Secrets

Secret values remain write-only:

```http
POST /skills/{name}/secrets
Content-Type: application/json

{ "name": "ACCESS_TOKEN", "value": "secret-value" }
```

`GET /skills/{name}/secrets` returns configured names only. Never cache secret
values or include them in analytics, logs, error reporting, chat requests, or
operation metadata.

## Compatibility

- Existing path-based install routes remain available to trusted local tools.
- Existing snake_case request fields remain accepted during migration.
- `/api/skills/*` remains an alias of `/skills/*`.
- The AI SDK UI Message Stream is unchanged.
- Existing configured-root discovery remains read-only.
- `POST /skills/reload` remains supported but now returns the refreshed
  catalog instead of only a message.

## Frontend Acceptance Checklist

- [ ] Upload sends one multipart ZIP to the sidecar, never the canonical server.
- [ ] The UI shows preview and trust warning before installation.
- [ ] Setup and update confirmations are explicit and not preselected.
- [ ] Update sends the installed source hash from the preview.
- [ ] Installation uses the operation status URL and survives transient network
      errors.
- [ ] Unknown operation phases are ignored safely.
- [ ] Terminal failure uses stable error codes.
- [ ] Pending/disconnected catalog sync is not shown as install failure.
- [ ] Catalog caches are scoped by device and generation.
- [ ] Install/update invalidates detail, tool-catalog, and HITL caches.
- [ ] Logout/device change clears upload and polling state.
- [ ] No local path, ZIP bytes, secret value, or raw setup output reaches the
      canonical server or analytics.
- [ ] AI SDK chat requests still send `deviceId`.
