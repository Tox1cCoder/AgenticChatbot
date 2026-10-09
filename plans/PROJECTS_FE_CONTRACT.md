# Projects Frontend Contract

**For:** frontend developers building the Projects UI. This documents the current wire format. For the full nested agent and conversation objects, see [Custom Agents Frontend Contract](CUSTOM_AGENTS_FE_CONTRACT.md) and [AI SDK Frontend Contract](AI_SDK_FE_CONTRACT.md).

**Backend as of:** 2026-10-09.

## Frontend changes in this revision

- Pass `deviceId` on project detail and default-agent GET/PUT requests, including through the sidecar.
- Render device capability availability separately from the agent's `enabled` switch; retain unavailable and disabled selections.
- Save project fields and ordered default-agent IDs in separate requests, with explicit handling for a partially completed save.
- Apply the inheritance, refresh, and error rules below. Membership and inherited attachments are now committed together.

## What a project is

A project groups conversations. It has:

- **Instructions** that apply to every conversation in the project, on top of each conversation's own persona.
- **Default custom agents** that are copied onto a conversation when it is created in, or moved into, the project.

A conversation belongs to at most one project. Projects are private to their owner.

## Paths

Project routes have **no `/ai` prefix**. Call `/projects/...` even if the rest of the app uses `/ai/...`. Through the local sidecar, the same routes also answer at `/api/projects/...`.

All requests require `Authorization: Bearer <token>`. Use the server access token when calling the canonical API. The local sidecar accepts its local session token or the active server access token and forwards authenticated requests upstream. Canonical Swagger is at `/docs` (schema at `/openapi.json`); the sidecar has its own `/docs` and `/openapi.json`.

## Response envelope

Successful single-resource and mutation operations return `{ "success": true, "message": string, "data": T, "error": null }`; `code` is omitted. The paginated `GET /projects` response has `{ "success": true, "message": string, "data": { "items": Project[], "meta": PaginationMeta } }` and no `error` field. Delete, attach, and detach use `data: null` with HTTP `200`. Creation uses HTTP `201`; all other successful Projects operations use `200`.

Application errors from the canonical API return `{ "success": false, "code": string, "message": string }`. Input validation uses HTTP `422`, `code: "invalid_input"`, and an additional `error` object mapping locations to arrays of messages, such as `{ "body.name": ["String should have at least 1 character"] }`. The local sidecar can return its own authentication or proxy errors before reaching the canonical API; handle HTTP status as well as `code`.

## Project object

```ts
type Project = {
  id: string;
  createdAt: string;
  updatedAt: string;
  deletedAt: string | null;
  ownerId: string;
  name: string;
  description: string | null;
  instructions: string | null;
  conversationCount: number;          // live conversations in the project
  customAgents: CustomAgent[] | null; // filled only by GET /projects/{projectId}; null elsewhere
};
```

`CustomAgent` is the same shape the custom-agent routes return.

Projects store default-agent IDs, not separate skills or MCP configuration. Configure skills and tools on each custom agent. Project create/update bodies support only `name`, `description`, and `instructions`; save default agents through their dedicated PUT route.

### Field details

| Field | Type | Nullable | Format and rules | Example |
|---|---|---|---|---|
| `id` | string | no | UUID v4, lowercase, hyphenated. Set by the server. | `"b3e95508-bd4e-491a-8b2a-982fea675e23"` |
| `createdAt` | string | no | ISO 8601 date-time with a timezone (`Z` or an offset); fractional seconds may be present. Parse as a date, not a sortable string. Never changes. | `"2026-09-22T03:03:59.822441Z"` |
| `updatedAt` | string | no | Same date-time format as `createdAt`. Project field edits update it; changing default agents or membership does not. | `"2026-09-22T03:03:59.822441Z"` |
| `deletedAt` | string | yes | Same format as `createdAt`. Always `null` in practice, because deleted projects are never returned. | `null` |
| `ownerId` | string | no | UUID of the user who owns the project. Always the signed-in user. | `"c3f04191-895a-4fb4-95c1-9f3fe66ac8f1"` |
| `name` | string | no | 1–255 characters before trimming; surrounding whitespace is removed and blank names are rejected. Not unique: key UI by `id`. | `"Roadmap"` |
| `description` | string | yes | Up to 2,000 characters, plain text. `null` and `""` are both possible; treat both as "no description". | `"Q3 planning conversations."` |
| `instructions` | string | yes | Up to 8,000 characters, plain text (newlines kept). `null`, `""` and whitespace-only all mean "no project instructions" to the assistant. | `"Be brief."` |
| `conversationCount` | number | no | Integer, 0 or more. Counts conversations currently in the project, excluding deleted ones. Computed on every read. | `3` |
| `customAgents` | array | yes | `CustomAgent[]` in the project's order, deleted agents excluded. An array (possibly `[]`) only on `GET /projects/{projectId}`; `null` on list, create and update. | `null` |

Request bodies use camelCase names. `POST /projects` requires `name` (string, 1–255 characters); `description` (string or `null`, at most 2,000 characters) and `instructions` (string or `null`, at most 8,000 characters) are optional and default to `null`. `PATCH /projects/{projectId}` accepts any subset: omitted fields stay unchanged; `null` clears `description` or `instructions`; `name: null` is a `422`. An empty PATCH body is accepted. The backend trims names; description and instructions retain their whitespace. Path IDs and `customAgentIds` entries are UUID strings; the ordered `customAgentIds` array must not contain duplicates. `PUT .../custom-agents` replaces the entire set; `{ "customAgentIds": [] }` clears it, and omitting the field also clears it.

## Endpoints

| Method | Path | Query | Body | `data` on success |
|---|---|---|---|---|
| `GET` | `/projects` | `page?`, `limit?`, `search?` | — | `{ items: Project[], meta: PaginationMeta }` |
| `POST` | `/projects` | — | `{ name, description?, instructions? }` | `Project` (status `201`) |
| `GET` | `/projects/{projectId}` | `deviceId?` | — | `Project` with `customAgents` filled |
| `PATCH` | `/projects/{projectId}` | — | any of `name`, `description`, `instructions` | `Project` |
| `DELETE` | `/projects/{projectId}` | — | — | `null` |
| `GET` | `/projects/{projectId}/custom-agents` | `deviceId?` | — | `CustomAgent[]` in order |
| `PUT` | `/projects/{projectId}/custom-agents` | `deviceId?` | `{ customAgentIds: string[] }` | `CustomAgent[]` in the new order |
| `PUT` | `/projects/{projectId}/conversations/{conversationId}` | — | — | `null` |
| `DELETE` | `/projects/{projectId}/conversations/{conversationId}` | — | — | `null` |

### Device availability on default agents

Send the active registered device UUID as `deviceId` on the three agent-bearing endpoints above. The Projects sidecar proxy forwards this query unchanged; it does **not** inject the active device automatically. Obtain the registered `deviceId` from the local authentication/session response, rather than using the machine's `deviceIdentifier`.

Project agent responses compute `availability` against one owner-scoped device snapshot. The full agent resource is defined in the [Custom Agents contract](CUSTOM_AGENTS_FE_CONTRACT.md#core-resource); its capability fields are:

```ts
type CustomAgentAvailability = {
  status: "ready" | "degraded" | "device_unavailable";
  deviceId: string | null;
  sessionId: string | null;
  missingTools: {
    serverName: string;
    qualifiedToolId: string;
    toolName: string | null;
  }[];
  missingSkills: { lookupName: string; name: string }[];
  warnings: string[];
};
// On CustomAgent: availability: CustomAgentAvailability | null
```

Example `availability` fragment for an agent whose selected client tool is absent from an otherwise ready device:

```json
{
  "status": "degraded",
  "deviceId": "cf24a938-c0a7-46dc-afc7-d57b6c9568f5",
  "sessionId": "d2e4977f-9ff5-4533-86fd-7c857b79894f",
  "missingTools": [{
    "serverName": "csv",
    "qualifiedToolId": "client__csv__profile",
    "toolName": "profile"
  }],
  "missingSkills": [],
  "warnings": ["Selected MCP tool 'client__csv__profile' is not available on this device."]
}
```

| Status | Meaning | UI behavior |
|---|---|---|
| `ready` | Selected local dependencies resolve, or the agent has no local dependencies. | Show available capabilities; do not infer provider or external-service health. |
| `degraded` | The device snapshot is ready, but some selected local tools or skills are missing. | Show missing entries and warnings; retain the agent selection. Runtime skips unresolved capabilities. |
| `device_unavailable` | Local dependencies exist, but there is no usable snapshot for the requested device. | Show that the device must connect/sync; retain the agent selection. |

Omitted, invalid, foreign-owned, offline, or not-yet-synced device IDs produce unavailable local capabilities rather than a project access error. An agent with no local dependencies can still be `ready`. A missing/null availability value means unknown; do not render it as ready.

`enabled` is independent of availability. Disabled agents remain visible, selectable, and inheritable, but are excluded from runtime rosters loaded for subsequent turns. Re-enabling preserves existing attachments. Changing the switch does not cancel an in-flight turn.

Saved selections are account-wide intent; local tools and skills resolve only against the requesting user's active device. Changing devices changes availability without rewriting defaults. Legacy skill refs with `source: "server"` resolve to their current client-owned skill. Typed `availability` keys are camelCase; persisted `toolRefs`/`skillRefs` dictionary keys remain snake_case.

Backend MCP selections are saved hints. Agents can discover and use the globally enabled backend MCP catalog regardless of those selections. Describe the selector as a preference, not a permissions control.

### Saving the settings form

1. Create the project with `POST /projects`, or save changed fields with `PATCH /projects/{projectId}`. Retain the returned ID and normalized name.
2. If default agents changed, send the complete ordered list to `PUT /projects/{projectId}/custom-agents?deviceId={deviceId}`. Use agent database IDs, not `runtimeAgentId` values. Owned live agents can be saved even when disabled or unavailable on this device.
3. Refetch project detail with the same `deviceId` after both operations succeed. Do not replace an existing default-agent list with `customAgents: null` from a create/PATCH response.

These two settings requests are **not one transaction**. If fields save but agent replacement fails, report the partial save, preserve the draft agent list, and retry the PUT against the existing project. Do not create a duplicate project to retry. Disable overlapping submissions; concurrent settings updates have no revision/ETag conflict check.

### Project listing

`GET /projects?page=1&limit=10&search=roadmap` searches the signed-in user's live projects. `search` is optional, case-insensitive substring matching over `name` and `description`; surrounding whitespace is ignored, and blank search means no filter. Results are newest first, with ID as the tie-breaker. `page` defaults to `1` (minimum `1`), `limit` defaults to `10` (range `1`–`100`), and `search` is at most 200 characters. Invalid values return `422 invalid_input`.

```ts
type PaginationMeta = {
  total: number;       // count of all matching projects before paging
  perPage: number;     // requested limit
  currentPage: number; // requested page, 1-based
  lastPage: number;    // at least 1, including when total is 0
};
type ProjectListData = { items: Project[]; meta: PaginationMeta };
```

An out-of-range page has `items: []` and keeps the requested `currentPage`. `GET /projects` now returns this paginated object, so clients expecting a bare `Project[]` must read `data.items`.

Example `GET /projects?page=1&limit=10` response (timestamps and IDs vary):

```json
{
  "success": true,
  "message": "Projects retrieved",
  "data": {
    "items": [{
      "id": "b3e95508-bd4e-491a-8b2a-982fea675e23",
      "createdAt": "2026-09-22T03:03:59.822441Z",
      "updatedAt": "2026-09-22T03:03:59.822441Z",
      "deletedAt": null,
      "ownerId": "c3f04191-895a-4fb4-95c1-9f3fe66ac8f1",
      "name": "Roadmap",
      "description": "Q3 planning conversations.",
      "instructions": "Be brief.",
      "conversationCount": 3,
      "customAgents": null
    }],
    "meta": { "total": 1, "perPage": 10, "currentPage": 1, "lastPage": 1 }
  }
}
```

## Changes to conversations

- Every conversation response now includes `projectId: string | null`.
- **Create inside a project:** send `projectId` in the body of `POST /conversations/` or `POST /ai/conversations`. The project's default agents are attached in the same request.
- **List a project's conversations:** `GET /conversations/?projectId={projectId}&page=1&limit=10&search=term`. The response has `data: { items: Conversation[], meta: PaginationMeta }`. `page` is 1-based and `limit` is `1`–`100`; the server defaults to 10. The local sidecar defaults to 20 when the UI omits `limit`, so send it explicitly for consistent paging. `search` is optional (maximum 200 characters), trims surrounding whitespace, and matches title or non-deleted message content without case sensitivity. With search, relevance determines order (exact title, title prefix, title substring, then message match); without search, `orderBy=updatedAt|createdAt` and `orderDirection=desc|asc` apply (defaults `updatedAt` and `desc`). `GET /ai/conversations` ignores `projectId` and search, so use the non-`/ai` route here.
- **Change membership** only with the two `/projects/{projectId}/conversations/{conversationId}` routes. `projectId` sent to `PATCH /conversations/{conversationId}` is ignored.
- **Chat is unchanged.** The backend applies project instructions itself; send nothing extra.

## Behavior the UI needs to handle

- **Names are trimmed by the backend.** Blank names return `422`; client validation can catch them earlier.
- **PATCH:** an omitted field is left unchanged. `null` clears `description` or `instructions`. `name: null` is rejected with `422`.
- **Always send `customAgentIds`.** `PUT .../custom-agents` with `{}` clears the project's agent list. `[]` clears it on purpose.
- **Default agents are copied on create/attach.** Changing defaults alone does not update existing conversations. Existing attachments keep their order; missing defaults append in project order. For example, existing `[C, B]` plus defaults `[A, B]` becomes `[C, B, A]`. Reattaching to the same project does not duplicate existing agents, but can append newly selected defaults. Leaving the project keeps all attachments. Deleting a custom agent removes it from project defaults and conversation attachments.
- **Membership and inheritance are atomic.** Create/attach either persist membership and inherited agents together or roll back together. Concurrent retries do not duplicate agents. A stale detach cannot clear a newer move to another project.
- **Attaching is a move.** Attaching a conversation that is already in another project moves it, with no error. Add a confirmation step in the UI if you want one.
- **Deleting a project keeps its conversations but deletes its memories.** Conversations are detached (`projectId` becomes `null`), not deleted. Memories saved inside the project are deleted with it; global memories are untouched. Say so in the delete dialog.
- **Refetch after `data: null`.** After attach or detach, refetch the conversation (`projectId`) and affected project(s) (`conversationCount`). After project deletion, refresh the project list and affected conversations; fetching the deleted project returns `404`.
- **Instruction edits apply on the next message** in every conversation in the project.
- **Membership decides what the assistant can recall.** In a project, the assistant recalls memories saved in that project plus global ones, and can search past conversations in that same project only. Moving a conversation changes this from its next message, which is worth one line in a move confirmation.

## Errors

Canonical API errors use the envelope described above. A `422` carries `error` keyed by location, for example `{ "body.name": ["String should have at least 1 character"] }`.

| Status | `code` | When | Suggested UI |
|---|---|---|---|
| `404` | `PROJECT_NOT_FOUND` | Project is missing or deleted, on any project route or on conversation create/list with `projectId` | Remove it from lists and clear the selection |
| `403` | `PROJECT_FORBIDDEN` | Project belongs to another user, or a `customAgentIds` entry is not one of the user's live agents | Return to the project list, or refetch owned agents and reconcile the selection |
| `404` | `CONVERSATION_NOT_FOUND` | Attach/detach: conversation is missing or deleted | Refetch the conversation list |
| `403` | `CONVERSATION_ACCESS_DENIED` | Attach/detach: conversation belongs to another user | Refetch the conversation list |
| `404` | `PROJECT_CONVERSATION_NOT_FOUND` | Detach: the conversation is not in this project | Refetch its `projectId`; do not retry |
| `422` | `invalid_input` | Blank name, field too long, invalid path/body UUID, duplicate agent ids, or `name: null` | Show the message on the field |

A `PROJECT_FORBIDDEN` error does not identify the failing agent ID in a structured field. Refetch owned live agents and compare their IDs with the submitted selection. On network/server errors, retain form state and refetch to reconcile a possibly completed write before retrying.

## Refresh and frontend acceptance checklist

- Read project lists from `data.items`; keep pagination metadata and send explicit conversation paging limits.
- Validate trimmed names, retain ordered agent IDs, and explicitly send `[]` only when clearing defaults.
- Key project-detail/default-agent availability caches by signed-in user, project ID, and device ID. Discard stale responses when the device changes. Refresh on reconnect, session/catalog changes, and custom-agent edits or deletion; options responses expose `deviceSnapshot` for catalog identity.
- After moving a conversation, refresh its detail and dedicated custom-agent attachments, plus counts for both source and destination projects. After detach, refresh the conversation and source count. After deletion, refresh project lists and affected conversations.
- Explain that defaults apply on create/attach, instruction edits apply on the next message, and project deletion preserves conversations while deleting project memories.
- Keep disabled/unavailable defaults visible with separate enabled and capability indicators. Show missing capabilities without clearing saved selections or blocking an otherwise valid default-agent save.
- Handle partial settings saves, `403`/`404`, field-level `422`, and network failures without losing edits. Render unknown additive fields safely.
