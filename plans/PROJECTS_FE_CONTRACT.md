# Projects Frontend Contract

**Audience:** Frontend developers integrating project management (grouping conversations under shared instructions and default agents) with the chat UI.

**Contract status:** Documents the backend implementation as of 2026-09-21, derived from `app/api/projects.py`, `app/services/project_service.py`, `app/services/project_context_service.py`, `app/repositories/project.py`, and the passing tests in `tests/test_projects_api.py`.

All HTTP routes require:

```http
Authorization: Bearer <jwt>
```

## Endpoint summary

| Endpoint | Method | Purpose | Success |
|---|---:|---|---:|
| `/projects` | `GET` | List the authenticated user's live projects. | `200` |
| `/projects` | `POST` | Create a project. | `201` |
| `/projects/{projectId}` | `GET` | Read one project, including its ordered default agents. | `200` |
| `/projects/{projectId}` | `PATCH` | Update name, description, or instructions. | `200` |
| `/projects/{projectId}` | `DELETE` | Soft-delete a project; its conversations are detached, not deleted. | `200` |
| `/projects/{projectId}/custom-agents` | `GET` | Read the project's ordered default agents. | `200` |
| `/projects/{projectId}/custom-agents` | `PUT` | Replace the project's ordered default agent set. | `200` |
| `/projects/{projectId}/conversations/{conversationId}` | `PUT` | Move a conversation into the project and seed its default agents. | `200` |
| `/projects/{projectId}/conversations/{conversationId}` | `DELETE` | Release a conversation from the project. | `200` |

There is no `/ai`-prefixed alias family for projects. This router is registered once.

## Common response and error envelopes

Successful responses use the same envelope as every other resource in this backend:

```ts
type ApiResponse<T> = {
  success: true;
  message: string;
  data: T;
  error: null;
};
```

Delete and the two attach/detach endpoints return `data: null`:

```json
{
  "success": true,
  "message": "Project deleted",
  "data": null,
  "error": null
}
```

Application errors:

```ts
type ApiError = {
  success: false;
  code: string;
  message: string;
};
```

Request-shape validation (e.g. an over-length `instructions` field) uses `422 invalid_input`, keyed by a path such as `body.instructions`.

## Naming and JSON casing

Request bodies accept camelCase and snake_case. Response bodies are entirely camelCase — unlike the custom-agent options/ref payloads, there are no nested snake_case dictionaries here.

## Core resource

```ts
type UUID = string;

type Project = {
  id: UUID;
  createdAt: string;
  updatedAt: string;
  deletedAt: string | null;
  ownerId: UUID;
  name: string;
  description: string | null;
  instructions: string | null;
  conversationCount: number;
  customAgents: CustomAgent[] | null;
};
```

`customAgents` is `null` on list responses and on plain reads that do not request agent detail; it is always populated (as an array, possibly empty) on `GET /projects/{projectId}`. See [Read one project](#read-one-project).

`conversationCount` is the number of *live* (non-deleted) conversations currently attached to the project, recomputed on every read.

## Create

```http
POST /projects
Content-Type: application/json
```

```json
{
  "name": "Roadmap",
  "description": "Q3 planning conversations.",
  "instructions": "Be brief."
}
```

| Field | Required | Rules |
|---|---:|---|
| `name` | yes | Trimmed non-empty string; maximum 255 characters. |
| `description` | no | String or `null`; maximum 2,000 characters. |
| `instructions` | no | String or `null`; maximum 8,000 characters. Applied to every conversation in the project — see [Composition](#composition-of-the-system-instruction). |

Response `data` is a complete `Project` with `conversationCount: 0` and `customAgents: null`.

```json
{
  "success": true,
  "message": "Project created",
  "data": {
    "id": "b1f6c8b0-4b1a-4b3a-9b3a-1f6c8b0b1a4b",
    "createdAt": "2026-09-21T10:00:00Z",
    "updatedAt": "2026-09-21T10:00:00Z",
    "deletedAt": null,
    "ownerId": "0c73f7d3-5aca-4a7e-8b71-4cf802f4d901",
    "name": "Roadmap",
    "description": "Q3 planning conversations.",
    "instructions": "Be brief.",
    "conversationCount": 0,
    "customAgents": null
  },
  "error": null
}
```

Sending `instructions` longer than 8,000 characters returns `422 invalid_input` and creates nothing.

## List

```http
GET /projects
```

Response `data` is `Project[]`, live projects only, newest first. Not paginated. Each entry's `customAgents` is `null`; fetch `GET /projects/{projectId}` or `GET /projects/{projectId}/custom-agents` for agent detail.

## Read one project

```http
GET /projects/{projectId}
```

Response `data` is one `Project` with `customAgents` populated as the ordered default-agent array (empty array if none are set).

Errors:

- A missing or soft-deleted `projectId` returns `404 PROJECT_NOT_FOUND`.
- A live project owned by another user returns `403 PROJECT_FORBIDDEN`. Existence is therefore observable to a caller who already has the UUID, matching the existing custom-agent convention.

## Update

```http
PATCH /projects/{projectId}
Content-Type: application/json
```

```json
{ "name": "Renamed" }
```

Body accepts any subset of `name`, `description`, `instructions`. Omitted fields are unchanged (the schema uses `exclude_unset`). Sending `null` explicitly clears `description` or `instructions` back to `null`. `name` is not nullable: sending `{"name": null}` is rejected with `422 invalid_input` rather than clearing it — omit `name` to leave it unchanged, or send a non-empty string to rename.

Response `data` is the updated `Project` (`customAgents: null`, matching the list/create shape — this endpoint does not include agent detail).

Errors match [Read one project](#read-one-project): `404 PROJECT_NOT_FOUND` / `403 PROJECT_FORBIDDEN`. An explicit `null` `name` is `422 invalid_input`, not a 403/404.

## Delete

```http
DELETE /projects/{projectId}
```

Soft-deletes the project and **detaches** (does not delete) every conversation currently in it — each detached conversation's `projectId` becomes `null` and it reverts to an unaffiliated conversation. There is no restore endpoint; restoring a soft-deleted project would restore it with zero conversations even if the row were un-deleted directly.

Errors match [Read one project](#read-one-project).

## Project default agents

### Read

```http
GET /projects/{projectId}/custom-agents
```

Response `data` is `CustomAgent[]` (the same shape documented in `CUSTOM_AGENTS_FE_CONTRACT.md`) in persisted order.

### Replace

```http
PUT /projects/{projectId}/custom-agents
Content-Type: application/json
```

```json
{
  "customAgentIds": [
    "0c73f7d3-5aca-4a7e-8b71-4cf802f4d901",
    "4e256b3e-bb8d-4fb4-88f8-296ee37c1932"
  ]
}
```

- Replace-all semantics, ordered by array position.
- Every id must be a live custom agent owned by the caller, or the request fails with `403 PROJECT_FORBIDDEN`.
- Duplicate ids are rejected with `422 invalid_input` before the service is called.
- `[]` clears the project's default set.

**This endpoint does not reach conversations already in the project.** It only changes what gets *seeded* onto conversations created or attached from this point forward. See [Seeding](#seeding).

## Conversation membership

### Attach (move)

```http
PUT /projects/{projectId}/conversations/{conversationId}
```

Points the conversation at this project and seeds the project's current default agents onto it (see [Seeding](#seeding)). `data` is `null` on success.

**Attaching a conversation that already belongs to a different project is a move, not an error.** The conversation is simply repointed; nothing about its prior project membership blocks the call, and there is no confirmation step.

Errors:

- `403 PROJECT_FORBIDDEN` if the caller does not own the *project*.
- `404 CONVERSATION_NOT_FOUND` (from the shared conversation-validation path, `validate_conversation_exists`) if `conversationId` does not identify any conversation.
- `403 CONVERSATION_ACCESS_DENIED` (same shared path, `validate_user_owns_conversation`) if the caller does not own the *conversation*. A conversation owned by someone else cannot be attached to your project even if you own the project.

### Detach

```http
DELETE /projects/{projectId}/conversations/{conversationId}
```

Releases the conversation from the project (`projectId` becomes `null`). The agents seeded while it was in the project are **not** removed from the conversation.

Errors:

- `403 PROJECT_FORBIDDEN` / `404 CONVERSATION_NOT_FOUND` / `403 CONVERSATION_ACCESS_DENIED` as above.
- `404 PROJECT_CONVERSATION_NOT_FOUND` if the conversation is not currently a member of *this* project — including when it was never attached, or is currently attached to a different project. Detach from the wrong project does not silently succeed and does not touch the conversation's actual project.

## Seeding

"Seeding" means: copy the project's current default custom agents onto a conversation's own attachment list.

- Seeding happens at exactly two moments: **conversation create** (when `POST /conversations` is called with a `projectId`) and **attach** (`PUT /projects/{projectId}/conversations/{conversationId}`).
- Seeding is insert-if-absent: an agent already attached to the conversation (by any means) is left alone and never duplicated or reordered. Newly-seeded agents are appended after the conversation's existing attachments.
- Seeding never removes a conversation's existing agents, including agents that are not part of the project's default set.
- `PUT /projects/{projectId}/custom-agents` (replacing the project's default set) does **not** retroactively touch any conversation already in the project. Conversations already attached keep whatever agents they were seeded with (or have added/removed since) regardless of later changes to the project's defaults.
- Detach does not undo seeding — a conversation keeps its seeded agents after leaving the project.

## Conversations integration

### `projectId` on conversation list

```http
GET /conversations?projectId={projectId}
```

`projectId` is an optional query parameter (alias `projectId`) that restricts the list to conversations currently in that project. Omit it to list all of the caller's conversations regardless of project.

The project must be owned by the caller, or the request fails with `403 PROJECT_FORBIDDEN` before the list query runs — a foreign `projectId` never returns an empty page, matching every other project path.

### `projectId` on conversation create

```json
POST /conversations
{
  "title": "New chat",
  "projectId": "b1f6c8b0-4b1a-4b3a-9b3a-1f6c8b0b1a4b"
}
```

`projectId` is optional on `ConversationCreate`. When present:

- The project must be owned by the caller, or the request fails with `403 PROJECT_FORBIDDEN`.
- The project's default agents are seeded onto the new conversation in the same request (see [Seeding](#seeding)).

### `projectId` on conversation read

`ConversationRead` (returned from create/get/list/update) includes `projectId: UUID | null`, reflecting current membership. Unlike `customAgents` on that same schema (see `CUSTOM_AGENTS_FE_CONTRACT.md` IR-5), `projectId` is populated by every conversation read path.

## Composition of the system instruction

When a conversation belongs to a live project, the system instruction sent to the model on every turn is composed from **both** the project's instructions and the conversation's own persona (`personaPrompt`), not just one or the other:

- Each part is independently sanitized and capped at 8,000 characters before composition. A project's `instructions` and a conversation's `personaPrompt` can each independently be up to 8,000 characters — the cap is per-part, not shared.
- When both are present, they are joined with two literal headers, project first:

  ```text
  Project instructions:
  <project instructions, up to 8000 chars>

  Conversation-specific instructions:
  <conversation persona, up to 8000 chars>
  ```

- If only one part is present (no project, or a project with empty instructions), the composed instruction is exactly that one part with no header — byte-identical to how a project-less conversation has always behaved.
- The composed instruction is resolved fresh on every turn, live against the project's current `instructions`. Editing a project's instructions takes effect on the *next* turn of every conversation in it, with no republish or migration step.
- The frontend can reproduce this exact preview client-side (e.g. to show "what the model will see") by truncating each part independently to 8,000 characters and joining with the two headers above only when both parts are non-empty. The combined string can be up to roughly 16,060 characters.

## Error matrix

| HTTP/status | Code | Applies to | Frontend behavior |
|---:|---|---|---|
| `403` | `PROJECT_FORBIDDEN` | Project exists but belongs to another user; the custom-agent ids in a `PUT .../custom-agents` body are not all owned by the caller; or `GET /conversations?projectId=` names a project the caller does not own. | Do not expose or cache the resource. Return to the project list. |
| `403` | `CONVERSATION_ACCESS_DENIED` | The conversation in an attach/detach call belongs to another user. | Return to the conversation list. |
| `404` | `PROJECT_NOT_FOUND` | Missing or soft-deleted project. | Remove it from local lists and selections. |
| `404` | `CONVERSATION_NOT_FOUND` | The conversation in an attach/detach call does not exist. | Close the conversation and preserve any unsent draft. |
| `404` | `PROJECT_CONVERSATION_NOT_FOUND` | Detach where the conversation is not currently a member of that project. | Refetch the conversation's actual `projectId` and reconcile local state; do not assume detach succeeded. |
| `422` | `invalid_input` | `name`/`description`/`instructions` over length, invalid UUID, duplicate ids in `customAgentIds`, or an explicit `null` `name` on update. | Map `error` paths to fields where possible; keep form state. |
| `500` | `internal_server_error` | Unexpected backend failure. | Keep unsaved state, show a generic retry action. |

## Recommended frontend state model

```ts
type ProjectState = {
  all: Project[];
  byId: Map<UUID, Project>;
  agentsByProject: Map<UUID, CustomAgent[]>;
  conversationsByProject: Map<UUID, UUID[]>;
};
```

Recommended invalidation:

- After create: refetch the list or insert the returned `data` (with `conversationCount: 0`, `customAgents: null`).
- After update: replace the `byId` entry; note the response does not include `customAgents`, so do not overwrite a cached agent list with it.
- After delete: remove the project globally, and clear `projectId` on any locally cached conversation that pointed at it.
- After attach/detach: refetch the conversation (for its `projectId`) and the project (for `conversationCount`); these endpoints return `data: null`, not an updated resource.
- After `PUT .../custom-agents`: replace the cached project agent list with the returned order. Do not assume any already-open conversation's attached-agent cache changed — it did not.
- On `PROJECT_NOT_FOUND`: evict the missing project.

## FE acceptance checklist

- Creates conversations with `projectId` when the user is composing inside a project; omits it otherwise.
- Treats attach as a move: no confirmation is required by the backend, but the frontend should warn the user before moving a conversation out of a project it currently displays elsewhere.
- Treats `PUT /projects/{projectId}/custom-agents` as forward-looking only; refreshes any open conversation's attached-agent list separately, and does not expect it to change from this call.
- Handles `404 PROJECT_CONVERSATION_NOT_FOUND` on detach as "already not a member," not as a generic failure — refetch state rather than retrying the same call.
- Distinguishes `403 PROJECT_FORBIDDEN` (bad project) from `403 CONVERSATION_ACCESS_DENIED` (bad conversation) when attaching, since only one of the two resources may be at fault.
- Never assumes a project's `instructions` length bounds the composed system instruction the model receives; the composed value can be roughly double a single part's cap when both a project and a conversation persona are present.
- Ignores unknown additive response fields, consistent with every other resource in this backend.
