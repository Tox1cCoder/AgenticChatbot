# Projects Frontend Contract

**For:** frontend developers building the Projects UI. This documents the current wire format. For the full nested agent and conversation objects, see [Custom Agents Frontend Contract](CUSTOM_AGENTS_FE_CONTRACT.md) and [AI SDK Frontend Contract](AI_SDK_FE_CONTRACT.md).

**Backend as of:** 2026-09-28.

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

### Field details

| Field | Type | Nullable | Format and rules | Example |
|---|---|---|---|---|
| `id` | string | no | UUID v4, lowercase, hyphenated. Set by the server. | `"b3e95508-bd4e-491a-8b2a-982fea675e23"` |
| `createdAt` | string | no | ISO 8601 date-time with a timezone (`Z` or an offset); fractional seconds may be present. Parse as a date, not a sortable string. Never changes. | `"2026-09-22T03:03:59.822441Z"` |
| `updatedAt` | string | no | Same date-time format as `createdAt`. Project field edits update it; changing default agents or membership does not. | `"2026-09-22T03:03:59.822441Z"` |
| `deletedAt` | string | yes | Same format as `createdAt`. Always `null` in practice, because deleted projects are never returned. | `null` |
| `ownerId` | string | no | UUID of the user who owns the project. Always the signed-in user. | `"c3f04191-895a-4fb4-95c1-9f3fe66ac8f1"` |
| `name` | string | no | 1–255 characters. Not trimmed. Not unique: two projects can share a name, so key UI by `id`. | `"Roadmap"` |
| `description` | string | yes | Up to 2,000 characters, plain text. `null` and `""` are both possible; treat both as "no description". | `"Q3 planning conversations."` |
| `instructions` | string | yes | Up to 8,000 characters, plain text (newlines kept). `null`, `""` and whitespace-only all mean "no project instructions" to the assistant. | `"Be brief."` |
| `conversationCount` | number | no | Integer, 0 or more. Counts conversations currently in the project, excluding deleted ones. Computed on every read. | `3` |
| `customAgents` | array | yes | `CustomAgent[]` in the project's order, deleted agents excluded. An array (possibly `[]`) only on `GET /projects/{projectId}`; `null` on list, create and update. | `null` |

Request bodies use camelCase names. `POST /projects` requires `name` (string, 1–255 characters); `description` (string or `null`, at most 2,000 characters) and `instructions` (string or `null`, at most 8,000 characters) are optional and default to `null`. `PATCH /projects/{projectId}` accepts any subset: omitted fields stay unchanged; `null` clears `description` or `instructions`; `name: null` is a `422`. An empty PATCH body is accepted. Project text fields are not trimmed by the backend. Path IDs and `customAgentIds` entries are UUID strings; the ordered `customAgentIds` array must not contain duplicates. `PUT .../custom-agents` replaces the entire set; `{ "customAgentIds": [] }` clears it, and omitting the field also clears it.

## Endpoints

| Method | Path | Body | `data` on success |
|---|---|---|---|
| `GET` | `/projects` | Query: `page?`, `limit?`, `search?` | `{ items: Project[], meta: PaginationMeta }` |
| `POST` | `/projects` | `{ name, description?, instructions? }` | `Project` (status `201`) |
| `GET` | `/projects/{projectId}` | — | `Project` with `customAgents` filled |
| `PATCH` | `/projects/{projectId}` | any of `name`, `description`, `instructions` | `Project` |
| `DELETE` | `/projects/{projectId}` | — | `null` |
| `GET` | `/projects/{projectId}/custom-agents` | — | `CustomAgent[]` in order |
| `PUT` | `/projects/{projectId}/custom-agents` | `{ customAgentIds: string[] }` | `CustomAgent[]` in the new order |
| `PUT` | `/projects/{projectId}/conversations/{conversationId}` | — | `null` |
| `DELETE` | `/projects/{projectId}/conversations/{conversationId}` | — | `null` |

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

- **Trim the name before sending.** The backend does not trim, and accepts `"   "` as a name.
- **PATCH:** an omitted field is left unchanged. `null` clears `description` or `instructions`. `name: null` is rejected with `422`.
- **Always send `customAgentIds`.** `PUT .../custom-agents` with `{}` clears the project's agent list. `[]` clears it on purpose.
- **Default agents only reach new members.** Changing a project's agents does not change conversations already in it. A conversation that leaves the project keeps its agents. Deleting a custom agent removes it from every project's list.
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
| `403` | `PROJECT_FORBIDDEN` | Project belongs to another user, or a `customAgentIds` entry is not one of the user's live agents | Return to the project list, or refetch agents and show which one failed |
| `404` | `CONVERSATION_NOT_FOUND` | Attach/detach: conversation is missing or deleted | Refetch the conversation list |
| `403` | `CONVERSATION_ACCESS_DENIED` | Attach/detach: conversation belongs to another user | Refetch the conversation list |
| `404` | `PROJECT_CONVERSATION_NOT_FOUND` | Detach: the conversation is not in this project | Refetch its `projectId`; do not retry |
| `422` | `invalid_input` | Field too long, invalid UUID, duplicate agent ids, or `name: null` | Show the message on the field |
