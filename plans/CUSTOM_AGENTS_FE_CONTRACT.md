# Custom Agents Frontend Contract

**Audience:** Frontend developers integrating custom-agent management with the AI SDK chat UI.

**Contract status:** Documents the backend implementation as of 2026-07-22. Where intended behavior and current wire behavior differ, this document describes the current wire behavior and calls out the discrepancy in [Implementation review](#implementation-review).

## Integration decision

Use the `/ai/...` routes for new AI SDK frontend work. The non-`/ai` routes are compatibility aliases backed by the same handlers; they do not provide different behavior.

All HTTP routes require:

```http
Authorization: Bearer <jwt>
```

When the browser talks through `client_backend`, the local sidecar session supplies the upstream bearer token and injects the active `deviceId` on options/create/update requests. A browser calling the main API directly must supply both itself.

## Endpoint summary

| Recommended endpoint | Method | Purpose | Success |
|---|---:|---|---:|
| `/ai/custom-agents` | `GET` | List the authenticated user's live custom agents, newest first. | `200` |
| `/ai/custom-agents` | `POST` | Create a custom agent. | `201` |
| `/ai/custom-agents/options?deviceId={deviceId}` | `GET` | Load current provider/model, server tool, active client tool, client server, and skill options. | `200` |
| `/ai/custom-agents/{customAgentId}` | `GET` | Read one owned custom agent. | `200` |
| `/ai/custom-agents/{customAgentId}?deviceId={deviceId}` | `PATCH` | Partially update an owned custom agent. | `200` |
| `/ai/custom-agents/{customAgentId}` | `DELETE` | Soft-delete an agent and detach it from all conversations. | `200` |
| `/ai/conversations/{conversationId}/custom-agents` | `GET` | Read the conversation's attached agents in attachment order. | `200` |
| `/ai/conversations/{conversationId}/custom-agents` | `PUT` | Replace the conversation's complete ordered attachment set. | `200` |
| `/ai/conversations` | `POST` | Create a conversation before attaching agents. | `201` |
| `/api/chat/{conversationId}` | `POST` | Start a chat turn. Attached agents are resolved server-side; do not add them to this body. | SSE |
| `/ai/resume-interrupt` | `POST` | Resume a paused HITL run. Custom-agent attachment is revalidated. | SSE |
| `/ai/conversations/{conversationId}/messages` | `GET` | Refetch durable messages and responding-agent metadata. | `200` |

### Equivalent compatibility aliases

| Recommended | Equivalent alias |
|---|---|
| `/ai/custom-agents` | `/custom-agents` |
| `/ai/custom-agents/options` | `/custom-agents/options` |
| `/ai/custom-agents/{customAgentId}` | `/custom-agents/{customAgentId}` |
| `/ai/conversations/{conversationId}/custom-agents` | `/conversations/{conversationId}/custom-agents` |

The routes in each row are the same FastAPI handler mounted twice. Do not alternate namespaces within one frontend. The current Streamlit client uses the canonical aliases; the AI SDK frontend should consistently use the recommended `/ai` routes.

## Common response and error envelopes

Successful management responses use:

```ts
type ApiResponse<T> = {
  success: true;
  message: string;
  data: T;
  error: null;
};
```

Delete succeeds without a resource payload and currently serializes nullable envelope fields:

```json
{
  "success": true,
  "message": "Custom agent deleted",
  "data": null,
  "error": null
}
```

Application errors use:

```ts
type ApiError = {
  success: false;
  code: string;
  message: string;
  error?: Record<string, string[]>;
};
```

Example:

```json
{
  "success": false,
  "code": "CUSTOM_AGENT_IN_USE",
  "message": "Custom agent is currently active or paused and cannot be modified"
}
```

Request-shape validation uses `422 invalid_input`; `error` is keyed by a path such as `body.name` or `path.custom_agent_id`.

```json
{
  "success": false,
  "code": "invalid_input",
  "message": "Invalid input",
  "error": {
    "body.customAgentIds": ["Value error, custom_agent_ids must not contain duplicates"]
  }
}
```

Unknown additive response fields and unknown stream event types must be ignored.

## Naming and JSON casing

Request schemas accept camelCase and snake_case. Use camelCase for new request code.

Top-level response models use camelCase, but catalog entries and persisted `toolRefs`/`skillRefs` are currently untyped dictionaries whose nested keys remain snake_case. The mixed casing is current wire behavior:

```json
{
  "providerType": "openai",
  "runtimeAgentId": "custom_agent:...",
  "toolRefs": [
    {
      "type": "client",
      "device_id": "desktop-1",
      "qualified_tool_id": "client__csv__profile"
    }
  ]
}
```

Recommended frontend boundary: normalize nested dictionary keys once in the API client. Until the backend is corrected, tolerate both spellings when reading:

```ts
function readKey<T>(value: Record<string, any>, camel: string, snake: string): T | undefined {
  return value[camel] ?? value[snake];
}
```

## Core resource

```ts
type UUID = string;

type CustomAgent = {
  id: UUID;
  createdAt: string;
  updatedAt: string;
  ownerId: UUID;
  name: string;
  slug: string;
  description: string | null;
  prompt: string;
  providerType: string;
  model: string;
  temperature: number | null;
  reasoningEffort: string | null;
  toolRefs: ToolRefWire[];
  skillRefs: SkillRefWire[];
  enabled: boolean;
  runtimeAgentId: `custom_agent:${string}`;
};
```

`runtimeAgentId`, not `id` or `slug`, is the workflow identity used in routing and chat metadata. It is derived as `custom_agent:${id}` and remains stable for the life of the resource.

The frontend should use:

- `id` for CRUD and attachment request bodies.
- `runtimeAgentId` for stream-event and message-metadata lookup.
- `name` for display.
- `slug` only as informational data; there is no slug-addressed endpoint.

## Options endpoint

```http
GET /ai/custom-agents/options?deviceId=desktop-1
```

`deviceId` is optional at the HTTP layer but required to receive active client-side tools and client-side skills. The sidecar injects its active device automatically. Direct API clients must pass it.

Response shape:

```ts
type CustomAgentOptions = {
  providers: ProviderOptionWire[];
  serverDefaultTools: ServerToolOptionWire[];
  serverTools: Record<string, unknown>[];
  clientTools: ClientToolOptionWire[];
  clientServers: ClientServerOptionWire[];
  skills: SkillOptionWire[];
};

type ProviderOptionWire = {
  provider_type: string;
  models: ModelOptionWire[];
};

type ModelOptionWire = {
  id: string;
  display_name: string;
  [key: string]: unknown;
};

type ServerToolOptionWire = {
  type: "server_mcp";
  server_name: string;
  tool_name: string;
  qualified_tool_id: string;
  description: string;
  args_schema: Record<string, unknown>;
};

type ClientToolOptionWire = {
  type: "client";
  device_id: string;
  session_id: string;
  catalog_version: string;
  tool_instance_id: string;
  server_name: string;
  qualified_tool_id: string;
  tool_name: string;
};

type ClientServerOptionWire = {
  server_name: string;
  device_id: string | null;
  tool_count: number;
};

type SkillOptionWire = {
  source: "server" | "client";
  lookup_name: string;
  name: string;
};
```

Example:

```json
{
  "success": true,
  "message": "Custom agent options retrieved",
  "data": {
    "providers": [
      {
        "provider_type": "openai",
        "models": [
          { "id": "gpt-4.1-mini", "display_name": "gpt-4.1-mini" }
        ]
      }
    ],
    "serverDefaultTools": [
      {
        "type": "server_mcp",
        "server_name": "calculator",
        "tool_name": "calculate",
        "qualified_tool_id": "calculator::calculate",
        "description": "Calculate an expression",
        "args_schema": {}
      }
    ],
    "serverTools": [],
    "clientTools": [
      {
        "type": "client",
        "device_id": "desktop-1",
        "session_id": "session-1",
        "catalog_version": "7",
        "tool_instance_id": "csv-profile-instance",
        "server_name": "csv",
        "qualified_tool_id": "client__csv__profile",
        "tool_name": "profile"
      }
    ],
    "clientServers": [
      { "server_name": "csv", "device_id": "desktop-1", "tool_count": 1 }
    ],
    "skills": [
      { "source": "server", "lookup_name": "data-analysis", "name": "Data Analysis" }
    ]
  },
  "error": null
}
```

Frontend rules:

- Use `providers[].provider_type` as the create/update `providerType` and `models[].id` as `model`.
- Preserve the complete selected tool or skill option when building a ref. Do not reconstruct client refs from display names.
- Treat client options as a live snapshot. Refresh options after reconnect/device change and before editing refs.
- Use `clientServers` only to group all tools from a sidecar MCP server. Saving a group still sends each underlying `clientTools` entry in `toolRefs`.
- Ignore `serverTools` for now; the implementation always returns `[]`. Backend MCP options are in `serverDefaultTools`.
- An options call may refresh provider and MCP catalogs and can therefore be slower than a normal read. Show a loading state.

## Tool and skill references

Create/update request bodies use this logical union. CamelCase is shown; snake_case is also accepted.

```ts
type ToolRefRequest =
  | {
      type: "server_mcp";
      serverName: string;
      toolName: string;
      qualifiedToolId: string;
      displayMetadata?: Record<string, unknown> | null;
    }
  | {
      type: "server_default";
      qualifiedToolId: string;
      displayMetadata?: Record<string, unknown> | null;
    }
  | {
      type: "client";
      deviceId: string;
      sessionId: string;
      catalogVersion: string;
      toolInstanceId: string;
      serverName: string;
      qualifiedToolId: string;
      toolName: string;
      displayMetadata?: Record<string, unknown> | null;
    };

type SkillRefRequest = {
  source: "server" | "client";
  lookupName: string;
  name: string;
  displayMetadata?: Record<string, unknown> | null;
};
```

`ToolRefWire` and `SkillRefWire` in read responses contain the same data but currently use nested snake_case keys.

Client tool refs are exact, device-scoped identities. Create/update validation requires the tool to be present in the active device catalog. At runtime the backend can rebase volatile `session_id`, `catalog_version`, and `tool_instance_id` fields after the same device reconnects, using `(device_id, qualified_tool_id)` as the stable identity. A tool missing from the current session is skipped, and the final assistant metadata may contain `custom_agent_warnings`.

Skill refs are restrictive: only selected, currently resolvable skills appear in the custom agent's prompt and can be activated.

Backend MCP refs currently are not restrictive at runtime. See [IR-2](#ir-2-server-mcp-selection-does-not-restrict-runtime-access-high).

## Create

```http
POST /ai/custom-agents?deviceId=desktop-1
Content-Type: application/json
```

```json
{
  "name": "Data Analyst",
  "description": "Answers analytics questions.",
  "prompt": "You are a precise data analyst. Ask before making assumptions.",
  "providerType": "openai",
  "model": "gpt-4.1-mini",
  "temperature": 0.2,
  "reasoningEffort": null,
  "toolRefs": [
    {
      "type": "server_mcp",
      "serverName": "calculator",
      "toolName": "calculate",
      "qualifiedToolId": "calculator::calculate"
    },
    {
      "type": "client",
      "deviceId": "desktop-1",
      "sessionId": "session-1",
      "catalogVersion": "7",
      "toolInstanceId": "csv-profile-instance",
      "serverName": "csv",
      "qualifiedToolId": "client__csv__profile",
      "toolName": "profile"
    }
  ],
  "skillRefs": [
    {
      "source": "server",
      "lookupName": "data-analysis",
      "name": "Data Analysis"
    }
  ],
  "enabled": true
}
```

| Field | Required | Rules |
|---|---:|---|
| `name` | yes | Trimmed non-empty string; maximum 255 characters. Its normalized slug must be unique among this user's live agents. |
| `description` | no | String or `null`; maximum 2,000 characters. |
| `prompt` | yes | Trimmed non-empty string. No configured maximum length. |
| `providerType` | yes | Non-empty; maximum 64 characters; must be configured/supported for this user. |
| `model` | yes | Non-empty; maximum 255 characters; must validate against the user's provider model catalog. |
| `temperature` | no | `null` or number from `0` through `2`. |
| `reasoningEffort` | no | `null` or string up to 32 characters. The schema does not publish an allowed enum. |
| `toolRefs` | no | Defaults to `[]`; duplicate refs are silently collapsed, preserving first occurrence. |
| `skillRefs` | no | Defaults to `[]`; each ref must exist in the current server/active-device skill catalog. |
| `enabled` | no | Defaults to `true`, but is not enforced by the runtime; see [IR-1](#ir-1-enabled-is-persisted-but-not-enforced-critical). |

The successful `data` is a complete `CustomAgent`.

Name uniqueness is slug-based and user-scoped. Names such as `Data Analyst`, `data-analyst`, and accent-normalized equivalents can collide.

## List and read

```http
GET /ai/custom-agents
GET /ai/custom-agents/{customAgentId}
```

List response `data` is `CustomAgent[]`, ordered newest first. The endpoint is not paginated.

Read response `data` is one `CustomAgent`. A live agent owned by another user returns `403 CUSTOM_AGENT_FORBIDDEN`; a missing or soft-deleted ID returns `404 CUSTOM_AGENT_NOT_FOUND`.

## Update

```http
PATCH /ai/custom-agents/{customAgentId}?deviceId=desktop-1
Content-Type: application/json
```

The body accepts any subset of create fields. Omitted fields stay unchanged.

```json
{
  "description": "Updated description",
  "temperature": 0.4,
  "toolRefs": [],
  "skillRefs": []
}
```

Rules:

- Send `[]` to clear tools or skills.
- `description: null`, `temperature: null`, and `reasoningEffort: null` clear those optional values.
- Never send `null` for `name`, `prompt`, `providerType`, `model`, or `enabled`; omit an unchanged field. Current validation allows some of these values through to a later failure. See [IR-7](#ir-7-patch-nullability-can-produce-server-errors-high).
- Updating `providerType` or `model` validates the resulting pair.
- Updating a tool or skill list revalidates the entire supplied list against the current catalogs.
- An empty body succeeds and returns the unchanged agent.
- Update returns `409 CUSTOM_AGENT_IN_USE` while that runtime agent is active or paused according to the backend lock registry.
- Editing is live: future turns resolve the latest saved configuration. Existing in-flight execution does not mutate in place.

## Delete

```http
DELETE /ai/custom-agents/{customAgentId}
```

Delete is a soft delete and removes every conversation attachment for this agent in the same database transaction. There is no restore endpoint.

Delete returns `409 CUSTOM_AGENT_IN_USE` while the agent is active or paused according to the lock registry. The frontend should require confirmation because the operation affects all conversations and cannot be undone through the API.

## Conversation attachments

### Read attachments

```http
GET /ai/conversations/{conversationId}/custom-agents
```

Response `data` is the complete `CustomAgent[]` in persisted attachment order.

This dedicated endpoint is authoritative. Do not expect `customAgents` to be populated by normal conversation CRUD responses; see [IR-5](#ir-5-conversationread-advertises-customagents-but-never-populates-it-medium).

### Replace attachments

```http
PUT /ai/conversations/{conversationId}/custom-agents
Content-Type: application/json
```

```json
{
  "customAgentIds": [
    "0c73f7d3-5aca-4a7e-8b71-4cf802f4d901",
    "4e256b3e-bb8d-4fb4-88f8-296ee37c1932f"
  ]
}
```

This is replace-all semantics, not append semantics:

- The array contains database `id` values, not `runtimeAgentId` values.
- Array order is persisted and returned. It orders router/planning descriptors but is not a guaranteed routing priority.
- Every ID must identify a live agent owned by the authenticated user.
- Duplicate IDs return `422 invalid_input`.
- `[]` detaches every custom agent.
- Detaching an active/paused selected custom agent returns `409 CUSTOM_AGENT_IN_USE`.
- Adding an attachment or reordering the same set can succeed during an active turn, but the running turn continues with its existing snapshot. The change applies to a later turn.
- There is currently no maximum attachment count.

Successful `data` is the complete updated ordered `CustomAgent[]`.

## New-conversation flow

Custom-agent attachments are not accepted by `POST /ai/conversations` or by the chat request. For a new conversation:

1. Create the conversation with `POST /ai/conversations`.
2. Persist selected agents with `PUT /ai/conversations/{conversationId}/custom-agents`.
3. Only after the attachment request succeeds, submit the first message to `POST /api/chat/{conversationId}`.

If step 2 fails, do not silently send the first message without the selected custom agents. Keep the draft and offer retry or explicitly tell the user the message will use base agents.

For an existing conversation, load attachments from the dedicated `GET` endpoint when opening the conversation. Cache a map keyed by `runtimeAgentId` for stream display:

```ts
const customAgentByRuntimeId = new Map(
  attachedAgents.map((agent) => [agent.runtimeAgentId, agent]),
);
```

## Chat and routing behavior

The chat body is unchanged from [`AI_SDK_FE_CONTRACT.md`](AI_SDK_FE_CONTRACT.md). Do not send `customAgentIds`, definitions, prompts, models, tool refs, or skill refs in the chat payload.

On every new user turn, the backend resolves the conversation's current attachments into workflow state. With no attachments, the existing base-agent behavior is unchanged.

Attached custom agents participate in:

- Initial router selection.
- Deterministic selection when the user explicitly mentions an attached agent's display name or full runtime ID.
- Cross-agent `hand_off` in both directions: base to custom, custom to base, and custom to another attached custom agent.
- Planning-mode worker dispatch using the runtime ID as the worker target.
- ReAct tool execution with selected client tools and skills.
- Turn-to-turn stickiness: after a custom agent answers, a natural follow-up stays with it unless the user explicitly names a different attached custom agent, the agent has been detached, or an active plan takes control.

Routing is server-controlled. The frontend does not choose an agent by adding a request field. A UI affordance such as “Ask Data Analyst” should put an unambiguous display-name mention in the user's message unless a future explicit-selection endpoint is added.

## Stream contract additions

The full SSE lifecycle remains defined by [`AI_SDK_FE_CONTRACT.md`](AI_SDK_FE_CONTRACT.md). Custom agents add identities to existing events and metadata; they do not introduce a new event type.

### Agent selected

Current AI SDK projection:

```json
{
  "type": "data-agent-selected",
  "data": {
    "agent": "custom_agent:0c73f7d3-5aca-4a7e-8b71-4cf802f4d901"
  },
  "transient": true
}
```

Use `data.agent` as the stable key and resolve the name through the attachment/list cache. Although the internal event carries `agent_name`, the AI SDK adapter currently drops it; see [IR-4](#ir-4-live-agent-selected-drops-the-display-name-medium).

Multiple `data-agent-selected` events may appear in one turn when a handoff changes the responding agent. Update the active-agent indicator each time.

### Final and history metadata

The authoritative responding identity is in final `data-assistant-message.data.message.metadata` and later message history:

```json
{
  "agent": {
    "id": "custom_agent:0c73f7d3-5aca-4a7e-8b71-4cf802f4d901",
    "kind": "custom",
    "name": "Data Analyst",
    "custom_agent_id": "0c73f7d3-5aca-4a7e-8b71-4cf802f4d901",
    "source": "response"
  },
  "custom_agent_warnings": [
    "Selected client tool 'client__csv__profile' is unavailable in the active device session and was skipped."
  ]
}
```

Metadata uses snake_case. Prefer `metadata.agent` over legacy compatibility fields such as `runtime_agent_id`, `custom_agent_id`, and `custom_agent_name`.

`agent.source` is:

- `response` when identity comes from the agent that produced the response.
- `selected_agent` when the backend only has the selected workflow identity.

Show `custom_agent_warnings` as a nonfatal warning on the message. The answer can still complete with unavailable optional tools or skills.

### Handoff metadata

When present:

```json
{
  "handoff": {
    "from_agent_id": "chat_agent",
    "to_agent_id": "custom_agent:0c73f7d3-5aca-4a7e-8b71-4cf802f4d901",
    "reason": "The custom analyst is better suited.",
    "tool_call_id": "call-1"
  }
}
```

Treat both IDs as runtime identities. Resolve custom IDs through the cache and base IDs through the normal base-agent name map.

### Planning subagents

Live planning progress uses the existing `data-subagent` event. For a custom worker, `data.subagent.name` can be the runtime ID:

```json
{
  "type": "data-subagent",
  "transient": true,
  "data": {
    "phase": "start",
    "subagent": {
      "id": "worker-1",
      "name": "custom_agent:0c73f7d3-5aca-4a7e-8b71-4cf802f4d901",
      "path": ["planning_agent", "worker-1"],
      "status": "running"
    },
    "task": "Analyze the metrics."
  }
}
```

Resolve custom worker names through the same runtime-ID cache. The durable `metadata.subagent_results[]` projection contains normalized identity fields:

```json
{
  "id": "worker-1",
  "agent": "custom_agent:0c73f7d3-5aca-4a7e-8b71-4cf802f4d901",
  "agent_name": "Data Analyst",
  "agent_kind": "custom",
  "custom_agent_id": "0c73f7d3-5aca-4a7e-8b71-4cf802f4d901",
  "status": "completed",
  "summary": "..."
}
```

## HITL, locks, and conflicts

The backend attempts to prevent mutation of the runtime definition while it is executing or paused:

| Action | Conflict condition |
|---|---|
| `PATCH` agent | That runtime agent is active/paused, or an attached conversation has an active turn whose selected agent is not known yet. |
| `DELETE` agent | Same as update. |
| `PUT` attachments | The request would detach an agent selected by an active/paused run in that conversation. |

On `409 CUSTOM_AGENT_IN_USE`:

- Preserve unsaved form state.
- Explain that the agent is active or awaiting approval.
- Offer retry after the turn/approval completes.
- Refetch the agent or attachment list after a successful retry.

There is no status endpoint that lets the frontend reliably pre-disable edit/delete/detach controls. Optimistic controls plus explicit `409` handling are required.

Resume reloads the current attached-agent map. If a paused checkpoint selected a custom agent that is no longer attached, the resume stream emits a terminal error with:

```json
{
  "type": "error",
  "errorText": "The custom agent for this paused conversation is no longer attached and cannot be resumed.",
  "statusCode": 409,
  "errorCode": "CUSTOM_AGENT_RESUME_CONFLICT"
}
```

Treat it as terminal under the normal AI SDK stream rules.

## Error matrix

| HTTP/status | Code | Applies to | Frontend behavior |
|---:|---|---|---|
| `400` | `CUSTOM_AGENT_VALIDATION_FAILED` | Invalid model, unavailable tool/skill, duplicate live slug/name, or other service validation. | Keep form values and show `message`. Refresh options when a previously selected catalog item is unavailable. |
| `401` | authentication code | Any route. | Run normal token refresh/sign-in flow. |
| `403` | `CUSTOM_AGENT_FORBIDDEN` | Agent exists but belongs to another user. | Do not expose or cache the resource. Return to list. |
| `403` | `CONVERSATION_ACCESS_DENIED` | Conversation exists but belongs to another user. | Return to conversation list. |
| `404` | `CUSTOM_AGENT_NOT_FOUND` | Missing or soft-deleted agent. | Remove it from local lists and selections. |
| `404` | `CONVERSATION_NOT_FOUND` | Missing/soft-deleted conversation. | Close the conversation and preserve any unsent draft. |
| `409` | `CUSTOM_AGENT_IN_USE` | Update/delete/detach conflicts with active or paused use. | Preserve state and offer retry. |
| stream `409` | `CUSTOM_AGENT_RESUME_CONFLICT` | Paused run's selected custom agent is no longer attached. | End pending approval UI and show a non-retryable conflict for that checkpoint. |
| `422` | `invalid_input` | Invalid UUID, missing required field, wrong type, duplicate attachment IDs, schema bounds. | Map `error` paths to fields where possible. |
| `500` | `internal_server_error` | Unexpected backend failure. | Keep unsaved state, show a generic retry action, and log correlation context if available. |

## Recommended frontend state model

```ts
type CustomAgentState = {
  all: CustomAgent[];
  byId: Map<UUID, CustomAgent>;
  byRuntimeId: Map<string, CustomAgent>;
  attachedByConversation: Map<UUID, UUID[]>;
  optionsByDevice: Map<string, CustomAgentOptions>;
};
```

Recommended invalidation:

- After create: refetch/list or insert returned `data`.
- After update: replace both `byId` and `byRuntimeId` entries with returned `data`.
- After delete: remove the agent globally and from every locally cached attachment set.
- After attachment `PUT`: replace the entire conversation attachment array with returned order.
- On device/session change: invalidate options and refresh before editing client tools/skills.
- On `CUSTOM_AGENT_NOT_FOUND`: evict the missing ID.
- On `CUSTOM_AGENT_IN_USE`: do not discard local form changes.

## Implementation review

These findings are based on the current routes, schemas, services, runtime policy, stream adapter, and tests. They are backend issues or contract hazards; the frontend mitigations below do not replace backend fixes.

### IR-1: `enabled` is persisted but not enforced (critical)

`enabled` is accepted, stored, returned, and patchable, but list/attachment/runtime-state queries do not filter it and workflow state does not carry it. An attached agent with `enabled: false` can still be routed to and executed.

Frontend mitigation: do not present `enabled` as a working runtime switch. Hide the toggle or label it unavailable until the backend filters disabled agents consistently. Do not rely on it for safety or access control.

Recommended backend fix: define disabled semantics, prevent new attachments/routing for disabled agents, and decide whether disabling should detach existing attachments or merely suppress them.

### IR-2: Server MCP selection does not restrict runtime access (high)

The API validates and stores selected `server_mcp` refs, but the runtime sets `allow_all_server_tools=True`. A custom agent can discover and load all backend MCP tools regardless of its stored server selections. Client tool and skill restrictions are enforced more narrowly.

Frontend mitigation: do not describe server tool selection as a permission boundary. If the picker remains visible, describe it as a saved preference/hint.

Recommended backend fix: either enforce stored server refs in discovery, binding, and execution, or remove the misleading selector and fields from this product surface.

### IR-3: In-use locks are process-local and expire before default HITL timeout (high)

The mutation lock is an in-memory `TTLCache` with a 10-minute TTL. The default HITL approval timeout is 30 minutes. The registry is not shared across API workers and does not survive restart. Consequently, edit/delete/detach can become possible while a durable interrupt is still resumable.

Frontend mitigation: handle `409` when it is returned, but do not assume the absence of `409` proves no paused run exists.

Recommended backend fix: derive paused locks from the durable interrupt/checkpoint store or use a shared lock store with a TTL at least as long as the interrupt lifecycle.

### IR-4: Live `agent-selected` drops the display name (medium)

The internal event adds `agent_name` for custom agents, but the AI SDK adapter emits only `data.agent`. This conflicts with implementation comments that imply the enriched field reaches AI SDK clients.

Frontend mitigation: map `data.agent` through `byRuntimeId`; fall back to `Custom Agent` plus a short ID if the definition was deleted or not loaded.

Recommended backend fix: add an optional `agentName` (or a normalized `agent` identity object) to `data-agent-selected` without changing the existing `agent` key.

### IR-5: `ConversationRead` advertises `customAgents` but never populates it (medium)

The conversation response schema contains `customAgents`, but `ConversationService` never fills it for create/get/list/update. Clients will receive `null`, including when agents are attached.

Frontend mitigation: always call the dedicated conversation custom-agents endpoint.

Recommended backend fix: either implement an explicit `include=custom_agents` path consistently, including for AI SDK conversation routes, or remove the misleading field.

### IR-6: Nested response casing is inconsistent and weakly typed (medium)

Top-level Pydantic response fields are camelCase, while nested option/ref dictionaries remain snake_case. OpenAPI describes these nested entries only as arbitrary objects, so generated clients cannot produce useful types.

Frontend mitigation: normalize at the API boundary and tolerate both spellings.

Recommended backend fix: return typed option/ref response models and choose one public casing convention.

### IR-7: PATCH nullability can produce server errors (high)

The update schema declares required persisted fields such as `name`, `prompt`, and `enabled` nullable. The service later passes invalid nulls into slug generation or non-null database columns. Some bodies can therefore produce `500` instead of a field-level `422`/`400`.

Frontend mitigation: never send null for `name`, `prompt`, `providerType`, `model`, or `enabled`; omit unchanged values.

Recommended backend fix: make non-clearable update fields `Optional` only in the “omittable” sense while rejecting explicit null with validators.

### IR-8: The `/ai` and canonical route families are duplicate aliases (low)

The backend mounts the same routers twice and the sidecar duplicates the proxy methods. There is no semantic difference. This increases documentation, test, and maintenance surface and permits clients to drift between namespaces.

Frontend mitigation: use `/ai` consistently in the AI SDK client.

Recommended backend fix: declare one family canonical, mark the other deprecated, instrument usage, migrate existing clients, then remove aliases only after a compatibility window.

### IR-9: No attachment limit or pagination (medium)

Agent lists are unpaginated, and a conversation can attach an unbounded number of agents. Every attachment is included in workflow routing/handoff/planning context, so very large sets can increase prompt size, ambiguity, and latency.

Frontend mitigation: apply a conservative product limit in the picker until the backend publishes one, while still rendering larger server-returned sets safely.

Recommended backend fix: enforce a maximum attachment count and paginate the global agent list if expected user counts warrant it.

### IR-10: Duplicate-name precheck is race-prone (low)

The service checks for a live slug before insert/update and the database also has a unique partial index. Concurrent requests can pass the precheck and then surface an unhandled integrity error as `500` instead of `400 CUSTOM_AGENT_VALIDATION_FAILED`.

Frontend mitigation: disable duplicate form submission and keep form state on `500`.

Recommended backend fix: catch the unique-constraint error and map it to the documented validation error.

### IR-11: Cross-owner lookup reveals existence via `403` vs `404` (low/security)

Reading a known live UUID owned by another user returns `403`, while a missing or soft-deleted UUID returns `404`. This is ownership-safe but leaks resource existence to a caller who already has a UUID.

Frontend mitigation: none required beyond normal error handling.

Recommended backend fix: if non-discoverability is a requirement, query by `(owner_id, id)` and return `404` for both cases.

## FE acceptance checklist

- Uses only the recommended `/ai` management/attachment namespace.
- Creates the conversation and saves attachments before sending its first chat message.
- Uses agent database IDs for attachment bodies and runtime IDs for stream/metadata lookup.
- Normalizes mixed nested key casing at the API boundary.
- Refreshes device-scoped options after reconnect/device changes.
- Preserves complete client tool identity refs instead of matching by display name.
- Treats attachment `PUT` as replace-all and preserves server-returned order.
- Handles `409 CUSTOM_AGENT_IN_USE` without losing edits.
- Handles terminal `CUSTOM_AGENT_RESUME_CONFLICT` in resume streams.
- Resolves live custom-agent names from the runtime-ID cache.
- Displays final `custom_agent_warnings` as nonfatal message warnings.
- Does not present `enabled` or server MCP selections as enforced runtime/security controls until backend issues IR-1 and IR-2 are fixed.
- Ignores unknown additive response fields and unknown SSE event types.

## Device capability resolution (account-wide intent, device-local execution)

Saved MCP/skill selections are account-wide *desired* capabilities keyed by stable
logical identity. They are resolved against the requesting device's live catalog for
availability and runtime binding. The frontend MUST follow these rules:

- Cache options by returned `deviceSnapshot`, not user ID alone.
- Cache client options only when `deviceSnapshot.status=ready`; `unavailable` includes offline and not-yet-synced sessions.
- Match saved client MCP selections by `(server_name, qualified_tool_id)`.
- Require the full current device/session/catalog/tool-instance/tool-name identity for newly submitted client refs.
- Show `availability.status=degraded`/`device_unavailable` as non-blocking.
- Preserve missing saved refs on unrelated edits; do not silently clear them.
- Provide an explicit control to remove retained unavailable refs.
- Never merge catalogs or HITL settings from two devices.
