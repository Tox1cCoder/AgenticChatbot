# AI SDK Frontend Contract

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/chat/{conversationId}` | Chat stream (AI SDK UI Message Stream, SSE). |
| `POST /ai/chat/{conversationId}` | Alias of the above. Use `/api/chat` for new integrations. |
| `POST /ai/resume-interrupt` | Resume after a HITL interrupt. Same SSE stream format. |
| `GET /hitl/interrupts/{interruptId}` | Read the authenticated owner's durable HITL lifecycle state for reconciliation; other owners cannot discover it. |
| `GET /ai/conversations/{conversationId}/messages` | Message history as AI SDK `UIMessage` objects. |
| `POST /ai/conversations`, `GET /ai/conversations`, `GET/PATCH/DELETE /ai/conversations/{id}` | Conversation CRUD (standard `ApiResponse` envelope). |
| `POST /documents/uploads` | Upload document files for RAG/search. Multipart form route; not part of the chat JSON payload. |
| `POST {connection_endpoint}` | Mint a widget WebSocket token (endpoint comes from the widget payload). |
| `POST /widgets/{widgetId}/actions/{actionKey}` | Execute a widget action. |
| `GET /tool-results/{blobId}` | Fetch an offloaded tool output. |

All endpoints require the normal `Authorization: Bearer <jwt>` header.

## Chat Request

```http
POST /api/chat/{conversationId}
Content-Type: application/json
```

```json
{
  "messages": [
    { "role": "user", "content": "Run the report" }
  ],
  "inlineRichResponseV1": true
}
```

| Field | Type | Rules |
|---|---|---|
| `messages` | array | AI SDK UI message history. The backend reads the latest `user` message. |
| `messages[].role` | string | `user` or `assistant`. |
| `messages[].content` | string or array | Plain text, or a parts array. |
| `messages[].parts` | array | Optional. Text parts are `{ "type": "text", "text": "..." }`; image parts and AI SDK file parts carrying image media are accepted (see Request Attachments). |
| `messages[].attachments` | array | Optional image attachments. `experimental_attachments` and `files` are accepted equivalents. |
| `message` | object or string | Optional latest UI message for custom transports. A dict is appended to `messages` when its `id` differs from the last entry. |
| `content` | string | Fallback user text when `messages` is empty. |
| `userId` | string | Client hint only. Ownership and user-aware features use the authenticated JWT identity, never this field. |
| `inlineRichResponseV1` | boolean | Rich-response v1 capability flag. `inline_rich_response_v1` is an accepted equivalent. **Required for rich UI**: without it the response contains no `rich_items`, no markers, no widgets, no canvas. |
| `deviceId` | string | Injected by the sidecar for local runtime tools. `device_id` is an accepted equivalent. |

Unknown extra fields are accepted and ignored.

Error responses (non-stream JSON):

| Status | Body | Cause |
|---|---|---|
| 400 | `{"success": false, "code": "http_error", "message": "No user message found"}` | No user text and no supported image attachments in the request. |
| 400 | `{"success": false, "code": "http_error", "message": "No supported image attachments found. Send image data URLs, raw base64, or http(s) image URLs; upload documents via /documents/uploads."}` | Attachment-like items were sent, but none were usable chat images. |
| 405 | — | Wrong method. |

### Request Attachments

The latest user message may carry chat image inputs in `parts`, `content` (as
an array), `attachments`, `experimental_attachments`, or `files`. This is
image-only: AI SDK `type: "file"` is accepted as a wrapper only when the item
contains image media. Documents such as PDF/DOCX/TXT must be uploaded through
`POST /documents/uploads`, then referenced by normal chat text.

Accepted item fields:

| Field | Rules |
|---|---|
| `type` | `image` or `file`. `file` must carry image media. Other types are ignored. |
| `name` / `filename` | Optional display name. Default `attachment`. |
| `mime` / `mimeType` / `mediaType` / `contentType` | MIME type. Must resolve to `image/*`. Default/guess is `image/jpeg` when omitted. |
| `data` / `base64` | Image data URL or raw base64. Raw base64 must be decodable. |
| `url` | `data:` image URL or `http(s):` image URL. Browser `blob:` URLs are ignored because the backend cannot fetch page-local blobs. |
| `path` / `image` / `source` | Fallback source fields. Local filesystem paths are ignored. |

Unsupported attachment items are not forwarded to the model. If a request has
no user text and all attachment-like items are unsupported, the route returns
the 400 "No supported image attachments found" error above.

Example image request:

```json
{
  "messages": [
    {
      "role": "user",
      "parts": [
        { "type": "text", "text": "Inspect this screenshot" },
        {
          "type": "file",
          "name": "screen.png",
          "mediaType": "image/png",
          "url": "data:image/png;base64,..."
        }
      ]
    }
  ],
  "inlineRichResponseV1": true
}
```

### Document Uploads

Use the document route for non-image files that should be searchable by RAG:

```http
POST /documents/uploads
Content-Type: multipart/form-data
```

Form fields:

| Field | Rules |
|---|---|
| `files` | One or more files. Supported extensions: `.txt`, `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.md`. |
| `conversation_id` | Conversation UUID. |

The response uses the standard `ApiResponse` envelope and reports per-file
accepted/rejected status. Uploading a document does not itself call the
assistant; submit a normal chat message after upload when the user wants an
answer based on the document.

## Stream Protocol

Response headers:

```http
content-type: text/event-stream; charset=utf-8
x-vercel-ai-ui-message-stream: v1
```

Framing: each event is one `data: <json>` line. The stream ends with
`data: [DONE]`. A `{"type": "heartbeat"}` event is emitted after every 15
seconds of source silence; ignore it.

Every stream begins:

```json
{ "type": "start", "messageId": "<assistant-message-uuid>" }
{ "type": "start-step" }
{ "type": "text-start", "id": "<text-part-uuid>" }
```

`messageId` equals the persisted assistant message id — do not create a
duplicate row after refetching history.

Terminal sequences (exactly one occurs per stream):

| Outcome | Sequence |
|---|---|
| Complete | optional `file` events → `data-assistant-message` → `text-end` → [`reasoning-end`] → `finish-step` → `finish` → `[DONE]` |
| Interrupt | [`text-delta` with pause copy] → `text-end` → [`reasoning-end`] → `data-interrupt` → `finish-step` → `finish` → `[DONE]` |
| Error | `error` → `text-end` → [`reasoning-end`] → [`data-error-message`] → `finish-step` → `finish` → `[DONE]` |

Event catalog:

| `type` | Transient | Purpose |
|---|---|---|
| `start`, `start-step`, `text-start`, `text-delta`, `text-end`, `finish-step`, `finish` | — | AI SDK lifecycle and answer text. |
| `reasoning-start`, `reasoning-delta`, `reasoning-end` | — | Model reasoning text. Emitted only when reasoning exists. |
| `tool-input-start`, `tool-input-available`, `tool-output-available` | — | Tool calls. |
| `file` | — | Generated/selected image as a file part. |
| `data-user-message` | yes | Persisted user message echo. |
| `data-agent-selected` | yes | Routing decision. |
| `data-continuation`, `data-node-complete` | yes | Planning-loop progress. |
| `data-rich-items` | yes | Live rich-item upserts (capable requests only; see emission rules). |
| `data-subagent` | yes | Live worker progress. |
| `data-assistant-message` | no | Final assistant message side-channel (metadata + parts). |
| `data-interrupt` | no | HITL pause. Terminal. |
| `data-error-message` | yes | Persisted error message echo. |
| `error` | — | Error text. Terminal. |
| `heartbeat` | — | Keep-alive. |

Unknown event types must be ignored.

## Stream Events

Text:

```json
{ "type": "text-delta", "id": "text-part-id", "delta": "hello" }
{ "type": "text-end", "id": "text-part-id" }
```

Reasoning:

```json
{ "type": "reasoning-start", "id": "reasoning-id" }
{ "type": "reasoning-delta", "id": "reasoning-id", "delta": "..." }
{ "type": "reasoning-end", "id": "reasoning-id" }
```

Tool call:

```json
{ "type": "tool-input-start", "toolCallId": "tool-call-id", "toolName": "tool_name" }
{ "type": "tool-input-available", "toolCallId": "tool-call-id", "toolName": "tool_name", "input": {} }
{ "type": "tool-output-available", "toolCallId": "tool-call-id", "output": "any JSON value", "render": {} }
```

`input` and `output` are any JSON value. JSON-shaped strings are parsed before
emission (never double-encoded). `render` is present only when the tool
produced a structured render payload (see Tool Artifacts).

File (image) part:

```json
{ "type": "file", "url": "data:image/png;base64,...", "mediaType": "image/png" }
```

User message echo — a wire-safe projection. Database fields
(`conversation_id`, `sender`, `updated_at`) are never included:

```json
{ "type": "data-user-message", "transient": true,
  "data": { "message": {
    "id": "user-message-id",
    "role": "user",
    "content": "user text",
    "createdAt": "ISO-8601",
    "metadata": {}
  } } }
```

Routing and planning progress:

```json
{ "type": "data-agent-selected", "data": { "agent": "chat_agent" }, "transient": true }
{ "type": "data-continuation", "data": { "round": 1, "max_rounds": 3, "reason": "tool_budget" }, "transient": true }
{ "type": "data-node-complete", "data": { "node": "chat_agent" }, "transient": true }
```

Rich item upsert (only for requests that sent `inlineRichResponseV1: true`):

```json
{ "type": "data-rich-items",
  "data": { "operation": "upsert", "items": [] },
  "transient": true }
```

`data-rich-items` is a partial live-progress channel, not the full registry:

| Item type | Streams transiently? | Delivered via |
|---|---|---|
| `live_widget`, `tool_render` | Yes, as each safe tool completes. | Upsert **and** final `data-assistant-message.metadata.rich_items`. |
| `image` | Never. | Final `data-assistant-message.metadata.rich_items` / history for placement, plus selected image `file` parts for media. |
| `canvas_artifact` | Never. | Final `data-assistant-message.metadata.rich_items` / history only. |
| Any item whose payload carries inline binary `data` / `base64`, including nested renderer content | Never. | Final / history only. |

A turn with zero `data-rich-items` events is normal (canvas-only,
image-only, or no rich activity). The authoritative registry is always
`metadata.rich_items` on the final `data-assistant-message` and on history.
Transient upserts use the same public rich-item serialization as final metadata:
null-valued keys are omitted, and `provenance` is `{}` when there is no origin
metadata.
A marker streamed in `text-delta` whose item has not arrived yet renders as a
pending placeholder until `finish`.

Final assistant message side-channel:

```json
{ "type": "data-assistant-message",
  "data": { "message": {
    "id": "assistant-message-id",
    "role": "assistant",
    "createdAt": "ISO-8601",
    "metadata": {},
    "parts": []
  } } }
```

- `content` is never included — the answer text already arrived as `text-delta`.
- `metadata` is the scrubbed backend metadata (see Assistant Metadata). There
  is no `messageMetadata` mirror.
- `parts`, when present, contains the message's file parts. For v1 rich
  messages the file parts are exactly the selected images.
- The event is suppressed when the message carries nothing beyond its id.

Error:

```json
{
  "type": "error",
  "errorText": "This interrupt has already been resolved.",
  "statusCode": 409,
  "errorCode": "INTERRUPT_ALREADY_RESOLVED"
}
{ "type": "data-error-message", "data": { "message": {} }, "transient": true }
```

`data-error-message.data.message` uses the same wire-safe projection as
`data-user-message` (with `content` included).

`statusCode` and `errorCode` are optional and appear only for known domain
failures. They are the AI SDK camel-case projection of the internal
`status_code` and `error_code` values. Unknown exceptions retain `type` and
`errorText` only. Treat the error event as terminal regardless of whether
those optional fields are present.

## Subagent Progress

When the Planning Agent dispatches workers, each worker streams transient
`data-subagent` events interleaved with the rest of the stream:

```json
{ "type": "data-subagent", "transient": true,
  "data": { "phase": "start",
            "subagent": { "id": "worker-a", "name": "search_agent", "path": ["planning_agent", "worker-a"], "status": "running" },
            "task": "Find source material." } }
{ "type": "data-subagent", "transient": true,
  "data": { "phase": "delta",
            "subagent": { "id": "worker-a", "name": "search_agent", "path": ["planning_agent", "worker-a"], "status": "running" },
            "text": "Weighing which sources are authoritative…", "channel": "reasoning" } }
{ "type": "data-subagent", "transient": true,
  "data": { "phase": "tool",
            "subagent": { "id": "worker-a", "name": "search_agent", "path": ["planning_agent", "worker-a"], "status": "running" },
            "toolCallId": "sub-call-1", "toolName": "search_documents", "status": "success" } }
{ "type": "data-subagent", "transient": true,
  "data": { "phase": "end",
            "subagent": { "id": "worker-a", "name": "search_agent", "path": ["planning_agent", "worker-a"], "status": "completed" },
            "summary": "Full worker answer…", "thinking": "Final reasoning…", "elapsedMs": 1234 } }
```

| Field | Presence | Meaning |
|---|---|---|
| `data.phase` | always | `start`, `delta`, `tool`, or `end`. |
| `data.subagent.id` | always | Stable key — upsert the same UI row across phases. |
| `data.subagent.name` | always | Worker agent name. |
| `data.subagent.path` | always | Hierarchy path. |
| `data.subagent.status` | always | `running`, `completed`, `failed`, `timeout`, or `requires_approval`. |
| `data.task` | `start` | The worker's instruction. |
| `data.text` | `delta` | Live worker model token(s). |
| `data.channel` | `delta` | `reasoning` (worker thinking) or `text` (worker answer). |
| `data.toolCallId` / `data.toolName` | `tool` | The tool the worker invoked. |
| `data.status` | `tool` | Per-tool status (`success` / `error` / …). |
| `data.render` | `tool`, optional | Structured render payload for the worker's tool result. |
| `data.output` | `tool` | The worker tool's output. |
| `data.summary` | `end` | The worker's full, untruncated answer. |
| `data.thinking` | `end`, optional | Reasoning from the worker's final model call. Untruncated. |
| `data.elapsedMs` | `end` | Worker wall-clock duration in ms. |
| `data.error` | `end`, on failure | Error text. |
| `data.requestedModel` / `data.resolvedModel` | `end`, optional | Task-local model override and the model that actually answered. |

Rendering rules:

- `data-subagent` is transient — keep your own map keyed by `subagent.id` for
  a live panel; drop it when the run finishes.
- Accumulate `delta` events per worker per `channel`. `reasoning` deltas are
  the worker's live thinking; `text` deltas are its answer-in-progress. The
  `end` event's `thinking` is the final model call's reasoning, not the
  concatenation of the deltas.
- Worker model output never appears in the top-level `text-delta` /
  `reasoning-delta` stream — those channels carry only the responding agent.
- The durable record is `metadata.subagent_results` on the final
  `data-assistant-message`.
- Resume-path (`/ai/resume-interrupt`) dispatches do not stream live progress.
- Ignore unknown phases.

## Human-in-the-Loop

### Detection

```ts
const isHitl = event.type === "data-interrupt";
const requests = event.data?.interrupt?.action_requests ?? [];
```

The stream terminates immediately after `data-interrupt`.

Interrupt kinds:

| Kind | Detection | FE behavior |
|---|---|---|
| `tool_approval` | `data-interrupt` with `data.interrupt.action_requests[]` | Render approval UI; resume via `POST /ai/resume-interrupt`. |
| `planning_pause` | Final metadata has `planning_budget_reached: true`, `execution_paused: true`, or `execution_pause_reason` | No decision UI. Show the message; the user continues with a normal chat message. |
| `subagent_requires_approval` | Live `data-subagent` `status === "requires_approval"`, or `metadata.subagent_results[].status === "requires_approval"` | Show the worker as blocked. Nested workers are not resumable via `/ai/resume-interrupt`. |

Pause reasons on persisted messages (`metadata.pause_reason` /
`metadata.execution_pause_reason`) and their resume behavior:

| Value | Meaning | Resume |
|---|---|---|
| `tool_approval_required` | Paused for HITL tool approval. | Rebuild approval UI from `metadata.interrupt`. |
| `awaiting_approval` | Workflow/subagent stopped for a tool approval. | Use `metadata.interrupt` if present; otherwise show blocked state. |
| `max_iterations_reached` | Planning loop hit its budget. | No decision. User sends another message. |
| `consecutive_errors_limit` | Planning loop stopped after repeated errors. | No decision. Show retry/continue affordance. |
| `max_tasks_reached` (`execution_pause_reason`) | Planning execution pause. | No decision. Display `execution_pause_message`. |
| `recursion_limit` / `rate_limit` | Execution guard pauses. | No decision unless a `data-interrupt` payload exists. |

### Interrupt Event

```json
{
  "type": "data-interrupt",
  "data": {
    "threadId": "conversation-thread-id",
    "next": ["approval-node"],
    "interrupt": {
      "interrupt_id": "interrupt-id",
      "thread_id": "conversation-thread-id",
      "conversation_id": "conversation-id",
      "action_requests": [
        {
          "action": "tool_name",
          "args": {},
          "description": "optional",
          "task_id": "tool-call-id",
          "tool_call_id": "tool-call-id",
          "allowed_decisions": ["approve", "reject", "edit", "respond"]
        }
      ],
      "metadata": {
        "message": "optional display message",
        "reason": "optional display reason",
        "timeout_deadline": "ISO-8601",
        "device_id": "optional-device-id",
        "tool_provenance": {
          "tool-call-id": {
            "device_id": "optional-device-id",
            "tool_origin": "client_mcp | server_mcp | internal",
            "server_name": "optional-mcp-server",
            "qualified_tool_id": "optional",
            "tool_instance_id": "optional",
            "session_id": "optional",
            "catalog_version": 1
          }
        }
      }
    },
    "message": {
      "id": "paused-assistant-message-id",
      "role": "assistant",
      "content": "",
      "createdAt": "ISO-8601",
      "metadata": {
        "interrupt": {},
        "paused": true,
        "pause_reason": "tool_approval_required",
        "thread_id": "conversation-thread-id",
        "next": ["approval-node"]
      }
    }
  }
}
```

| Field | Rules |
|---|---|
| `data.threadId` | Send back as `threadId` on resume. |
| `data.interrupt.interrupt_id` | Send back as `interruptId` on resume. |
| `data.interrupt.conversation_id` | Send back as `conversationId` on resume. |
| `data.interrupt.action_requests[]` | The list of tool calls awaiting decisions. This is the only such list. |
| `action_requests[].action` | Tool/action name. |
| `action_requests[].args` | The arguments that will run. Display these. |
| `action_requests[].tool_call_id` | Primary decision target. When present, echo it as `toolCallId` on the matching decision. |
| `action_requests[].task_id` | UI correlation id. It is the decision target only when `tool_call_id` is absent. |
| `action_requests[].allowed_decisions` | When present, restrict the decision buttons to these values. |
| `data.interrupt.metadata.timeout_deadline` | Approval window deadline; render countdown/expired state. |
| `data.interrupt.metadata.tool_provenance` | Audit/debug map keyed by tool call id. Not primary UI copy. |
| `data.message` | Wire-safe projection of the persisted paused assistant message. `metadata.interrupt` carries the same payload for rebuild-from-history. |

### Resume

```http
POST /ai/resume-interrupt
```

```json
{
  "threadId": "conversation-thread-id",
  "conversationId": "conversation-id",
  "interruptId": "interrupt-id",
  "decisions": [
    { "type": "approve", "toolCallId": "tool-call-id", "action": "tool_name", "args": {} }
  ],
  "inlineRichResponseV1": true
}
```

`deviceId` is accepted for sidecar-scoped interrupts. Snake-case equivalents
(`thread_id`, `tool_call_id`, `task_id`, `inline_rich_response_v1`) are
accepted everywhere camelCase is shown.

Decision fields:

| Field | Required | Rules |
|---|---|---|
| `type` | yes | `approve`, `edit`, `reject`, or `respond`. |
| `toolCallId` | when the matching request has `tool_call_id` | Must equal that value. |
| `taskId` | no | UI correlation only. Not a substitute for `toolCallId`. |
| `action` | no | Tool name, for audit display. |
| `args` | per type | See below. |

Coverage rules:

- Send exactly one decision per `action_requests[]` entry.
- Incomplete coverage → `422 INTERRUPT_INCOMPLETE_DECISIONS`.
- Set an interrupt-ID-scoped submission lock before this POST and disable every
  control that can submit another resume request.
- Resume is first-write-wins: a duplicate discovered after streaming starts is
  a terminal `200` SSE stream whose `error` event has `errorCode`
  `INTERRUPT_ALREADY_RESOLVED` or `INTERRUPT_CONFLICT`; it is not a second
  successful resume. JSON validation failures that happen before streaming can
  still be ordinary non-stream HTTP errors.

Decision behavior:

| Type | Effect |
|---|---|
| `approve` | Run the original tool call. |
| `edit` | Run with replacement `args`. |
| `reject` | Do not run. Feedback in `args.message` / `args.reason` / `args.feedback`. |
| `respond` | Do not run. Human answer in `args.response` / `args.message`. |

The resume response is the same SSE stream format as chat.

### Durable Interrupt State and Duplicate Reconciliation

```http
GET /hitl/interrupts/{interruptId}
```

Use `fetch(..., { cache: "no-store" })` (or an equivalent uncached client
request) for this endpoint. It is authenticated and owner-filtered; a missing
or foreign ID returns HTTP `404` with `INTERRUPT_NOT_FOUND`:

```json
{
  "success": false,
  "code": "INTERRUPT_NOT_FOUND",
  "message": "HITL interrupt not found."
}
```

Successful responses contain only lifecycle data:

```json
{
  "success": true,
  "message": "HITL interrupt state retrieved",
  "data": {
    "interruptId": "interrupt-id",
    "conversationId": "conversation-id",
    "status": "resolving",
    "expiresAt": "ISO-8601",
    "updatedAt": "ISO-8601"
  }
}
```

`status` is exactly `pending`, `resolving`, `resolved`, `failed`, or `expired`.
The response never contains decisions, tool arguments, device/session data,
user IDs, or secret material.

When the terminal stream error's `errorCode` is exactly
`INTERRUPT_ALREADY_RESOLVED` or `INTERRUPT_CONFLICT`, do exactly one uncached
state GET. Do not automatically replay the resume POST; use the durable state
to choose the next UI state:

| State | Required frontend behavior |
|---|---|
| `pending` | Remove the submission lock and exact-ID suppression marker, then restore the approval form. |
| `resolving` | Clear local decisions, retain the exact-ID suppression marker, and render a non-submittable “already processing” state with one manual **Check status** action. Do not poll. |
| `resolved` | Clear the paused approval UI, retain suppression for the stale paused message, refetch conversation history, and return to normal chat. |
| `failed` | Clear the paused approval UI, retain suppression for the stale paused message, explain that a new message is required, and return to normal chat. |
| `expired` | Clear the paused approval UI, retain suppression for the stale paused message, explain that a new message is required, and return to normal chat. |
| unavailable (including `404`) | Clear the paused approval UI, retain suppression for the stale paused message, explain that a new message is required, and return to normal chat. This is not an additional lifecycle status. |

Do not treat `INTERRUPT_EXPIRED`, `INTERRUPT_FAILED`, or device/session/catalog/
tool mismatch errors as successful completion. Surface their message and do not
automatically retry the resume POST. A claimed resume that later fails reaches
durable `failed`; `failed` and `expired` are terminal, require a new user
message, and neither client may POST resume for that interrupt again.

## Message History

```http
GET /ai/conversations/{conversationId}/messages
```

Query: `page`, `limit`, `orderBy` (`createdAt` default, `updatedAt`),
`orderDirection` (`asc`/`desc`), `inlineRichResponseV1` (also
`inline_rich_response_v1`). Without the rich flag, markers are stripped from
`content` and `rich_items` / `rich_items_version` / `rich_reference_warnings`
are removed from metadata.

```json
{
  "success": true,
  "message": "Messages retrieved successfully",
  "data": {
    "messages": [
      {
        "id": "message-id",
        "role": "assistant",
        "content": "Assistant markdown",
        "parts": [
          { "type": "text", "text": "Assistant markdown" },
          { "type": "file", "url": "data:image/png;base64,...", "mediaType": "image/png" }
        ],
        "metadata": {},
        "createdAt": "ISO-8601"
      }
    ],
    "meta": { "total": 1, "perPage": 20, "currentPage": 1, "lastPage": 1 }
  },
  "error": null
}
```

| Field | Rules |
|---|---|
| `id` | Message UUID. |
| `role` | `user` or `assistant`. |
| `content` | Markdown/plain text. Contains `<!--rich:<id>-->` markers only when the request opted in with `inlineRichResponseV1=true`. |
| `parts` | Always present. A leading `text` part carrying `content` is guaranteed; images appear as `file` parts. |
| `metadata` | Backend metadata, scrubbed (see Assistant Metadata). Null for messages persisted without metadata. |
| `createdAt` | ISO-8601 timestamp. |

The total count is `data.meta.total`.

## Assistant Metadata

```ts
const backendMeta = message.metadata ?? {};
```

`metadata` is the single metadata field on history messages and on every
stream side-channel payload. All fields are optional; ignore unknown keys.
The scrub applied to every AI SDK response removes the legacy renderer
fields, database-redundant keys, and `_`-prefixed internal keys listed in
[Not on the Wire](#not-on-the-wire).

Core runtime/model fields:

| Field | Type | Use |
|---|---|---|
| `persona_used` | string | Persona prompt applied this turn. Debug display. |
| `provider` / `model` | string | Provider and model of the final response. |
| `key_source` | string | Credential source (`env`, `db`, `settings`, `none`). Debug/admin. |
| `config_source` | string | Runtime model-config source. Debug/admin. |
| `config_warnings` | string[] | Non-fatal model/config warnings. |
| `custom_model_override` | boolean | Ad hoc model override was used. Debug/admin. |
| `provider_fallback` | object | `{from, to, reason}` when the backend fell back between providers. |
| `reasoning_effort` | string | Requested effort level. |
| `thinking` | string | Persisted provider thinking text. Treat as debug-only. |
| `thinking_summary` / `reasoning_summary` | string | Short reasoning summary. |
| `reasoning_tokens` | number | Provider-reported reasoning token count. |
| `context_overflow_retry` | boolean | The turn was retried after a context overflow. |

Token/context fields:

| Field | Type | Use |
|---|---|---|
| `token_breakdown.estimated` | object | Estimated prompt/tool/history token counts. |
| `token_breakdown.actual` | object | Provider-reported `input_tokens`, `output_tokens`, `total_tokens`, `reasoning_tokens`. |
| `token_breakdown.counts` | object | History/tool message and bound-tool counts. |
| `token_breakdown.bound_tool_names` | string[] | Tools bound to the model request. Debug/admin. |
| `context_window` | object | Context-window meter. Shape below. |

```json
{
  "context_window": {
    "provider": "openai",
    "model": "gpt-4o",
    "context_window_tokens": 128000,
    "max_input_tokens": 128000,
    "max_output_tokens": 16384,
    "source": "registry",
    "known": true,
    "used_tokens": 12500,
    "used_token_source": "actual_total",
    "usage_ratio": 0.09765625,
    "display_state": "ok"
  }
}
```

When `known` is `false`, the numeric fields are null.

Agent/routing fields:

| Field | Type | Use |
|---|---|---|
| `agent` | object | `{id, kind, name, custom_agent_id, source}` — the responding agent. `kind` is `base` or `custom`; `source` is `response` or `selected_agent`. |
| `handoff` | object | `{from_agent_id, to_agent_id, reason, tool_call_id}`. |
| `custom_agent_warnings` | array | Custom-agent tool/skill availability warnings. |
| `subagent_dispatches` | array | Planning dispatch records. Debug; prefer `data-subagent` live and `subagent_results` durable. |
| `subagent_results` | array | Durable worker records: `{id, agent, agent_name, agent_kind, status, summary}` plus, when present, `custom_agent_id`, `thinking`, `error`, `elapsed_ms`, `related_todo_ids`, `requested_model`, `resolved_model`. `summary` and `thinking` are untruncated. Absent keys are omitted, never null. |
| `subagent_worker_artifacts` | object | Per-worker debug artifacts. |

Renderer/media fields:

| Field | Type | Use |
|---|---|---|
| `tool_artifacts` | array | Persisted tool execution records (see Tool Artifacts). |
| `rich_items_version` | number | `1`. Present only on messages with rich-item activity. |
| `rich_items` | array | The authoritative rich-item registry for widgets, canvas artifacts, tool renders, and selected image placement. |
| `rich_reference_warnings` | array | `{code, id}` validation warnings: `unknown_rich_item`, `invalid_rich_item`. |
| `documents_cited` / `citations` | array | RAG citation metadata (see RAG Citations). |
| `chunks_retrieved` / `documents_found` | number | RAG retrieval counters. |
| `canvas_artifact` (persisted only) | — | Legacy `metadata.canvas_artifact` is never on the AI SDK wire. Canvas is delivered as a normal `message.metadata.rich_items` item. |

Planning/UX fields:

| Field | Type | Use |
|---|---|---|
| `suggested_questions` | string[] | Follow-up suggestions. |
| `reply_to_user_message_id` | string | The user message this reply answers. |
| `interrupt` | object | Persisted HITL payload for paused messages. |
| `paused` / `pause_reason` | boolean/string | Paused-message markers (see HITL). |
| `thread_id` / `next` | string / string[] | Resume identifiers for paused messages. |
| `todos` | array | Task-plan state for planning UI. |
| `planning_call_count` | number | Planning loop calls this turn. |
| `all_tasks_completed` | boolean | Planning finished all tasks. |
| `planning_budget_reached` | boolean | Planning paused at its budget. |
| `todos_synced` | boolean | Task-plan state was synced at persistence. |
| `next_task` | object | Active/next task summary. |
| `planning_rubric` | object | Plan-quality grading metadata. |
| `execution_paused` / `execution_pause_reason` / `execution_pause_message` | boolean/string/string | Planning execution pause (not a HITL approval). |
| `agentic_mode` / `disable_tools` | boolean | RAG agent execution flags. |

### Assistant Images

Selected images are represented twice for v1 rich messages: `image` entries in
`metadata.rich_items` provide placement/provenance, and AI SDK `file` events /
`parts[].type === "file"` provide the renderable media payload. The file parts
are exactly the selected images; unselected candidates are never exposed. Do not
build an image gallery from legacy metadata.

## Rich Response v1

Activation: the request sends `inlineRichResponseV1: true` **and** the server
setting `inline_rich_response_enabled` is on (default on; it is a kill
switch). Non-capable responses contain no markers and no rich keys.

### Serialization

Rich items are serialized with null-valued keys omitted. A key is present
with a real value or absent — `"title": null` and `"data": null` never occur.
Check for presence, not for null.

### Emitted Types

| `type` | Emitted | `id` format | `display_policy` |
|---|---|---|---|
| `image` | yes | `image:tool:<tool-call-id>:<index>`, `image:document:<image-id>` | `inline_only` |
| `live_widget` | yes | `widget:<widget-id>` | `inline_or_append` |
| `tool_render` | yes | `tool:<tool-call-id>` | `inline_or_append` |
| `canvas_artifact` | yes | `canvas:main` | `inline_or_append` |
| `citation` | **no — reserved**, never constructed | — | — |
| `resource_link` | **no — reserved**, never constructed | — | — |

Common fields:

| Field | Presence | Rules |
|---|---|---|
| `id` | always | Stable marker target. |
| `type` | always | One of the emitted types. Render unknown types as an ignorable placeholder. |
| `source` | optional | `web_search` / `tool_image` / `rag_document` (images), `widget_tool` (widgets), `tool` (tool renders). Absent on canvas. |
| `display_policy` | always | `inline_only` or `inline_or_append`. |
| `title` | optional | Always present on canvas; on others only when the source had one. |
| `alt_text` | image only | Always present on image items; absent on every other type. |
| `provenance` | always | Origin metadata; `{}` when there is none. |
| `payload` | always | Type-specific payload. |

Display policies:

| Policy | Behavior |
|---|---|
| `inline_only` | Render only at the marker. Never append. Images use this. |
| `inline_or_append` | Render at the marker when referenced; otherwise append below the answer. |

### Markers

```text
<!--rich:<id>-->
```

| Rule | Value |
|---|---|
| Placement | The marker resolves as a standalone line or embedded in prose, outside code contexts. Standalone lines (up to 3 leading spaces) are preferred. |
| ID charset | ASCII letters, digits, `_`, `-`, `.`, `:`. |
| Max ID length | 128. |
| Code contexts | Markers inside fenced, 4-space-indented, or inline-code spans are literal text. |

### Rich Image Item

```json
{
  "id": "image:tool:tool-call-id:0",
  "type": "image",
  "source": "tool_image",
  "display_policy": "inline_only",
  "alt_text": "Image from tool result",
  "payload": {
    "url": "https://example.com/image.png",
    "mime_type": "image/png",
    "source_url": "https://example.com/source",
    "description": "Optional description"
  },
  "provenance": { "tool_call_id": "tool-call-id", "tool": "tool_name", "index": 0 }
}
```

- Exactly one of `payload.url` or `payload.data` (raw base64) is present. The
  unused key is absent.
- `payload.mime_type` is always present. Allowed: `image/png`, `image/jpeg`,
  `image/webp`, `image/gif`.
- `payload.source_url`, `payload.description`, and item `title` appear only
  when the source provided them.
- `source` is `web_search` (Tavily), `tool_image` (other tools), or
  `rag_document` (document images: id `image:document:<image-id>`,
  provenance `{document_image_id, page_number}`, title `"Page <n>"` when
  known).
- Tool-image provenance may add `thumbnail_url`, `width`, `height`,
  `source_domain`, `provider`. Audit/debug only.
- Images are selection-only: an image item exists only when the final
  markdown references it inline.

### Rich Live Widget Item

```json
{
  "id": "widget:widget-id",
  "type": "live_widget",
  "source": "widget_tool",
  "display_policy": "inline_or_append",
  "title": "Widget title",
  "provenance": {},
  "payload": {
    "widget_id": "widget-id",
    "session_id": "conversation-id",
    "widget_type": "html",
    "status": "active",
    "version": 1,
    "connection_endpoint": "/widgets/widget-id/connection"
  }
}
```

All payload keys are always present. The payload never contains widget
`state` — state arrives over the widget WebSocket. `title` is absent when the
widget has none.

### Rich Tool Render Item

```json
{
  "id": "tool:tool-call-id",
  "type": "tool_render",
  "source": "tool",
  "display_policy": "inline_or_append",
  "payload": {
    "render": {
      "version": 1,
      "type": "mcp_app",
      "template_uri": "ui://server/component.html",
      "structured_content": {}
    }
  },
  "provenance": { "tool_call_id": "tool-call-id", "tool": "tool_name" }
}
```

`payload.render` is renderer-specific (`mcp_app` and other app renders).
Widget renders use the `live_widget` type; error/text renders are not
promoted. `title` is present only when the render carries one.

### Rich Canvas Artifact Item

```json
{
  "id": "canvas:main",
  "type": "canvas_artifact",
  "display_policy": "inline_or_append",
  "title": "Canvas title",
  "provenance": {},
  "payload": {
    "language": "html",
    "title": "Canvas title",
    "content": "<!doctype html>..."
  }
}
```

- At most one per message; the id is always `canvas:main`.
- Created only for capable requests (or messages that already have other
  rich-item activity). Non-capable canvas messages expose no canvas on the AI
  SDK wire.
- Arrives only in the final `data-assistant-message` and history — never as a
  transient upsert, never as a `file` part.
- `payload.content` is executable, untrusted browser content. Render only in
  a sandboxed iframe (`sandbox`, `referrerpolicy="no-referrer"`).
- `payload.language` is `html`, `svg`, or `react`; normalize anything else to
  `html`. CanvasAgent fence-hint normalization: `svg` → `svg`;
  `react`/`jsx`/`js`/`javascript`/`tsx`/`ts` → `react`; everything else →
  `html`.
- `payload.title` falls back to `Canvas`.
- `payload.preferred_height` is schema-accepted but never emitted. Use your
  default frame height.
- A truncated generation still produces an artifact; the assistant text then
  contains a visible "output was cut off" note. There is no wire flag.

### Reserved Types

`citation` and `resource_link` exist in the schema and are never constructed.
You will not receive them. RAG citations are delivered through
`metadata.documents_cited` / `metadata.citations`.

### Auto-Placement

- With `rich_auto_place_enabled`, the backend inserts markers for relevant
  unreferenced image candidates and live widgets into the final persisted
  markdown. Defaults: ≤ 3 auto-placed images per answer, one placed item per
  paragraph, minimum keyword-overlap score 0.25. Widgets are not capped by
  the image limit.
- Auto-placed markers are inserted at persistence — after text deltas have
  streamed. For the exact final layout, refetch history with
  `inlineRichResponseV1=true` after `finish`.
- The authoritative layout is persisted `message.content` +
  `metadata.rich_items`.

### Renderer Algorithm

1. Send `inlineRichResponseV1: true`.
2. Maintain an `items_by_id` map; apply `data-rich-items` upserts into it.
3. Accumulate `text-delta`.
4. Split content on rich markers outside code (standalone line or embedded in prose).
5. Render known items at their markers; render a pending placeholder for
   markers whose item has not arrived.
6. On the final `data-assistant-message`, replace the map with
   `metadata.rich_items`. Refetch history when the exact auto-placed layout
   is required.
7. Append unreferenced `inline_or_append` items below the answer. Never
   append `inline_only` images.

## Live Widgets

The only widget type is `html`: a self-contained micro-app. Widget state
renders exclusively as a sandboxed iframe from `state.html` — never inject it
into the chat DOM. There are no structured widget renderers; do not pick a
renderer by `widget_type`. Legacy persisted items carrying a removed
structured type render as a legacy placeholder.

Widget state (delivered over the WebSocket):

```json
{ "html": "<!doctype html>...", "height": 620, "caption": "Optional short caption" }
```

`height` is 260–960. `caption` is optional.

Mount flow:

1. Render a placeholder from the rich item payload.
2. `POST {payload.connection_endpoint}` with normal auth.
3. Open the returned `ws_url`.
4. Render `widget_state_sync.state.html` in a sandboxed iframe (`srcdoc`).
5. Re-render the iframe on each `widget_update.state`.

Connection response:

```json
{
  "widget_id": "widget-id",
  "session_id": "conversation-id",
  "widget_type": "html",
  "title": "Widget title",
  "status": "active",
  "version": 1,
  "ws_url": "/widgets/widget-id/connect?session_id=conversation-id&token=jwt",
  "token": "jwt",
  "expires_at": "ISO-8601"
}
```

Server WebSocket events — `widget_state_sync` (initial), `widget_update`,
`widget_close`, each carrying `{type, widget_id, widget_type, title, state,
status, version}`; plus:

```json
{ "type": "ping", "timestamp": "ISO-8601" }
{ "type": "error", "message": "Widget is no longer available." }
```

Client WebSocket events:

```json
{ "type": "pong" }
{ "type": "user_state_patch", "patch": { "selection": "row-id" } }
```

Widget actions:

```http
POST /widgets/{widgetId}/actions/{actionKey}
```

```json
{ "input_values": { "note": "demo" }, "state_patch": { "selection": "row-id" } }
```

Response:

```json
{
  "widget_id": "widget-id",
  "session_id": "conversation-id",
  "action_key": "explain_current_state",
  "content": "Message to submit through normal chat stream"
}
```

The action endpoint does not call the assistant. Submit `content` through
`POST /api/chat/{conversationId}`.

## Tool Artifacts

`metadata.tool_artifacts[]` — persisted tool execution records:

```json
{
  "tool_call_id": "tool-call-id",
  "tool": "tool_name",
  "args": {},
  "output": "tool output text or compact JSON string",
  "error": null,
  "status": "success",
  "render": {}
}
```

| Field | Rules |
|---|---|
| `tool_call_id` | Tool call id, or null. |
| `tool` | Tool name. |
| `args` | Tool input. |
| `output` | Output text, capped. Widget outputs are compacted and exclude `state`. |
| `error` | Error text or null. |
| `status` | `success`, `error`, `failed`, or `rejected`. |
| `render` | Optional structured render payload (same shapes as `tool-output-available.render`). |
| `blob_id` | Present when a large output was offloaded. Fetch the full text with `GET /tool-results/{blob_id}`. |
| `blob_size_bytes` | Size of the offloaded output. |
| `output_truncated` | `true` when `output` is a preview plus an offload notice. |

Tool artifacts are plain persisted dicts — unlike rich items, null values
(e.g. `"error": null`) do appear.

## RAG Citations

```json
{
  "documents_cited": [
    {
      "document_id": "document-id",
      "source": "report.pdf",
      "document_number": 1,
      "chunks": [
        { "chunk_index": 0, "score": 0.85, "character_count": 1500, "content": "Chunk text...", "page_number": 5 }
      ],
      "total_chunks": 1,
      "avg_score": 0.85
    }
  ],
  "chunks_retrieved": 3,
  "documents_found": 1,
  "citations": [
    { "document_id": "document-id", "source": "report.pdf", "chunk_index": 0, "page_number": 5, "score": 0.85, "content": "Chunk text..." }
  ]
}
```

Presence depends on the RAG tool path used.

## Detection Helpers

```ts
function getBackendMeta(message: any) {
  return message?.metadata ?? {};
}

function isHitlEvent(event: any) {
  return event?.type === "data-interrupt";
}

function getHitlRequests(event: any) {
  return event?.data?.interrupt?.action_requests ?? [];
}

function getInterruptKind(event: any, backendMeta: any = {}) {
  if (event?.type === "data-interrupt") return "tool_approval";
  if (event?.type === "data-subagent" && event?.data?.subagent?.status === "requires_approval") {
    return "subagent_requires_approval";
  }
  if (backendMeta?.planning_budget_reached || backendMeta?.execution_paused || backendMeta?.execution_pause_reason) {
    return "planning_pause";
  }
  if (Array.isArray(backendMeta?.subagent_results) &&
      backendMeta.subagent_results.some((r: any) => r?.status === "requires_approval")) {
    return "subagent_requires_approval";
  }
  return null;
}

function getRichItems(meta: any, type: string) {
  return Array.isArray(meta?.rich_items)
    ? meta.rich_items.filter((i: any) => i?.type === type)
    : [];
}

const getLiveWidgets = (meta: any) => getRichItems(meta, "live_widget");
const getCanvasItems = (meta: any) => getRichItems(meta, "canvas_artifact");

function getImageParts(message: any) {
  return Array.isArray(message?.parts)
    ? message.parts.filter((p: any) => p?.type === "file" && p?.mediaType?.startsWith("image/"))
    : [];
}

function normalizeCanvasLanguage(payload: any) {
  const value = String(payload?.language ?? "").toLowerCase();
  return value === "svg" || value === "react" ? value : "html";
}
```

## Not on the Wire

Definitive list of things AI SDK responses never contain (removed 2026-07-02
or never implemented). If you find any of these, it is a bug — report it.

| Absent | Use instead |
|---|---|
| `messageMetadata` mirror, `message_metadata` alias | `message.metadata`. |
| `data-interrupt.data.pendingToolCalls` | `data.interrupt.action_requests[]`. |
| `data.total` on the messages listing | `data.meta.total`. |
| `metadata.images`, `has_images`, `images_count`, `agentic_images_count` | `file` parts / image rich items. |
| `metadata.live_widgets` | `live_widget` rich items. |
| `metadata.canvas_artifact` | The `canvas:main` rich item. |
| `metadata.pending_tool_calls` | `metadata.interrupt.action_requests`. |
| `metadata.conversation_id`, `has_tool_calls`, `context_messages` | The route's conversation id; nothing (write-only debug counters). |
| `_`-prefixed metadata keys (`_rich_item_candidates`, `_inline_rich_response_v1`, …) | Internal only. |
| `citation` / `resource_link` rich items | `metadata.documents_cited` / `metadata.citations`. |
| `content` on `data-assistant-message` | Accumulated `text-delta`, or history. |
| Widget `state` in rich item payloads | The widget WebSocket. |
| `payload.preferred_height` on canvas items | Your default frame height. |
| Null-valued keys on rich items | Absent keys. |
