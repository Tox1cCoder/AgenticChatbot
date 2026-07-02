# AI SDK Frontend Contract

This document describes the frontend-facing response shape for the AI SDK-compatible chat path.

## BREAKING CHANGES — 2026-07-02 response-format cleanup

> **Read this first if you built against an earlier revision of this document.**
> The legacy compatibility fields were removed from every AI SDK-visible surface
> (`plans/ai_sdk_response_cleanup.md` records the decision). Persisted data and
> the internal `/messages/stream` path are unchanged.

| # | Change | Migration |
|---|---|---|
| 1 | `messageMetadata` mirror **removed** from history messages and the `data-assistant-message` / `data-interrupt` message payloads. | Read `message.metadata` only. |
| 2 | `data-interrupt.data.pendingToolCalls` **removed**. | Read `data.interrupt.action_requests[]`. |
| 3 | `data.total` **removed** from the messages listing payload. | Read `data.meta.total`. |
| 4 | Legacy renderer metadata **scrubbed** from all AI SDK responses: `images`, `has_images`, `images_count`, `agentic_images_count`, `live_widgets`, `canvas_artifact`, `pending_tool_calls`, internal `_`-prefixed keys, and database-redundant debug keys (`conversation_id`, `has_tool_calls`, `context_messages`). | Images: render `parts[].type === "file"`. Widgets, canvas artifacts, tool renders: render `metadata.rich_items` — send `inlineRichResponseV1: true`. The conversation id comes from the route. |
| 5 | `data-user-message` / `data-error-message` payloads are now projected (`id`, `role`, `content`, `createdAt`, `metadata`, `parts`) — raw DB fields (`conversation_id`, `sender`, `updated_at`) no longer appear. | Use the projected fields. |
| 6 | History messages always carry `parts` with a guaranteed leading `text` part. | Render from `parts` per the AI SDK v5+ `UIMessage` spec. |

Because of change 4, **`inlineRichResponseV1: true` is effectively required** for
clients that want widgets, canvas artifacts, or tool renders. The server-side
`inline_rich_response_enabled` kill switch now disables rich UI entirely for AI
SDK clients (they still get text and image `file` parts).

## Chat Path

Use:

```http
POST /api/chat/{conversationId}
```

Alias:

```http
POST /ai/chat/{conversationId}
```

Request body:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Run the report"
    }
  ],
  "userId": "optional-user-id",
  "inlineRichResponseV1": false
}
```

Request fields:

| Field | Type | Notes |
|---|---|---|
| `messages` | array | AI SDK UI message history. Backend reads the latest user message. |
| `messages[].role` | string | Usually `user` or `assistant`. |
| `messages[].content` | string or parts array | Plain text, or parts-style content. |
| `messages[].parts` | array | Optional. Text parts use `{ "type": "text", "text": "..." }`; file/image parts are supported. |
| `messages[].attachments` | array | Optional image attachments. Also accepts `experimental_attachments` or `files`. |
| `userId` | string, optional | Reserved/client hint only on authenticated deployments. The backend uses the authenticated user identity, not this body field, for ownership and user-aware features. |
| `inlineRichResponseV1` | boolean, optional | Opt in to marker-positioned rich UI items and article-style inline placement. Also accepted as `inline_rich_response_v1`. |
| `deviceId` | string, optional | Usually injected by the sidecar/client backend for local runtime tools. |

The response is `text/event-stream` and includes this header. Header names are case-insensitive; the backend currently emits it in lowercase.

```http
x-vercel-ai-ui-message-stream: v1
```

Parse each `data:` line as JSON and switch on `type`. The stream ends with `data: [DONE]`.

## Stream Event Shapes

The stream always starts like this:

```json
{ "type": "start", "messageId": "assistant-message-id" }
{ "type": "start-step" }
{ "type": "text-start", "id": "text-part-id" }
```

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
{
  "type": "tool-input-start",
  "toolCallId": "tool-call-id",
  "toolName": "tool_name"
}
```

```json
{
  "type": "tool-input-available",
  "toolCallId": "tool-call-id",
  "toolName": "tool_name",
  "input": {}
}
```

```json
{
  "type": "tool-output-available",
  "toolCallId": "tool-call-id",
  "output": "any JSON value",
  "render": {}
}
```

`tool-input-available.input` and `tool-output-available.output` may be any JSON-serializable value: object, array, string, number, boolean, or null.

File/image part:

```json
{
  "type": "file",
  "url": "data:image/png;base64,...",
  "mediaType": "image/png"
}
```

Assistant metadata:

```json
{
  "type": "data-assistant-message",
  "data": {
    "message": {
      "id": "assistant-message-id",
      "role": "assistant",
      "createdAt": "ISO-8601 timestamp",
      "metadata": {},
      "parts": []
    }
  }
}
```

Notes:

- `data-assistant-message.data.message.content` is intentionally omitted on the stream because answer text already arrived as `text-delta`.
- Stream metadata is projected to the AI SDK `metadata` field only. The `messageMetadata` mirror and the `message_metadata` wire alias are not emitted on AI SDK stream messages.
- `metadata` is scrubbed of legacy renderer fields (see the breaking-changes table) before it reaches the wire.
- `parts` may contain generated image/file parts. For v1 rich responses, only selected images appear as file parts.
- Article-style auto-placement can add final `<!--rich:<id>-->` markers at persistence time after text deltas have already streamed. For the AI SDK stream, use the persisted message from history after `finish` when you need the exact final marker layout.

Rich item upsert:

```json
{
  "type": "data-rich-items",
  "data": {
    "operation": "upsert",
    "items": []
  },
  "transient": true
}
```

Emission rules — `data-rich-items` is a **partial, live-progress channel**, not the
full registry:

| Item type | Streams as `data-rich-items`? | Arrives via |
|---|---|---|
| `live_widget`, `tool_render` | Yes, as each tool completes (only for capable requests). | Transient upsert **and** final `data-assistant-message.metadata.rich_items`. |
| `image` | Never (selection happens at finalization; candidates must not leak). | Final `data-assistant-message` / history only. |
| `canvas_artifact` | Never (the source is the asset; it is promoted at persistence). | Final `data-assistant-message` / history only. |
| Any item whose payload carries inline binary `data` | Never. | Final `data-assistant-message` / history only. |

Do not treat the absence of `data-rich-items` on a turn as an error: a
canvas-only or image-only answer emits none. The authoritative registry is
always `metadata.rich_items` on the final `data-assistant-message` (and
history). A marker streamed in `text-delta` whose item has not been upserted
yet should render as a pending placeholder until `finish`.

Other data events:

```json
{ "type": "data-agent-selected", "data": { "agent": "chat_agent" }, "transient": true }
{ "type": "data-user-message", "data": { "message": {} }, "transient": true }
{ "type": "data-continuation", "data": { "round": 1, "max_rounds": 3, "reason": "tool_budget" }, "transient": true }
{ "type": "data-node-complete", "data": { "node": "chat_agent" }, "transient": true }
{ "type": "heartbeat" }
```

`data-user-message.data.message` is a wire-safe projection of the persisted user
message — `id`, `role`, `content`, `createdAt`, plus `metadata`/`parts` when
present. Database-only fields (`conversation_id`, `sender`, `updated_at`) are
never included.

## Subagent Progress

> **Status: shipped** (2026-06-11, "Live Subagent Progress" — `event_streaming.md` Tasks 11–15). The backend emits these parts on all AI SDK chat endpoints; the wire shape below matches `ai_sdk_v6.py::_subagent`. Resume-path (`/ai/resume-interrupt`) dispatches do not stream live progress (documented out of scope); token-level worker deltas (`phase: "delta"`) are reserved but not emitted today.

When the Planning Agent dispatches workers (`dispatch_subagents`), each worker streams live progress as transient `data-subagent` parts, interleaved with the rest of the stream:

```json
{ "type": "data-subagent", "transient": true,
  "data": { "phase": "start",
            "subagent": { "id": "worker-a", "name": "search_agent", "path": ["planning_agent", "worker-a"], "status": "running" },
            "task": "Find source material." } }
{ "type": "data-subagent", "transient": true,
  "data": { "phase": "tool",
            "subagent": { "id": "worker-a", "name": "search_agent", "path": ["planning_agent", "worker-a"], "status": "running" },
            "toolCallId": "sub-call-1", "toolName": "search_documents", "status": "success" } }
{ "type": "data-subagent", "transient": true,
  "data": { "phase": "end",
            "subagent": { "id": "worker-a", "name": "search_agent", "path": ["planning_agent", "worker-a"], "status": "completed" },
            "output": "…", "summary": "Short worker summary", "elapsedMs": 1234 } }
```

Fields:

| Field | Type | Notes |
|---|---|---|
| `data.phase` | string | `start`, `tool`, or `end`. The lifecycle of one worker. |
| `data.subagent.id` | string | **Stable key.** Upsert/update the same UI row across `start` → `tool` → `end`. |
| `data.subagent.name` | string | Worker agent name (e.g. `search_agent`). |
| `data.subagent.path` | array | Hierarchy path, e.g. `["planning_agent", "worker-a"]`. |
| `data.subagent.status` | string | `running`, `completed`, `failed`, `timeout`, or `requires_approval`. Reflects the worker's current state on each event. |
| `data.task` | string | `start` only. The instruction given to the worker. |
| `data.toolName` / `data.toolCallId` | string | `tool` only. A tool the worker invoked. |
| `data.output` / `data.summary` | string | `tool`/`end`. Worker tool output / final summary. |
| `data.status` | string | `tool` only. Per-tool status (`success`/`error`/…). |
| `data.render` | object | `tool` only, optional. Structured render payload for the worker's tool result. |
| `data.text` | string | Reserved for `phase: "delta"` worker token deltas (not emitted today). |
| `data.elapsedMs` | number | `end` only. Worker wall-clock duration. |
| `data.error` | string or null | `end` only, when the worker failed. |

Rendering rules:

- `data-subagent` is **transient** — it is not added to message state. Maintain your own map keyed by `data.subagent.id` for a live "subagents working" panel, then drop/collapse it when the run finishes.
- The **durable** record of subagent results stays in `backendMeta.subagent_results` on the final `data-assistant-message` (see Agent metadata). Use that for the persisted/historical view; use `data-subagent` only for live progress.
- Unknown future phases should be ignored gracefully.

Finish:

```json
{ "type": "finish-step" }
{ "type": "finish" }
```

Error:

```json
{ "type": "error", "errorText": "error message" }
{ "type": "data-error-message", "data": { "message": {} }, "transient": true }
```

## HITL Auto-Detection

Detect HITL by checking for `data-interrupt`:

```ts
const isHitl = event.type === "data-interrupt";
const interrupt = event.data?.interrupt;
const requests = interrupt?.action_requests ?? [];
```

When `isHitl` is true, render a human-in-the-loop UI from `requests`. The stream will finish immediately after the interrupt event.

## HITL Interrupt Types

There is currently one AI SDK stream event for human-in-the-loop pauses:
`data-interrupt`. The backend does not emit a separate `interrupt.type` field
today. Frontend should derive the interrupt kind from the event and metadata:

| Derived kind | How to detect | FE behavior |
|---|---|---|
| `tool_approval` | `event.type === "data-interrupt"` and `data.interrupt.action_requests[]` is present | Render approval UI and resume through `POST /ai/resume-interrupt`. This is the current HITL interrupt shape. |
| `planning_pause` | Final assistant metadata has `planning_budget_reached: true`, `execution_paused: true`, or `execution_pause_reason` | Do not render HITL decisions. Show the assistant content/message and let the user continue with a normal chat message. |
| `subagent_requires_approval` | Live `data-subagent.data.subagent.status === "requires_approval"` or final `backendMeta.subagent_results[].status === "requires_approval"` | Show worker as blocked. The nested worker is not resumable through `/ai/resume-interrupt`; the user must approve or rerun the operation from the main conversation. |

Pause/reason fields are not the same as decision types:

| Field/value | Meaning | Resume behavior |
|---|---|---|
| `metadata.pause_reason = "tool_approval_required"` | Persisted assistant message is paused for HITL tool approval. | Use the persisted `interrupt` payload to rebuild approval UI. |
| `pause_reason = "awaiting_approval"` | Workflow/subagent stopped because a tool needs approval. | Prefer the `interrupt` object if present. Without `interrupt`, show blocked state only. |
| `pause_reason = "max_iterations_reached"` | Planning loop hit its iteration/budget limit. | No HITL decision. Let the user send another message to continue. |
| `pause_reason = "consecutive_errors_limit"` | Planning loop stopped after repeated errors. | No HITL decision. Show retry/continue affordance. |
| `execution_pause_reason = "max_tasks_reached"` | Persisted planning execution pause derived from `planning_budget_reached`. | No HITL decision. Use `execution_pause_message` for display. |
| `recursion_limit` / `rate_limit` | Possible internal/planning pause reasons from execution guards. | No HITL decision unless a `data-interrupt` payload also exists. |

Decision types are the user actions sent back on resume: `approve`, `edit`,
`reject`, and `respond`.

## HITL Interrupt Shape

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
        "timeout_deadline": "ISO-8601 timestamp",
        "device_id": "optional-device-id",
        "tool_provenance": {
          "tool-call-id": {
            "device_id": "optional-device-id",
            "tool_origin": "client_or_server_origin",
            "server_name": "optional-mcp-server",
            "qualified_tool_id": "optional-qualified-tool-id",
            "tool_instance_id": "optional-tool-instance-id",
            "session_id": "optional-runtime-session",
            "catalog_version": 1
          }
        }
      }
    },
    "message": {
      "id": "paused-assistant-message-id",
      "content": "",
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

Important HITL fields:

| Field | Type | Notes |
|---|---|---|
| `data.threadId` | string | Required for resume as `threadId`. |
| `data.interrupt.interrupt_id` | string | Required for resume as `interruptId`. |
| `data.interrupt.conversation_id` | string | Required for resume as `conversationId`. |
| `data.interrupt.action_requests[]` | array | Canonical (and only) list of tool calls awaiting human input. |
| `action_requests[].action` | string | Tool/action name. |
| `action_requests[].args` | object | Original tool arguments. Use this to show what will run. |
| `action_requests[].description` | string or null | Optional human-readable tool description. |
| `action_requests[].tool_call_id` | string or null | Primary decision target. If present, send it back as `toolCallId` on the matching decision. |
| `action_requests[].task_id` | string or null | UI/request identifier. Send as `taskId` if useful, but do not use it instead of `toolCallId` when `tool_call_id` is present. It is only the decision target when `tool_call_id` is missing. |
| `action_requests[].allowed_decisions` | array or null | If present, restrict UI buttons to these decisions. |
| `data.interrupt.metadata` | object | Display/recovery metadata for the whole interrupt. See below. |

Interrupt metadata fields:

| Field | Type | Notes |
|---|---|---|
| `message` | string or null | Optional display copy for the approval UI. |
| `reason` | string or null | Optional display reason when `message` is absent or too terse. |
| `timeout_deadline` | string or null | ISO-8601 deadline for the approval window. FE may show countdown/expired state. |
| `device_id` | string or null | Client device that owns a local-tool interrupt, when applicable. |
| `tool_provenance` | object | Map keyed by tool call id or action name. Use for audit/debug and stale-runtime checks, not primary user copy. |
| `tool_provenance.*.device_id` | string or null | Device associated with that tool call. |
| `tool_provenance.*.tool_origin` | string or null | Tool origin such as `client_mcp`, `server_mcp`, or `internal`. |
| `tool_provenance.*.server_name` | string or null | MCP server name, usually for client/server MCP tools. |
| `tool_provenance.*.qualified_tool_id` | string or null | Stable qualified tool identifier when available. |
| `tool_provenance.*.tool_instance_id` | string or null | Runtime tool instance id used for stale resume validation. |
| `tool_provenance.*.session_id` | string or null | Runtime session id used for stale resume validation. |
| `tool_provenance.*.catalog_version` | number or null | Tool catalog version used for stale resume validation. |

## Resume HITL

Call:

```http
POST /ai/resume-interrupt
```

Body:

```json
{
  "threadId": "conversation-thread-id",
  "conversationId": "conversation-id",
  "interruptId": "interrupt-id",
  "decisions": [
    {
      "type": "approve",
      "toolCallId": "tool-call-id",
      "action": "tool_name",
      "args": {}
    }
  ],
  "inlineRichResponseV1": false
}
```

Decision fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | string | yes | `approve`, `edit`, `reject`, or `respond`. |
| `toolCallId` | string | conditionally required | Required whenever the corresponding `action_requests[]` item has `tool_call_id`. Snake case `tool_call_id` is also accepted. |
| `taskId` | string | optional | UI/request identifier. Snake case `task_id` is also accepted. Fallback target only when `toolCallId`/`tool_call_id` is unavailable. |
| `action` | string | optional | Tool/action name. Useful for audit/debug display. |
| `args` | object | optional | Meaning depends on decision type. |

Resume coverage rules:

- Send exactly one decision for every item in `data.interrupt.action_requests[]`.
- If an action request contains `tool_call_id`, the matching decision must include the same value as `toolCallId` (or `tool_call_id`).
- `taskId` may be included for UI correlation/back-compat, but `taskId` alone is not sufficient when the pending request has a distinct `tool_call_id`.
- The backend rejects incomplete coverage with `422 INTERRUPT_INCOMPLETE_DECISIONS`; retrying the same already-claimed interrupt can return `409 INTERRUPT_ALREADY_RESOLVED`, so build the complete decision set before the first resume request.

Decision behavior:

| Type | Meaning |
|---|---|
| `approve` | Run the original tool call. |
| `edit` | Run the tool call with replacement `args`. |
| `reject` | Do not run the tool. Put feedback in `args.message`, `args.reason`, or `args.feedback`. |
| `respond` | Do not run the tool. Pass a human answer back with `args.response` or `args.message`. |

Examples:

```json
{
  "type": "reject",
  "toolCallId": "tool-call-id",
  "action": "tool_name",
  "args": { "reason": "Do not access this account." }
}
```

```json
{
  "type": "respond",
  "toolCallId": "tool-call-id",
  "action": "ask_user",
  "args": { "response": "Use the quarterly report." }
}
```

The resume response uses the same AI SDK stream event format as chat.

## Message History Shape

`GET /ai/conversations/{conversationId}/messages` returns AI SDK `UIMessage` objects.

Query fields:

| Field | Type | Notes |
|---|---|---|
| `page` / `limit` / `orderBy` / `orderDirection` | query | Standard pagination and ordering. |
| `inlineRichResponseV1` | boolean, optional | Opt in to marker-bearing persisted rich-response content. Snake case `inline_rich_response_v1` is also accepted. Without this flag, standalone rich markers are stripped and `rich_items`, `rich_items_version`, and `rich_reference_warnings` are removed from metadata. |

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
        "createdAt": "ISO-8601 timestamp"
      }
    ],
    "meta": {
      "total": 1,
      "perPage": 20,
      "currentPage": 1,
      "lastPage": 1
    }
  }
}
```

Message fields:

| Field | Type | Notes |
|---|---|---|
| `id` | string | Message UUID. |
| `role` | string | `user` or `assistant`. |
| `content` | string | Markdown/plain text. For rich v1, may contain `<!--rich:<id>-->` markers only when the history request opts in with `inlineRichResponseV1=true` and the server rich-response setting is enabled. |
| `parts` | array | AI SDK UI parts. Always present; a leading `text` part carrying the message content is guaranteed. Images are exposed as `file` parts. |
| `parts[].type` | string | `text`, `file`, or `reasoning`. |
| `parts[].text` | string | Text part content. |
| `parts[].url` | string | File URL, data URL, remote URL, or blob URL. |
| `parts[].mediaType` | string | MIME type for file part. |
| `parts[].reasoning` | string | Reasoning content for reasoning parts, if present. |
| `metadata` | object or null | Canonical AI SDK UI message metadata field, scrubbed of legacy renderer fields. |
| `createdAt` | string or null | ISO-8601 timestamp. |

## Assistant Metadata Shape

Use:

```ts
const backendMeta = message.metadata ?? {};
```

Common `backendMeta` fields:

```json
{
  "persona_used": "optional persona prompt",
  "provider": "openai",
  "model": "gpt-4o",
  "key_source": "env",
  "config_source": "request",
  "config_warnings": ["optional warning"],
  "custom_model_override": true,
  "provider_fallback": {
    "from": "openai",
    "to": "gemini",
    "reason": "provider_error"
  },
  "reasoning_effort": "high",
  "thinking": "provider thinking text if persisted",
  "thinking_summary": "provider thinking summary if available",
  "reasoning_summary": "provider reasoning summary if available",
  "reasoning_tokens": 123,
  "token_breakdown": {},
  "context_window": {},
  "agent": {},
  "handoff": {},
  "custom_agent_warnings": [],
  "subagent_dispatches": [],
  "subagent_results": [],
  "tool_artifacts": [],
  "documents_cited": [],
  "chunks_retrieved": 3,
  "documents_found": 1,
  "citations": [],
  "rich_items_version": 1,
  "rich_items": [],
  "rich_reference_warnings": [],
  "suggested_questions": [],
  "reply_to_user_message_id": "optional-user-message-id",
  "todos_synced": true,
  "interrupt": {},
  "paused": true,
  "pause_reason": "tool_approval_required",
  "execution_paused": true,
  "execution_pause_reason": "max_tasks_reached",
  "execution_pause_message": "Completed a planning iteration. Send a message to continue."
}
```

All fields are optional. Frontend should ignore unknown keys.

Metadata rules:

- `message.metadata` is the only metadata field. The `messageMetadata` mirror and the `message_metadata` wire alias are not emitted by the AI SDK response path.
- Legacy renderer fields (`images`, `has_images`, `images_count`, `agentic_images_count`, `live_widgets`, `canvas_artifact`, `pending_tool_calls`) are scrubbed from AI SDK responses even when present in persisted data. Images arrive as `file` parts; everything else arrives as `rich_items`.
- Database-redundant debug keys (`conversation_id`, `has_tool_calls`, `context_messages`) are also scrubbed — the conversation id is always known from the route.
- `rich_items`, when present, is a field inside backend metadata at the same level as `tool_artifacts`. It is not a top-level message field.
- Internal keys beginning with `_`, such as `_rich_item_candidates` and `_inline_rich_response_v1`, are scrubbed from AI SDK responses.

Core runtime/model fields:

| Field | Type | FE usage |
|---|---|---|
| `persona_used` | string | Optional trace/debug display for the persona prompt applied to this turn. |
| `provider` | string | Model provider id, e.g. `openai`, `gemini`, `anthropic`. |
| `model` | string | Model id used for the final response. |
| `key_source` | string | Credential source such as `env`, `db`, `settings`, or `none`. Usually debug/admin only. |
| `config_source` | string | Runtime model config source, e.g. request, saved agent config, default, or fallback. |
| `config_warnings` | string[] | Non-fatal model/config warnings. Show only in debug/admin surfaces unless product wants them user-visible. |
| `custom_model_override` | boolean | `true` when the request used an ad hoc model override. Debug/admin only. |
| `provider_fallback` | object | Present when backend fell back from one provider/model to another. |
| `provider_fallback.from` / `to` | string | Provider/model fallback source and target. |
| `provider_fallback.reason` | string | Why fallback happened. |
| `reasoning_effort` | string | Requested effort level when supported by the provider. |
| `thinking` | string | Persisted provider thinking text if exposed by the provider/config. Treat as sensitive/debug-only unless product policy says otherwise. |
| `thinking_summary` / `reasoning_summary` | string | Short reasoning/thinking summary when available. |
| `reasoning_tokens` | number | Provider-reported reasoning/thinking token count when available. |

Token/context fields:

| Field | Type | FE usage |
|---|---|---|
| `token_breakdown.estimated` | object | Estimated prompt/tool/history token counts. Debug or context-meter UI. |
| `token_breakdown.actual` | object | Provider-reported `input_tokens`, `output_tokens`, `total_tokens`, and `reasoning_tokens` when available. |
| `token_breakdown.counts` | object | Counts for history messages, tool messages, and bound tools. |
| `token_breakdown.bound_tool_names` | string[] | Tool names included in the model request. Debug/admin only. |
| `context_window` | object | Context-window metadata and usage meter fields. See Context and Model Metadata. |

Agent/routing fields:

| Field | Type | FE usage |
|---|---|---|
| `agent.id` | string | Runtime agent id, e.g. `chat_agent`, `canvas_agent`, or `custom_agent:<uuid>`. |
| `agent.kind` | string | `base` or `custom`. |
| `agent.name` | string | Display name for the responding/selected agent. |
| `agent.custom_agent_id` | string or null | Stable custom-agent database id when applicable. |
| `agent.source` | string | `response` when the final response identifies itself, otherwise `selected_agent`. |
| `handoff.from_agent_id` / `to_agent_id` | string | Agent handoff source and target. |
| `handoff.reason` | string | Short reason the model/tool supplied for the handoff. |
| `handoff.tool_call_id` | string or null | Tool call id associated with the handoff. |
| `custom_agent_warnings` | array | Warnings about custom-agent tool/skill availability. |
| `subagent_dispatches` | array | Planning dispatch debug/activity records. Prefer `data-subagent` for live progress and `subagent_results` for durable summaries. |
| `subagent_results` | array | Durable compact worker summaries. See Subagent Progress. |

Renderer/media fields:

| Field | Type | FE usage |
|---|---|---|
| `tool_artifacts` | array | Persisted compact tool execution records. Use for trace, audit, and tool-render fallback. |
| `rich_items_version` | number | Current version is `1`. Present only for messages with rich-item activity. |
| `rich_items` | array | Final authoritative rich-item registry for inline/append rendering. The only source for widgets, canvas artifacts, and tool renders. |
| `rich_reference_warnings` | array | Validation warnings such as `unknown_rich_item` or `invalid_rich_item`. |
| `documents_cited` / `citations` | array | RAG citation metadata. |
| `chunks_retrieved` / `documents_found` | number | RAG retrieval counters. |

Removed legacy renderer fields (`images`, image counters, `live_widgets`,
`canvas_artifact`) are scrubbed from AI SDK responses — see the
breaking-changes table.

Planning/HITL/user-experience fields:

| Field | Type | FE usage |
|---|---|---|
| `suggested_questions` | string[] | Optional follow-up suggestions. |
| `reply_to_user_message_id` | string | Assistant message was generated as a reply to a specific user message. |
| `interrupt` | object | Persisted HITL interrupt payload. Rebuild approval UI from this when message is paused. |
| `paused` | boolean | `true` for persisted paused assistant messages. |
| `pause_reason` | string | Pause reason; see HITL Interrupt Types. |
| `thread_id` | string | Resume thread id for persisted interrupt messages. |
| `next` | string[] | Next graph node(s) for resume/debug. |
| `todos` | array | Current task-plan/todo state for planning UI. |
| `planning_call_count` | number | Number of planning loop calls this turn. |
| `all_tasks_completed` | boolean | Planning execution completed all tasks. |
| `planning_budget_reached` | boolean | Planning loop paused after reaching budget/iteration limit. |
| `todos_synced` | boolean | Backend synced response metadata back into persisted task-plan state. |
| `next_task` | object | Active/next task summary for planning UI. |
| `planning_rubric` | object | Plan-quality grading metadata. |
| `subagent_worker_artifacts` | object | Debug/detail artifacts keyed by worker when available. |
| `execution_paused` | boolean | Persisted planning execution pause marker. Not a HITL approval interrupt. |
| `execution_pause_reason` | string | Current persisted value is usually `max_tasks_reached`; see HITL Interrupt Types. |
| `execution_pause_message` | string | User-facing copy for planning pauses. |

## Images and Attachments

### Request Attachments

The latest user message may send images in `parts`, `content`, `attachments`, `experimental_attachments`, or `files`.

Accepted input item variants:

```json
{
  "type": "file",
  "name": "chart.png",
  "mediaType": "image/png",
  "url": "data:image/png;base64,..."
}
```

```json
{
  "type": "image",
  "name": "chart.png",
  "mime": "image/png",
  "data": "raw-base64-without-data-url-prefix"
}
```

Accepted attachment fields:

| Field | Type | Notes |
|---|---|---|
| `type` | string | `image` or `file`. Other types are ignored. |
| `name` / `filename` | string | Optional display filename. Defaults to `attachment`. |
| `mime` / `mimeType` / `mediaType` / `contentType` | string | MIME type. Defaults to `image/jpeg` on request extraction. |
| `data` | string | Data URL or raw base64. |
| `base64` | string | Raw base64. |
| `url` | string | `data:`, `http://`, `https://`, or `blob:` URL. |
| `path` / `image` / `source` | string or object | Fallback source fields. Local filesystem-like paths are ignored. |

### Assistant Images

Assistant images are delivered exclusively as AI SDK `file` parts (streamed
`file` chunks during generation, `parts[].type === "file"` on history and the
final `data-assistant-message`). The legacy `backendMeta.images` array and its
counters are scrubbed from AI SDK responses.

Frontend rendering rule:

- Render AI SDK `parts[].type === "file"` for images.
- If `rich_items_version === 1`, image rich items carry placement (`inline_only`
  markers); the emitted file parts contain only the selected images.
- Do not build a separate image gallery from metadata.

## Rich Response v1

When `inlineRichResponseV1` is true and server config enables it, assistant markdown may contain standalone markers:

```markdown
Here is the chart:

<!--rich:widget:widget-id-->
```

Marker format:

```text
<!--rich:<id>-->
```

Rules:

| Rule | Value |
|---|---|
| Marker line | Must be on its own line. |
| Allowed ID chars | ASCII letters, digits, `_`, `-`, `.`, `:`. |
| Max ID length | 128 chars. |
| Code blocks | Markers inside fenced or 4-space indented code blocks are literal markdown. |

`backendMeta.rich_items` item shape:

```json
{
  "id": "widget:widget-id",
  "type": "live_widget",
  "source": "widget_tool",
  "display_policy": "inline_or_append",
  "title": "Optional title",
  "alt_text": "Optional alt text",
  "provenance": {},
  "payload": {}
}
```

Common fields:

| Field | Type | Notes |
|---|---|---|
| `id` | string | Stable marker target. |
| `type` | string | `image`, `live_widget`, `tool_render`, `canvas_artifact`, `citation`, or `resource_link`. |
| `source` | string or null | Origin, such as `widget_tool`, `tool`, `web_search`, `tool_image`. |
| `display_policy` | string | `inline_only` or `inline_or_append`. |
| `title` | string or null | Display title. |
| `alt_text` | string or null | Accessibility text. Required for image rich items. |
| `provenance` | object | Optional origin metadata. |
| `payload` | object | Type-specific payload. |

Display policies:

| Policy | Behavior |
|---|---|
| `inline_only` | Render only at marker. Images use this. Do not append elsewhere. |
| `inline_or_append` | Render at marker if present; otherwise append below the answer. |

### Article-Style Auto-Placement

The article rich-response implementation does not add new stream event types or
new rich item types. It changes when markers can appear:

- `inlineRichResponseV1: true` is still the client capability flag.
- The server-side `inline_rich_response_enabled` setting now defaults to enabled and remains a kill switch. If it is disabled, the backend strips/omits v1 marker behavior even when the client opts in.
- When `rich_auto_place_enabled` is enabled, the backend may insert markers for relevant unreferenced image candidates and live widgets into the final persisted assistant markdown.
- Auto-placement is deterministic and bounded by server settings. Current defaults: at most 3 auto-placed images per answer, one placed item per paragraph, and a minimum keyword-overlap score of 0.25. Widget placement is not capped by the image limit.
- Images still use `display_policy: "inline_only"`. Unplaced image candidates are never exposed to AI SDK clients; do not render a separate gallery.
- The final authoritative layout is the pair of persisted `message.content` plus `metadata.rich_items`.

Streaming note:

- If the model wrote markers itself, those markers can appear in `text-delta`.
- If the backend inserted markers during persistence, the earlier `text-delta` stream may not contain those markers.
- The AI SDK `data-assistant-message` event intentionally omits `content`, so clients that need exact article placement during/after a live run should refetch `GET /ai/conversations/{conversationId}/messages?inlineRichResponseV1=true` after `finish`, or use an application-level finalized message source if one is available.

### Rich Image Item

```json
{
  "id": "image:tool:tool-call-id:0",
  "type": "image",
  "source": "tool_image",
  "display_policy": "inline_only",
  "alt_text": "Image from tool result",
  "title": "Optional title",
  "payload": {
    "url": "https://example.com/image.png",
    "data": null,
    "mime_type": "image/png",
    "source_url": "https://example.com/source",
    "description": "Optional description"
  },
  "provenance": {
    "tool_call_id": "tool-call-id",
    "tool": "tool_name",
    "index": 0
  }
}
```

Exactly one of `payload.url` or `payload.data` is present. Allowed MIME types are `image/png`, `image/jpeg`, `image/webp`, and `image/gif`.

### Rich Live Widget Item

```json
{
  "id": "widget:widget-id",
  "type": "live_widget",
  "source": "widget_tool",
  "display_policy": "inline_or_append",
  "title": "Widget title",
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

### Rich Tool Render Item

```json
{
  "id": "tool:tool-call-id",
  "type": "tool_render",
  "source": "tool",
  "display_policy": "inline_or_append",
  "title": "Optional title",
  "payload": {
    "render": {
      "version": 1,
      "type": "mcp_app",
      "template_uri": "ui://server/component.html",
      "structured_content": {}
    }
  },
  "provenance": {
    "tool_call_id": "tool-call-id",
    "tool": "tool_name"
  }
}
```

`payload.render` is renderer-specific. Known examples include `mcp_app`, `image`, `json`, `text`, `error`, `live_widget`, and `subagent_dispatch`. Only meaningful non-widget app renders become `tool_render` rich items.

### Rich Canvas Artifact Item

```json
{
  "id": "canvas:main",
  "type": "canvas_artifact",
  "display_policy": "inline_or_append",
  "title": "Canvas title",
  "payload": {
    "language": "html",
    "title": "Canvas title",
    "content": "<!doctype html>...",
    "preferred_height": null
  }
}
```

Emission notes:

- A message carries at most one canvas artifact, promoted from the CanvasAgent
  output at persistence with the stable id `canvas:main`.
- The canvas rich item is created for requests that sent
  `inlineRichResponseV1: true` (or any message that already has other
  rich-item activity). Canvas messages created by non-capable clients keep the
  pre-v1 persisted shape and expose no canvas on the AI SDK wire.
- `payload.preferred_height` is currently always `null` for promoted canvas
  items.

Canvas artifacts are standalone browser-rendered artifacts. They may appear as
legacy `backendMeta.canvas_artifact` and/or as a rich item with
`type: "canvas_artifact"`.

Canvas artifact kinds FE should expect:

| Kind | Typical user request | `language` | Rendering expectation |
|---|---|---|---|
| Full HTML page/app | Website, landing page, calculator, dashboard, game, form, animation, chart, HTML canvas visualization | `html` | Render `content` in the existing sandboxed canvas/iframe path as a full self-contained document. |
| SVG artifact | Icon, logo, static vector illustration, simple diagram | `svg` | Render `content` as standalone SVG in the same reviewed canvas boundary. |
| React browser artifact | React component/app, JSX/TSX/JS interactive artifact | `react` | Render as a self-contained browser artifact. Current prompt expects React/ReactDOM UMD CDN usage and a root mount point when React is used. |

`payload.language` is a renderer/editor hint, not the human language of the
answer and not a MIME type. `CanvasAgent` currently normalizes fenced-code
language hints like this:

| LLM code fence hint | Emitted `language` |
|---|---|
| `svg` | `svg` |
| `react`, `jsx`, `js`, `javascript`, `tsx`, `ts` | `react` |
| `html`, blank, unknown, or anything else | `html` |

The rich-item schema accepts `language` as a string for forward compatibility,
but FE should only special-case `html`, `svg`, and `react` today. Unknown future
values should fall back to the safest existing canvas renderer or an
unsupported-artifact placeholder.

Canvas payload fields:

| Field | Type | Notes |
|---|---|---|
| `payload.language` | string | Renderer/editor hint. Current emitted values are `html`, `svg`, and `react`. |
| `payload.title` | string | Display title. Often derived from the HTML `<title>` tag; fallback is `Canvas`. |
| `payload.content` | string | Full artifact source. Treat as executable/untrusted browser content and render only inside the approved sandbox boundary. |
| `payload.preferred_height` | number or null | Optional rich-item height hint in pixels. Legacy `canvas_artifact` does not currently emit this field. |

### Rich Citation Item

```json
{
  "id": "citation:assistant-message-id:0",
  "type": "citation",
  "display_policy": "inline_or_append",
  "title": "Optional title",
  "payload": {
    "source": "report.pdf",
    "document_id": "document-id",
    "page_number": 5,
    "chunk_index": 12
  }
}
```

### Rich Resource Link Item

```json
{
  "id": "resource:0",
  "type": "resource_link",
  "display_policy": "inline_or_append",
  "title": "Optional title",
  "payload": {
    "url": "https://example.com",
    "title": "Example",
    "description": "Optional description"
  }
}
```

Renderer algorithm:

1. Opt in with `inlineRichResponseV1: true`.
2. Store transient `data-rich-items.data.items` by `id`.
3. Accumulate `text-delta`.
4. Split complete standalone `<!--rich:<id>-->` marker lines.
5. Render known rich item at marker position.
6. On final `data-assistant-message`, replace transient items with `backendMeta.rich_items` when present. If exact auto-placed marker layout is required, refetch history with `inlineRichResponseV1=true` after `finish`.
7. Append only unreferenced `inline_or_append` items. Never append `inline_only` images.

## Live Widgets

Live widgets have one supported type: `html`. A widget is a self-contained micro-app.
The frontend renders the widget **state only as a sandboxed iframe** from `state.html` —
never inject `state.html` into the main chat DOM. `state.html` is untrusted, executable
content; treat it with the same safety boundary as a canvas artifact (`sandbox` iframe,
`referrerpolicy="no-referrer"`).

Expected widget state (delivered over the WebSocket, see below):

```json
{
  "html": "<!doctype html>...",
  "height": 620,
  "caption": "Optional short caption"
}
```

`height` is a number between 260 and 960; `caption` is optional. There are no structured
widget renderers (`table`/`chart`/`dashboard`/`form`/`list`) — do **not** choose a React
renderer by `widget_type`. Legacy persisted metadata may still carry a removed structured
type; render those as an unsupported/legacy placeholder rather than a structured renderer.

Live widgets reach AI SDK clients only through
`backendMeta.rich_items[]` where `type === "live_widget"` (the legacy
`backendMeta.live_widgets[]` array is scrubbed from AI SDK responses).

Widget rich-item `payload` fields:

| Field | Type | Notes |
|---|---|---|
| `widget_id` | string | Widget identifier. |
| `session_id` | string | Conversation/session id. |
| `widget_type` | string | Always `html` — the only supported live widget type. Older persisted items may carry a removed structured type; render those as a legacy placeholder. |
| `status` | string | `active` or `closed`. |
| `version` | number | Incrementing state version. |
| `connection_endpoint` | string | POST this endpoint to mint a widget WebSocket token. |

Mount flow:

1. Render placeholder from metadata.
2. POST `connection_endpoint` with normal auth.
3. Open the returned `ws_url`.
4. Render `widget_state_sync.state.html` inside a sandboxed iframe (`srcdoc`).
5. Apply future `widget_update.state` by re-rendering the iframe.

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
  "expires_at": "ISO-8601 timestamp"
}
```

Widget WebSocket server events:

```json
{
  "type": "widget_state_sync",
  "widget_id": "widget-id",
  "widget_type": "html",
  "title": "Widget title",
  "state": {},
  "status": "active",
  "version": 1
}
```

```json
{
  "type": "widget_update",
  "widget_id": "widget-id",
  "widget_type": "html",
  "title": "Widget title",
  "state": {},
  "status": "active",
  "version": 2
}
```

```json
{
  "type": "widget_close",
  "widget_id": "widget-id",
  "widget_type": "html",
  "title": "Widget title",
  "state": {},
  "status": "closed",
  "version": 3
}
```

```json
{ "type": "ping", "timestamp": "ISO-8601 timestamp" }
{ "type": "error", "message": "Widget is no longer available." }
```

Widget WebSocket client events:

```json
{ "type": "pong" }
```

```json
{
  "type": "user_state_patch",
  "patch": {
    "selection": "row-id"
  }
}
```

Widget action endpoint:

```http
POST /widgets/{widgetId}/actions/{actionKey}
```

Request:

```json
{
  "input_values": { "note": "demo" },
  "state_patch": { "selection": "row-id" }
}
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

The action endpoint does not call the assistant. Submit `content` through `/api/chat/{conversationId}`.

## Tool Artifacts

Persisted tool artifacts live in `backendMeta.tool_artifacts[]`:

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

Fields:

| Field | Type | Notes |
|---|---|---|
| `tool_call_id` | string or null | Tool call ID. |
| `tool` | string | Tool name. Legacy shape may use `tool_name`. |
| `args` | any | Tool input args. |
| `output` | string or null | Tool output, capped for metadata. Widget outputs are compacted and exclude large `state`. |
| `error` | string or null | Error text, if failed. |
| `status` | string | `success`, `error`, `failed`, or `rejected`. |
| `render` | object, optional | Structured render payload for UI. |
| `blob_id` | string, optional | Present when a large tool output was offloaded. Fetch full text with `GET /tool-results/{blob_id}` using normal auth. |
| `blob_size_bytes` | number, optional | Size of the full offloaded output in bytes. |
| `output_truncated` | boolean, optional | `true` when `output` is only a preview plus an offload notice. |

Offloaded tool-result storage is backend-internal. New blobs are stored in the
database, while legacy file-backed blobs remain readable through the same
`GET /tool-results/{blob_id}` endpoint; frontend behavior does not change.

Render payload is tool-specific. Examples:

```json
{
  "version": 1,
  "type": "mcp_app",
  "template_uri": "ui://canva/presentation-viewer.html",
  "structured_content": {
    "presentation_id": "deck_123"
  }
}
```

```json
{
  "type": "image",
  "content": [
    { "type": "text", "text": "Here is the chart." },
    { "type": "image", "mimeType": "image/png", "data": "base64" }
  ]
}
```

## RAG Citations

Document/RAG responses can include:

```json
{
  "documents_cited": [
    {
      "document_id": "document-id",
      "source": "report.pdf",
      "document_number": 1,
      "chunks": [
        {
          "chunk_index": 0,
          "score": 0.85,
          "character_count": 1500,
          "content": "Chunk text...",
          "page_number": 5
        }
      ],
      "total_chunks": 1,
      "avg_score": 0.85
    }
  ],
  "chunks_retrieved": 3,
  "documents_found": 1,
  "citations": [
    {
      "document_id": "document-id",
      "source": "report.pdf",
      "chunk_index": 0,
      "page_number": 5,
      "score": 0.85,
      "content": "Chunk text..."
    }
  ]
}
```

Fields are optional and depend on the RAG tool path.

## Canvas Artifact

Canvas artifacts reach AI SDK clients only as `canvas_artifact` rich items
(see Rich Canvas Artifact Item). The legacy `backendMeta.canvas_artifact`
object is scrubbed from AI SDK responses.

## Context and Model Metadata

Optional model/runtime fields:

```json
{
  "provider": "openai",
  "model": "gpt-4o",
  "key_source": "env",
  "config_source": "request",
  "config_warnings": [],
  "custom_model_override": true,
  "provider_fallback": {
    "from": "openai",
    "to": "gemini",
    "reason": "provider_error"
  },
  "reasoning_effort": "high",
  "reasoning_tokens": 300,
  "context_overflow_retry": true
}
```

Context window:

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

If `known` is false, `context_window_tokens`, `max_input_tokens`, `max_output_tokens`, `used_tokens`, and `usage_ratio` may be null.

Agent metadata:

```json
{
  "agent": {
    "id": "search_agent",
    "kind": "base",
    "name": "Search Agent",
    "custom_agent_id": null,
    "source": "selected_agent"
  },
  "handoff": {
    "from_agent_id": "planning_agent",
    "to_agent_id": "search_agent",
    "reason": "Needs web search",
    "tool_call_id": "tool-call-id"
  },
  "subagent_results": [
    {
      "id": "worker-id",
      "agent": "search_agent",
      "agent_name": "Search Agent",
      "agent_kind": "base",
      "custom_agent_id": null,
      "status": "completed",
      "summary": "Short worker summary"
    }
  ],
  "custom_agent_warnings": []
}
```

Planning metadata may include `todos`, `planning_call_count`, `all_tasks_completed`, `planning_budget_reached`, `pause_reason`, `execution_paused`, `execution_pause_reason`, `execution_pause_message`, `next_task`, `planning_rubric`, and `subagent_worker_artifacts`. These are optional and mostly useful for task-plan UI.

## Frontend Detection Summary

Recommended detection:

```ts
function getBackendMeta(message: any) {
  return message?.metadata ?? {};
}

function isHitlEvent(event: any) {
  return event?.type === "data-interrupt";
}

function getInterruptKind(event: any, backendMeta: any = {}) {
  if (event?.type === "data-interrupt") {
    return "tool_approval";
  }
  if (
    event?.type === "data-subagent" &&
    event?.data?.subagent?.status === "requires_approval"
  ) {
    return "subagent_requires_approval";
  }
  if (
    backendMeta?.planning_budget_reached ||
    backendMeta?.execution_paused ||
    backendMeta?.execution_pause_reason
  ) {
    return "planning_pause";
  }
  if (
    Array.isArray(backendMeta?.subagent_results) &&
    backendMeta.subagent_results.some((item: any) => item?.status === "requires_approval")
  ) {
    return "subagent_requires_approval";
  }
  return null;
}

function getHitlRequests(event: any) {
  return event?.data?.interrupt?.action_requests ?? [];
}

function getLiveWidgets(meta: any) {
  return Array.isArray(meta?.rich_items)
    ? meta.rich_items.filter((item: any) => item?.type === "live_widget")
    : [];
}

function getImageParts(message: any) {
  return Array.isArray(message?.parts)
    ? message.parts.filter((part: any) => part?.type === "file" && part?.mediaType?.startsWith("image/"))
    : [];
}

function normalizeCanvasLanguage(artifactOrPayload: any) {
  const value = String(artifactOrPayload?.language ?? "").toLowerCase();
  return value === "svg" || value === "react" ? value : "html";
}

function getCanvasItems(meta: any) {
  return Array.isArray(meta?.rich_items)
    ? meta.rich_items.filter((item: any) => item?.type === "canvas_artifact")
    : [];
}
```
