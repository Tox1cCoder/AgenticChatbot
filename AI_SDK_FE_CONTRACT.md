# AI SDK Frontend Contract

This document describes the frontend-facing response shape for the AI SDK-compatible chat path.

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
| `userId` | string, optional | Enables server-side user-aware features. |
| `inlineRichResponseV1` | boolean, optional | Opt in to marker-positioned rich UI items. Also accepted as `inline_rich_response_v1`. |
| `deviceId` | string, optional | Usually injected by the sidecar/client backend for local runtime tools. |

The response is `text/event-stream` and includes:

```http
X-Vercel-Ai-UI-Message-Stream: v1
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
  "output": {},
  "render": {}
}
```

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
      "created_at": "ISO-8601 timestamp",
      "message_metadata": {},
      "messageMetadata": {},
      "metadata": {},
      "parts": []
    }
  }
}
```

Notes:

- `data-assistant-message.data.message.content` is intentionally omitted on the stream because answer text already arrived as `text-delta`.
- Metadata may appear under `message_metadata`, `messageMetadata`, and/or `metadata` depending on path/projection. Treat `messageMetadata ?? message_metadata ?? metadata` as the backend metadata.
- `parts` may contain generated image/file parts. For v1 rich responses, only selected images appear as file parts.

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

Other data events:

```json
{ "type": "data-agent-selected", "data": { "agent": "chat_agent" }, "transient": true }
{ "type": "data-user-message", "data": { "message": {} }, "transient": true }
{ "type": "data-continuation", "data": { "round": 1, "max_rounds": 3, "reason": "tool_budget" }, "transient": true }
{ "type": "data-node-complete", "data": { "node": "chat_agent" }, "transient": true }
{ "type": "heartbeat" }
```

## Subagent Progress

> **Status: planned.** Ships with the "Live Subagent Progress" work (see `event_streaming.md`, Tasks 11–15). Not yet emitted by the deployed backend — this section documents the agreed target shape so the frontend can build in parallel.

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
const requests = interrupt?.action_requests ?? event.data?.pendingToolCalls ?? [];
```

When `isHitl` is true, render a human-in-the-loop UI from `requests`. The stream will finish immediately after the interrupt event.

## HITL Interrupt Shape

```json
{
  "type": "data-interrupt",
  "data": {
    "threadId": "conversation-thread-id",
    "next": ["approval-node"],
    "pendingToolCalls": [
      {
        "id": "tool-call-id",
        "name": "tool_name",
        "args": {},
        "tool_call_id": "tool-call-id"
      }
    ],
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
      "message_metadata": {
        "interrupt": {},
        "paused": true,
        "pause_reason": "tool_approval_required",
        "thread_id": "conversation-thread-id",
        "next": ["approval-node"],
        "pending_tool_calls": []
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
| `data.interrupt.action_requests[]` | array | Canonical list of tool calls awaiting human input. Prefer this over `pendingToolCalls` for UI. |
| `action_requests[].action` | string | Tool/action name. |
| `action_requests[].args` | object | Original tool arguments. Use this to show what will run. |
| `action_requests[].description` | string or null | Optional human-readable tool description. |
| `action_requests[].tool_call_id` | string or null | Preferred decision target. Send back as `toolCallId`. |
| `action_requests[].task_id` | string or null | Fallback decision target if `tool_call_id` is missing. |
| `action_requests[].allowed_decisions` | array or null | If present, restrict UI buttons to these decisions. |

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
| `toolCallId` | string | recommended | Target tool call ID. Snake case `tool_call_id` is also accepted. |
| `taskId` | string | optional | Fallback target ID. Snake case `task_id` is also accepted. |
| `action` | string | optional | Tool/action name. Useful for audit/debug display. |
| `args` | object | optional | Meaning depends on decision type. |

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

`GET /ai/conversations/{conversationId}/messages` returns:

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
        "messageMetadata": {},
        "createdAt": "ISO-8601 timestamp"
      }
    ],
    "total": 1
  }
}
```

Message fields:

| Field | Type | Notes |
|---|---|---|
| `id` | string | Message UUID. |
| `role` | string | `user` or `assistant`. |
| `content` | string | Markdown/plain text. For rich v1, may contain `<!--rich:<id>-->` markers only when client opts in. |
| `parts` | array or null | AI SDK UI parts. Images are exposed as `file` parts. |
| `parts[].type` | string | `text`, `file`, or `reasoning`. |
| `parts[].text` | string | Text part content. |
| `parts[].url` | string | File URL, data URL, remote URL, or blob URL. |
| `parts[].mediaType` | string | MIME type for file part. |
| `parts[].reasoning` | string | Reasoning content for reasoning parts, if present. |
| `metadata` | object or null | Mirror of backend metadata for UI compatibility. |
| `messageMetadata` | object or null | Canonical AI SDK metadata field. Same content as `metadata` for assistant messages. |
| `createdAt` | string or null | ISO-8601 timestamp. |

## Assistant Metadata Shape

Use:

```ts
const backendMeta =
  message.messageMetadata ?? message.message_metadata ?? message.metadata ?? {};
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
  "reasoning_summary": "provider reasoning summary if available",
  "reasoning_tokens": 123,
  "context_window": {},
  "agent": {},
  "handoff": {},
  "subagent_results": [],
  "tool_artifacts": [],
  "live_widgets": [],
  "images": [],
  "has_images": true,
  "images_count": 1,
  "agentic_images_count": 1,
  "documents_cited": [],
  "chunks_retrieved": 3,
  "documents_found": 1,
  "citations": [],
  "canvas_artifact": {},
  "rich_items_version": 1,
  "rich_items": [],
  "rich_reference_warnings": [],
  "suggested_questions": [],
  "interrupt": {},
  "paused": true,
  "pause_reason": "tool_approval_required"
}
```

All fields are optional. Frontend should ignore unknown keys.

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

### Assistant Image Metadata

Legacy image metadata may appear in `backendMeta.images`:

```json
{
  "images": [
    {
      "data": "base64_encoded_image_data",
      "mime": "image/jpeg",
      "name": "Image description or caption",
      "page_number": 5,
      "caption": "Original image caption",
      "url": "https://optional.remote/image.png",
      "source_url": "https://optional.source/page",
      "description": "Optional description",
      "rich_item_id": "image:document:image-id"
    }
  ],
  "has_images": true,
  "images_count": 1
}
```

Image metadata fields:

| Field | Type | Notes |
|---|---|---|
| `data` | string | Raw base64. AI SDK adapter emits a `file` part as `data:<mime>;base64,<data>`. |
| `url` | string | Remote/data/blob URL. Used directly for `file.url`. |
| `base64` / `image` / `source` | string | Fallback image source fields accepted by the adapter. |
| `mime` / `mimeType` / `mediaType` / `contentType` | string | MIME type. Defaults to `image/png` for response projection. |
| `name` | string | Optional display name. |
| `caption` | string | Optional caption from document/image extraction. |
| `page_number` | number | Optional document page number. |
| `source_url` | string | Optional source page URL. |
| `description` | string | Optional description. |
| `rich_item_id` / `id` | string | Used to filter v1 rich images. |

Frontend rendering rule:

- Prefer AI SDK `parts[].type === "file"` for images.
- If `rich_items_version === 1`, render selected image rich items and file parts only. Do not build a separate gallery from `backendMeta.images`.
- If there is no `rich_items_version`, legacy clients may render `backendMeta.images` as a gallery.

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
    "widget_type": "table",
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
  "id": "canvas:assistant-message-id",
  "type": "canvas_artifact",
  "display_policy": "inline_or_append",
  "title": "Canvas title",
  "payload": {
    "language": "html",
    "title": "Canvas title",
    "content": "<!doctype html>...",
    "preferred_height": 640
  }
}
```

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
6. On final `data-assistant-message`, replace transient items with `backendMeta.rich_items`.
7. Append only unreferenced `inline_or_append` items. Never append `inline_only` images.

## Live Widgets

Live widgets can appear in two places:

1. Legacy metadata: `backendMeta.live_widgets[]`
2. Rich response v1: `backendMeta.rich_items[]` where `type === "live_widget"`

Legacy `live_widgets[]` shape:

```json
{
  "widget_id": "widget-id",
  "session_id": "conversation-id",
  "widget_type": "table",
  "title": "Widget title",
  "status": "active",
  "version": 1,
  "connection_endpoint": "/widgets/widget-id/connection"
}
```

Fields:

| Field | Type | Notes |
|---|---|---|
| `widget_id` | string | Widget identifier. |
| `session_id` | string | Conversation/session id. |
| `widget_type` | string | Renderer type, such as `table`, `chart`, or a custom widget type. |
| `title` | string or null | Optional title. |
| `status` | string | `active` or `closed`. |
| `version` | number | Incrementing state version. |
| `connection_endpoint` | string | POST this endpoint to mint a widget WebSocket token. |

Mount flow:

1. Render placeholder from metadata.
2. POST `connection_endpoint` with normal auth.
3. Open the returned `ws_url`.
4. Render `widget_state_sync.state`.
5. Apply future `widget_update.state`.

Connection response:

```json
{
  "widget_id": "widget-id",
  "session_id": "conversation-id",
  "widget_type": "table",
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
  "widget_type": "table",
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
  "widget_type": "table",
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
  "widget_type": "table",
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
| `blob_id` / related blob fields | optional | May appear when large tool output is offloaded. |

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

Canvas responses can include legacy metadata:

```json
{
  "canvas_artifact": {
    "content": "<full self-contained HTML / SVG document>",
    "language": "html",
    "title": "Short title",
    "editable": true
  }
}
```

Rich v1 may also expose this as a `canvas_artifact` rich item.

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
  return message?.messageMetadata ?? message?.message_metadata ?? message?.metadata ?? {};
}

function isHitlEvent(event: any) {
  return event?.type === "data-interrupt";
}

function getHitlRequests(event: any) {
  return event?.data?.interrupt?.action_requests ?? event?.data?.pendingToolCalls ?? [];
}

function getLiveWidgets(meta: any) {
  const richWidgets = Array.isArray(meta?.rich_items)
    ? meta.rich_items.filter((item: any) => item?.type === "live_widget")
    : [];
  return richWidgets.length ? richWidgets : meta?.live_widgets ?? [];
}

function getImageParts(message: any) {
  return Array.isArray(message?.parts)
    ? message.parts.filter((part: any) => part?.type === "file" && part?.mediaType?.startsWith("image/"))
    : [];
}
```
