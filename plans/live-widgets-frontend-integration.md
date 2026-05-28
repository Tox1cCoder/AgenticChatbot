# Live Widgets — Frontend Integration Guide

Live widgets are interactive in-chat UI components created by the backend during a normal assistant turn.
They are delivered through assistant message metadata, then hydrated over HTTP + WebSocket.

This guide is for frontend developers building against the AI SDK endpoints, including Next.js clients.

---

## Overview

```text
AI agent calls widget_create()
        │
        ▼
Assistant message metadata contains live_widgets[]
        │
        ▼
Frontend renders widget cards / placeholders
        │
        ▼
Frontend POSTs /widgets/{id}/connection to mint a short-lived token
        │
        ▼
Frontend opens the returned ws_url
        │
        ▼
Frontend receives widget_state_sync + widget_update events with full widget state
```

Key point:

- `live_widgets` metadata tells you that a widget exists.
- The actual widget `state` arrives over the widget WebSocket.

---

## 1. Detecting widgets in assistant messages

The backend exposes widgets in assistant message metadata under `messageMetadata.live_widgets`.

You may encounter this in two places:

- AI SDK streaming side-channel event: `data-assistant-message`
- Message history APIs that return assistant messages with `messageMetadata`

### Materialized assistant message shape

This is the shape most frontend code should use after the AI SDK has materialized the message:

```json
{
  "id": "msg-uuid",
  "role": "assistant",
  "createdAt": "2026-04-06T12:00:00Z",
  "messageMetadata": {
    "live_widgets": [
      {
        "widget_id": "b3f2c1a0-...",
        "session_id": "sess-uuid",
        "widget_type": "chart",
        "title": "Revenue Explorer",
        "status": "active",
        "version": 1,
        "connection_endpoint": "/widgets/b3f2c1a0-.../connection"
      }
    ]
  }
}
```

### Raw AI SDK streaming side-channel event

If you consume the raw stream directly, the widget metadata is nested under `data.message`:

```json
{
  "type": "data-assistant-message",
  "data": {
    "message": {
      "id": "msg-uuid",
      "role": "assistant",
      "createdAt": "2026-04-06T12:00:00Z",
      "messageMetadata": {
        "live_widgets": [
          {
            "widget_id": "b3f2c1a0-...",
            "session_id": "sess-uuid",
            "widget_type": "chart",
            "title": "Revenue Explorer",
            "status": "active",
            "version": 1,
            "connection_endpoint": "/widgets/b3f2c1a0-.../connection"
          }
        ]
      }
    }
  }
}
```

Notes:

- The backend strips `content` from this side-channel event because streamed text is already delivered through `text-delta` events.
- `live_widgets` is absent when the turn did not produce widgets. Treat missing as "no widgets", not as an error.

---

## 2. `live_widgets` metadata contract

Each widget entry has this shape:

| Field | Type | Description |
|---|---|---|
| `widget_id` | `string` | Stable identifier for this widget instance |
| `session_id` | `string` | Conversation / thread ID the widget belongs to |
| `widget_type` | `string` | `"table"` \| `"chart"` \| `"dashboard"` \| `"form"` \| `"list"` \| `"html"` |
| `title` | `string \| null` | Optional human-readable title |
| `status` | `string` | `"active"` or `"closed"` |
| `version` | `number` | Current widget version |
| `connection_endpoint` | `string` | Relative API path used to mint a short-lived widget connection token |

Stable public widget types:

- `table`
- `chart`
- `dashboard`
- `form`
- `list`
- `html`

Use `widget_type` to choose the frontend renderer.

---

## 3. Minting a connection token

Before opening a widget WebSocket, call:

```http
POST {API_BASE}/widgets/{widget_id}/connection
Authorization: Bearer {user_access_token}
Content-Type: application/json
```

Request body is empty.

### Response

```json
{
  "widget_id": "b3f2c1a0-...",
  "session_id": "sess-uuid",
  "widget_type": "chart",
  "title": "Revenue Explorer",
  "status": "active",
  "version": 1,
  "ws_url": "/widgets/b3f2c1a0-.../connect?session_id=sess-uuid&token=eyJ...",
  "token": "eyJ...",
  "expires_at": "2026-04-06T12:05:00Z"
}
```

| Field | Type | Description |
|---|---|---|
| `ws_url` | `string` | Relative WebSocket path including token query parameter |
| `token` | `string` | Widget-scoped JWT, valid for 5 minutes |
| `expires_at` | `string` | ISO-8601 token expiry |

Security notes:

- The token is scoped to a single widget and session.
- Do not persist it beyond the active connection lifecycle.

---

## 4. WebSocket contract

Open:

```text
ws://{host}/widgets/{widget_id}/connect?session_id={session_id}&token={token}
```

For TLS, use `wss://`.

### Server → client events

| `type` | Purpose |
|---|---|
| `widget_state_sync` | Full state snapshot immediately after connect |
| `widget_update` | Full state after any server-side change |
| `widget_close` | Widget is now closed / read-only |
| `ping` | Keepalive |
| `error` | Protocol / auth / availability error |

### Event payload shape

For `widget_state_sync`, `widget_update`, and `widget_close`, the server sends:

```json
{
  "type": "widget_update",
  "widget_id": "b3f2c1a0-...",
  "widget_type": "chart",
  "title": "Revenue Explorer",
  "state": {
    "chart_type": "bar",
    "labels": ["Jan", "Feb"],
    "datasets": [{ "label": "Revenue", "data": [12, 18] }]
  },
  "status": "active",
  "version": 2
}
```

### Client → server events

| `type` | Purpose | Required fields |
|---|---|---|
| `user_state_patch` | Shallow-merge a UI patch into widget state | `patch` |
| `pong` | Keepalive reply | none |

Example:

```json
{ "type": "user_state_patch", "patch": { "table_ui": { "sort_by": "price", "sort_dir": "asc" } } }
```

Merge rule:

- `user_state_patch` is a shallow merge against the current stored state.
- After a successful patch, the widget version increments and a new `widget_update` is broadcast.

---

## 5. Renderer strategy

### Structured widgets

For `table`, `chart`, `dashboard`, `form`, and `list`, build normal React renderers.

Recommended approach:

- Render a stable outer card from `live_widgets[]`
- Connect lazily or eagerly depending on your UX
- Re-render from the latest WebSocket `state`
- Send `user_state_patch` for user-driven UI state changes

### HTML widgets

For `widget_type="html"`, render the widget state as a sandboxed iframe micro-app.

Expected state shape:

```json
{
  "html": "<!doctype html>...",
  "height": 540,
  "caption": "Optional note shown above the iframe"
}
```

Recommended rendering:

```tsx
<iframe
  sandbox="allow-scripts allow-forms allow-modals allow-downloads"
  referrerPolicy="no-referrer"
  srcDoc={state.html}
  style={{ width: "100%", height: `${state.height ?? 540}px`, border: "none" }}
/>
```

Important:

- Do not inject `state.html` directly into the chat DOM.
- Use an iframe boundary.
- `html` widgets are intended for bounded in-chat micro experiences, not full standalone websites.
- There is currently no standardized iframe-to-parent `postMessage` contract for syncing arbitrary internal iframe UI events back into widget state. If you need that later, define it explicitly as a follow-up feature.

---

## 6. Recommended widget state conventions

The `state` object is model-authored, so render defensively. The following conventions are the current preferred shapes.

| `widget_type` | Typical top-level keys |
|---|---|
| `table` | `columns`, `rows` |
| `chart` | `chart_type`, `labels`, `datasets` |
| `dashboard` | `panels` |
| `form` | `fields`, `values` |
| `list` | `items`, `selection` |
| `html` | `html`, optional `height`, optional `caption` |

### Interactive wrapper conventions

Structured widgets may also include an interactive control model:

```json
{
  "controls": [
    {
      "key": "metric",
      "label": "Metric",
      "type": "segmented",
      "options": [
        { "value": "revenue", "label": "Revenue" },
        { "value": "profit", "label": "Profit" }
      ],
      "value": "revenue"
    }
  ],
  "control_values": {
    "metric": "revenue"
  },
  "views": {
    "metric=revenue": {
      "chart_type": "bar",
      "labels": ["Jan", "Feb"],
      "datasets": [{ "label": "Revenue", "data": [12, 18] }]
    },
    "metric=profit": {
      "chart_type": "line",
      "labels": ["Jan", "Feb"],
      "datasets": [{ "label": "Profit", "data": [3, 5] }]
    }
  }
}
```

Equivalent alternative:

```json
{
  "controls": [...],
  "control_values": { "metric": "revenue" },
  "variants": [
    {
      "match": { "metric": "revenue" },
      "state": { "...": "..." }
    },
    {
      "match": { "metric": "profit" },
      "state": { "...": "..." }
    }
  ]
}
```

Frontend guidance:

- Resolve `controls` using `control_values`
- Then resolve the active `views` / `variants` payload
- Then render the resolved payload as the widget body
- When users change controls, send back a shallow patch updating `control_values`

Additional optional UI state conventions:

- `table_ui`: search, sorting, pagination-like table view state
- `chart_ui`: chart type overrides, hidden series, presentation controls

These are still part of normal widget `state`.

---

## 7. End-to-end TypeScript example

```typescript
type WidgetType = "table" | "chart" | "dashboard" | "form" | "list" | "html";

interface LiveWidget {
  widget_id: string;
  session_id: string;
  widget_type: WidgetType;
  title: string | null;
  status: "active" | "closed";
  version: number;
  connection_endpoint: string;
}

interface WidgetRealtimeEvent {
  type: "widget_state_sync" | "widget_update" | "widget_close" | "ping" | "error";
  widget_id?: string;
  widget_type?: WidgetType;
  title?: string | null;
  state?: unknown;
  status?: "active" | "closed";
  version?: number;
  timestamp?: string;
  message?: string;
}

async function connectWidget(widget: LiveWidget, accessToken: string) {
  const resp = await fetch(widget.connection_endpoint, {
    method: "POST",
    headers: { Authorization: `Bearer ${accessToken}` },
  });

  if (!resp.ok) {
    throw new Error(`Token mint failed: ${resp.status}`);
  }

  const { ws_url } = await resp.json();
  const ws = new WebSocket(ws_url.startsWith("ws") ? ws_url : `${location.origin.replace(/^http/, "ws")}${ws_url}`);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data) as WidgetRealtimeEvent;

    switch (msg.type) {
      case "widget_state_sync":
      case "widget_update":
      case "widget_close":
        renderWidget(widget, msg.state, msg.version ?? 0, msg.status ?? widget.status);
        break;
      case "ping":
        ws.send(JSON.stringify({ type: "pong" }));
        break;
      case "error":
        console.error("Widget error:", msg.message);
        break;
    }
  };

  function applyUserPatch(patch: Record<string, unknown>) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "user_state_patch", patch }));
    }
  }

  return { ws, applyUserPatch };
}

function renderWidget(
  widget: LiveWidget,
  state: unknown,
  version: number,
  status: "active" | "closed",
) {
  console.log("render widget", widget.widget_id, widget.widget_type, version, status, state);
}
```

---

## 8. Parsing `live_widgets` from the stream

### If you use AI SDK message objects

Read:

```ts
const widgets = message.messageMetadata?.live_widgets ?? [];
```

### If you consume the raw side-channel event

Read:

```ts
if (event.type === "data-assistant-message") {
  const payload = JSON.parse(event.data);
  const widgets = payload.data?.message?.messageMetadata?.live_widgets ?? [];
}
```

### If you load message history from the API

Read:

```ts
const widgets = message.messageMetadata?.live_widgets ?? [];
```

The history API also mirrors metadata under `metadata`, but `messageMetadata` should be preferred for AI SDK-facing code.

---

## 9. Token expiry and reconnection

Widget tokens expire after 5 minutes.

To reconnect:

1. Call `POST /widgets/{widget_id}/connection` again with normal bearer auth.
2. Open a new WebSocket using the fresh `ws_url`.
3. Use the next `widget_state_sync` event as the source of truth.

Recommended:

- exponential backoff on reconnect
- stop patching when widget status becomes `closed`
- render closed widgets as read-only

---

## 10. Checklist for implementers

- [ ] Parse `messageMetadata.live_widgets`
- [ ] Support the stable public widget types: `table`, `chart`, `dashboard`, `form`, `list`, `html`
- [ ] Render structured widgets with normal React components
- [ ] Render `html` widgets in a sandboxed iframe, not inline HTML
- [ ] Call `POST /widgets/{id}/connection` before opening the widget socket
- [ ] Handle `widget_state_sync`, `widget_update`, and `widget_close`
- [ ] Reply with `pong` to every `ping`
- [ ] Treat `user_state_patch` as a shallow-merge patch contract
- [ ] Support the interactive wrapper conventions: `controls`, `control_values`, `views` / `variants`
- [ ] Reconnect with a fresh token when the widget token expires

## Migration note (2026-05-25): inline rich response v1

The inline rich response v1 contract (`response_format.md`) extends widget delivery with optional **inline placement**:

- Capable assistant messages now persist `messageMetadata.rich_items[]` alongside the existing `live_widgets[]` field. A widget appears with id `widget:<widget_id>` and `display_policy: inline_or_append`.
- When the assistant body contains a standalone marker `<!--rich:widget:<widget_id>-->`, render that widget at the marker position. The marker is model-authored placement data; the backend does not invent a position when it is omitted. While a streamed marker is waiting for its item upsert, reserve that inline position with a lightweight widget placeholder.
- The widget WebSocket protocol, token minting, and mount metadata are unchanged. Both `live_widgets[]` (legacy) and `tool_artifacts[]` (audit) continue to be persisted, so widget recovery and ownership checks remain identical.
- Frontend consumers wanting the inline experience should:
  1. Opt in with `inlineRichResponseV1: true` on chat/resume requests.
  2. Maintain a transient registry from `data-rich-items` upserts so a newly created widget can appear at the marker as soon as the marker text arrives in the stream.
  3. On final `data-assistant-message`, switch to `messageMetadata.rich_items` (authoritative).
  4. For repeated markers of the same widget id, mount the live WebSocket connection at most once; later occurrences should render a focus/open control rather than a second live mount.
- Non-opt-in clients receive only the existing `live_widgets[]` shape — no marker text — so no migration is forced on legacy implementations.

See [`README.md`](../README.md) → "Inline Rich Response (v1)" for the full contract.

## 11. Meaningful Widgets — Shared State Contract (2026-05-28)

Live widgets now read like article-quality inline visuals. All renderers (AI SDK clients **and** the Streamlit demo) should consume the same enriched state contract when present. Legacy widget shapes remain valid.

### State envelope

```json
{
  "presentation": {
    "title": "Phase-space intuition",
    "caption": "The same oscillator becomes easier to read when position and velocity are viewed together.",
    "layout": "article",
    "x_label": "Position",
    "y_label": "Velocity",
    "unit": "normalized",
    "x_kind": "time",
    "annotations": [
      {"label": "Stable orbit", "series": "Trajectory", "point_index": 8}
    ]
  },
  "chart_type": "line",
  "labels": ["t0", "t1", "t2"],
  "datasets": [{"label": "Trajectory", "data": [0.1, 0.4, 0.9]}],
  "controls": [
    {"key": "damping", "label": "Damping", "type": "slider", "min": 0, "max": 1, "step": 0.05, "value": 0.2}
  ],
  "control_values": {"damping": 0.2},
  "views": {
    "damping=0.2": {"chart_type": "line", "labels": ["t0", "t1", "t2"], "datasets": [{"label": "Trajectory", "data": [0.1, 0.4, 0.9]}]}
  },
  "actions": [
    {
      "key": "explain_current_state",
      "label": "Explain current state",
      "type": "assistant_message",
      "message_template": "Explain the widget state for damping={{control_values.damping}} in the context of the current answer."
    }
  ]
}
```

### Required frontend behavior

- Opt into `inlineRichResponseV1` on chat/resume requests.
- Read **transient** widgets from `data-rich-items` SSE events as they stream.
- Read **final** widgets from `messageMetadata.rich_items` after `data-assistant-message`.
- Hydrate widget state through `POST /widgets/{id}/connection` and the widget WebSocket. **No widget state is embedded in `rich_items` — state always arrives over the WebSocket.**
- Render `presentation`, `controls`, `views`/`variants`, chart hover values, and `actions` using the same conventions the Streamlit demo uses.
- For action buttons, call `POST /widgets/{id}/actions/{action_key}` with optional `state_patch` and `input_values`. Submit the returned `content` through the normal AI SDK chat stream (do **not** invoke the assistant directly from the action endpoint — that endpoint only renders the message).

### Action endpoint contract

Request body (`POST /widgets/{widget_id}/actions/{action_key}`):

```json
{
  "input_values": {"note": "optional user input"},
  "state_patch": {"control_values": {"damping": 0.4}}
}
```

Both fields are optional. `state_patch` is applied as a shallow merge to widget state **before** the action template is rendered.

Response:

```json
{
  "widget_id": "w-1",
  "session_id": "conversation-id",
  "action_key": "explain_current_state",
  "content": "Explain the widget state for damping=0.4 in the context of the current answer."
}
```

Errors:
- `404` — widget or action not found
- `400` — action is not of type `assistant_message`, or has no `message_template`
- `403` — access denied (widget belongs to another conversation)

The endpoint also records the latest action in `widget.state.last_action` so a subsequent `widget_get_state` call from the agent can inspect user intent.

### Hover values

Bars, line points, and donut slices include both `data-tooltip` (custom hover) and SVG `<title>` (accessibility fallback). The tooltip payload is the pipe-delimited string `series|label|value` (donut slices use `series||value`). Frontends are free to ignore the custom format and rely on `<title>` + `aria-label`, but the recommendation is to mirror the demo's `.lw-tooltip` floating box for consistent UX.

### Reference snippet — running a widget action

```ts
async function runWidgetAction(widgetId: string, actionKey: string, statePatch: unknown) {
  const response = await fetch(`/widgets/${widgetId}/actions/${actionKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state_patch: statePatch }),
  });
  if (!response.ok) throw new Error(`Widget action failed: ${response.status}`);
  const payload = await response.json();
  sendMessage({ text: payload.content }); // submit through the normal chat stream
}
```

### Implementer checklist

- [ ] Render `presentation.title`, `caption`, `x_label`, `y_label`, `unit` near the widget body, not as a heavy header card
- [ ] Render `presentation.annotations` as callouts/markers when their target series + point_index resolves
- [ ] Attach hover values to bars, line points, and donut slices (custom tooltip **or** SVG `<title>` minimum)
- [ ] Render `actions` of type `assistant_message` as buttons; submit returned `content` through the AI SDK chat stream with `inlineRichResponseV1: true`
- [ ] For repeated widget markers, mount the WebSocket once; later occurrences should focus/open the existing widget
