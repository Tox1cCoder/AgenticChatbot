# Live Widgets — Frontend Integration Guide

Live widgets are interactive in-chat UI components created by the backend during a normal assistant turn.
They are delivered through assistant message metadata, then hydrated over HTTP + WebSocket.

Live widgets have **one supported type: `html`**. A widget is a self-contained micro-app
(animation, sliders, live readouts, a small diagram/graph) authored by the model as a single
HTML document. The frontend renders the widget state **only as a sandboxed iframe** from
`state.html` — it never builds structured React renderers from the widget type.

This guide is for frontend developers building against the AI SDK endpoints, including Next.js clients.

---

## Overview

```text
AI agent calls widget_create(widget_type="html")
        │
        ▼
Assistant message metadata contains live_widgets[]
        │
        ▼
Frontend renders a widget card / placeholder
        │
        ▼
Frontend POSTs /widgets/{id}/connection to mint a short-lived token
        │
        ▼
Frontend opens the returned ws_url
        │
        ▼
Frontend receives widget_state_sync + widget_update events with full widget state
        │
        ▼
Frontend renders state.html inside a sandboxed iframe
```

Key point:

- `live_widgets` metadata tells you that a widget exists.
- The actual widget `state` (the HTML document, its height, an optional caption) arrives over the widget WebSocket.

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
        "widget_type": "html",
        "title": "Harmonic Oscillation",
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
            "widget_type": "html",
            "title": "Harmonic Oscillation",
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
| `widget_type` | `string` | Always `"html"` for new widgets |
| `title` | `string \| null` | Optional human-readable title |
| `status` | `string` | `"active"` or `"closed"` |
| `version` | `number` | Current widget version |
| `connection_endpoint` | `string` | Relative API path used to mint a short-lived widget connection token |

There is one supported `widget_type`: `html`. The removed structured types
(`table`/`chart`/`dashboard`/`form`/`list`) and the former `iframe`/`micro_app` aliases are
no longer created. Legacy persisted conversations may still surface a removed type — render
those as an unsupported/legacy placeholder, not a structured renderer (see § 5).

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
  "widget_type": "html",
  "title": "Harmonic Oscillation",
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
| `error` | Protocol / auth / availability / contract error |

### Event payload shape

For `widget_state_sync`, `widget_update`, and `widget_close`, the server sends:

```json
{
  "type": "widget_update",
  "widget_id": "b3f2c1a0-...",
  "widget_type": "html",
  "title": "Harmonic Oscillation",
  "state": {
    "html": "<!doctype html>...",
    "height": 620,
    "caption": "Drag the sliders to change amplitude and frequency."
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

Merge rule:

- `user_state_patch` is a shallow merge against the current stored state.
- After a successful patch the widget version increments and a new `widget_update` is broadcast.
- For an HTML widget, the server **re-validates the merged state against the HTML contract**
  before storing it. A patch that would produce empty `html` or an out-of-range `height` is
  rejected with an `error` event and is not applied. In practice an HTML micro-app manages its
  own interactivity inside the iframe, so most clients never send `user_state_patch`.

---

## 5. Renderer strategy

There is one renderer: a **sandboxed iframe**.

For `widget_type="html"`, render the widget state as a sandboxed iframe micro-app.

Expected state shape:

```json
{
  "html": "<!doctype html>...",
  "height": 620,
  "caption": "Optional note shown above the iframe"
}
```

Recommended rendering:

```tsx
<iframe
  sandbox="allow-scripts allow-forms allow-modals allow-downloads"
  referrerPolicy="no-referrer"
  srcDoc={state.html}
  style={{ width: "100%", height: `${state.height ?? 620}px`, border: "none" }}
/>
```

Important:

- `state.html` is untrusted, executable content. **Do not inject it into the chat DOM.** Use an iframe boundary.
- Read only the contract keys `html`, `height`, `caption`. The former aliases
  (`document`/`content`/`srcdoc`/`min_height`/`minHeight`) are not part of the contract.
- `html` widgets are bounded in-chat micro experiences, not full standalone websites.
- There is currently no standardized iframe-to-parent `postMessage` contract for syncing
  internal iframe UI events back into widget state. If you need that later, define it
  explicitly as a follow-up feature.

### Legacy structured widgets

The structured types (`table`/`chart`/`dashboard`/`form`/`list`) are removed. New turns never
create them. If a legacy conversation surfaces a widget whose `widget_type` is not `html`,
render a small unsupported/legacy placeholder (e.g. "This widget type is no longer supported")
rather than attempting a structured renderer. Do not branch the renderer on `widget_type`
beyond this legacy guard.

---

## 6. End-to-end TypeScript example

```typescript
type WidgetType = "html";

interface LiveWidget {
  widget_id: string;
  session_id: string;
  widget_type: string; // "html" for new widgets; legacy values may appear
  title: string | null;
  status: "active" | "closed";
  version: number;
  connection_endpoint: string;
}

interface HtmlWidgetState {
  html: string;
  height?: number;
  caption?: string;
}

interface WidgetRealtimeEvent {
  type: "widget_state_sync" | "widget_update" | "widget_close" | "ping" | "error";
  widget_id?: string;
  widget_type?: string;
  title?: string | null;
  state?: HtmlWidgetState;
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

  return { ws };
}

function renderWidget(
  widget: LiveWidget,
  state: HtmlWidgetState | undefined,
  version: number,
  status: "active" | "closed",
) {
  // Render state.html inside a sandboxed iframe (srcDoc). Never inject into the chat DOM.
  console.log("render widget", widget.widget_id, version, status, state?.height);
}
```

---

## 7. Parsing `live_widgets` from the stream

### If you use AI SDK message objects

```ts
const widgets = message.messageMetadata?.live_widgets ?? [];
```

### If you consume the raw side-channel event

```ts
if (event.type === "data-assistant-message") {
  const payload = JSON.parse(event.data);
  const widgets = payload.data?.message?.messageMetadata?.live_widgets ?? [];
}
```

### If you load message history from the API

```ts
const widgets = message.messageMetadata?.live_widgets ?? [];
```

The history API also mirrors metadata under `metadata`, but `messageMetadata` should be preferred for AI SDK-facing code.

---

## 8. Token expiry and reconnection

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

## 9. Checklist for implementers

- [ ] Parse `messageMetadata.live_widgets`
- [ ] Treat `html` as the only live widget type; render a legacy placeholder for any other type
- [ ] Render `html` widgets in a sandboxed iframe from `state.html`, never inline in the chat DOM
- [ ] Read only the contract keys `html`, `height`, `caption`
- [ ] Call `POST /widgets/{id}/connection` before opening the widget socket
- [ ] Handle `widget_state_sync`, `widget_update`, and `widget_close`
- [ ] Reply with `pong` to every `ping`
- [ ] Reconnect with a fresh token when the widget token expires

## Migration note (2026-05-25): inline rich response v1

The inline rich response v1 contract (`response_format.md`) extends widget delivery with optional **inline placement**:

- Capable assistant messages persist `messageMetadata.rich_items[]` alongside the existing `live_widgets[]` field. A widget appears with id `widget:<widget_id>` and `display_policy: inline_or_append`.
- When the assistant body contains a standalone marker `<!--rich:widget:<widget_id>-->`, render that widget at the marker position. The marker is model-authored placement data; the backend does not invent a position when it is omitted. While a streamed marker is waiting for its item upsert, reserve that inline position with a lightweight widget placeholder.
- The widget WebSocket protocol, token minting, and mount metadata are unchanged. Both `live_widgets[]` (legacy) and `tool_artifacts[]` (audit) continue to be persisted, so widget recovery and ownership checks remain identical.
- Frontend consumers wanting the inline experience should:
  1. Opt in with `inlineRichResponseV1: true` on chat/resume requests.
  2. Maintain a transient registry from `data-rich-items` upserts so a newly created widget can appear at the marker as soon as the marker text arrives in the stream.
  3. On final `data-assistant-message`, switch to `messageMetadata.rich_items` (authoritative).
  4. For repeated markers of the same widget id, mount the live WebSocket connection at most once; later occurrences should render a focus/open control rather than a second live mount.
- Non-opt-in clients receive only the existing `live_widgets[]` shape — no marker text — so no migration is forced on legacy implementations.

See [`README.md`](../README.md) → "Inline Rich Response (v1)" for the full contract.

## 10. HTML widget state contract (2026-06-22)

The one supported widget state shape:

```json
{
  "html": "<!doctype html>...self-contained responsive HTML/CSS/JS...",
  "height": 620,
  "caption": "Optional short caption"
}
```

- `html` — a complete, self-contained document. It owns all of its own interactivity
  (animation, sliders, live readouts, canvas/SVG diagrams) with vanilla JS and inline CSS;
  no external dependencies, auth assumptions, or cross-window requirements.
- `height` — a number between 260 and 960. The iframe is rendered at this pixel height.
- `caption` — optional short note rendered above the iframe.

The backend enforces this contract at `widget_create` / `widget_update` time and again when an
action `state_patch` or a WebSocket `user_state_patch` is applied. Invalid state never reaches
the frontend.

### Action endpoint (optional)

An HTML widget's state may include a top-level `actions[]` array of
`type: "assistant_message"` entries. `POST /widgets/{widget_id}/actions/{action_key}`:

- applies an optional `state_patch` (re-validated against the HTML contract before storing),
- renders the action's `message_template` from the widget state (supports `{{state.<path>}}`
  and `{{input_values.<key>}}`),
- records `last_action` on the widget state,
- returns `{widget_id, session_id, action_key, content}`.

Request body:

```json
{
  "input_values": {"note": "optional user input"},
  "state_patch": {"caption": "updated caption"}
}
```

Both fields are optional. Submit the returned `content` through the normal AI SDK chat stream —
the endpoint does **not** invoke the assistant directly.

Errors:
- `404` — widget or action not found
- `400` — action is not `assistant_message`, has no `message_template`, or the merged state breaks the HTML contract
- `403` — access denied (widget belongs to another conversation)
