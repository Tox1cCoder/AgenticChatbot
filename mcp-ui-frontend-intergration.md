# MCP UI — Frontend Integration Guide

Companion to `live-widgets-frontend-integration.md`. This guide covers the broader tool-result rendering contract introduced by `mcp_ui_plan.md`: static MCP results (tables, charts, images, MCP apps, errors) and how they coexist with live widgets.

Audience: frontend developers building a Next.js client against the AI SDK endpoints.

---

## Two contracts, one chat UI

| Surface | Scope | Source of truth | Covered in |
|---|---|---|---|
| `live_widgets` + WebSocket | Interactive, server-updated widgets (`table`, `chart`, `dashboard`, `form`, `list`, `html`) | `messageMetadata.live_widgets` + `/widgets/{id}/connection` + WS | [`live-widgets-frontend-integration.md`](live-widgets-frontend-integration.md) |
| `tool_artifacts[].render` + `tool-output-available.render` | One-shot tool results (text, tables, charts, images, MCP apps, errors, JSON fallbacks) | `messageMetadata.tool_artifacts[].render` + SSE `tool-output-available.render` | This document |

They are **additive, not alternative**. A single assistant turn can produce both a live widget and several static tool renders. Route them by the rules in [§5](#5-resolving-overlap-with-live_widgets) and there is no collision.

---

## 1. What the backend emits

### 1.1 Live SSE stream — `tool-output-available`

Every successful or failed tool call emits one `tool-output-available` chunk. The `render` sibling is new and optional:

```json
{
  "type": "tool-output-available",
  "toolCallId": "tc-1",
  "output": "<string or JSON — compact model-facing form>",
  "render": {
    "version": 1,
    "type": "mcp_app",
    "model_content": "Created presentation: Quarterly Roadmap",
    "text": "Created presentation: Quarterly Roadmap",
    "title": "Quarterly Roadmap",
    "structured_content": { "presentation_id": "deck_123" },
    "content": [{ "type": "text", "text": "Created presentation: Quarterly Roadmap" }],
    "resources": [
      { "uri": "ui://canva/presentation-viewer.html", "mime_type": "text/html", "title": "Presentation viewer" }
    ],
    "template_uri": "ui://canva/presentation-viewer.html",
    "ui_meta": { "openai/outputTemplate": "ui://canva/presentation-viewer.html" }
  }
}
```

`render` is absent when:

- the tool ran before this feature shipped, or
- the tool result is completely opaque and the normalizer produced no meta.

Always fall back to `output` when `render` is missing.

### 1.2 Message history — `messageMetadata.tool_artifacts[]`

`GET /ai/conversations/{id}/messages` returns `UIMessage[]`. For assistant messages:

```json
{
  "id": "msg-uuid",
  "role": "assistant",
  "createdAt": "2026-04-23T12:00:00Z",
  "messageMetadata": {
    "tool_artifacts": [
      {
        "tool_call_id": "tc-1",
        "tool": "canva_create_presentation",
        "args": { "prompt": "roadmap" },
        "output": "Created presentation: Quarterly Roadmap",
        "error": null,
        "status": "success",
        "render": { "version": 1, "type": "mcp_app", "...": "..." }
      }
    ],
    "live_widgets": [ "... see live-widgets doc ..." ]
  }
}
```

Each artifact mirrors the render it had during streaming. Persisted messages hydrate back into exactly the same render flow as the live stream.

---

## 2. Type definitions

Drop this into `types/tool-render.ts`:

```ts
export type ToolRenderType =
  | "mcp_app"
  | "live_widget"
  | "chart"
  | "table"
  | "image"
  | "resource"
  | "json"
  | "text"
  | "error";

export interface ToolResource {
  uri: string;
  mime_type: string;
  title?: string;
}

export interface ToolContentBlock {
  type: "text" | "image" | "audio" | "resource" | "resource_link" | (string & {});
  text?: string;
  mimeType?: string;
  data?: string;              // empty when _truncated === true
  uri?: string;
  _truncated?: boolean;
  _original_size?: number;
}

export interface TruncationMarker {
  _truncated: true;
  _original_size: number;
}

export interface ToolRender {
  version: 1;
  type: ToolRenderType;
  model_content: string;
  text: string;
  title?: string;
  structured_content?: unknown | TruncationMarker;
  content?: ToolContentBlock[];
  resources?: ToolResource[];
  template_uri?: string;
  ui_meta?: Record<string, unknown>;
  error?: string;             // only when type === "error"
}

export interface ToolArtifact {
  tool_call_id: string | null;
  tool: string;
  args: unknown;
  output: string | null;
  error: string | null;
  status: "success" | "error" | "rejected";
  render?: ToolRender;
}
```

For the `live_widgets` types, see the widgets guide — don't redefine them here.

---

## 3. What each `render.type` means

| `render.type` | Where it fires | How to render |
|---|---|---|
| `text` | Plain text tool result | Render `render.text` as text (never HTML) |
| `json` | Structured result without a richer shape | JSON tree viewer over `render.structured_content`; fall back to `output` |
| `table` | Structured result with `columns` + `rows` | Render from `render.structured_content` (see [§4.3](#43-table)) |
| `chart` | Structured result with `chart_type` + `labels` + `datasets` | Render from `render.structured_content` (see [§4.4](#44-chart)) |
| `image` | Content blocks contain an image | Render `render.content[]` image blocks (see [§4.5](#45-image)) |
| `resource` | MCP resources without an app template | List `render.resources[]` (see [§4.6](#46-resource)) |
| `mcp_app` | Result advertises a UI app template URI | Show title + template URI; full iframe mounting is out of scope for v1 (see [§4.7](#47-mcp_app)) |
| `live_widget` | Tool was `widget_create` / `widget_update` | **Do NOT render from `render`**. Mount via `messageMetadata.live_widgets` + WebSocket (see [§5](#5-resolving-overlap-with-live_widgets)) |
| `error` | Tool execution failed | Render `render.error` (or `render.text`) as an error block |

`render.type` is a closed set today, but treat unknown values as `json` so the client doesn't break when the backend adds new types.

---

## 4. Per-type renderer guidance

### 4.1 Dispatcher

One switch, one fallback:

```tsx
export function ToolRenderView({
  render,
  fallback,        // raw `output`
  liveWidgets,     // from messageMetadata.live_widgets
  toolCallId,
}: {
  render: ToolRender | undefined;
  fallback: string | null;
  liveWidgets: LiveWidget[] | undefined;
  toolCallId: string | null;
}) {
  if (!render) return <OutputFallback value={fallback} />;

  switch (render.type) {
    case "text":        return <TextBlock>{render.text}</TextBlock>;
    case "error":       return <ToolError reason={render.error ?? render.text} />;
    case "table":       return <TableRenderer data={render.structured_content} title={render.title} />;
    case "chart":       return <ChartRenderer data={render.structured_content} title={render.title} />;
    case "image":       return <ImageRenderer content={render.content} resources={render.resources} />;
    case "resource":    return <ResourceList resources={render.resources} />;
    case "mcp_app":     return <McpAppPreview render={render} />;
    case "live_widget": return <LiveWidgetMount render={render} liveWidgets={liveWidgets} />;
    case "json":
    default:            return <JsonViewer value={render.structured_content ?? fallback} />;
  }
}
```

### 4.2 `text` / `error`
Trivial. For `error`, surface `render.error` (clean — no `Error:` prefix). Do **not** surface `output` for errors because it contains the prefixed form intended for the model.

### 4.3 `table`
`render.structured_content` has shape:

```ts
{ columns: string[]; rows: (unknown[] | Record<string, unknown>)[] }
```

Rows can be either list-of-list (positional) or list-of-dict (keyed by column). Normalize in the renderer. TanStack Table or a plain `<table>` is fine.

### 4.4 `chart`
`render.structured_content` has shape:

```ts
{
  chart_type: "line" | "bar" | "area" | (string & {});
  labels: (string | number)[];
  datasets: { label?: string; data: number[] }[];
}
```

Recharts fits naturally. Important: this is the **static chart** path. If you want a live, user-interactive chart, it comes through the live-widget path instead — `render.type === "live_widget"` with `widget_type === "chart"`.

### 4.5 `image`
Iterate `render.content[]` and filter for `block.type === "image"`. Key fields:

- `block.mimeType` → content type
- `block.data` → base64 payload (already redacted when oversize — see [§6](#6-edge-cases))

Render as `data:<mime>;base64,<data>`. When `block._truncated === true`, **do not** try to decode `data` (it is `""`). Show a placeholder with the original size and a hint that the tool should return a URL instead.

### 4.6 `resource`
List `render.resources[]`:

- If `uri` starts with `http://` or `https://` → link button opens in new tab.
- Otherwise → show the URI and `mime_type` as read-only text.

### 4.7 `mcp_app`
v1 is a preview, not a live iframe host:

- Show `render.title`.
- Show `render.template_uri` (copyable).
- Expand `render.structured_content` in a collapsible JSON panel.

Future work: when you define a parent↔iframe bridge for MCP apps, mount the template here with a strict allowlist of template origins. **Do not** naively iframe arbitrary template URIs.

### 4.8 `live_widget`
Render nothing from `render` directly. See [§5](#5-resolving-overlap-with-live_widgets).

### 4.9 `json`
Generic tree viewer over `render.structured_content`. When structured_content is the truncation marker `{_truncated: true, _original_size: N}`, show a chip — "N bytes omitted" — and fall back to `output`.

---

## 5. Resolving overlap with `live_widgets`

Widget tool calls produce **both** surfaces. They are complementary:

- `tool_artifacts[].render.type === "live_widget"` → "this tool call created widget X" (descriptor only)
- `messageMetadata.live_widgets[]` → "here's how to mount widget X" (carries `connection_endpoint`)

### Rules

1. **Mount via `live_widgets`, never via `render.structured_content`.** The descriptor on the render is just enough to correlate; it does **not** contain the full widget state. Full state arrives over the WebSocket as `widget_state_sync`.
2. **Correlate by `widget_id`**, not `tool_call_id`. A single turn can produce multiple widgets, and `live_widgets` is keyed by widget. Read `widget_id` from `render.structured_content.widget_id`.
3. **Deduplicate.** If you loop over tool artifacts to render them, skip any artifact whose `render.type === "live_widget"` — you'll render those via the `live_widgets` pass. Or render a small "Widget X created" inline chip in the tool-call card and let the widget card carry the actual UI.
4. **Missing `live_widgets` entry.** If `render.type === "live_widget"` but the correlated widget is not in `live_widgets[]` (older message, server bug), degrade to a non-interactive JSON view of `render.structured_content`. Do not silently drop the call.

### `LiveWidgetMount` component sketch

```tsx
function LiveWidgetMount({
  render,
  liveWidgets,
}: { render: ToolRender; liveWidgets?: LiveWidget[] }) {
  const widgetId =
    (render.structured_content as { widget_id?: string } | undefined)?.widget_id ?? null;

  const live = widgetId
    ? liveWidgets?.find((w) => w.widget_id === widgetId)
    : undefined;

  if (!live) {
    return <WidgetStub render={render} reason="Widget metadata not available" />;
  }

  // Defers entirely to the widgets integration guide.
  return <LiveWidgetCard widget={live} />;
}
```

Everything below `<LiveWidgetCard>` — token mint, WebSocket, state hydration, `user_state_patch` — is owned by `live-widgets-frontend-integration.md`. This doc does not redefine it.

---

## 6. Edge cases

### 6.1 Truncation markers

The backend caps payload size to keep SSE and checkpoints bounded:

- Image-like content blocks with `data` > 64 KB are redacted to `{ ..., data: "", _truncated: true, _original_size: N }`.
- `render.structured_content` > 128 KB is replaced with `{ _truncated: true, _original_size: N }`.

Render both defensively. Show a chip ("42 KB omitted for transport") and, when applicable, suggest the tool return a URL.

### 6.2 Errors
`render.type === "error"`:

- `render.error` → clean reason (no `"Error:"` prefix). Primary field to display.
- `render.text` → the same reason, may equal `"Error: <reason>"`. Backup display.
- `output` → model-facing `"Error: <reason>"`. Don't show directly in the error card; it's redundant.

### 6.3 Unknown `render.type`
Treat as `json`. The backend may introduce new types (e.g. `video`, `map`) before the client catches up.

### 6.4 `render` absent
Fall back to the string `output`. This is the pre-feature behavior; never crash on missing render.

### 6.5 Security
- **Never** set `render.text`, `render.model_content`, or any content-block text via `dangerouslySetInnerHTML`. These are model-authored.
- For `mcp_app`, don't iframe arbitrary `template_uri` values in v1 — show the URI, don't mount it.
- For HTML widgets (live widget type, not render type `html` — that doesn't exist in the render enum), follow the iframe sandbox rules in the widgets guide.

---

## 7. Integration points in a Next.js client

### 7.1 Message parts (streaming)

AI SDK v1 UI Message Stream surfaces tool calls as parts of type `tool-<name>`. The `render` field rides on the `tool-output-available` chunk alongside `output`.

Version check: some AI SDK versions strip unknown sibling fields on tool parts during message materialization. If `render` does not survive, intercept raw SSE chunks (`useChat({ experimental_prepareRequestBody: … })` or a custom fetcher) and keep a parallel `renderByCallId` map keyed on `toolCallId`.

```tsx
{message.parts.map((part, i) => {
  if (!part.type?.startsWith("tool-")) return null;
  const toolCallId = (part as any).toolCallId as string;
  const render: ToolRender | undefined =
    (part as any).render ?? renderByCallId[toolCallId];
  const output = (part as any).output as string | null;

  return (
    <ToolCallCard key={i} toolName={part.type.slice("tool-".length)} input={(part as any).input}>
      <ToolRenderView
        render={render}
        fallback={output}
        toolCallId={toolCallId}
        liveWidgets={message.messageMetadata?.live_widgets}
      />
    </ToolCallCard>
  );
})}
```

### 7.2 Initial messages (history)

```ts
const response = await fetch(`/ai/conversations/${conversationId}/messages`);
const { data } = await response.json();

// Build a tool_call_id → render index for fast lookup when iterating parts.
const messages: UIMessage[] = data.messages.map((m: AISDKUIMessage) => {
  const artifacts: ToolArtifact[] =
    m.messageMetadata?.tool_artifacts ?? [];
  return {
    ...m,
    // Depending on your setup, either:
    //   (a) materialize tool parts from artifacts so useChat renders them, or
    //   (b) keep artifacts as a side channel and look up by tool_call_id.
  };
});
```

Choose (a) if you want a single rendering code path for live and historical messages. Choose (b) if AI SDK's initial-message hydration doesn't round-trip tool parts for you.

### 7.3 State shape

Per-conversation derived state:

```ts
type ConversationClientState = {
  artifactByCallId: Record<string, ToolArtifact>;      // from messageMetadata.tool_artifacts
  liveWidgetsByMessageId: Record<string, LiveWidget[]>; // from messageMetadata.live_widgets
  renderByCallId: Record<string, ToolRender>;          // optional: streaming intercept when SDK strips render
};
```

All three are recomputed on every message update — do not treat them as the source of truth.

---

## 8. Testing plan

### Unit
- Dispatcher selects the correct renderer per `render.type`, including `live_widget` → `LiveWidgetMount`.
- Unknown `render.type` renders as JSON.
- `render` absent → `OutputFallback` renders `output`.
- Truncation markers render a visible indicator; no decode attempted on empty base64.
- `error` surfaces `render.error`, not `output`.

### Integration
- Mock SSE source emits a render-bearing `tool-output-available`; UI shows correct renderer within one render pass.
- History endpoint returns `tool_artifacts[].render`; replayed message looks identical to the streamed original.
- Turn that creates a live widget renders exactly one widget card (no duplicate from `render.structured_content`).
- Turn with both a static `table` render and a live `chart` widget renders both.

### End-to-end
- Backend contract tests (already in repo) plus client-side parity: `render` shape survives from `app/ai/tool_result_rendering.py` to the component's prop boundary.

---

## 9. Rollout order

Ship renderers in priority order so each step is independently shippable:

1. Types + dispatcher + `OutputFallback`
2. `TextBlock`, `ToolError`, `JsonViewer`
3. `TableRenderer`, `ChartRenderer`
4. `ImageRenderer`, `ResourceList`
5. `McpAppPreview` (non-iframe)
6. `LiveWidgetMount` — only after `live-widgets-frontend-integration.md` is implemented on your client

Each step is backward-compatible: before step N, unsupported types fall through to the JSON renderer without errors.

---

## 10. Checklist for implementers

- [ ] Add `ToolRender` / `ToolArtifact` types.
- [ ] Build a single dispatcher keyed on `render.type` with a `json` default.
- [ ] Wire dispatcher into the tool-call part renderer inside `useChat()` message loops.
- [ ] Wire dispatcher into the message-history hydration path.
- [ ] Handle missing `render` by rendering `output` verbatim.
- [ ] Render truncation markers (`_truncated`) with a visible "omitted X bytes" chip.
- [ ] Display `render.error` for error renders; never surface the prefixed `output` text directly.
- [ ] For `live_widget`, correlate via `widget_id` against `messageMetadata.live_widgets` and defer mount to the widgets guide.
- [ ] Do not inject any render content via `dangerouslySetInnerHTML`.
- [ ] For `mcp_app`, show metadata only (template URI + structured data); no iframe mount in v1.
- [ ] Guard against AI SDK versions that strip unknown tool-part fields; if so, keep a `renderByCallId` map from raw SSE.
- [ ] Verify (a) a widget-only turn renders exactly one widget card; (b) a static-render-only turn renders zero widget cards; (c) a mixed turn renders both without collision.

---

## References

- [`mcp_ui_plan.md`](mcp_ui_plan.md) — backend implementation plan for the render contract
- [`live-widgets-frontend-integration.md`](live-widgets-frontend-integration.md) — full widget mount + WebSocket contract
- [`live-ui-plan.md`](live-ui-plan.md) — backend widget architecture and decisions
