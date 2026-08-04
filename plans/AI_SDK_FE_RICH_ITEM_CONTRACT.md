# AI SDK and Streamlit Rich-Item Contract

**Status:** Normative production contract

**Version:** Rich response v1, updated 2026-08-03

This is the sole normative frontend contract for rich items, assistant images,
protected media, AI SDK rendering, and Streamlit parity. Images remain rich
items beside widgets, canvas artifacts, and tool results.

## 1. Core invariant

For any assistant message with `metadata.rich_items_version === 1`:

- `content` contains Markdown plus standalone `<!--rich:<id>-->` markers.
- `metadata.rich_items` is the final typed registry.
- Each marker resolves by exact `id` to one registry item.
- The final registry replaces transient state; it is never merged into it.
- `image` and `image_group` have `display_policy: "inline_only"` and render only
  at their marker.
- An unreferenced image is not appended as a gallery.
- Internal/unselected candidates are never sent to the client.

`image` and `image_group` are first-class `RichItem` variants, exactly like
`live_widget`, `canvas_artifact`, and `tool_render`. They are not a parallel
attachment system.

## 2. Capability and projection matrix

Send `inlineRichResponseV1: true` on chat/resume requests and
`?inlineRichResponseV1=true` on history requests when the client implements this
contract. `inline_rich_response_v1` is an accepted alias.

| Client path | Markers | Final `rich_items` | Image `file` parts | Authority |
|---|---:|---:|---:|---|
| Streamlit v1 stream/history | yes | yes | no | marker + rich registry |
| AI SDK rich-capable stream/history | yes | yes | no | marker + rich registry |
| AI SDK compatibility projection | no | no | selected images | standard AI SDK parts |
| Pre-v1 legacy message | no | no | legacy-compatible | legacy projection |

### AI SDK rich-capable

A rich-capable v1 response contains finalized image items in terminal
`data-assistant-message.data.message.metadata.rich_items` and history metadata.
It contains no selected-image `file` event and no selected-image `file` part.
Render the image only through its marker-resolved rich item.

### AI SDK compatibility

A client that omits the capability receives marker-free text and no rich
registry. Finalized selected images are projected as standard AI SDK file data.
During streaming, the normal `file` event builds the message part; the terminal
`data-assistant-message` snapshot may repeat the same selected part for
reconciliation. Do not render the nested snapshot as a separate visual tree.
Render the assembled `message.parts` once.

## 3. Canonical message shape

```json
{
  "id": "message-id",
  "role": "assistant",
  "content": "Answer paragraph.\n\n<!--rich:image:tool:call-1:0-->\n\nMore text.",
  "parts": [
    { "type": "text", "text": "Answer paragraph..." }
  ],
  "metadata": {
    "rich_items_version": 1,
    "rich_items": [],
    "rich_reference_warnings": []
  }
}
```

For rich-capable v1 messages, `parts` carries ordinary AI SDK text/tool state;
assistant images come from `metadata.rich_items`.

## 4. RichItem types

### Shared base

```ts
type DisplayPolicy = "inline_only" | "inline_or_append";

type RichItemBase = {
  id: string;
  type: string;
  display_policy: DisplayPolicy;
  title?: string;
  provenance?: Record<string, unknown>;
};
```

### type: "image"

```ts
type RichImageItem = RichItemBase & {
  type: "image";
  display_policy: "inline_only";
  alt_text: string;
  payload: {
    url: string;
    mime_type: string;
    source_url?: string;
    caption?: string;
    width?: number;
    height?: number;
  };
};
```

### type: "image_group"

```ts
type RichImageGroupItem = RichItemBase & {
  type: "image_group";
  display_policy: "inline_only";
  alt_text: string;
  payload: {
    items: Array<{
      url: string;
      mime_type: string;
      source_url?: string;
      description?: string;
      width?: number;
      height?: number;
    }>;
  };
};
```

A persisted group can contain one cell when registration of other cells failed.
Render that as a single-image layout.

### Widget, canvas, and tool result

```ts
type LiveWidgetItem = RichItemBase & {
  type: "live_widget";
  display_policy: "inline_or_append";
  payload: {
    widget_id: string;
    session_id: string;
    widget_type: string;
    status: string;
    version: number;
    connection_endpoint: string;
  };
};

type CanvasArtifactItem = RichItemBase & {
  type: "canvas_artifact";
  display_policy: "inline_or_append";
  payload: Record<string, unknown>;
};

type ToolRenderItem = RichItemBase & {
  type: "tool_render";
  display_policy: "inline_or_append";
  payload: Record<string, unknown>;
};

type RichItem =
  | RichImageItem
  | RichImageGroupItem
  | LiveWidgetItem
  | CanvasArtifactItem
  | ToolRenderItem;
```

Unknown future item types degrade to a neutral unavailable/unsupported block;
never render unknown payloads as trusted HTML.

## 5. Marker resolver

1. Accumulate streamed Markdown text.
2. Recognize standalone `<!--rich:<id>-->` marker blocks.
3. Resolve an id from the current registry map.
4. If a transient item has not arrived yet, reserve a neutral pending block.
5. At terminal `data-assistant-message`, replace the registry map with
   `message.metadata.rich_items`.
6. Re-render markers from the final registry.
7. Append only unreferenced items whose policy is `inline_or_append`.
8. Never append `inline_only` images.

Transient `data-rich-items` events contain safe non-image upserts only:

```json
{
  "type": "data-rich-items",
  "data": { "operation": "upsert", "items": [] },
  "transient": true
}
```

Images never arrive through this transient candidate channel. Final image items
arrive only after backend selection/externalization.

## 6. Streaming and history parity

Use the same marker parser, registry store, item components, and protected-media
loader for active streams and history.

- Stream: transient upserts may populate safe non-image items; terminal metadata
  replaces the map.
- History: initialize directly from final message metadata.
- Generated `data-image-preview` state is temporary. Clear it when terminal
  message state becomes authoritative.
- A resumed turn follows the same terminal reconciliation rule.

Final stream rendering and reloaded history must show the same selected items,
order, source attribution, and failure behavior.

## 7. Protected media

Supported authenticated paths:

- `/web-images/{id}`
- `/api/web-images/{id}`
- `/chat-images/{id}`
- `/api/chat-images/{id}`

Do not assign these paths directly to `<img src>`. A browser image request
cannot attach the application Bearer token. Fetch through the authenticated app
transport and render a temporary Blob URL.

```ts
const PROTECTED_PREFIXES = [
  "/web-images/",
  "/api/web-images/",
  "/chat-images/",
  "/api/chat-images/",
] as const;

export function isProtectedImageUrl(url: string): boolean {
  return PROTECTED_PREFIXES.some((prefix) => url.startsWith(prefix));
}

export async function loadProtectedImage(
  url: string,
  token: string,
  signal?: AbortSignal,
): Promise<string> {
  if (!isProtectedImageUrl(url)) throw new Error("visual_unavailable");

  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
    credentials: "same-origin",
    signal,
  });
  if (!response.ok) throw new Error("visual_unavailable");

  const blob = await response.blob();
  if (!blob.type.toLowerCase().startsWith("image/") || blob.size === 0) {
    throw new Error("visual_unavailable");
  }
  return URL.createObjectURL(blob);
}
```

Component lifecycle:

```ts
useEffect(() => {
  const controller = new AbortController();
  let objectUrl: string | undefined;

  loadProtectedImage(url, token, controller.signal)
    .then((loaded) => {
      objectUrl = loaded;
      setSrc(loaded);
      setState("loaded");
    })
    .catch((error) => {
      if (error?.name !== "AbortError") setState("failed");
    });

  return () => {
    controller.abort();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  };
}, [url, token]);
```

Revoke the previous object URL on URL/token change, replacement, message
removal, and component unmount. Never put tokens in query strings, DOM
attributes, logs, persistent storage, or shared cache keys.

## 8. Image rendering

### Single image

- Reserve aspect ratio when positive width and height are available.
- Render at natural size, capped by the answer column; do not upscale.
- Put `alt_text` only in the image `alt` attribute.
- Show `payload.caption` only when present as structured caption data.
- Show one source link derived from `payload.source_url`, with
  `target="_blank" rel="noopener noreferrer"`.
- If fetch/decode fails, omit the visual for that attempt; keep the text answer.

### Image group

- Keep persisted cell order.
- Load every cell independently.
- Keep successful siblings visible when one cell fails.
- Replace a failed cell in place with `Visual unavailable`.
- Remove/hide that failed cell's source caption.
- Do not collapse or reorder the row when every cell fails.
- Treat `description` as alt metadata, not a visible trusted caption.

## 9. Streamlit path

Streamlit advertises rich-response v1, resolves the same markers and registry,
and uses the local authenticated backend for protected references. It validates
non-empty `image/*` responses before encoding/rendering. Final stream metadata
and history use the same rich-response view. V1 messages never invoke the
legacy appended image gallery.

## 10. Failure semantics

| Result | Client behavior |
|---|---|
| `401` | Apply existing refresh/reauth policy; fail only the visual. |
| `404` | Neutral unavailable result; reveal no ownership/existence detail. |
| `413` | Media exceeded the sidecar bound; fail only the visual. |
| `502` / `504` | Upstream visual failed/timed out; fail only the visual. |
| Invalid/missing `image/*` MIME | Reject the visual. |
| Empty body or decode failure | Reject the visual/cell. |
| Abort/disconnect | Stop fetch and release object/upstream resources. |

No media failure may fail, replace, truncate, or roll back the assistant text.

## 11. Attribution and prohibited fallbacks

- `payload.url` is the protected media reference.
- `payload.source_url` is a clickable attribution link only.
- `provenance.original_image_url` is not part of the public wire contract.
- Never fetch or render publisher asset URLs as fallback media.
- Never expand CSP `img-src` to arbitrary publishers to mask a protected-route
  failure.
- Never expose unselected candidates or build a legacy gallery for v1 messages.

The restrictive media CSP is intentional defense in depth.

## 12. Deployment diagnostic

The current sidecar OpenAPI must include all four protected paths listed in
section 7. If `/web-images/{image_id}` or its `/api` alias is missing, the local
sidecar is stale: rebuild/reinstall and restart it. Do not change frontend CSP or
substitute a publisher asset URL as fallback media.

## 13. FE acceptance checklist

- [ ] Chat, resume, and history negotiate `inlineRichResponseV1` consistently.
- [ ] One rendered image exists per selected marker.
- [ ] Rich-capable messages render no image `file` duplicate.
- [ ] Compatibility messages render the assembled AI SDK file part once.
- [ ] Stream and history converge on the same final registry and order.
- [ ] Protected references use authenticated Blob loading.
- [ ] All Blob URLs are revoked.
- [ ] No unselected image gallery appears.
- [ ] Single-image failure leaves the answer intact.
- [ ] Group cell failure preserves successful siblings and cell order.
- [ ] Alt text, structured caption, and source attribution each have one owner.
- [ ] `original_image_url` is absent from stream and history payloads.
- [ ] No source/provenance fallback or token leakage exists.
- [ ] Missing `/web-images` is handled as a stale-sidecar deployment issue.
